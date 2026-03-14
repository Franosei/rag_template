"""Hybrid lexical retrieval over processed chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.policies.folder_policy import AuthorityLevel, FolderPolicy
from src.ingestion.models import Chunk, Modality
from src.retrieval.tokenization import tokenize_text
from src.utils.fileio import load_json


class RetrievalHit(BaseModel):
    """Search result returned to the orchestrator and API layer."""

    chunk_id: str
    source_doc_id: str
    folder_id: str
    file_name: str
    file_path: str
    citation: str
    content: str
    excerpt: str
    score: float
    page_number: int | None = None
    modality: Modality
    metadata: dict[str, object] = Field(default_factory=dict)


class _ChunkRecord(BaseModel):
    """Internal sparse-vector record for one chunk."""

    chunk: Chunk
    tokens: list[str]
    term_counts: dict[str, int]
    length: int


class HybridRetriever:
    """Pure-Python hybrid retriever combining TF-IDF and BM25-style scoring."""

    def __init__(self) -> None:
        self.records: list[_ChunkRecord] = []
        self.folder_policies: dict[str, FolderPolicy] = {}
        self.idf: dict[str, float] = {}
        self.avg_doc_length: float = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into normalized search terms."""

        return tokenize_text(text)

    def rebuild(self, chunk_files: list[Path], policies: list[FolderPolicy]) -> None:
        """Load chunk files from disk and rebuild the in-memory index."""

        chunks: list[Chunk] = []
        for chunk_file in chunk_files:
            for payload in load_json(chunk_file):
                chunks.append(Chunk.model_validate(payload))
        self.build(chunks, policies)

    def build(self, chunks: list[Chunk], policies: list[FolderPolicy]) -> None:
        """Build the sparse index from chunk data."""

        self.folder_policies = {policy.folder_id: policy for policy in policies}
        self.records = []
        document_frequency: Counter[str] = Counter()
        total_length = 0

        for chunk in chunks:
            tokens = self.tokenize(chunk.content_text)
            counts = Counter(tokens)
            total_length += len(tokens)
            self.records.append(
                _ChunkRecord(
                    chunk=chunk,
                    tokens=tokens,
                    term_counts=dict(counts),
                    length=len(tokens),
                )
            )
            for token in counts:
                document_frequency[token] += 1

        corpus_size = max(len(self.records), 1)
        self.avg_doc_length = total_length / corpus_size if self.records else 0.0
        self.idf = {
            token: math.log((corpus_size + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        folder_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> list[RetrievalHit]:
        """Run hybrid retrieval and return ranked hits."""

        query_tokens = self.tokenize(query)
        if not query_tokens or not self.records:
            return []

        query_counts = Counter(query_tokens)
        query_norm = math.sqrt(
            sum((query_counts[token] * self.idf.get(token, 0.0)) ** 2 for token in query_counts)
        ) or 1.0
        query_subject = self._query_subject(query, query_tokens)

        candidate_rows: list[tuple[_ChunkRecord, float, float, float]] = []
        max_dense = 0.0
        max_sparse = 0.0

        for record in self.records:
            if folder_ids and record.chunk.folder_id not in folder_ids:
                continue
            overlap = set(query_counts).intersection(record.term_counts)
            if not overlap:
                continue

            dense_score = self._dense_score(record, query_counts, query_norm)
            sparse_score = self._bm25_score(record, query_counts)
            quality_adjustment = self._quality_adjustment(record.chunk, query, query_subject)

            max_dense = max(max_dense, dense_score)
            max_sparse = max(max_sparse, sparse_score)
            candidate_rows.append((record, dense_score, sparse_score, quality_adjustment))

        results: list[RetrievalHit] = []
        for record, dense_score, sparse_score, quality_adjustment in candidate_rows:
            policy = self.folder_policies.get(record.chunk.folder_id)
            dense_weight, sparse_weight = (0.55, 0.45) if policy is None else policy.retrieval_weights()
            normalized_dense = dense_score / max_dense if max_dense else 0.0
            normalized_sparse = sparse_score / max_sparse if max_sparse else 0.0
            final_score = dense_weight * normalized_dense + sparse_weight * normalized_sparse + quality_adjustment
            evidence_text, matched_terms = self._extract_evidence(record.chunk.content_text, query_tokens, query_subject)

            results.append(
                RetrievalHit(
                    chunk_id=record.chunk.chunk_id,
                    source_doc_id=record.chunk.source_doc_id,
                    folder_id=record.chunk.folder_id,
                    file_name=record.chunk.file_name,
                    file_path=record.chunk.file_path,
                    citation=record.chunk.get_citation(),
                    content=record.chunk.content_text,
                    excerpt=self._excerpt(evidence_text or record.chunk.content_text, query_tokens),
                    score=round(final_score, 6),
                    page_number=record.chunk.page_number,
                    modality=record.chunk.modality,
                    metadata={
                        **record.chunk.metadata,
                        "retrieval_mode": "hybrid",
                        "matched_terms": matched_terms,
                        "evidence_text": evidence_text,
                        "quality_adjustment": round(quality_adjustment, 6),
                    },
                )
            )

        return sorted(results, key=lambda hit: hit.score, reverse=True)[:top_k]

    def _dense_score(self, record: _ChunkRecord, query_counts: Counter[str], query_norm: float) -> float:
        """Compute cosine similarity over sparse TF-IDF vectors."""

        numerator = 0.0
        document_norm = 0.0
        for token, frequency in record.term_counts.items():
            weight = frequency * self.idf.get(token, 0.0)
            document_norm += weight * weight
            if token in query_counts:
                numerator += weight * query_counts[token] * self.idf.get(token, 0.0)
        denominator = math.sqrt(document_norm) * query_norm or 1.0
        return numerator / denominator

    def _bm25_score(self, record: _ChunkRecord, query_counts: Counter[str]) -> float:
        """Compute a BM25-style sparse retrieval score."""

        k1 = 1.5
        b = 0.75
        score = 0.0
        average_length = self.avg_doc_length or 1.0
        for token in query_counts:
            term_frequency = record.term_counts.get(token, 0)
            if term_frequency == 0:
                continue
            idf = self.idf.get(token, 0.0)
            denominator = term_frequency + k1 * (1 - b + b * (record.length / average_length))
            score += idf * ((term_frequency * (k1 + 1)) / denominator)
        return score

    def _quality_adjustment(self, chunk: Chunk, query: str, query_subject: str | None) -> float:
        """Return a content-quality adjustment for ranking."""

        text = chunk.content_text.lower()
        adjustment = self._policy_boost(chunk.folder_id, chunk.modality)

        if chunk.metadata.get("is_toc_like"):
            adjustment -= 0.45
        if query.lower() in text:
            adjustment += 0.14
        if query_subject and f"{query_subject}:" in text:
            adjustment += 0.34
        if query_subject and f"{query_subject} is" in text:
            adjustment += 0.22
        if query_subject and "glossary" in text:
            adjustment += 0.12
        if "table of contents" in text or "contents" in text:
            adjustment -= 0.35
        if text.count(".") < 1 and len(text) < 120:
            adjustment -= 0.1

        return adjustment

    def _policy_boost(self, folder_id: str, modality: Modality) -> float:
        """Return a small authority- and modality-aware boost."""

        policy = self.folder_policies.get(folder_id)
        if policy is None:
            return 0.0

        boost = 0.0
        if policy.authority_level == AuthorityLevel.PRIMARY:
            boost += 0.08
        elif policy.authority_level == AuthorityLevel.SECONDARY:
            boost += 0.03
        if modality == Modality.TABLE and policy.has_tables:
            boost += 0.02
        return boost

    def _query_subject(self, query: str, query_tokens: list[str]) -> str | None:
        """Extract the main subject of a natural-language query when possible."""

        lower = query.lower().strip()
        for prefix in ("what is ", "define ", "what are ", "tell me about "):
            if lower.startswith(prefix):
                subject = re.sub(r"[?]+$", "", lower[len(prefix) :]).strip()
                return subject or None
        if query_tokens:
            return query_tokens[0]
        return None

    def _extract_evidence(
        self,
        text: str,
        query_tokens: list[str],
        query_subject: str | None,
    ) -> tuple[str, list[str]]:
        """Extract the best supporting passage from a chunk."""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

        if not paragraphs:
            return text.strip(), []

        scored: list[tuple[float, str, list[str]]] = []
        for paragraph in paragraphs:
            lowered = paragraph.lower()
            matched_terms = sorted({token for token in query_tokens if token in lowered})
            score = float(len(matched_terms))
            if query_subject and f"{query_subject}:" in lowered:
                score += 2.4
            if query_subject and f"{query_subject} is" in lowered:
                score += 1.8
            if 60 <= len(paragraph) <= 420:
                score += 0.5
            if "contents" in lowered:
                score -= 1.2
            if paragraph.count(".") + paragraph.count(":") + paragraph.count(";") == 0:
                score -= 0.9
            if lowered.startswith("ich ") and "guideline" in lowered:
                score -= 1.1
            scored.append((score, paragraph, matched_terms))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_paragraph, best_terms = scored[0]
        if best_score <= 0:
            return text.strip(), []
        return best_paragraph.strip(), best_terms

    def _excerpt(self, text: str, query_tokens: list[str], limit: int = 280) -> str:
        """Return a compact snippet centered on the first matched term."""

        compact = re.sub(r"\s+", " ", text).strip()
        lower = compact.lower()
        for token in query_tokens:
            index = lower.find(token)
            if index >= 0:
                start = max(index - 80, 0)
                end = min(index + limit, len(compact))
                return compact[start:end].strip()
        return compact[:limit].strip()
