"""Semantic retrieval over indexed chunks using cached embeddings."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from pydantic import BaseModel

from src.app.settings import settings
from src.core.policies.folder_policy import AuthorityLevel, FolderPolicy
from src.ingestion.models import Chunk, Modality
from src.llm.client import LLMClient, LLMError
from src.retrieval.hybrid import RetrievalHit
from src.retrieval.tokenization import tokenize_text
from src.utils.fileio import load_json
from src.utils.fileio import save_json

logger = logging.getLogger(__name__)


class _SemanticRecord(BaseModel):
    """Internal semantic index record."""

    chunk: Chunk
    embedding: list[float]


class SemanticRetriever:
    """Embedding-backed semantic retrieval with a local JSON cache."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None,
        cache_path: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.cache_path = cache_path or settings.semantic_cache_path
        self.embedding_model = settings.openai_embedding_model
        self.batch_size = settings.embedding_batch_size
        self.folder_policies: dict[str, FolderPolicy] = {}
        self.records: list[_SemanticRecord] = []
        self.last_error: str | None = None

    @property
    def is_enabled(self) -> bool:
        """Return whether semantic retrieval is configured."""

        return bool(settings.semantic_retrieval_enabled and self.llm_client and self.llm_client.is_configured)

    @property
    def is_ready(self) -> bool:
        """Return whether the retriever has usable embeddings in memory."""

        return self.is_enabled and bool(self.records)

    def build(self, chunks: list[Chunk], policies: list[FolderPolicy]) -> None:
        """Build or refresh the semantic index from chunk data."""

        self.folder_policies = {policy.folder_id: policy for policy in policies}
        self.records = []
        self.last_error = None

        if not self.is_enabled:
            return

        cache = self._load_cache()
        missing_chunks: list[Chunk] = []
        for chunk in chunks:
            if not cache.get(chunk.content_hash):
                missing_chunks.append(chunk)

        if missing_chunks:
            try:
                self._embed_missing_chunks(cache, missing_chunks)
            except LLMError as exc:
                self.last_error = str(exc)
                logger.warning("Semantic embedding build failed", extra={"error": self.last_error})

        for chunk in chunks:
            embedding = cache.get(chunk.content_hash)
            if embedding:
                self.records.append(_SemanticRecord(chunk=chunk, embedding=embedding))

        if cache:
            self._save_cache(cache)

    def search(
        self,
        query: str,
        *,
        folder_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> list[RetrievalHit]:
        """Search indexed chunks by semantic similarity."""

        if not self.is_ready:
            return []

        try:
            query_embedding = self.llm_client.embed_texts(
                [self._embedding_text(query)],
                model=self.embedding_model,
            )[0]
        except LLMError as exc:
            self.last_error = str(exc)
            logger.warning("Semantic query embedding failed", extra={"error": self.last_error})
            return []

        query_terms = self._query_terms(query)
        query_subject = self._query_subject(query, query_terms)

        scored_records: list[tuple[Chunk, float, float]] = []
        max_similarity = 0.0
        for record in self.records:
            if folder_ids and record.chunk.folder_id not in folder_ids:
                continue

            similarity = self._cosine_similarity(query_embedding, record.embedding)
            if similarity <= 0.0:
                continue

            quality_adjustment = self._quality_adjustment(record.chunk, query, query_subject)
            max_similarity = max(max_similarity, similarity)
            scored_records.append((record.chunk, similarity, quality_adjustment))

        hits: list[RetrievalHit] = []
        for chunk, similarity, quality_adjustment in scored_records:
            evidence_text, matched_terms = self._extract_evidence(chunk.content_text, query_terms, query_subject)
            normalized_similarity = similarity / max_similarity if max_similarity else 0.0
            final_score = normalized_similarity + quality_adjustment
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    source_doc_id=chunk.source_doc_id,
                    folder_id=chunk.folder_id,
                    file_name=chunk.file_name,
                    file_path=chunk.file_path,
                    citation=chunk.get_citation(),
                    content=chunk.content_text,
                    excerpt=self._excerpt(evidence_text or chunk.content_text, query_terms),
                    score=round(final_score, 6),
                    page_number=chunk.page_number,
                    modality=chunk.modality,
                    metadata={
                        **chunk.metadata,
                        "retrieval_mode": "semantic",
                        "matched_terms": matched_terms,
                        "evidence_text": evidence_text,
                        "semantic_similarity": round(similarity, 6),
                        "quality_adjustment": round(quality_adjustment, 6),
                    },
                )
            )

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def _load_cache(self) -> dict[str, list[float]]:
        """Load the semantic embedding cache from disk."""

        if not self.cache_path.exists():
            return {}

        try:
            payload = load_json(self.cache_path)
        except Exception:
            logger.warning("Could not load semantic cache", extra={"path": str(self.cache_path)})
            return {}

        if payload.get("model") != self.embedding_model:
            return {}
        embeddings = payload.get("embeddings", {})
        return {str(key): list(map(float, value)) for key, value in embeddings.items() if value}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        """Persist the semantic embedding cache."""

        save_json({"model": self.embedding_model, "embeddings": cache}, self.cache_path)

    def _embed_missing_chunks(self, cache: dict[str, list[float]], chunks: list[Chunk]) -> None:
        """Request embeddings for chunks that are not yet cached."""

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [self._embedding_text(chunk.content_text) for chunk in batch]
            embeddings = self.llm_client.embed_texts(texts, model=self.embedding_model)
            for chunk, embedding in zip(batch, embeddings, strict=True):
                cache[chunk.content_hash] = embedding

    def _embedding_text(self, text: str) -> str:
        """Normalize text before sending it to the embedding model."""

        compact = re.sub(r"\s+", " ", text or "").strip()
        return compact[:4000]

    def _query_terms(self, query: str) -> list[str]:
        """Extract useful query terms for evidence presentation."""

        return tokenize_text(query, min_length=4)

    def _query_subject(self, query: str, query_terms: list[str]) -> str | None:
        """Extract the main subject of a question when possible."""

        lower = query.lower().strip()
        for prefix in ("what is ", "define ", "what are ", "tell me about ", "explain "):
            if lower.startswith(prefix):
                subject = re.sub(r"[?]+$", "", lower[len(prefix) :]).strip()
                return subject or None
        if query_terms:
            return query_terms[0]
        return None

    def _quality_adjustment(self, chunk: Chunk, query: str, query_subject: str | None) -> float:
        """Bias ranking toward authoritative, non-boilerplate evidence."""

        text = (chunk.content_text or "").lower()
        adjustment = self._policy_boost(chunk.folder_id, chunk.modality)

        if chunk.metadata.get("is_toc_like"):
            adjustment -= 0.5
        if "table of contents" in text or "contents" in text:
            adjustment -= 0.35
        if query_subject and f"{query_subject}:" in text:
            adjustment += 0.28
        if query_subject and f"{query_subject} is" in text:
            adjustment += 0.18
        if query.lower() in text:
            adjustment += 0.08

        return adjustment

    def _policy_boost(self, folder_id: str, modality: Modality) -> float:
        """Return a modest source-quality boost."""

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

    def _extract_evidence(
        self,
        text: str,
        query_terms: list[str],
        query_subject: str | None,
    ) -> tuple[str, list[str]]:
        """Extract the best human-readable passage for display and synthesis."""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

        if not paragraphs:
            return text.strip(), []

        scored: list[tuple[float, str, list[str]]] = []
        for paragraph in paragraphs:
            lowered = paragraph.lower()
            matched_terms = sorted({term for term in query_terms if term in lowered})
            score = float(len(matched_terms))
            if query_subject and f"{query_subject}:" in lowered:
                score += 2.3
            if query_subject and f"{query_subject} is" in lowered:
                score += 1.7
            if 80 <= len(paragraph) <= 520:
                score += 0.4
            if "contents" in lowered:
                score -= 1.2
            if paragraph.count(".") + paragraph.count(":") + paragraph.count(";") == 0:
                score -= 0.8
            if lowered.startswith("ich ") and "guideline" in lowered:
                score -= 1.0
            scored.append((score, paragraph, matched_terms))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_paragraph, best_terms = scored[0]
        if best_score <= 0:
            return text.strip(), []
        return best_paragraph.strip(), best_terms

    def _excerpt(self, text: str, query_terms: list[str], limit: int = 320) -> str:
        """Return a compact snippet centered on the first matched term."""

        compact = re.sub(r"\s+", " ", text).strip()
        lower = compact.lower()
        for term in query_terms:
            index = lower.find(term)
            if index >= 0:
                start = max(index - 90, 0)
                end = min(index + limit, len(compact))
                return compact[start:end].strip()
        return compact[:limit].strip()

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""

        if not left or not right or len(left) != len(right):
            return 0.0

        numerator = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right, strict=True):
            numerator += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value

        denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
        if denominator == 0.0:
            return 0.0
        return numerator / denominator
