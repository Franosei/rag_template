"""Lightweight graph-enhanced retrieval over indexed chunks."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field

from src.core.policies.folder_policy import FolderPolicy
from src.ingestion.models import Chunk
from src.retrieval.hybrid import RetrievalHit
from src.retrieval.tokenization import tokenize_text


class GraphPath(BaseModel):
    """Human-readable graph path connecting a query concept to evidence."""

    concept: str
    folder_id: str
    folder_name: str
    file_name: str
    chunk_id: str
    score: float


class GraphSearchResult(BaseModel):
    """Graph retrieval output and diagnostics."""

    query_concepts: list[str] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)


class GraphRetriever:
    """Build a lightweight concept graph and use it to expand retrieval."""

    def __init__(self) -> None:
        self.chunk_lookup: dict[str, Chunk] = {}
        self.folder_policies: dict[str, FolderPolicy] = {}
        self.doc_to_chunk_ids: dict[str, list[str]] = defaultdict(list)
        self.folder_to_chunk_ids: dict[str, list[str]] = defaultdict(list)
        self.concept_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
        self.chunk_to_concepts: dict[str, set[str]] = {}

    def build(self, chunks: list[Chunk], policies: list[FolderPolicy]) -> None:
        """Build the graph structures from chunk and policy data."""

        self.chunk_lookup = {}
        self.folder_policies = {policy.folder_id: policy for policy in policies}
        self.doc_to_chunk_ids = defaultdict(list)
        self.folder_to_chunk_ids = defaultdict(list)
        self.concept_to_chunk_ids = defaultdict(set)
        self.chunk_to_concepts = {}

        for chunk in chunks:
            self.chunk_lookup[chunk.chunk_id] = chunk
            self.doc_to_chunk_ids[chunk.source_doc_id].append(chunk.chunk_id)
            self.folder_to_chunk_ids[chunk.folder_id].append(chunk.chunk_id)

            policy = self.folder_policies.get(chunk.folder_id)
            concepts = self._extract_concepts(chunk.content_text, policy.key_topics if policy else [])
            self.chunk_to_concepts[chunk.chunk_id] = concepts
            for concept in concepts:
                self.concept_to_chunk_ids[concept].add(chunk.chunk_id)

    def search(
        self,
        query: str,
        *,
        folder_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> GraphSearchResult:
        """Search the concept graph and expand matches through document relationships."""

        query_concepts = sorted(self._extract_concepts(query, []))
        if not query_concepts or not self.chunk_lookup:
            return GraphSearchResult(query_concepts=query_concepts)

        chunk_scores: dict[str, float] = defaultdict(float)
        matched_concepts: dict[str, set[str]] = defaultdict(set)
        direct_matches: set[str] = set()
        paths: list[GraphPath] = []

        for concept in query_concepts:
            concept_weight = 1.0 + 0.35 * concept.count(" ")
            for chunk_id in self.concept_to_chunk_ids.get(concept, set()):
                chunk = self.chunk_lookup[chunk_id]
                if folder_ids and chunk.folder_id not in folder_ids:
                    continue
                adjusted_weight = concept_weight + self._quality_adjustment(chunk, concept, query)
                if adjusted_weight <= 0:
                    continue
                direct_matches.add(chunk_id)
                chunk_scores[chunk_id] += adjusted_weight
                matched_concepts[chunk_id].add(concept)
                policy = self.folder_policies.get(chunk.folder_id)
                paths.append(
                    GraphPath(
                        concept=concept,
                        folder_id=chunk.folder_id,
                        folder_name=policy.folder_name if policy else chunk.folder_id,
                        file_name=chunk.file_name,
                        chunk_id=chunk_id,
                        score=round(adjusted_weight, 4),
                    )
                )

        for chunk_id in list(direct_matches):
            chunk = self.chunk_lookup[chunk_id]
            sibling_ids = self.doc_to_chunk_ids.get(chunk.source_doc_id, [])
            for sibling_id in sibling_ids:
                if sibling_id == chunk_id:
                    continue
                sibling = self.chunk_lookup[sibling_id]
                if folder_ids and sibling.folder_id not in folder_ids:
                    continue
                if sibling.metadata.get("is_toc_like"):
                    continue
                distance = abs((chunk.chunk_index or 0) - (sibling.chunk_index or 0))
                expansion_weight = 0.25 if distance <= 2 else 0.08
                chunk_scores[sibling_id] += expansion_weight
                if matched_concepts.get(chunk_id):
                    matched_concepts[sibling_id].update(matched_concepts[chunk_id])

        hits: list[RetrievalHit] = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True):
            chunk = self.chunk_lookup[chunk_id]
            concepts = sorted(matched_concepts.get(chunk_id, set()))
            evidence_text = self._extract_evidence(chunk.content_text, query_concepts)
            evidence_terms = [concept for concept in concepts if concept in evidence_text.lower()]
            is_direct_match = chunk_id in direct_matches
            if self._is_boilerplate_evidence(evidence_text):
                continue
            if not is_direct_match and not evidence_terms:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    source_doc_id=chunk.source_doc_id,
                    folder_id=chunk.folder_id,
                    file_name=chunk.file_name,
                    file_path=chunk.file_path,
                    citation=chunk.get_citation(),
                    content=chunk.content_text,
                    excerpt=self._excerpt(evidence_text, query_concepts),
                    score=round(score, 6),
                    page_number=chunk.page_number,
                    modality=chunk.modality,
                    metadata={
                        "matched_concepts": concepts,
                        "matched_terms": evidence_terms,
                        "graph_match_type": "direct" if is_direct_match else "expanded",
                        "retrieval_mode": "graph",
                        "evidence_text": evidence_text,
                    },
                )
            )
            if len(hits) >= top_k:
                break

        return GraphSearchResult(
            query_concepts=query_concepts,
            hits=hits,
            paths=sorted(paths, key=lambda path: path.score, reverse=True)[:12],
        )

    def _extract_concepts(self, text: str, key_topics: list[str]) -> set[str]:
        """Extract graph concepts from text and policy topics."""

        normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
        tokens = tokenize_text(normalized)
        concepts = {token for token in tokens if len(token) >= 5}

        meaningful_tokens = [token for token in tokens if len(token) >= 4]
        for size in (2, 3):
            for index in range(len(meaningful_tokens) - size + 1):
                phrase = " ".join(meaningful_tokens[index : index + size])
                if len(phrase) <= 40:
                    concepts.add(phrase)

        for topic in key_topics:
            cleaned = re.sub(r"\s+", " ", topic.lower()).strip()
            if cleaned and cleaned in normalized:
                concepts.add(cleaned)

        return set(sorted(concepts)[:48])

    def _quality_adjustment(self, chunk: Chunk, concept: str, query: str) -> float:
        """Apply simple heuristics so graph retrieval favors real evidence over boilerplate."""

        text = (chunk.content_text or "").lower()
        adjustment = 0.0

        if chunk.metadata.get("is_toc_like"):
            adjustment -= 1.1
        if "table of contents" in text or "contents" in text:
            adjustment -= 0.65
        if concept and f"{concept}:" in text:
            adjustment += 0.3
        if concept and f"{concept} is" in text:
            adjustment += 0.2
        if query.lower() in text:
            adjustment += 0.08

        return adjustment

    def _extract_evidence(self, text: str, query_concepts: list[str]) -> str:
        """Extract the best paragraph or sentence for citation display."""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

        if not paragraphs:
            return text.strip()

        scored: list[tuple[float, str]] = []
        for paragraph in paragraphs:
            lowered = paragraph.lower()
            score = float(sum(1 for concept in query_concepts if concept in lowered))
            if any(f"{concept}:" in lowered for concept in query_concepts):
                score += 2.0
            if any(f"{concept} is" in lowered for concept in query_concepts):
                score += 1.5
            if 60 <= len(paragraph) <= 420:
                score += 0.4
            if "contents" in lowered:
                score -= 1.3
            if paragraph.count(".") + paragraph.count(":") + paragraph.count(";") == 0:
                score -= 0.9
            if lowered.startswith("ich ") and "guideline" in lowered:
                score -= 1.1
            scored.append((score, paragraph))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1].strip()

    def _is_boilerplate_evidence(self, text: str) -> bool:
        """Detect repeated document-title text that should not surface as evidence."""

        lowered = (text or "").lower().strip()
        punctuation_count = text.count(".") + text.count(":") + text.count(";")
        if punctuation_count == 0 and lowered.startswith("ich ") and "guideline" in lowered:
            return True
        if punctuation_count == 0 and "statistical principles for clinical trials" in lowered:
            return True
        return False

    def _excerpt(self, text: str, query_concepts: list[str], limit: int = 280) -> str:
        """Return a snippet centered on the first matched concept."""

        compact = re.sub(r"\s+", " ", text).strip()
        lower = compact.lower()
        for concept in query_concepts:
            index = lower.find(concept)
            if index >= 0:
                start = max(index - 80, 0)
                end = min(index + limit, len(compact))
                return compact[start:end].strip()
        return compact[:limit].strip()
