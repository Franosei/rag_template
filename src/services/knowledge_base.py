"""Application service for ingestion, indexing, and query orchestration."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from src.agents.folder_profiler import FolderProfilerAgent
from src.app.settings import settings
from src.core.graph.schema import EdgeType, NodeType
from src.core.orchestration.state import RunState
from src.core.policies.folder_policy import FolderPolicy
from src.core.policies.registry import FolderPolicyRegistry
from src.ingestion.models import Chunk
from src.ingestion.pipeline import IngestionPipeline
from src.llm.client import LLMClient, LLMError
from src.retrieval.graph import GraphRetriever
from src.retrieval.hybrid import HybridRetriever, RetrievalHit
from src.retrieval.semantic import SemanticRetriever
from src.utils.fileio import load_json
from src.utils.fileio import save_json

logger = logging.getLogger(__name__)

_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")
_SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md", ".rst"}


class SourceIngestionResult(BaseModel):
    """Summary returned after ingesting a source."""

    folder_id: str
    folder_name: str
    folder_path: str
    summary: str
    total_documents: int
    total_chunks: int
    key_topics: list[str] = Field(default_factory=list)


class KnowledgeBaseService:
    """Coordinate ingestion, indexing, and citation-aware answering."""

    def __init__(self) -> None:
        self.settings = settings
        self.registry = FolderPolicyRegistry(self.settings.folder_registry_dir)
        self.profiler = FolderProfilerAgent()
        self.llm_client = LLMClient() if self.settings.llm_enabled else None
        self.pipeline = IngestionPipeline(llm_client=self.llm_client, data_dir=self.settings.data_dir)
        self.semantic_retriever = SemanticRetriever(
            llm_client=self.llm_client,
            cache_path=self.settings.semantic_cache_path,
        )
        self.hybrid_retriever = HybridRetriever()
        self.graph_retriever = GraphRetriever()
        self._lock = asyncio.Lock()

    async def bootstrap(self) -> dict[str, object]:
        """Ingest the bundled corpus and rebuild the in-memory index."""

        async with self._lock:
            results: list[SourceIngestionResult] = []
            for folder in self._discover_inbound_folders():
                result = await self._profile_and_ingest(folder)
                results.append(result)
            self._rebuild_index()
            return {
                "bootstrapped_folders": len(results),
                "folders": [result.model_dump(mode="json") for result in results],
                "stats": self.stats(),
            }

    async def ingest_seed_dataset(self) -> dict[str, object]:
        """Explicitly bootstrap the bundled sample data."""

        return await self.bootstrap()

    async def ingest_local_path(self, local_path: str, folder_name: str | None = None) -> SourceIngestionResult:
        """Copy a safe local file or directory into managed storage and ingest it."""

        source_path = self.settings.resolve_path(local_path)
        self._assert_allowed_local_path(source_path)
        target_folder = self._create_managed_folder(folder_name or source_path.stem)

        if source_path.is_dir():
            shutil.copytree(source_path, target_folder, dirs_exist_ok=True)
        elif source_path.is_file():
            shutil.copy2(source_path, target_folder / source_path.name)
        else:
            raise FileNotFoundError(f"Local path does not exist: {source_path}")

        async with self._lock:
            result = await self._profile_and_ingest(target_folder)
            self._rebuild_index()
            return result

    async def ingest_remote_url(self, url: str, folder_name: str | None = None) -> SourceIngestionResult:
        """Download a remote file into managed storage and ingest it."""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Remote sources must use http or https URLs.")

        filename = Path(parsed.path).name or "downloaded_source.pdf"
        self._assert_supported_extension(filename)
        target_folder = self._create_managed_folder(folder_name or Path(filename).stem)
        await self._download_file(url, target_folder / filename)

        async with self._lock:
            result = await self._profile_and_ingest(target_folder)
            self._rebuild_index()
            return result

    async def ingest_api_manifest(self, manifest_url: str, folder_name: str | None = None) -> SourceIngestionResult:
        """Download multiple files from an API-delivered manifest and ingest them together."""

        parsed = urlparse(manifest_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Manifest sources must use http or https URLs.")

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(manifest_url)
            response.raise_for_status()
            payload = response.json()

        if isinstance(payload, dict):
            documents = payload.get("documents", [])
        else:
            documents = payload
        if not isinstance(documents, list):
            raise ValueError("Manifest payload must be a list or an object with a 'documents' list.")
        if len(documents) > self.settings.max_manifest_items:
            raise ValueError(f"Manifest exceeds the limit of {self.settings.max_manifest_items} documents.")

        target_folder = self._create_managed_folder(folder_name or "api_manifest")
        for item in documents:
            if isinstance(item, str):
                url = item
                filename = Path(urlparse(url).path).name
            elif isinstance(item, dict):
                url = str(item.get("url", ""))
                filename = str(item.get("filename") or Path(urlparse(url).path).name)
            else:
                raise ValueError("Manifest entries must be strings or objects with a 'url'.")

            if not url:
                raise ValueError("Manifest entry is missing a URL.")
            self._assert_supported_extension(filename)
            await self._download_file(url, target_folder / filename)

        async with self._lock:
            result = await self._profile_and_ingest(target_folder)
            self._rebuild_index()
            return result

    async def ingest_inline_upload(
        self,
        *,
        filename: str,
        content_base64: str,
        folder_name: str | None = None,
    ) -> SourceIngestionResult:
        """Persist a browser-uploaded file and ingest it."""

        self._assert_supported_extension(filename)
        decoded = base64.b64decode(content_base64, validate=True)
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(decoded) > max_bytes:
            raise ValueError(f"Uploaded file exceeds the {self.settings.max_upload_mb} MB limit.")

        target_folder = self._create_managed_folder(folder_name or Path(filename).stem)
        target_file = target_folder / self._safe_name(filename)
        target_file.write_bytes(decoded)

        async with self._lock:
            result = await self._profile_and_ingest(target_folder)
            self._rebuild_index()
            return result

    def list_sources(self) -> list[dict[str, object]]:
        """Return indexed folder metadata for the UI."""

        return [policy.model_dump(mode="json") for policy in self.registry.list()]

    def stats(self) -> dict[str, object]:
        """Return a compact operational status summary."""

        policies = self.registry.list()
        chunk_files = list(self.settings.processed_dir.glob("*/chunks.json"))
        chunk_count = 0
        for path in chunk_files:
            try:
                chunk_count += len(load_json(path))
            except Exception:
                continue
        return {
            "folder_count": len(policies),
            "indexed_chunk_files": len(chunk_files),
            "indexed_chunks": chunk_count,
            "llm_enabled": bool(self.llm_client and self.llm_client.is_configured),
            "semantic_retrieval_enabled": self.semantic_retriever.is_enabled,
            "semantic_retrieval_ready": self.semantic_retriever.is_ready,
            "graph_retrieval_enabled": True,
            "allowed_local_roots": [str(path) for path in self.settings.allowed_local_root_paths],
        }

    async def query(
        self,
        question: str,
        *,
        folder_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, object]:
        """Run the hybrid agentic retrieval flow and return an answer with citations."""

        question = self._normalize_text(question.strip())
        if not question:
            raise ValueError("A question is required.")

        if not self.registry.list() and self.settings.auto_bootstrap_sample_data:
            await self.bootstrap()

        run_state = RunState(query=question, max_retrieval_results=top_k or self.settings.default_top_k)
        query_node = run_state.graph.add_simple_node(NodeType.QUERY, "User Query", data={"query": question})

        selected_policies = self._select_policies(question, folder_ids)
        if folder_ids and not selected_policies:
            raise ValueError("None of the requested folder IDs are indexed.")
        run_state.selected_folders = selected_policies
        scope_node = run_state.graph.add_simple_node(
            NodeType.AGENT,
            "Scope Selection",
            data={"folders": [policy.folder_name for policy in selected_policies]},
            agent_name="ScopeSelector",
        )
        run_state.graph.link(query_node, scope_node, EdgeType.TRIGGERS)

        semantic_hits = self.semantic_retriever.search(
            question,
            folder_ids=[policy.folder_id for policy in selected_policies] or None,
            top_k=top_k or self.settings.default_top_k,
        )
        semantic_node = run_state.graph.add_simple_node(
            NodeType.RETRIEVAL,
            "Semantic Retrieval",
            data={"hit_count": len(semantic_hits), "embedding_model": self.settings.openai_embedding_model},
        )
        run_state.graph.link(scope_node, semantic_node, EdgeType.RETRIEVES)

        hybrid_hits = self.hybrid_retriever.search(
            question,
            folder_ids=[policy.folder_id for policy in selected_policies] or None,
            top_k=top_k or self.settings.default_top_k,
        )
        hybrid_node = run_state.graph.add_simple_node(
            NodeType.RETRIEVAL,
            "Hybrid Retrieval",
            data={"hit_count": len(hybrid_hits)},
        )
        run_state.graph.link(scope_node, hybrid_node, EdgeType.RETRIEVES)

        graph_result = self.graph_retriever.search(
            question,
            folder_ids=[policy.folder_id for policy in selected_policies] or None,
            top_k=top_k or self.settings.default_top_k,
        )
        graph_node = run_state.graph.add_simple_node(
            NodeType.RETRIEVAL,
            "Graph Retrieval",
            data={
                "hit_count": len(graph_result.hits),
                "query_concepts": graph_result.query_concepts[:8],
            },
        )
        run_state.graph.link(scope_node, graph_node, EdgeType.RETRIEVES)

        hits = self._merge_hits(
            semantic_hits,
            hybrid_hits,
            graph_result.hits,
            top_k or self.settings.default_top_k,
        )
        run_state.retrieved_chunks = [hit.model_dump(mode="json") for hit in hits]
        fusion_node = run_state.graph.add_simple_node(
            NodeType.RETRIEVAL,
            "Retrieval Fusion",
            data={
                "semantic_hits": len(semantic_hits),
                "hybrid_hits": len(hybrid_hits),
                "graph_hits": len(graph_result.hits),
                "final_hits": len(hits),
            },
        )
        run_state.graph.link(semantic_node, fusion_node, EdgeType.SYNTHESIZES)
        run_state.graph.link(hybrid_node, fusion_node, EdgeType.SYNTHESIZES)
        run_state.graph.link(graph_node, fusion_node, EdgeType.SYNTHESIZES)
        run_state.graph.total_chunks_retrieved = len(hits)

        answer, citations, warnings = await self._synthesize_answer(question, hits)
        for warning in warnings:
            run_state.add_warning(warning)

        run_state.answer = answer
        run_state.citations = citations
        answer_node = run_state.graph.add_simple_node(
            NodeType.ANSWER,
            "Answer",
            data={"citation_count": len(citations)},
        )
        run_state.graph.link(fusion_node, answer_node, EdgeType.SYNTHESIZES)

        trace_file = self.settings.runs_dir / f"{run_state.run_id}.json"
        save_json(
            {
                "run_id": run_state.run_id,
                "query": question,
                "answer": answer,
                "citations": citations,
                "warnings": run_state.warnings,
                "graph": run_state.graph.to_dict(),
            },
            trace_file,
        )

        return {
            "run_id": run_state.run_id,
            "answer": answer,
            "citations": citations,
            "warnings": run_state.warnings,
            "selected_folders": [policy.model_dump(mode="json") for policy in selected_policies],
            "retrieval_hits": [hit.model_dump(mode="json") for hit in hits],
            "retrieval_diagnostics": {
                "strategy": "semantic_hybrid_graph" if semantic_hits else "hybrid_plus_graph",
                "query_concepts": graph_result.query_concepts,
                "semantic_hits": [self._compact_hit(hit) for hit in semantic_hits[:5]],
                "hybrid_hits": [self._compact_hit(hit) for hit in hybrid_hits[:5]],
                "graph_hits": [self._compact_hit(hit) for hit in graph_result.hits[:5]],
                "graph_paths": [path.model_dump(mode="json") for path in graph_result.paths],
            },
            "trace": run_state.graph.to_dict(),
        }

    async def _profile_and_ingest(self, folder_path: Path) -> SourceIngestionResult:
        """Profile a folder, ingest it, and persist its policy."""

        policy = await self.profiler.profile_folder(folder_path)
        stats = await self.pipeline.ingest_folder(folder_path, policy)
        policy.total_chunks = int(stats["total_chunks"])
        policy.updated_at = datetime.now(tz=timezone.utc)
        self.registry.upsert(policy)

        return SourceIngestionResult(
            folder_id=policy.folder_id,
            folder_name=policy.folder_name,
            folder_path=policy.folder_path,
            summary=policy.summary,
            total_documents=policy.total_documents,
            total_chunks=policy.total_chunks,
            key_topics=policy.key_topics,
        )

    def _rebuild_index(self) -> None:
        """Refresh the in-memory hybrid index from processed chunk files."""

        policies = self.registry.list()
        chunks = self._load_all_chunks()
        self.semantic_retriever.build(chunks, policies)
        self.hybrid_retriever.build(chunks, policies)
        self.graph_retriever.build(chunks, policies)

    def _discover_inbound_folders(self) -> list[Path]:
        """Return immediate child folders plus the root if it contains files."""

        inbound_dir = self.settings.inbound_dir
        folders = [path for path in sorted(inbound_dir.iterdir()) if path.is_dir()]
        has_root_files = any(
            item.is_file() and item.suffix.lower() in _SUPPORTED_SOURCE_EXTENSIONS for item in inbound_dir.iterdir()
        )
        if has_root_files:
            folders.insert(0, inbound_dir)
        return folders

    def _select_policies(self, question: str, folder_ids: list[str] | None) -> list[FolderPolicy]:
        """Select folders either explicitly or via lightweight scope ranking."""

        policies = self.registry.list()
        if folder_ids:
            selected = [policy for policy in policies if policy.folder_id in set(folder_ids)]
            return selected

        question_tokens = set(self.hybrid_retriever.tokenize(question))
        scored: list[tuple[int, FolderPolicy]] = []
        for policy in policies:
            searchable = " ".join(
                [
                    policy.folder_name.lower(),
                    policy.summary.lower(),
                    " ".join(policy.key_topics).lower(),
                    " ".join(item.value for item in policy.document_types).lower(),
                ]
            )
            score = sum(1 for token in question_tokens if token in searchable)
            scored.append((score, policy))

        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] > 0:
            return [policy for score, policy in scored if score > 0][:4]
        return policies

    async def _synthesize_answer(
        self,
        question: str,
        hits: list[RetrievalHit],
    ) -> tuple[str, list[dict[str, object]], list[str]]:
        """Synthesize an answer from retrieval hits with optional LLM support."""

        if not hits:
            return (
                "I could not find supporting evidence for that question in the indexed documents.",
                [],
                ["No relevant indexed evidence was retrieved."],
            )

        warnings: list[str] = []
        if self.llm_client and self.llm_client.is_configured:
            try:
                return self._llm_answer(question, hits), self._build_citations(question, hits), warnings
            except LLMError as exc:
                logger.warning("Falling back to extractive answer synthesis", extra={"error": str(exc)})
                warnings.append("Remote LLM synthesis was unavailable, so the answer was composed extractively.")

        return self._extractive_answer(question, hits), self._build_citations(question, hits), warnings

    def _llm_answer(self, question: str, hits: list[RetrievalHit]) -> str:
        """Ask the configured LLM to synthesize an answer from evidence."""

        context_blocks = []
        for index, hit in enumerate(hits[: self.settings.max_answer_context_chunks], start=1):
            context_text = self._answer_context_text(hit)
            context_blocks.append(f"[{index}] {hit.citation}\n{context_text}")

        answer = self.llm_client.generate_text(
            prompt=(
                "Question:\n"
                f"{question}\n\n"
                "Evidence:\n"
                + "\n\n".join(context_blocks)
                + "\n\nWrite a professional answer grounded only in the evidence. "
                "Return valid Markdown. Use short sections or bullet lists when they improve clarity. "
                "Put headings, numbered items, and bullet points on separate lines with blank lines between sections. "
                "Bold key concepts, use italics sparingly, and prefer readable structure over dense prose. "
                "If the evidence is incomplete, say so briefly. "
                "Do not list citation numbers in the prose because citations are handled separately."
            ),
            system_prompt=(
                "You are a clinical-trials research assistant. "
                "Prefer direct synthesis over copying text. "
                "Avoid repetition, page headers, and document boilerplate. "
                "Use precise, regulatory-grade language when the evidence supports it."
            ),
            temperature=0.1,
        )
        return self._clean_model_answer(answer)

    def _extractive_answer(self, question: str, hits: list[RetrievalHit]) -> str:
        """Build an answer directly from the strongest retrieved sentences."""

        query_tokens = set(self.hybrid_retriever.tokenize(question))
        selected_sentences: list[str] = []
        for hit in hits[:5]:
            sentences = re.split(r"(?<=[.!?])\s+", self._normalize_text(hit.content))
            ranked = sorted(
                sentences,
                key=lambda sentence: self._score_sentence(sentence, query_tokens, question),
                reverse=True,
            )
            for sentence in ranked:
                cleaned = self._normalize_text(sentence)
                if cleaned and cleaned not in selected_sentences and len(cleaned) >= 40:
                    selected_sentences.append(cleaned)
                    break

        if not selected_sentences:
            selected_sentences = [self._normalize_text(hit.excerpt) for hit in hits[:3] if hit.excerpt]

        summary = " ".join(selected_sentences[:2]).strip()
        if not summary:
            return "I found relevant evidence, but I could not construct a concise extractive answer."
        return summary

    def _build_citations(self, question: str, hits: list[RetrievalHit]) -> list[dict[str, object]]:
        """Return unique citations for the top-ranked hits."""

        citations: list[dict[str, object]] = []
        seen: set[str] = set()
        ordered_hits = self._order_citation_hits(hits)
        for hit in ordered_hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            citations.append(
                {
                    "index": len(citations) + 1,
                    "chunk_id": hit.chunk_id,
                    "source_doc_id": hit.source_doc_id,
                    "citation": hit.citation,
                    "file_name": hit.file_name,
                    "file_path": hit.file_path,
                    "page_number": hit.page_number,
                    "score": hit.score,
                    "excerpt": hit.excerpt,
                    "matched_terms": hit.metadata.get("matched_terms", []),
                    "evidence_text": self._normalize_text(str(hit.metadata.get("evidence_text") or hit.content)),
                    "retrieval_mode": hit.metadata.get("retrieval_mode", "hybrid"),
                }
            )
            if len(citations) >= self.settings.max_answer_citations:
                break
        return citations

    def _assert_allowed_local_path(self, source_path: Path) -> None:
        """Reject local paths outside configured safe roots."""

        try:
            source_path = source_path.resolve()
        except FileNotFoundError:
            source_path = source_path.absolute()

        allowed = any(root == source_path or root in source_path.parents for root in self.settings.allowed_local_root_paths)
        if not allowed:
            allowed_roots = ", ".join(str(root) for root in self.settings.allowed_local_root_paths)
            raise PermissionError(
                "Local path ingestion is restricted to configured safe roots. "
                f"Current allowed roots: {allowed_roots}"
            )

    def _create_managed_folder(self, folder_name: str) -> Path:
        """Create a unique managed folder under the inbound directory."""

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        slug = self._safe_name(folder_name or "source")
        folder = self.settings.inbound_dir / f"{slug}_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    async def _download_file(self, url: str, destination: Path) -> None:
        """Stream a remote file to disk with size validation."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = self.settings.max_remote_file_mb * 1024 * 1024

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                size_hint = int(response.headers.get("content-length", "0") or 0)
                if size_hint and size_hint > max_bytes:
                    raise ValueError(f"Remote file exceeds the {self.settings.max_remote_file_mb} MB limit.")

                written = 0
                with open(destination, "wb") as handle:
                    async for chunk in response.aiter_bytes():
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError(
                                f"Remote file exceeded the {self.settings.max_remote_file_mb} MB limit during download."
                            )
                        handle.write(chunk)

    def _assert_supported_extension(self, filename: str) -> None:
        """Reject unsupported file types early."""

        suffix = Path(filename).suffix.lower()
        if suffix not in _SUPPORTED_SOURCE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or '<none>'}")

    def _safe_name(self, raw_name: str) -> str:
        """Return a filesystem-safe name."""

        cleaned = _FILENAME_SANITIZER.sub("_", raw_name).strip("._")
        return cleaned or "source"

    def _load_all_chunks(self) -> list:
        """Load all processed chunks from disk."""

        chunks: list[Chunk] = []
        for chunk_file in sorted(self.settings.processed_dir.glob("*/chunks.json")):
            for payload in load_json(chunk_file):
                chunks.append(Chunk.model_validate(payload))
        return chunks

    def _merge_hits(
        self,
        semantic_hits: list[RetrievalHit],
        hybrid_hits: list[RetrievalHit],
        graph_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        """Fuse semantic, lexical, and graph results into a final ranked set."""

        merged: dict[str, RetrievalHit] = {}
        self._merge_channel(merged, semantic_hits, channel_name="semantic", channel_weight=0.58, rank_bonus=0.16)
        self._merge_channel(merged, hybrid_hits, channel_name="hybrid", channel_weight=0.27, rank_bonus=0.1)
        self._merge_channel(merged, graph_hits, channel_name="graph", channel_weight=0.15, rank_bonus=0.08)

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return self._select_diverse_hits(ranked, top_k)

    def _compact_hit(self, hit: RetrievalHit) -> dict[str, object]:
        """Return a small hit summary for the UI."""

        return {
            "chunk_id": hit.chunk_id,
            "citation": hit.citation,
            "score": hit.score,
            "matched_concepts": hit.metadata.get("matched_concepts", []),
            "matched_terms": hit.metadata.get("matched_terms", []),
            "mode": hit.metadata.get("retrieval_mode", "hybrid"),
            "page_number": hit.page_number,
        }

    def _answer_context_text(self, hit: RetrievalHit) -> str:
        """Build a larger context window for answer synthesis."""

        evidence_text = self._normalize_text(str(hit.metadata.get("evidence_text") or ""))
        content_text = self._normalize_text(hit.content)
        max_chars = self.settings.max_answer_context_chars

        if evidence_text and len(evidence_text) >= min(500, max_chars):
            return evidence_text[:max_chars]
        if content_text:
            return content_text[:max_chars]
        return evidence_text[:max_chars]

    def _normalize_text(self, text: str) -> str:
        """Clean whitespace and remove obvious PDF boilerplate artifacts."""

        cleaned = re.sub(r"\s+", " ", text or "").strip()
        cleaned = re.sub(r"\bPage \d+/\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"/[A-Z]{2,}[A-Z0-9/-]*", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _clean_model_answer(self, answer: str) -> str:
        """Normalize a model-written Markdown answer while preserving line breaks."""

        cleaned = (answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"^(answer|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _score_sentence(self, sentence: str, query_tokens: set[str], question: str) -> float:
        """Score a candidate sentence for extractive fallback."""

        lowered = sentence.lower()
        score = float(sum(1 for token in query_tokens if token in lowered))
        if question.lower().startswith("what is") and ":" in sentence[:80]:
            score += 2.0
        if 50 <= len(sentence) <= 260:
            score += 0.8
        if "glossary" in lowered:
            score += 0.5
        if any(marker in sentence for marker in ("/", " CHMP ", " E9-R1 ")):
            score -= 0.6
        return score

    def _order_citation_hits(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Prefer citation-ready evidence over contents-style or weak boilerplate hits."""

        return sorted(hits, key=lambda hit: (self._citation_penalty(hit), -hit.score))

    def _merge_channel(
        self,
        merged: dict[str, RetrievalHit],
        hits: list[RetrievalHit],
        *,
        channel_name: str,
        channel_weight: float,
        rank_bonus: float,
    ) -> None:
        """Blend one retrieval channel into the merged ranking map."""

        if not hits:
            return

        top_score = max(hit.score for hit in hits) or 1.0
        for rank, hit in enumerate(hits, start=1):
            normalized_score = hit.score / top_score if top_score else 0.0
            increment = channel_weight * normalized_score + max(0.0, rank_bonus - rank * 0.01)

            if hit.chunk_id in merged:
                existing = merged[hit.chunk_id]
                metadata = dict(existing.metadata)
                metadata.update(hit.metadata)
                metadata[f"{channel_name}_score"] = hit.score
                merged[hit.chunk_id] = existing.model_copy(
                    update={"score": round(existing.score + increment, 6), "metadata": metadata}
                )
            else:
                metadata = dict(hit.metadata)
                metadata[f"{channel_name}_score"] = hit.score
                merged[hit.chunk_id] = hit.model_copy(update={"score": round(increment, 6), "metadata": metadata})

    def _citation_penalty(self, hit: RetrievalHit) -> float:
        """Return a penalty value used to deprioritize weak citation candidates."""

        penalty = 0.0
        evidence_text = str(hit.metadata.get("evidence_text") or hit.excerpt or "").lower()

        if hit.metadata.get("is_toc_like"):
            penalty += 3.0
        if "table of contents" in evidence_text or "contents" in evidence_text:
            penalty += 2.0
        if len(evidence_text.strip()) < 60:
            penalty += 0.4
        if not hit.metadata.get("matched_terms") and not hit.metadata.get("matched_concepts"):
            penalty += 0.3

        return penalty

    def _select_diverse_hits(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        """Prefer high-scoring but non-duplicative evidence hits."""

        selected: list[RetrievalHit] = []
        doc_counts: Counter[str] = Counter()

        for hit in hits:
            if doc_counts[hit.source_doc_id] >= 3:
                continue
            if any(self._is_near_duplicate(hit, existing) for existing in selected):
                continue
            selected.append(hit)
            doc_counts[hit.source_doc_id] += 1
            if len(selected) >= top_k:
                return selected

        for hit in hits:
            if hit.chunk_id in {item.chunk_id for item in selected}:
                continue
            if any(self._is_near_duplicate(hit, existing) for existing in selected):
                continue
            if doc_counts[hit.source_doc_id] >= 5:
                continue
            selected.append(hit)
            doc_counts[hit.source_doc_id] += 1
            if len(selected) >= top_k:
                break

        return selected[:top_k]

    def _is_near_duplicate(self, left: RetrievalHit, right: RetrievalHit) -> bool:
        """Check whether two hits carry essentially the same evidence."""

        if left.file_path == right.file_path and left.page_number == right.page_number:
            return True

        left_tokens = set(self.hybrid_retriever.tokenize(str(left.metadata.get("evidence_text") or left.excerpt)))
        right_tokens = set(self.hybrid_retriever.tokenize(str(right.metadata.get("evidence_text") or right.excerpt)))
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
        return overlap >= 0.82
