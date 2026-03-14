"""Chunking utilities for extracted text."""

from __future__ import annotations

import hashlib
import logging
import re

from src.app.settings import settings
from src.ingestion.models import Chunk, ConfidenceLevel, ExtractionMethod, Modality
from src.utils.ids import generate_chunk_id

logger = logging.getLogger(__name__)
_PAGE_MARKER_PATTERN = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)


class TextChunker:
    """Split text into page-aware, paragraph-preserving chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_text(
        self,
        *,
        text: str,
        doc_id: str,
        folder_id: str,
        file_path: str,
        file_name: str,
    ) -> list[Chunk]:
        """Split a document into chunks while preserving provenance."""

        if not (text or "").strip():
            logger.debug("Skipping empty text during chunking", extra={"file_name": file_name})
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        for page_number, page_text in self._split_pages(text):
            for raw_segment in self._segment_text(page_text):
                cleaned_segment = self._normalize_segment(raw_segment)
                if not cleaned_segment:
                    continue

                chunks.append(
                    Chunk(
                        chunk_id=generate_chunk_id(doc_id, chunk_index),
                        folder_id=folder_id,
                        source_doc_id=doc_id,
                        modality=Modality.TEXT,
                        content_text=cleaned_segment,
                        file_path=file_path,
                        file_name=file_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        extraction_method=ExtractionMethod.TEXT_EXTRACT,
                        confidence=ConfidenceLevel.HIGH,
                        metadata={
                            "is_toc_like": self._is_toc_like(raw_segment),
                            "char_count": len(cleaned_segment),
                        },
                        content_hash=hashlib.md5(cleaned_segment.encode("utf-8")).hexdigest(),
                    )
                )
                chunk_index += 1

        return chunks

    def _split_pages(self, text: str) -> list[tuple[int | None, str]]:
        """Split text into page segments when PDF page markers are present."""

        matches = list(_PAGE_MARKER_PATTERN.finditer(text))
        if not matches:
            return [(None, text)]

        segments: list[tuple[int | None, str]] = []
        for index, match in enumerate(matches):
            page_number = int(match.group(1))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            page_text = text[start:end].strip()
            if page_text:
                segments.append((page_number, page_text))
        return segments or [(None, text)]

    def _segment_text(self, text: str) -> list[str]:
        """Create chunks while preserving paragraph boundaries whenever possible."""

        normalized_text = text.strip()
        if not normalized_text:
            return []
        if len(normalized_text) <= self.chunk_size:
            return [normalized_text]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized_text) if part.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [part.strip() for part in re.split(r"\n+", normalized_text) if part.strip()]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            compact_paragraph = self._normalize_segment(paragraph)
            if not compact_paragraph:
                continue

            if len(compact_paragraph) > self.chunk_size:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_length = 0
                chunks.extend(self._split_large_paragraph(compact_paragraph))
                continue

            projected_length = current_length + len(compact_paragraph) + (2 if current_parts else 0)
            if projected_length > self.chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = [compact_paragraph]
                current_length = len(compact_paragraph)
            else:
                current_parts.append(compact_paragraph)
                current_length = projected_length

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        """Split an oversized paragraph on sentence boundaries with no duplicate overlap."""

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        if len(sentences) <= 1:
            return [paragraph[index : index + self.chunk_size].strip() for index in range(0, len(paragraph), self.chunk_size)]

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    def _normalize_segment(self, text: str) -> str:
        """Normalize whitespace while keeping human-readable sentence flow."""

        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return ""

        paragraphs = []
        for paragraph in re.split(r"\n\s*\n+", normalized):
            compact = re.sub(r"\s*\n\s*", " ", paragraph)
            compact = re.sub(r"[ \t\f\v]+", " ", compact).strip()
            if compact:
                paragraphs.append(compact)

        return "\n\n".join(paragraphs)

    def _is_toc_like(self, text: str) -> bool:
        """Heuristically detect contents/index-like text that should rank lower."""

        compact = (text or "").lower()
        dot_leaders = len(re.findall(r"\.{4,}", compact))
        repeated_page_numbers = len(re.findall(r"\b\d{1,3}\b", compact))
        if "table of contents" in compact or "contents" in compact:
            return True
        return dot_leaders >= 3 and repeated_page_numbers >= 4
