"""Pydantic models for extracted documents and chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Modality(str, Enum):
    """Content modality for a chunk."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ExtractionMethod(str, Enum):
    """How a chunk was produced."""

    TEXT_EXTRACT = "text_extract"
    TABLE_PARSE = "table_parse"
    OCR = "ocr"
    CAPTION = "caption"
    PDF_IMAGE_EXTRACT = "pdf_image_extract"
    DOCX_IMAGE_EXTRACT = "docx_image_extract"


class ConfidenceLevel(str, Enum):
    """Confidence label assigned to extracted content."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ImageMetadata(BaseModel):
    """Metadata specific to image chunks."""

    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    figure_number: int | None = None
    image_index: int | None = None
    bbox: dict[str, float] | None = None
    ocr_text: str | None = None
    caption: str | None = None
    detected_elements: list[str] = Field(default_factory=list)


class TableMetadata(BaseModel):
    """Metadata specific to table chunks."""

    rows: int
    columns: int
    headers: list[str] = Field(default_factory=list)
    table_index: int | None = None
    structured_data: list[dict[str, object]] | None = None


class Chunk(BaseModel):
    """Unified retrieval unit used by the search layer."""

    chunk_id: str
    folder_id: str
    source_doc_id: str
    modality: Modality
    content_text: str
    file_path: str
    file_name: str
    page_number: int | None = None
    chunk_index: int | None = None
    extraction_method: ExtractionMethod
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    image_metadata: ImageMetadata | None = None
    table_metadata: TableMetadata | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def get_citation(self) -> str:
        """Return a concise human-readable citation label."""

        parts = [self.file_name]
        if self.page_number is not None:
            parts.append(f"page {self.page_number}")
        if self.modality == Modality.TABLE and self.table_metadata and self.table_metadata.table_index is not None:
            parts.append(f"table {self.table_metadata.table_index}")
        if self.modality == Modality.IMAGE and self.image_metadata and self.image_metadata.image_index is not None:
            parts.append(f"image {self.image_metadata.image_index}")
        return ", ".join(parts)


class Document(BaseModel):
    """Source document metadata recorded during ingestion."""

    doc_id: str
    folder_id: str
    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    page_count: int | None = None
    is_scanned: bool = False
    has_images: bool = False
    has_tables: bool = False
    total_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    processed_at: datetime | None = None
