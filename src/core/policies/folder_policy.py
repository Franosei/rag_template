"""Folder policy models used by ingestion and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Document families recognized by the profiler."""

    GUIDELINE = "guideline"
    PUBLICATION = "publication"
    SOFTWARE_REFERENCE = "software_reference"
    DATA_TABLE = "data_table"
    NOTE = "note"
    UNKNOWN = "unknown"


class RetrievalStrategy(str, Enum):
    """Search behavior preferred for a folder."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    STRUCTURED = "structured"


class AuthorityLevel(str, Enum):
    """Trust posture assigned to a folder."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DRAFT = "draft"
    ARCHIVED = "archived"


class EntitySchema(BaseModel):
    """High-value entities expected in a folder."""

    entity_type: str
    examples: list[str] = Field(default_factory=list)


class FolderPolicy(BaseModel):
    """Profile and governance configuration for a document folder."""

    folder_id: str
    folder_path: str
    folder_name: str

    document_types: list[DocumentType] = Field(default_factory=lambda: [DocumentType.UNKNOWN])
    primary_domain: str = "clinical research"
    authority_level: AuthorityLevel = AuthorityLevel.SECONDARY

    summary: str = ""
    key_topics: list[str] = Field(default_factory=list)
    expected_entities: list[EntitySchema] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=lambda: ["en"])

    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    keyword_boost: float = 1.0
    requires_exact_match: bool = False

    total_documents: int = 0
    total_chunks: int = 0
    has_tables: bool = False
    has_images: bool = False
    average_doc_length: float | None = None

    access_restrictions: list[str] = Field(default_factory=list)
    citation_required: bool = True

    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    profiler_version: str = "1.0.0"

    image_processing_required: bool = False
    ocr_enabled: bool = False
    caption_enabled: bool = False
    min_confidence_threshold: float = 0.7
    query_time_fallback_allowed: bool = False

    has_scanned_pdfs: bool = False
    has_standalone_images: bool = False
    image_types: list[str] = Field(default_factory=list)
    estimated_image_count: int = 0
    images_may_contain_pii: bool = False
    redaction_required: bool = False

    def retrieval_weights(self) -> tuple[float, float]:
        """Return dense and sparse scoring weights for hybrid retrieval."""

        if self.retrieval_strategy == RetrievalStrategy.DENSE:
            return 0.75, 0.25
        if self.retrieval_strategy == RetrievalStrategy.SPARSE:
            return 0.25, 0.75
        if self.retrieval_strategy == RetrievalStrategy.STRUCTURED:
            return 0.35, 0.65
        return 0.55, 0.45
