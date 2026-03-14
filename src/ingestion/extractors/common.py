"""Common extraction primitives."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.ingestion.models import Chunk, Document

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Normalized extractor output."""

    document: Document
    text_content: str
    chunks: list[Chunk] = Field(default_factory=list)
    success: bool = True
    error_message: str | None = None


class BaseExtractor(ABC):
    """Base interface for file extractors."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"extractors.{self.__class__.__name__}")

    @abstractmethod
    async def can_extract(self, file_path: Path) -> bool:
        """Return whether the extractor can handle the file."""

    @abstractmethod
    async def extract(self, file_path: Path, doc_id: str, folder_id: str) -> ExtractionResult:
        """Extract content and metadata from a file."""

    def _create_document_metadata(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str,
        **kwargs: Any,
    ) -> Document:
        """Create a document metadata record from a file path."""

        stat = file_path.stat()
        return Document(
            doc_id=doc_id,
            folder_id=folder_id,
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_extension=file_path.suffix.lower(),
            file_size_bytes=stat.st_size,
            **kwargs,
        )
