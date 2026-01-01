import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from src.ingestion.models import Chunk, Document

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Result from document extraction"""
    document: Document
    text_content: str
    chunks: List[Chunk] = []
    success: bool = True
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class BaseExtractor(ABC):
    """Base class for document extractors"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"extractors.{self.__class__.__name__}")
    
    @abstractmethod
    async def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file"""
        pass
    
    @abstractmethod
    async def extract(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str
    ) -> ExtractionResult:
        """Extract content from file"""
        pass
    
    def _create_document_metadata(
        self,
        file_path: Path,
        doc_id: str,
        folder_id: str,
        **kwargs
    ) -> Document:
        """Create Document metadata object"""
        from src.ingestion.models import Document
        from datetime import datetime
        
        stat = file_path.stat()
        
        return Document(
            doc_id=doc_id,
            folder_id=folder_id,
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_extension=file_path.suffix.lower(),
            file_size_bytes=stat.st_size,
            created_at=datetime.utcnow(),
            **kwargs
        )