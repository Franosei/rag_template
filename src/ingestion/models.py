from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime
from enum import Enum


class Modality(str, Enum):
    """Type of content in this chunk"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ExtractionMethod(str, Enum):
    """How was this content extracted?"""
    # Text
    TEXT_EXTRACT = "text_extract"
    
    # Tables
    TABLE_PARSE = "table_parse"
    
    # Images
    OCR = "ocr"
    CAPTION = "caption"
    PDF_IMAGE_EXTRACT = "pdf_image_extract"
    PDF_PAGE_RENDER = "pdf_page_render"
    DOCX_IMAGE_EXTRACT = "docx_image_extract"
    EXCEL_CHART_EXPORT = "excel_chart_export"


class ConfidenceLevel(str, Enum):
    """Extraction quality assessment"""
    HIGH = "high"      # >0.9 confidence or clean extraction
    MEDIUM = "medium"  # 0.7-0.9 confidence
    LOW = "low"        # <0.7 confidence
    UNKNOWN = "unknown"


class ImageMetadata(BaseModel):
    """Metadata specific to image chunks"""
    image_path: Optional[str] = None  # Path to saved image file
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None  # png, jpg, etc.
    figure_number: Optional[int] = None
    image_index: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None  # {x, y, width, height}
    ocr_text: Optional[str] = None
    caption: Optional[str] = None
    detected_elements: List[str] = []  # ["chart", "table", "text", "diagram"]


class TableMetadata(BaseModel):
    """Metadata specific to table chunks"""
    rows: int
    columns: int
    headers: List[str] = []
    table_index: Optional[int] = None
    structured_data: Optional[List[Dict[str, Any]]] = None  # Row dicts


class Chunk(BaseModel):
    """
    Unified chunk model for all retrievable evidence.
    Used for text, tables, and images.
    """
    
    # Identity
    chunk_id: str
    folder_id: str
    source_doc_id: str
    
    # Content
    modality: Modality
    content_text: str  # For images: OCR + caption; for tables: linearized text
    
    # Provenance
    file_path: str
    file_name: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None  # Position within document
    
    # Extraction metadata
    extraction_method: ExtractionMethod
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    
    # Type-specific metadata
    image_metadata: Optional[ImageMetadata] = None
    table_metadata: Optional[TableMetadata] = None
    
    # Generic metadata
    metadata: Dict[str, Any] = {}
    
    # Deduplication
    content_hash: str  # MD5 or similar
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_citation(self) -> str:
        """Generate human-readable citation"""
        citation_parts = [self.file_name]
        
        if self.page_number is not None:
            citation_parts.append(f"page {self.page_number}")
        
        if self.modality == Modality.IMAGE:
            if self.image_metadata and self.image_metadata.figure_number:
                citation_parts.append(f"figure {self.image_metadata.figure_number}")
            elif self.image_metadata and self.image_metadata.image_index is not None:
                citation_parts.append(f"image {self.image_metadata.image_index}")
        
        if self.modality == Modality.TABLE:
            if self.table_metadata and self.table_metadata.table_index is not None:
                citation_parts.append(f"table {self.table_metadata.table_index}")
        
        return ", ".join(citation_parts)
    
    def is_low_confidence(self) -> bool:
        """Check if this chunk has low extraction confidence"""
        return self.confidence == ConfidenceLevel.LOW


class Document(BaseModel):
    """Metadata for a source document"""
    doc_id: str
    folder_id: str
    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    
    # Document characteristics
    page_count: Optional[int] = None
    is_scanned: bool = False  # Is this a scanned PDF?
    has_images: bool = False
    has_tables: bool = False
    
    # Processing results
    total_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None