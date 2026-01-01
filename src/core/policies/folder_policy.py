from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Types of documents we recognize"""
    REPORT = "report"
    MANUAL = "manual"
    SPECIFICATION = "specification"
    CORRESPONDENCE = "correspondence"
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"

class RetrievalStrategy(str, Enum):
    """How should we search this folder?"""
    DENSE = "dense"          # Primarily vector search
    SPARSE = "sparse"        # Primarily keyword search
    HYBRID = "hybrid"        # Balance of both
    STRUCTURED = "structured" # Table/metadata focused

class AuthorityLevel(str, Enum):
    """How trustworthy is this source?"""
    PRIMARY = "primary"       # Official, authoritative
    SECONDARY = "secondary"   # Derived, commentary
    DRAFT = "draft"          # Work in progress
    ARCHIVED = "archived"    # Historical, possibly outdated

class EntitySchema(BaseModel):
    """Expected entities in this folder"""
    entity_type: str  # e.g., "person", "product", "regulation"
    examples: List[str] = []
    
class FolderPolicy(BaseModel):
    """
    Generated profile of a document folder.
    Guides retrieval and reasoning about this collection.
    """
    
    # Identity
    folder_id: str  # Unique identifier
    folder_path: str  # Original filesystem path
    folder_name: str
    
    # Classification
    document_types: List[DocumentType]
    primary_domain: str  # e.g., "healthcare", "legal", "engineering"
    authority_level: AuthorityLevel = AuthorityLevel.SECONDARY
    
    # Content Profile
    summary: str  # What's in this folder?
    key_topics: List[str] = []
    expected_entities: List[EntitySchema] = []
    date_range: Optional[tuple[datetime, datetime]] = None
    languages: List[str] = ["en"]
    
    # Retrieval Configuration
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    keyword_boost: float = 1.0  # For hybrid: >1 favors keywords, <1 favors vectors
    requires_exact_match: bool = False  # Should we be strict about keywords?
    
    # Metadata
    total_documents: int = 0
    total_chunks: int = 0
    has_tables: bool = False
    has_images: bool = False
    average_doc_length: Optional[float] = None
    
    # Governance
    access_restrictions: List[str] = []  # e.g., ["pii_redaction", "internal_only"]
    citation_required: bool = True
    
    # Versioning
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    profiler_version: str = "0.1.0"
    
    # Image processing configuration
    image_processing_required: bool = False
    ocr_enabled: bool = True
    caption_enabled: bool = True
    min_confidence_threshold: float = 0.7
    query_time_fallback_allowed: bool = False
    
    # Image characteristics
    has_scanned_pdfs: bool = False
    has_standalone_images: bool = False
    image_types: List[str] = []  # ["forms", "charts", "screenshots", "diagrams"]
    estimated_image_count: int = 0
    
    # Governance for images
    images_may_contain_pii: bool = False
    redaction_required: bool = False
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }