import logging
from typing import List
import hashlib

from src.ingestion.models import Chunk, Modality, ExtractionMethod, ConfidenceLevel
from src.utils.ids import generate_chunk_id
from src.app.settings import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Split text into overlapping chunks for retrieval.
    
    Uses simple character-based chunking with overlap.
    Can be enhanced with semantic chunking later.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        folder_id: str,
        file_path: str,
        file_name: str
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Full document text
            doc_id: Document ID
            folder_id: Folder ID
            file_path: Source file path
            file_name: Source file name
        
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for chunking: {file_name}")
            return []
        
        chunks = []
        text = text.strip()
        
        # Simple character-based chunking with overlap
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within overlap zone
                search_start = max(start, end - self.chunk_overlap)
                search_text = text[search_start:end + 100]  # Look ahead a bit
                
                # Find last sentence ending
                for delimiter in ['. ', '.\n', '! ', '?\n', '? ']:
                    last_idx = search_text.rfind(delimiter)
                    if last_idx != -1:
                        end = search_start + last_idx + len(delimiter)
                        break
            
            chunk_text = text[start:end].strip()
            
            if not chunk_text:
                break
            
            # Create chunk
            chunk_id = generate_chunk_id(doc_id, chunk_index)
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            chunk = Chunk(
                chunk_id=chunk_id,
                folder_id=folder_id,
                source_doc_id=doc_id,
                modality=Modality.TEXT,
                content_text=chunk_text,
                file_path=file_path,
                file_name=file_name,
                chunk_index=chunk_index,
                extraction_method=ExtractionMethod.TEXT_EXTRACT,
                confidence=ConfidenceLevel.HIGH,
                content_hash=content_hash
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position
            start = end - self.chunk_overlap
            
            # Safety check
            if chunk_index > 10000:
                logger.warning(f"Too many chunks for {file_name}, stopping at 10000")
                break
        
        logger.debug(
            f"Created {len(chunks)} chunks for {file_name}",
            extra={"text_length": len(text), "chunk_count": len(chunks)}
        )
        
        return chunks