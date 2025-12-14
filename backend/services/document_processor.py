"""
Document Processing Service

This module handles document upload, processing, and chunking for the RAG system.
It manages document metadata, extracts text from various file formats, and
prepares documents for embedding and vector storage.

Classes:
    DocumentProcessor: Main document processing service

Author: Francis Osei
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

from app.utils.file_handlers import FileHandler
from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentProcessor:
    """
    Document processing service for handling document uploads and text extraction.
    
    This class manages the entire document processing pipeline including:
    - Document metadata tracking
    - Text extraction from multiple formats
    - Document chunking for RAG
    - File storage management
    
    Attributes:
        file_handler (FileHandler): Handler for file operations
        processed_docs (Dict[str, Dict]): Cache of processed document metadata
        
    Example:
        >>> processor = DocumentProcessor()
        >>> result = await processor.process_document(
        ...     file_path="/tmp/report.pdf",
        ...     filename="report.pdf"
        ... )
        >>> print(result["chunk_count"])
        45
    """
    
    def __init__(self):
        """
        Initialize the document processor.
        
        Sets up the file handler and initializes the document cache.
        """
        self.file_handler = FileHandler()
        self.processed_docs: Dict[str, Dict] = {}
        logger.info("DocumentProcessor initialized")
    
    async def process_document(
        self,
        file_path: str,
        filename: str
    ) -> Dict:
        """
        Process a single document and return chunks with metadata.
        
        This method orchestrates the entire document processing pipeline:
        1. Validates file existence
        2. Extracts text from the document
        3. Chunks the text with overlap
        4. Stores metadata
        5. Returns processed data
        
        Args:
            file_path (str): Full path to the uploaded file
            filename (str): Original filename with extension
            
        Returns:
            Dict: Processing result containing:
                - doc_id (str): Unique document identifier
                - chunks (List[str]): List of text chunks
                - metadata (Dict): Document metadata
                
        Raises:
            ValueError: If document is empty or processing fails
            FileNotFoundError: If file doesn't exist at path
            
        Example:
            >>> result = await processor.process_document(
            ...     "/uploads/report.pdf",
            ...     "Q3_Report.pdf"
            ... )
            >>> print(f"Created {len(result['chunks'])} chunks")
            Created 42 chunks
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            logger.info(f"Processing document: {filename} (ID: {doc_id})")
            
            # Extract text from document
            logger.debug(f"Extracting text from {filename}")
            text = self.file_handler.extract_text(file_path, filename.lower())
            
            # Validate extracted text
            if not text or len(text.strip()) == 0:
                raise ValueError(
                    "Document appears to be empty or text extraction failed. "
                    "Please ensure the document contains readable text."
                )
            
            logger.info(
                f"Extracted {len(text)} characters from {filename}"
            )
            
            # Chunk the text
            logger.debug(f"Chunking document {filename}")
            chunks = self.file_handler.chunk_text(
                text,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError(
                    "No valid chunks created from document. "
                    "Document may be too short or improperly formatted."
                )
            
            # Store document metadata
            doc_metadata = {
                "id": doc_id,
                "filename": filename,
                "upload_date": datetime.now(),
                "chunk_count": len(chunks),
                "status": "processed",
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "character_count": len(text),
                "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks)
            }
            
            # Cache document metadata
            self.processed_docs[doc_id] = doc_metadata
            
            logger.info(
                f"Successfully processed {filename}: "
                f"{len(chunks)} chunks created, "
                f"avg size {doc_metadata['avg_chunk_size']} chars"
            )
            
            return {
                "doc_id": doc_id,
                "chunks": chunks,
                "metadata": doc_metadata
            }
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Validation error processing {filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document {filename}: {e}")
            raise ValueError(f"Failed to process document: {str(e)}")
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """
        Get metadata for a processed document.
        
        Args:
            doc_id (str): Document identifier
            
        Returns:
            Optional[Dict]: Document metadata if found, None otherwise
            
        Example:
            >>> info = processor.get_document_info("doc-123")
            >>> print(info["filename"])
            'report.pdf'
        """
        doc_info = self.processed_docs.get(doc_id)
        if doc_info:
            logger.debug(f"Retrieved info for document {doc_id}")
        else:
            logger.warning(f"Document {doc_id} not found in cache")
        return doc_info
    
    def list_documents(self) -> List[Dict]:
        """
        List all processed documents.
        
        Returns:
            List[Dict]: List of document metadata dictionaries
            
        Example:
            >>> docs = processor.list_documents()
            >>> print(f"Total documents: {len(docs)}")
            Total documents: 5
        """
        docs = list(self.processed_docs.values())
        logger.debug(f"Listed {len(docs)} documents")
        return docs
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its associated file.
        
        This method removes both the document metadata from cache
        and the physical file from storage.
        
        Args:
            doc_id (str): Document identifier to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
            
        Example:
            >>> success = processor.delete_document("doc-123")
            >>> if success:
            ...     print("Document deleted")
        """
        if doc_id not in self.processed_docs:
            logger.warning(f"Attempted to delete non-existent document {doc_id}")
            return False
        
        doc_info = self.processed_docs[doc_id]
        
        try:
            # Delete the physical file if it exists
            file_path = doc_info.get("file_path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
            
            # Remove from cache
            del self.processed_docs[doc_id]
            
            logger.info(f"Successfully deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dict: Statistics including total documents, chunks, sizes
            
        Example:
            >>> stats = processor.get_statistics()
            >>> print(f"Total chunks: {stats['total_chunks']}")
        """
        if not self.processed_docs:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_size_bytes": 0,
                "average_chunks_per_doc": 0
            }
        
        total_chunks = sum(
            doc["chunk_count"] for doc in self.processed_docs.values()
        )
        total_size = sum(
            doc.get("file_size", 0) for doc in self.processed_docs.values()
        )
        
        stats = {
            "total_documents": len(self.processed_docs),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "average_chunks_per_doc": (
                total_chunks // len(self.processed_docs)
                if self.processed_docs else 0
            )
        }
        
        logger.debug(f"Generated statistics: {stats}")
        return stats
    
    def clear_all(self) -> None:
        """
        Clear all processed documents from cache.
        
        Note: This does not delete the physical files.
        
        Example:
            >>> processor.clear_all()
            >>> print(len(processor.list_documents()))
            0
        """
        count = len(self.processed_docs)
        self.processed_docs.clear()
        logger.info(f"Cleared {count} documents from cache")


# Create singleton instance
_document_processor_instance: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """
    Get or create the singleton DocumentProcessor instance.
    
    Returns:
        DocumentProcessor: Singleton processor instance
        
    Example:
        >>> processor = get_document_processor()
        >>> docs = processor.list_documents()
    """
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance


# Export main classes and functions
__all__ = ["DocumentProcessor", "get_document_processor"]