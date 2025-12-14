"""
Vector Store Service

This module manages the vector database for storing and retrieving document embeddings.
It uses ChromaDB for persistent storage and similarity search.

Classes:
    VectorStore: Service for vector database operations

Author: Francis Osei
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Tuple, Optional
import logging

from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """
    Vector database service for document storage and retrieval.
    
    This class provides a high-level interface to ChromaDB for storing
    document embeddings and performing semantic similarity searches.
    
    Attributes:
        client (chromadb.Client): ChromaDB client instance
        collection (chromadb.Collection): Active collection for documents
        collection_name (str): Name of the collection
        
    Example:
        >>> store = VectorStore()
        >>> store.add_documents(
        ...     doc_id="doc-123",
        ...     chunks=["text 1", "text 2"],
        ...     embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ...     metadata={"filename": "test.pdf"}
        ... )
    """
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector store with ChromaDB.
        
        Args:
            collection_name (str): Name for the document collection
            
        Raises:
            RuntimeError: If ChromaDB initialization fails
            
        Example:
            >>> store = VectorStore()
            >>> # Or with custom collection
            >>> store = VectorStore("my_documents")
        """
        self.collection_name = collection_name
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """
        Initialize ChromaDB client and collection.
        
        Creates a persistent ChromaDB instance and initializes
        or retrieves the document collection.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info(f"Initializing ChromaDB at {settings.VECTOR_DB_PATH}")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "RAG system document collection",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )
            
            logger.info(
                f"ChromaDB initialized successfully. "
                f"Collection: {self.collection_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(
                f"Could not initialize vector database: {str(e)}"
            )
    
    def add_documents(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict
    ) -> None:
        """
        Add document chunks to the vector store.
        
        This method stores document chunks along with their embeddings
        and metadata for later retrieval.
        
        Args:
            doc_id (str): Unique document identifier
            chunks (List[str]): List of text chunks
            embeddings (List[List[float]]): Corresponding embedding vectors
            metadata (Dict): Document metadata (filename, upload_date, etc.)
            
        Raises:
            RuntimeError: If adding documents fails
            ValueError: If chunks and embeddings length mismatch
            
        Example:
            >>> store.add_documents(
            ...     doc_id="doc-123",
            ...     chunks=["chunk 1", "chunk 2"],
            ...     embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ...     metadata={"filename": "report.pdf"}
            ... )
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have same length"
            )
        
        if not chunks:
            logger.warning(f"No chunks to add for document {doc_id}")
            return
        
        try:
            # Generate unique IDs for each chunk
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Create metadata for each chunk
            metadatas = [
                {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_index": i,
                    "upload_date": str(metadata.get("upload_date", "")),
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            logger.info(
                f"Added {len(chunks)} chunks for document {doc_id} "
                f"({metadata.get('filename', 'unknown')})"
            )
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise RuntimeError(f"Failed to add documents: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Search for similar documents using query embedding.
        
        This method performs semantic similarity search and returns
        the most relevant document chunks.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            top_k (int): Number of results to return (default: 5)
            filter_metadata (Optional[Dict]): Metadata filters for search
            
        Returns:
            Tuple containing:
                - List[str]: Retrieved document chunks
                - List[Dict]: Chunk metadata
                - List[float]: Similarity scores (0-1, higher is better)
                
        Raises:
            RuntimeError: If search fails
            
        Example:
            >>> docs, metadata, scores = store.search(
            ...     query_embedding=[0.1, 0.2, ...],
            ...     top_k=5
            ... )
            >>> for doc, score in zip(docs, scores):
            ...     print(f"Score: {score:.2f} - {doc[:50]}")
        """
        try:
            logger.debug(f"Searching for top {top_k} similar documents")
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add metadata filter if provided
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            # Perform search
            results = self.collection.query(**query_params)
            
            if not results or not results['documents'][0]:
                logger.info("No matching documents found")
                return [], [], []
            
            # Extract results
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Convert distances to similarity scores
            # ChromaDB uses squared L2 distance, convert to similarity (0-1)
            similarities = [1 / (1 + dist) for dist in distances]
            
            logger.info(
                f"Found {len(documents)} relevant chunks, "
                f"best similarity: {max(similarities):.3f}"
            )
            
            return documents, metadatas, similarities
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise RuntimeError(f"Failed to search documents: {str(e)}")
    
    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id (str): Document identifier
            
        Returns:
            int: Number of chunks deleted
            
        Raises:
            RuntimeError: If deletion fails
            
        Example:
            >>> deleted = store.delete_document("doc-123")
            >>> print(f"Deleted {deleted} chunks")
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if not results or not results['ids']:
                logger.warning(f"No chunks found for document {doc_id}")
                return 0
            
            chunk_ids = results['ids']
            
            # Delete chunks
            self.collection.delete(ids=chunk_ids)
            
            logger.info(f"Deleted {len(chunk_ids)} chunks for document {doc_id}")
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Error deleting document from vector store: {e}")
            raise RuntimeError(f"Failed to delete document: {str(e)}")
    
    def get_document_count(self) -> int:
        """
        Get total number of chunks in the collection.
        
        Returns:
            int: Total chunk count
            
        Example:
            >>> count = store.get_document_count()
            >>> print(f"Total chunks: {count}")
        """
        try:
            count = self.collection.count()
            logger.debug(f"Collection contains {count} chunks")
            return count
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_unique_documents(self) -> List[str]:
        """
        Get list of unique document IDs in the collection.
        
        Returns:
            List[str]: List of unique document IDs
            
        Example:
            >>> doc_ids = store.get_unique_documents()
            >>> print(f"Unique documents: {len(doc_ids)}")
        """
        try:
            # Get all metadata
            results = self.collection.get()
            
            if not results or not results['metadatas']:
                return []
            
            # Extract unique doc_ids
            doc_ids = set(
                meta['doc_id'] 
                for meta in results['metadatas'] 
                if 'doc_id' in meta
            )
            
            return list(doc_ids)
            
        except Exception as e:
            logger.error(f"Error getting unique documents: {e}")
            return []
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        
        Warning: This operation cannot be undone.
        
        Raises:
            RuntimeError: If clearing fails
            
        Example:
            >>> store.clear_collection()
            >>> print(store.get_document_count())
            0
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "RAG system document collection",
                    "hnsw:space": "cosine"
                }
            )
            
            logger.info(f"Collection '{self.collection_name}' cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise RuntimeError(f"Failed to clear collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dict: Collection statistics
            
        Example:
            >>> stats = store.get_collection_stats()
            >>> print(stats)
            {'total_chunks': 150, 'unique_documents': 5, ...}
        """
        try:
            total_chunks = self.get_document_count()
            unique_docs = self.get_unique_documents()
            
            stats = {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "unique_documents": len(unique_docs),
                "average_chunks_per_doc": (
                    total_chunks // len(unique_docs) if unique_docs else 0
                )
            }
            
            logger.debug(f"Collection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }
    
    def peek(self, limit: int = 5) -> Dict:
        """
        Peek at a sample of documents in the collection.
        
        Args:
            limit (int): Number of documents to retrieve
            
        Returns:
            Dict: Sample documents with metadata
            
        Example:
            >>> sample = store.peek(3)
            >>> print(len(sample['documents']))
            3
        """
        try:
            results = self.collection.peek(limit=limit)
            logger.debug(f"Peeked at {limit} documents")
            return results
        except Exception as e:
            logger.error(f"Error peeking collection: {e}")
            return {}


# Create singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create the singleton VectorStore instance.
    
    Returns:
        VectorStore: Singleton store instance
        
    Example:
        >>> store = get_vector_store()
        >>> count = store.get_document_count()
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


# Export main classes and functions
__all__ = ["VectorStore", "get_vector_store"]