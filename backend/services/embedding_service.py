"""
Embedding Service

This module handles text embedding generation using sentence transformers.
It converts text into high-dimensional vectors for semantic similarity search.

Classes:
    EmbeddingService: Service for generating text embeddings

Author: Francis Osei
"""

from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional
import numpy as np

from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """
    Service for generating text embeddings using sentence transformers.
    
    This class provides methods to encode text into vector embeddings
    for semantic search and similarity comparison. It uses pre-trained
    models from the sentence-transformers library.
    
    Attributes:
        model (SentenceTransformer): Loaded embedding model
        model_name (str): Name of the loaded model
        embedding_dimension (int): Dimension of output vectors
        
    Example:
        >>> service = EmbeddingService()
        >>> embedding = service.encode_query("What is the policy?")
        >>> print(len(embedding))
        384
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service with a specified model.
        
        Args:
            model_name (Optional[str]): Name of the sentence-transformer model.
                Defaults to value from settings.
                
        Raises:
            RuntimeError: If model fails to load
            
        Example:
            >>> service = EmbeddingService()
            >>> # Or with custom model
            >>> service = EmbeddingService("all-mpnet-base-v2")
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension: int = 0
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the sentence transformer model.
        
        This method downloads (if necessary) and loads the embedding model.
        The model is cached locally after first download.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {self.embedding_dimension}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Could not load embedding model '{self.model_name}': {str(e)}"
            )
    
    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of document chunks.
        
        This method processes multiple texts in batches for efficiency.
        It's optimized for encoding large numbers of documents.
        
        Args:
            texts (List[str]): List of text strings to encode
            batch_size (int): Number of texts to process at once (default: 32)
            show_progress (bool): Whether to show progress bar (default: False)
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            RuntimeError: If encoding fails
            ValueError: If texts list is empty
            
        Example:
            >>> chunks = ["Text 1", "Text 2", "Text 3"]
            >>> embeddings = service.encode_documents(chunks)
            >>> print(len(embeddings))
            3
            >>> print(len(embeddings[0]))
            384
        """
        if not texts:
            logger.warning("Attempted to encode empty text list")
            return []
        
        try:
            logger.info(f"Encoding {len(texts)} document chunks")
            
            # Encode with batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list format
            embeddings_list = embeddings.tolist()
            
            logger.info(
                f"Successfully encoded {len(embeddings_list)} documents"
            )
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
            raise RuntimeError(f"Failed to encode documents: {str(e)}")
    
    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single query.
        
        This method is optimized for encoding single queries for search.
        
        Args:
            query (str): Query text to encode
            normalize (bool): Whether to normalize the embedding (default: True)
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            RuntimeError: If encoding fails
            ValueError: If query is empty
            
        Example:
            >>> query = "What is the refund policy?"
            >>> embedding = service.encode_query(query)
            >>> print(len(embedding))
            384
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            logger.debug(f"Encoding query: '{query[:50]}...'")
            
            # Encode single query
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            logger.debug(f"Query encoded successfully")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise RuntimeError(f"Failed to encode query: {str(e)}")
    
    def encode_batch_queries(
        self,
        queries: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.
        
        Useful for encoding multiple query variations in agentic RAG.
        
        Args:
            queries (List[str]): List of query strings
            batch_size (int): Batch size for encoding (default: 32)
            
        Returns:
            List[List[float]]: List of query embeddings
            
        Raises:
            RuntimeError: If encoding fails
            
        Example:
            >>> queries = [
            ...     "What is the policy?",
            ...     "Tell me about the policy",
            ...     "Policy details?"
            ... ]
            >>> embeddings = service.encode_batch_queries(queries)
            >>> print(len(embeddings))
            3
        """
        if not queries:
            logger.warning("Attempted to encode empty queries list")
            return []
        
        try:
            logger.info(f"Encoding {len(queries)} queries")
            
            embeddings = self.model.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            embeddings_list = embeddings.tolist()
            logger.info(f"Successfully encoded {len(queries)} queries")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error encoding batch queries: {e}")
            raise RuntimeError(f"Failed to encode queries: {str(e)}")
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
            
        Returns:
            float: Similarity score between 0 and 1
            
        Example:
            >>> emb1 = service.encode_query("refund policy")
            >>> emb2 = service.encode_query("return policy")
            >>> similarity = service.compute_similarity(emb1, emb2)
            >>> print(f"Similarity: {similarity:.2f}")
            Similarity: 0.87
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding model.
        
        Returns:
            int: Embedding vector dimension
            
        Example:
            >>> dim = service.get_embedding_dimension()
            >>> print(f"Embedding dimension: {dim}")
            Embedding dimension: 384
        """
        return self.embedding_dimension
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including name, dimension, and max length
            
        Example:
            >>> info = service.get_model_info()
            >>> print(info["model_name"])
            'sentence-transformers/all-MiniLM-L6-v2'
        """
        if not self.model:
            return {
                "model_name": self.model_name,
                "status": "not loaded"
            }
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": self.model.max_seq_length,
            "status": "loaded"
        }
    
    def reload_model(self, model_name: Optional[str] = None) -> None:
        """
        Reload the embedding model.
        
        Useful for switching models or recovering from errors.
        
        Args:
            model_name (Optional[str]): New model name, or None to reload current
            
        Example:
            >>> service.reload_model("all-mpnet-base-v2")
        """
        if model_name:
            self.model_name = model_name
        
        logger.info(f"Reloading embedding model: {self.model_name}")
        self._load_model()


# Create singleton instance
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the singleton EmbeddingService instance.
    
    Returns:
        EmbeddingService: Singleton service instance
        
    Example:
        >>> service = get_embedding_service()
        >>> embedding = service.encode_query("test query")
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance


# Export main classes and functions
__all__ = ["EmbeddingService", "get_embedding_service"]