"""
LLM Service

This module handles integration with Large Language Models (OpenAI, HuggingFace)
and orchestrates the agentic RAG workflow for high-accuracy question answering.

Classes:
    LLMService: Service for LLM integration and response generation

Author: Francis Osei
"""

import openai
from typing import List, Dict, Optional
import logging

from app.config import get_settings
from app.services.rag_agent import AgenticRAG
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """
    Service for LLM integration and response generation.
    
    This class provides methods to generate responses using various LLM providers
    with support for both simple RAG and advanced agentic RAG workflows.
    
    Attributes:
        openai_client (openai.OpenAI): OpenAI API client
        agent (AgenticRAG): Agentic RAG workflow orchestrator
        
    Example:
        >>> service = LLMService()
        >>> response = await service.generate_response(
        ...     question="What is the refund policy?",
        ...     embedding_service=embedding_svc,
        ...     vector_store=vector_store,
        ...     use_agentic=True
        ... )
    """
    
    def __init__(self):
        """
        Initialize the LLM service.
        
        Sets up OpenAI client if API key is available and prepares
        the agentic RAG system.
        
        Example:
            >>> service = LLMService()
            >>> print(service.is_configured())
            True
        """
        self.openai_client: Optional[openai.OpenAI] = None
        self.agent: Optional[AgenticRAG] = None
        
        # Initialize OpenAI client if configured
        if settings.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OpenAI API key not configured")
    
    def is_configured(self) -> bool:
        """
        Check if LLM service is properly configured.
        
        Returns:
            bool: True if service is ready to use
            
        Example:
            >>> if service.is_configured():
            ...     print("Ready to generate responses")
        """
        return self.openai_client is not None
    
    def initialize_agent(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ) -> None:
        """
        Initialize the agentic RAG system.
        
        This method sets up the intelligent multi-step reasoning system
        for high-accuracy question answering.
        
        Args:
            embedding_service (EmbeddingService): Service for embeddings
            vector_store (VectorStore): Vector database service
            
        Raises:
            ValueError: If OpenAI client is not configured
            
        Example:
            >>> service.initialize_agent(embed_svc, vector_store)
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        if not self.agent:
            self.agent = AgenticRAG(
                llm_client=self.openai_client,
                embedding_service=embedding_service,
                vector_store=vector_store
            )
            logger.info("Agentic RAG system initialized")
    
    async def generate_response_agentic(
        self,
        question: str,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ) -> Dict:
        """
        Generate response using agentic RAG for maximum accuracy.
        
        This method uses the full agentic workflow including:
        - Query decomposition
        - Query rewriting
        - Hybrid retrieval
        - Re-ranking
        - Generation with verification
        - Iterative refinement
        
        Args:
            question (str): User's question
            embedding_service (EmbeddingService): Embedding service instance
            vector_store (VectorStore): Vector store instance
            
        Returns:
            Dict: Response containing:
                - answer (str): Generated answer
                - sources (List[Dict]): Source documents
                - confidence_score (float): Confidence (0-1)
                - has_answer_in_documents (bool): Verification status
                - warning (Optional[str]): Warning message
                - verification_details (Dict): Verification info
                - workflow_steps (Dict): Workflow statistics
                
        Raises:
            ValueError: If services not properly configured
            RuntimeError: If generation fails
            
        Example:
            >>> response = await service.generate_response_agentic(
            ...     "What is the policy?",
            ...     embed_svc,
            ...     vector_store
            ... )
            >>> print(response["confidence_score"])
            0.92
        """
        # Initialize agent if needed
        if not self.agent:
            self.initialize_agent(embedding_service, vector_store)
        
        if not self.agent:
            raise ValueError("Agentic RAG system not initialized")
        
        try:
            logger.info(f"Starting agentic RAG workflow for: '{question}'")
            
            # Execute agentic workflow
            result = await self.agent.answer_question(question)
            
            # Format response
            response = {
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence_score": result["confidence"],
                "has_answer_in_documents": result["verified"],
                "warning": (
                    None if result["verified"] 
                    else "Answer verification failed - may not be fully accurate"
                ),
                "verification_details": result.get("verification", {}),
                "workflow_steps": result.get("workflow_steps", {})
            }
            
            logger.info(
                f"Agentic RAG completed. "
                f"Verified: {result['verified']}, "
                f"Confidence: {result['confidence']:.2f}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in agentic response generation: {e}")
            raise RuntimeError(f"Failed to generate agentic response: {str(e)}")
    
    async def generate_response_simple(
        self,
        question: str,
        context_chunks: List[str],
        metadata: List[Dict],
        similarities: List[float]
    ) -> Dict:
        """
        Generate response using simple RAG (legacy mode).
        
        This is a single-pass RAG without verification or refinement.
        Faster but less accurate than agentic mode.
        
        Args:
            question (str): User's question
            context_chunks (List[str]): Retrieved document chunks
            metadata (List[Dict]): Chunk metadata
            similarities (List[float]): Similarity scores
            
        Returns:
            Dict: Response with answer and sources
            
        Raises:
            ValueError: If OpenAI not configured
            RuntimeError: If generation fails
            
        Example:
            >>> response = await service.generate_response_simple(
            ...     "What is X?",
            ...     ["chunk 1", "chunk 2"],
            ...     [{"filename": "doc.pdf"}],
            ...     [0.9, 0.8]
            ... )
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            # Build context from chunks
            context_parts = []
            for i, (chunk, meta) in enumerate(zip(context_chunks, metadata), 1):
                source = meta.get('filename', 'Unknown')
                context_parts.append(f"[Source {i}: {source}]\n{chunk}\n")
            
            context = "\n".join(context_parts)
            
            # Build prompt
            prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided documents.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context below
2. If the answer is not in the context, say "I cannot find this information in the uploaded documents"
3. Always cite which source document you used (e.g., "According to [Source 1]...")
4. Do not make up information or use knowledge outside of these documents

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (with citations):"""
            
            # Generate response
            logger.debug("Generating simple RAG response")
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based only on provided documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            answer = response.choices[0].message.content
            
            # Simple relevance check
            avg_similarity = (
                sum(similarities[:3]) / min(len(similarities), 3)
                if similarities else 0
            )
            has_answer = avg_similarity >= settings.MIN_SIMILARITY_SCORE
            
            # Format sources
            sources = []
            for i, (chunk, meta, score) in enumerate(
                zip(context_chunks, metadata, similarities)
            ):
                sources.append({
                    "content": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "document_name": meta.get('filename', 'Unknown'),
                    "chunk_id": f"chunk_{i}",
                    "similarity_score": round(score, 3)
                })
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence_score": round(avg_similarity, 3),
                "has_answer_in_documents": has_answer,
                "warning": (
                    None if has_answer 
                    else "Low confidence - answer may not be in documents"
                )
            }
            
            logger.info("Simple RAG response generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    async def generate_response(
        self,
        question: str,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        use_agentic: bool = True,
        context_chunks: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        similarities: Optional[List[float]] = None
    ) -> Dict:
        """
        Main entry point for response generation.
        
        This method routes to either agentic or simple RAG based on parameters.
        
        Args:
            question (str): User's question
            embedding_service (Optional[EmbeddingService]): For agentic mode
            vector_store (Optional[VectorStore]): For agentic mode
            use_agentic (bool): Whether to use agentic RAG (default: True)
            context_chunks (Optional[List[str]]): For simple mode
            metadata (Optional[List[Dict]]): For simple mode
            similarities (Optional[List[float]]): For simple mode
            
        Returns:
            Dict: Generated response
            
        Raises:
            ValueError: If required parameters missing
            
        Example:
            >>> # Agentic mode (recommended)
            >>> response = await service.generate_response(
            ...     question="What is X?",
            ...     embedding_service=embed_svc,
            ...     vector_store=vector_store,
            ...     use_agentic=True
            ... )
            
            >>> # Simple mode
            >>> response = await service.generate_response(
            ...     question="What is X?",
            ...     use_agentic=False,
            ...     context_chunks=["chunk 1"],
            ...     metadata=[{"filename": "doc.pdf"}],
            ...     similarities=[0.9]
            ... )
        """
        if not self.is_configured():
            raise ValueError(
                "LLM service not configured. Please set OPENAI_API_KEY in .env"
            )
        
        if use_agentic:
            if not embedding_service or not vector_store:
                raise ValueError(
                    "Agentic mode requires embedding_service and vector_store"
                )
            return await self.generate_response_agentic(
                question,
                embedding_service,
                vector_store
            )
        else:
            if not context_chunks or not metadata or not similarities:
                raise ValueError(
                    "Simple mode requires context_chunks, metadata, and similarities"
                )
            return await self.generate_response_simple(
                question,
                context_chunks,
                metadata,
                similarities
            )
    
    def get_service_info(self) -> Dict:
        """
        Get information about the LLM service.
        
        Returns:
            Dict: Service configuration and status
            
        Example:
            >>> info = service.get_service_info()
            >>> print(info["llm_provider"])
            'openai'
        """
        return {
            "llm_provider": settings.LLM_PROVIDER,
            "model": settings.OPENAI_MODEL,
            "configured": self.is_configured(),
            "agentic_mode": settings.AGENTIC_MODE,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS
        }


# Create singleton instance
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get or create the singleton LLMService instance.
    
    Returns:
        LLMService: Singleton service instance
        
    Example:
        >>> service = get_llm_service()
        >>> info = service.get_service_info()
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance


# Export main classes and functions
__all__ = ["LLMService", "get_llm_service"]