"""
Agentic RAG Agent

This module implements an intelligent agentic RAG system with multi-step reasoning,
self-verification, and iterative refinement for near-perfect accuracy.

The agent uses six key steps:
1. Query Decomposition - Break complex questions into sub-questions
2. Query Rewriting - Generate alternative phrasings
3. Hybrid Retrieval - Search with multiple query variants
4. LLM Re-ranking - Intelligently rank results by relevance
5. Generation with Verification - Generate and verify answers
6. Iterative Refinement - Improve answers until verified

Classes:
    AgenticRAG: Main agentic RAG orchestrator

Author: Francis Osei
"""

import openai
from typing import List, Dict, Tuple, Optional
import logging
import json

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class AgenticRAG:
    """
    Agentic RAG system with intelligent multi-step reasoning.
    
    This class implements an advanced RAG workflow that goes far beyond
    simple retrieval and generation. It uses multiple techniques to
    achieve near-perfect accuracy:
    
    - Query decomposition for complex questions
    - Multiple query variants for comprehensive retrieval
    - LLM-based re-ranking for better relevance
    - Answer verification against sources
    - Iterative refinement until verified
    
    Attributes:
        llm_client (openai.OpenAI): OpenAI API client
        embedding_service (EmbeddingService): Embedding generation service
        vector_store (VectorStore): Vector database
        max_iterations (int): Maximum refinement attempts
        
    Example:
        >>> agent = AgenticRAG(llm_client, embed_svc, vector_store)
        >>> result = await agent.answer_question("What is the policy?")
        >>> print(result["verified"])
        True
        >>> print(result["confidence"])
        0.94
    """
    
    def __init__(
        self,
        llm_client: openai.OpenAI,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        max_iterations: Optional[int] = None
    ):
        """
        Initialize the agentic RAG system.
        
        Args:
            llm_client (openai.OpenAI): OpenAI client instance
            embedding_service (EmbeddingService): Service for embeddings
            vector_store (VectorStore): Vector database service
            max_iterations (Optional[int]): Max refinement iterations
            
        Example:
            >>> agent = AgenticRAG(client, embed_svc, store)
        """
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.max_iterations = max_iterations or settings.MAX_ITERATIONS
        
        logger.info(
            f"AgenticRAG initialized with max_iterations={self.max_iterations}"
        )
    
    async def query_decomposition(self, question: str) -> List[str]:
        """
        Break complex question into simpler sub-questions.
        
        This helps handle multi-part questions by addressing each
        component separately for more comprehensive answers.
        
        Args:
            question (str): Original complex question
            
        Returns:
            List[str]: List of sub-questions (including original)
            
        Example:
            >>> sub_qs = await agent.query_decomposition(
            ...     "What were Q3 revenues and main growth drivers?"
            ... )
            >>> print(sub_qs)
            ['What were Q3 revenues?', 'What were main growth drivers?']
        """
        if not settings.ENABLE_QUERY_DECOMPOSITION:
            return [question]
        
        try:
            prompt = f"""Break down this complex question into 2-4 simpler sub-questions that, when answered together, fully address the main question.

Main Question: {question}

Return ONLY a JSON array of sub-questions, nothing else:
["sub-question 1", "sub-question 2", ...]"""

            response = self.llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            sub_questions = json.loads(content)
            
            # Validate and clean
            if not isinstance(sub_questions, list):
                logger.warning("Query decomposition returned non-list, using original")
                return [question]
            
            # Filter out empty strings
            sub_questions = [q.strip() for q in sub_questions if q.strip()]
            
            if not sub_questions:
                return [question]
            
            logger.info(
                f"Decomposed question into {len(sub_questions)} sub-questions"
            )
            return sub_questions
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using original")
            return [question]
    
    async def query_rewriting(
        self,
        question: str,
        iteration: int = 0
    ) -> List[str]:
        """
        Generate alternative phrasings of the question.
        
        Different phrasings help retrieve documents that use
        different terminology for the same concepts.
        
        Args:
            question (str): Original question
            iteration (int): Current refinement iteration
            
        Returns:
            List[str]: List including original and alternatives
            
        Example:
            >>> variants = await agent.query_rewriting("What is the policy?")
            >>> print(len(variants))
            4
        """
        if not settings.ENABLE_QUERY_REWRITING:
            return [question]
        
        try:
            prompt = f"""Generate 3 alternative ways to ask this question that might help find relevant information:

Original: {question}

Return ONLY a JSON array of rephrased questions:
["rephrase 1", "rephrase 2", "rephrase 3"]"""

            response = self.llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            alternatives = json.loads(content)
            
            if not isinstance(alternatives, list):
                return [question]
            
            alternatives = [q.strip() for q in alternatives if q.strip()]
            
            # Include original question
            all_variants = [question] + alternatives
            
            logger.info(f"Generated {len(all_variants)} query variants")
            return all_variants
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original")
            return [question]
    
    async def hybrid_retrieval(
        self,
        questions: List[str],
        top_k: int = 10
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Retrieve documents using multiple query variations.
        
        This comprehensive retrieval ensures we don't miss relevant
        documents due to terminology differences.
        
        Args:
            questions (List[str]): List of query variants
            top_k (int): Number of results per query
            
        Returns:
            Tuple of:
                - List[str]: Retrieved document chunks
                - List[Dict]: Metadata for each chunk
                - List[float]: Similarity scores
                
        Example:
            >>> docs, meta, scores = await agent.hybrid_retrieval(
            ...     ["query 1", "query 2"],
            ...     top_k=10
            ... )
        """
        all_docs = []
        all_metadata = []
        all_scores = []
        seen_chunks = set()
        
        logger.debug(f"Performing hybrid retrieval with {len(questions)} queries")
        
        for q in questions:
            try:
                # Get embedding
                embedding = self.embedding_service.encode_query(q)
                
                # Search
                docs, metadata, scores = self.vector_store.search(
                    query_embedding=embedding,
                    top_k=top_k
                )
                
                # Deduplicate and collect
                for doc, meta, score in zip(docs, metadata, scores):
                    # Create unique ID for deduplication
                    doc_id = f"{meta.get('doc_id')}_{meta.get('chunk_index')}"
                    
                    if doc_id not in seen_chunks:
                        seen_chunks.add(doc_id)
                        all_docs.append(doc)
                        all_metadata.append(meta)
                        all_scores.append(score)
                        
            except Exception as e:
                logger.warning(f"Error retrieving for query '{q}': {e}")
                continue
        
        # Sort by score
        if all_docs:
            sorted_results = sorted(
                zip(all_docs, all_metadata, all_scores),
                key=lambda x: x[2],
                reverse=True
            )
            
            # Return top results
            top_n = min(top_k, len(sorted_results))
            docs, meta, scores = zip(*sorted_results[:top_n])
            
            logger.info(
                f"Hybrid retrieval found {len(docs)} unique chunks, "
                f"best score: {max(scores):.3f}"
            )
            
            return list(docs), list(meta), list(scores)
        
        logger.warning("Hybrid retrieval found no documents")
        return [], [], []
    
    async def rerank_results(
        self,
        question: str,
        documents: List[str],
        metadata: List[Dict],
        scores: List[float]
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Re-rank retrieved documents using LLM for better relevance.
        
        Vector similarity doesn't always correlate with semantic relevance.
        The LLM can better judge which documents truly answer the question.
        
        Args:
            question (str): Original question
            documents (List[str]): Retrieved documents
            metadata (List[Dict]): Document metadata
            scores (List[float]): Original similarity scores
            
        Returns:
            Tuple of re-ranked documents, metadata, and boosted scores
            
        Example:
            >>> docs, meta, scores = await agent.rerank_results(
            ...     "What is X?", docs, meta, scores
            ... )
        """
        if not documents or len(documents) <= settings.RERANK_TOP_K:
            return documents, metadata, scores
        
        try:
            # Prepare documents for ranking (limit to top candidates)
            top_candidates = min(settings.RERANK_TOP_K, len(documents))
            doc_texts = []
            
            for i in range(top_candidates):
                preview = documents[i][:300]  # First 300 chars
                doc_texts.append(f"Document {i+1}:\n{preview}...")
            
            prompt = f"""Rank these documents by relevance to the question. Return ONLY a JSON array of document numbers in order of relevance (most relevant first).

Question: {question}

{chr(10).join(doc_texts)}

Return format: [3, 1, 5, 2, 4, ...]"""

            response = self.llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            rankings = json.loads(content)
            
            # Validate rankings
            if not isinstance(rankings, list):
                logger.warning("Invalid re-ranking format, using original order")
                return documents, metadata, scores
            
            # Reorder based on LLM ranking
            reranked_docs = []
            reranked_meta = []
            reranked_scores = []
            
            for rank_pos, rank in enumerate(rankings):
                idx = rank - 1  # Convert to 0-indexed
                if 0 <= idx < len(documents):
                    reranked_docs.append(documents[idx])
                    reranked_meta.append(metadata[idx])
                    # Boost score based on LLM ranking position
                    boost = 1.0 + (len(rankings) - rank_pos) / len(rankings)
                    reranked_scores.append(scores[idx] * boost)
            
            # Add any remaining documents not in rankings
            ranked_indices = {r - 1 for r in rankings if 0 <= r - 1 < len(documents)}
            for i in range(len(documents)):
                if i not in ranked_indices:
                    reranked_docs.append(documents[i])
                    reranked_meta.append(metadata[i])
                    reranked_scores.append(scores[i] * 0.5)  # Lower boost
            
            logger.info("Documents re-ranked by LLM")
            return reranked_docs, reranked_meta, reranked_scores
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}, using original order")
            return documents, metadata, scores
    
    async def generate_with_verification(
        self,
        question: str,
        documents: List[str],
        metadata: List[Dict]
    ) -> Dict:
        """
        Generate answer and verify it against sources.
        
        This two-step process ensures the answer is grounded in
        the provided documents and catches hallucinations.
        
        Args:
            question (str): Original question
            documents (List[str]): Top relevant documents
            metadata (List[Dict]): Document metadata
            
        Returns:
            Dict containing:
                - answer (str): Generated answer
                - verified (bool): Verification status
                - verification_details (Dict): Verification info
                - sources_used (List[str]): Documents used
                - metadata (List[Dict]): Metadata
                
        Example:
            >>> result = await agent.generate_with_verification(
            ...     "What is X?", docs, metadata
            ... )
            >>> print(result["verified"])
            True
        """
        if not settings.ENABLE_VERIFICATION:
            # Skip verification if disabled
            logger.info("Verification disabled, generating without verification")
        
        # Build context
        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents[:5], metadata[:5]), 1):
            filename = meta.get('filename', 'Unknown')
            context_parts.append(f"[Source {i} - {filename}]\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        # Generation prompt with strict instructions
        prompt = f"""You are a precise AI assistant. Answer the question using ONLY the provided sources.

CRITICAL RULES:
1. Answer MUST be directly supported by the sources
2. Quote exact phrases from sources in your answer
3. If information is not in sources, say "This information is not available in the provided documents"
4. Cite source numbers for every claim [Source N]

SOURCES:
{context}

QUESTION: {question}

ANSWER (with citations):"""

        try:
            response = self.llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=settings.MAX_TOKENS
            )
            
            answer = response.choices[0].message.content
            
            # Verify answer if enabled
            if settings.ENABLE_VERIFICATION:
                verification = await self.verify_answer(question, answer, documents)
            else:
                verification = {
                    "is_valid": True,
                    "confidence": 0.8,
                    "issues": [],
                    "supporting_quotes": []
                }
            
            result = {
                "answer": answer,
                "verified": verification["is_valid"],
                "verification_details": verification,
                "sources_used": documents[:5],
                "metadata": metadata[:5]
            }
            
            logger.info(
                f"Generated answer, verified={verification['is_valid']}, "
                f"confidence={verification['confidence']:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
    
    async def verify_answer(
        self,
        question: str,
        answer: str,
        sources: List[str]
    ) -> Dict:
        """
        Verify that the answer is supported by sources.
        
        This critical step catches hallucinations and unsupported claims.
        
        Args:
            question (str): Original question
            answer (str): Generated answer
            sources (List[str]): Source documents
            
        Returns:
            Dict with verification results:
                - is_valid (bool): Whether answer is valid
                - confidence (float): Confidence in verification
                - issues (List[str]): Any problems found
                - supporting_quotes (List[str]): Supporting evidence
                
        Example:
            >>> verification = await agent.verify_answer(
            ...     "What is X?", "X is Y", sources
            ... )
            >>> print(verification["is_valid"])
            True
        """
        sources_text = "\n\n".join([
            f"Source {i+1}:\n{s}" for i, s in enumerate(sources[:5])
        ])
        
        prompt = f"""Verify if this answer is fully supported by the provided sources. Check for:
1. Factual accuracy
2. Source support
3. No hallucinations

QUESTION: {question}

ANSWER: {answer}

SOURCES:
{sources_text}

Return ONLY a JSON object:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue1", "issue2"] or [],
    "supporting_quotes": ["quote1", "quote2"]
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            verification = json.loads(content)
            
            logger.info(
                f"Verification complete: valid={verification.get('is_valid')}, "
                f"confidence={verification.get('confidence', 0):.2f}"
            )
            
            return verification
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues": ["Verification process failed"],
                "supporting_quotes": []
            }
    
    async def iterative_refinement(
        self,
        question: str,
        initial_answer: Dict,
        iteration: int = 0
    ) -> Dict:
        """
        Refine answer if verification fails.
        
        This recursive process tries alternative approaches until
        a verified answer is obtained or max iterations reached.
        
        Args:
            question (str): Original question
            initial_answer (Dict): Previous answer attempt
            iteration (int): Current iteration number
            
        Returns:
            Dict: Refined answer (same format as generate_with_verification)
            
        Example:
            >>> refined = await agent.iterative_refinement(
            ...     "What is X?", initial_result, iteration=0
            ... )
        """
        # Stop if verified or max iterations reached
        if initial_answer["verified"] or iteration >= self.max_iterations:
            logger.info(
                f"Refinement stopping: verified={initial_answer['verified']}, "
                f"iteration={iteration}"
            )
            return initial_answer
        
        logger.info(f"Starting refinement iteration {iteration + 1}")
        
        # Try alternative query phrasings
        alt_questions = await self.query_rewriting(question, iteration)
        
        # Retrieve with new queries
        docs, meta, scores = await self.hybrid_retrieval(alt_questions, top_k=15)
        
        if not docs:
            logger.warning("No documents in refinement, returning initial answer")
            return initial_answer
        
        # Re-rank
        docs, meta, scores = await self.rerank_results(question, docs, meta, scores)
        
        # Generate new answer
        refined = await self.generate_with_verification(question, docs, meta)
        
        # Recursive refinement if still not verified
        if not refined["verified"] and iteration < self.max_iterations - 1:
            logger.info("Answer still unverified, continuing refinement")
            return await self.iterative_refinement(question, refined, iteration + 1)
        
        return refined
    
    async def answer_question(self, question: str) -> Dict:
        """
        Main agentic workflow to answer question with high accuracy.
        
        This orchestrates all six steps of the agentic RAG process:
        1. Query decomposition
        2. Query rewriting
        3. Hybrid retrieval
        4. Re-ranking
        5. Generation with verification
        6. Iterative refinement
        
        Args:
            question (str): User's question
            
        Returns:
            Dict containing:
                - answer (str): Final answer
                - verified (bool): Verification status
                - confidence (float): Confidence score
                - sources (List[Dict]): Source information
                - verification (Dict): Verification details
                - workflow_steps (Dict): Workflow statistics
                
        Example:
            >>> result = await agent.answer_question(
            ...     "What were the main risks in 2023?"
            ... )
            >>> print(f"Answer: {result['answer']}")
            >>> print(f"Verified: {result['verified']}")
            >>> print(f"Confidence: {result['confidence']}")
        """
        logger.info(f"=== Starting Agentic RAG for: {question} ===")
        
        # Step 1: Query Decomposition
        sub_questions = await self.query_decomposition(question)
        logger.info(f"Step 1: Decomposed into {len(sub_questions)} questions")
        
        # Step 2: Query Rewriting
        all_queries = []
        for sq in sub_questions:
            variants = await self.query_rewriting(sq)
            all_queries.extend(variants)
        logger.info(f"Step 2: Generated {len(all_queries)} query variants")
        
        # Step 3: Hybrid Retrieval
        documents, metadata, scores = await self.hybrid_retrieval(
            all_queries,
            top_k=20
        )
        logger.info(f"Step 3: Retrieved {len(documents)} unique documents")
        
        if not documents:
            logger.warning("No documents found for question")
            return {
                "answer": "I could not find any relevant information in the uploaded documents to answer this question.",
                "verified": False,
                "confidence": 0.0,
                "sources": [],
                "verification": {
                    "is_valid": False,
                    "confidence": 0.0,
                    "issues": ["No documents found"],
                    "supporting_quotes": []
                },
                "workflow_steps": {
                    "sub_questions": sub_questions,
                    "queries_generated": len(all_queries),
                    "documents_retrieved": 0,
                    "refinement_iterations": 0
                }
            }
        
        # Step 4: Re-ranking
        documents, metadata, scores = await self.rerank_results(
            question, documents, metadata, scores
        )
        logger.info("Step 4: Re-ranked documents")
        
        # Step 5: Generate with Verification
        initial_result = await self.generate_with_verification(
            question, documents, metadata
        )
        logger.info(f"Step 5: Generated answer, verified={initial_result['verified']}")
        
        # Step 6: Iterative Refinement if needed
        if not initial_result["verified"] and self.max_iterations > 0:
            logger.info("Step 6: Answer not verified, starting refinement...")
            final_result = await self.iterative_refinement(question, initial_result)
        else:
            final_result = initial_result
        
        # Prepare final response
        response = {
            "answer": final_result["answer"],
            "verified": final_result["verified"],
            "confidence": final_result["verification_details"].get("confidence", 0.0),
            "sources": [
                {
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "document_name": meta.get("filename", "Unknown"),
                    "chunk_id": f"chunk_{i}",
                    "similarity_score": round(score, 3)
                }
                for i, (doc, meta, score) in enumerate(
                    zip(
                        final_result["sources_used"],
                        final_result["metadata"],
                        scores[:len(final_result["sources_used"])]
                    )
                )
            ],
            "verification": final_result["verification_details"],
            "workflow_steps": {
                "sub_questions": len(sub_questions),
                "queries_generated": len(all_queries),
                "documents_retrieved": len(documents),
                "refinement_iterations": self.max_iterations
            }
        }
        
        logger.info(
            f"=== Agentic RAG Complete: verified={response['verified']}, "
            f"confidence={response['confidence']:.2f} ==="
        )
        
        return response


# Export main class
__all__ = ["AgenticRAG"]