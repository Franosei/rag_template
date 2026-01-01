from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4

from src.core.graph.schema import ExecutionGraph
from src.core.orchestration.errors import RunError, DegradedMode
from src.core.policies.folder_policy import FolderPolicy


class RunState(BaseModel):
    """
    Mutable state object passed through the agent pipeline.
    Contains query, intermediate results, and trace graph.
    """
    
    # Identity
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Input
    query: str
    user_context: Dict[str, Any] = {}  # Optional user metadata
    
    # Configuration
    max_retrieval_results: int = 20
    enable_tables: bool = True
    enable_vision: bool = True
    require_citations: bool = True
    
    # Trace Graph
    graph: ExecutionGraph = Field(default_factory=lambda: ExecutionGraph(
        query="", query_id=""
    ))
    
    # Agent Outputs (accumulated)
    selected_folders: List[FolderPolicy] = []
    retrieved_chunks: List[Dict[str, Any]] = []  # We'll define proper Chunk model later
    analyzed_tables: List[Dict[str, Any]] = []
    analyzed_images: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []  # We'll define Claim model later
    verifications: List[Dict[str, Any]] = []
    
    # Final Output
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    
    # Error Tracking
    errors: List[RunError] = []
    degraded_mode: Optional[DegradedMode] = None
    
    # Metrics
    total_duration_ms: float = 0.0
    llm_call_count: int = 0
    total_llm_tokens: int = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_error(self, error: RunError):
        """Add an error to the run"""
        self.errors.append(error)
        logger.warning(
            f"Run error recorded: {error.stage} - {error.message}",
            extra={"error": error.model_dump()}
        )
    
    def is_degraded(self) -> bool:
        """Check if run is in degraded mode"""
        return self.degraded_mode is not None
    
    def has_fatal_error(self) -> bool:
        """Check if any fatal error occurred"""
        from src.core.orchestration.errors import ErrorSeverity
        return any(e.severity == ErrorSeverity.FATAL for e in self.errors)