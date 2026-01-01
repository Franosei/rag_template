from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ErrorSeverity(str, Enum):
    """How bad is this error?"""
    FATAL = "fatal"           # Cannot proceed at all
    DEGRADABLE = "degradable" # Proceed with reduced capability
    RECOVERABLE = "recoverable" # Try fallback


class ErrorStage(str, Enum):
    """Where did the error occur?"""
    INITIALIZATION = "initialization"
    FOLDER_PROFILING = "folder_profiling"
    SCOPE_SELECTION = "scope_selection"
    RETRIEVAL = "retrieval"
    TABLE_ANALYSIS = "table_analysis"
    VISION_ANALYSIS = "vision_analysis"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    WRITING = "writing"
    GRAPH_EXPORT = "graph_export"


class RunError(BaseModel):
    """Structured error that gets logged to trace graph"""
    error_id: str = Field(default_factory=lambda: f"err_{int(datetime.utcnow().timestamp() * 1000)}")
    stage: ErrorStage
    severity: ErrorSeverity
    message: str
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    remediation: Optional[str] = None  # What we tried to do about it
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_exception(
        cls,
        stage: ErrorStage,
        severity: ErrorSeverity,
        exception: Exception,
        remediation: Optional[str] = None
    ) -> "RunError":
        """Create RunError from a caught exception"""
        return cls(
            stage=stage,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            remediation=remediation
        )


class DegradedMode(BaseModel):
    """Describes what capabilities are unavailable"""
    reason: str
    missing_capabilities: list[str]
    available_alternatives: list[str] = []