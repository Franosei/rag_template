"""Mutable query run state."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.graph.schema import ExecutionGraph
from src.core.orchestration.errors import DegradedMode, ErrorSeverity, RunError
from src.core.policies.folder_policy import FolderPolicy

logger = logging.getLogger(__name__)


class RunState(BaseModel):
    """Shared state passed through the retrieval and reasoning flow."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    query: str
    user_context: dict[str, object] = Field(default_factory=dict)
    max_retrieval_results: int = 20
    enable_tables: bool = True
    enable_vision: bool = False
    require_citations: bool = True
    graph: ExecutionGraph | None = None
    selected_folders: list[FolderPolicy] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, object]] = Field(default_factory=list)
    analyzed_tables: list[dict[str, object]] = Field(default_factory=list)
    analyzed_images: list[dict[str, object]] = Field(default_factory=list)
    claims: list[dict[str, object]] = Field(default_factory=list)
    verifications: list[dict[str, object]] = Field(default_factory=list)
    answer: str | None = None
    citations: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[RunError] = Field(default_factory=list)
    degraded_mode: DegradedMode | None = None
    total_duration_ms: float = 0.0
    llm_call_count: int = 0
    total_llm_tokens: int = 0

    def model_post_init(self, __context: object) -> None:
        """Create the execution graph lazily after the query is known."""

        if self.graph is None:
            self.graph = ExecutionGraph(query=self.query, query_id=self.run_id)

    def add_error(self, error: RunError) -> None:
        """Record a structured error for the run."""

        self.errors.append(error)
        logger.warning(
            "Run error recorded",
            extra={"stage": error.stage.value, "severity": error.severity.value, "message": error.message},
        )

    def add_warning(self, message: str) -> None:
        """Record a non-fatal warning for the run."""

        self.warnings.append(message)
        logger.info("Run warning recorded", extra={"warning": message})

    def is_degraded(self) -> bool:
        """Return whether the run is operating in degraded mode."""

        return self.degraded_mode is not None

    def has_fatal_error(self) -> bool:
        """Return whether a fatal error has been recorded."""

        return any(error.severity == ErrorSeverity.FATAL for error in self.errors)
