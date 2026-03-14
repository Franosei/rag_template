"""Shared base class for optional agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel

from src.core.graph.schema import NodeType
from src.core.orchestration.errors import ErrorSeverity, ErrorStage, RunError
from src.core.orchestration.state import RunState
from src.llm.client import LLMClient
from src.llm.prompts import prompt_manager

T = TypeVar("T", bound=BaseModel)


class AgentResult(BaseModel):
    """Base payload returned by an agent."""

    success: bool = True
    error_message: str | None = None
    duration_ms: float = 0.0


class Agent(ABC, Generic[T]):
    """Minimal async agent wrapper that records trace nodes and failures."""

    error_stage: ErrorStage = ErrorStage.REASONING

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client
        self.agent_name = self.__class__.__name__
        self.logger = logging.getLogger(f"agents.{self.agent_name}")

    async def run(self, state: RunState) -> T:
        """Execute the agent and capture trace metadata."""

        started_at = datetime.now(tz=timezone.utc)
        node_id = state.graph.add_simple_node(
            NodeType.AGENT,
            self.agent_name,
            data={"started_at": started_at.isoformat()},
            agent_name=self.agent_name,
        )

        try:
            result = await self._execute(state)
            duration_ms = (datetime.now(tz=timezone.utc) - started_at).total_seconds() * 1000
            for node in state.graph.nodes:
                if node.node_id == node_id:
                    node.duration_ms = duration_ms
                    node.success = True
                    break
            state.total_duration_ms += duration_ms
            return result
        except Exception as exc:
            for node in state.graph.nodes:
                if node.node_id == node_id:
                    node.success = False
                    node.error = str(exc)
                    break
            state.add_error(
                RunError.from_exception(
                    stage=self.error_stage,
                    severity=ErrorSeverity.DEGRADABLE,
                    exception=exc,
                )
            )
            raise

    @abstractmethod
    async def _execute(self, state: RunState) -> T:
        """Run agent-specific logic."""

    def load_prompt(self, prompt_name: str, variables: dict[str, object] | None = None) -> str:
        """Load or render a prompt template."""

        if variables:
            return prompt_manager.render_prompt(prompt_name, variables)
        return prompt_manager.load_prompt(prompt_name)
