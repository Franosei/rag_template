import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from datetime import datetime
from pydantic import BaseModel

from src.core.orchestration.state import RunState
from src.core.graph.schema import GraphNode, NodeType
from src.llm.client import LLMClient
from src.llm.prompts import prompt_manager

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class AgentResult(BaseModel):
    """Base class for structured agent outputs"""
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: float = 0.0


class Agent(ABC, Generic[T]):
    """
    Base class for all agents in the system.
    
    Agents are async, return structured results, and automatically
    log their execution to the trace graph.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.agent_name = self.__class__.__name__
        self.logger = logging.getLogger(f"agents.{self.agent_name}")
    
    async def run(self, state: RunState) -> T:
        """
        Execute the agent. This is the public interface.
        Handles timing, graph logging, and error catching.
        """
        start_time = datetime.utcnow()
        node_id = None
        
        try:
            self.logger.info(f"Starting {self.agent_name}")
            
            # Add agent node to graph
            node_id = state.graph.add_node(GraphNode(
                node_type=NodeType.AGENT,
                label=self.agent_name,
                agent_name=self.agent_name,
                data={"started_at": start_time.isoformat()}
            ))
            
            # Execute agent logic
            result = await self._execute(state)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update node with results
            if node_id:
                for node in state.graph.nodes:
                    if node.node_id == node_id:
                        node.duration_ms = duration_ms
                        node.success = result.success if isinstance(result, AgentResult) else True
                        node.data["completed_at"] = end_time.isoformat()
                        break
            
            # Update state metrics
            state.llm_call_count += 1  # Approximate
            state.total_duration_ms += duration_ms
            
            self.logger.info(
                f"Completed {self.agent_name}",
                extra={"duration_ms": duration_ms, "success": True}
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"{self.agent_name} failed: {e}", exc_info=True)
            
            # Update node with error
            if node_id:
                for node in state.graph.nodes:
                    if node.node_id == node_id:
                        node.success = False
                        node.error = str(e)
                        break
            
            # Record error in state
            from src.core.orchestration.errors import RunError, ErrorSeverity, ErrorStage
            
            error = RunError.from_exception(
                stage=ErrorStage[self.agent_name.upper().replace("AGENT", "")],
                severity=ErrorSeverity.DEGRADABLE,  # Override in subclasses if needed
                exception=e
            )
            state.add_error(error)
            
            raise
    
    @abstractmethod
    async def _execute(self, state: RunState) -> T:
        """
        Implement agent-specific logic here.
        This is where the actual work happens.
        """
        pass
    
    def load_prompt(self, prompt_name: str, variables: dict = None) -> str:
        """Helper to load and render prompts"""
        if variables:
            return prompt_manager.render_prompt(prompt_name, variables)
        else:
            return prompt_manager.load_prompt(prompt_name)