"""Execution graph models for lightweight agent tracing."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Node categories recorded during a run."""

    QUERY = "query"
    AGENT = "agent"
    RETRIEVAL = "retrieval"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CLAIM = "claim"
    VERIFICATION = "verification"
    ANSWER = "answer"


class EdgeType(str, Enum):
    """Relationship types in the execution graph."""

    TRIGGERS = "triggers"
    RETRIEVES = "retrieves"
    SOURCES_FROM = "sources_from"
    SUPPORTS = "supports"
    VERIFIES = "verifies"
    SYNTHESIZES = "synthesizes"


class GraphNode(BaseModel):
    """A node in the execution trace graph."""

    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: NodeType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    label: str
    data: dict[str, object] = Field(default_factory=dict)
    agent_name: str | None = None
    duration_ms: float | None = None
    success: bool = True
    error: str | None = None


class GraphEdge(BaseModel):
    """A directed edge in the execution trace graph."""

    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    weight: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)


class ExecutionGraph(BaseModel):
    """Structured trace emitted for each user query."""

    graph_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    query: str
    query_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    total_agents_invoked: int = 0
    total_chunks_retrieved: int = 0
    total_claims_made: int = 0
    total_duration_ms: float = 0.0

    def add_node(self, node: GraphNode) -> str:
        """Add a node and return its identifier."""

        self.nodes.append(node)
        if node.node_type == NodeType.AGENT:
            self.total_agents_invoked += 1
        return node.node_id

    def add_simple_node(
        self,
        node_type: NodeType,
        label: str,
        *,
        data: dict[str, object] | None = None,
        agent_name: str | None = None,
    ) -> str:
        """Create and register a graph node with minimal boilerplate."""

        return self.add_node(
            GraphNode(
                node_type=node_type,
                label=label,
                data=data or {},
                agent_name=agent_name,
            )
        )

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge after validating both endpoints exist."""

        node_ids = {node.node_id for node in self.nodes}
        if edge.source_node_id not in node_ids:
            raise ValueError(f"Source node {edge.source_node_id} not found")
        if edge.target_node_id not in node_ids:
            raise ValueError(f"Target node {edge.target_node_id} not found")
        self.edges.append(edge)

    def link(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: EdgeType,
        *,
        weight: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Create and add a graph edge."""

        self.add_edge(
            GraphEdge(
                edge_type=edge_type,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                weight=weight,
                metadata=metadata or {},
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Export the graph as JSON-serializable data."""

        return self.model_dump(mode="json")
