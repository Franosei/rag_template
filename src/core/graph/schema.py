from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from uuid import uuid4

class NodeType(str, Enum):
    """Types of nodes in our execution graph"""
    QUERY = "query"
    AGENT = "agent"
    RETRIEVAL = "retrieval"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CLAIM = "claim"
    VERIFICATION = "verification"
    ANSWER = "answer"

class EdgeType(str, Enum):
    """Types of relationships"""
    TRIGGERS = "triggers"       # Query triggers agent
    RETRIEVES = "retrieves"     # Agent retrieves chunks
    SOURCES_FROM = "sources_from"  # Chunk from document
    SUPPORTS = "supports"       # Chunk supports claim
    VERIFIES = "verifies"       # Verification checks claim
    SYNTHESIZES = "synthesizes" # Answer from claims

class GraphNode(BaseModel):
    """A node in the execution trace graph"""
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: NodeType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Content depends on type
    label: str  # Human-readable label
    data: Dict[str, Any] = {}  # Type-specific payload
    
    # Metadata
    agent_name: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

class GraphEdge(BaseModel):
    """A directed edge in the execution trace graph"""
    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    
    # Optional metadata
    weight: float = 1.0  # Relevance score, confidence, etc.
    metadata: Dict[str, Any] = {}

class ExecutionGraph(BaseModel):
    """Complete trace of a query execution"""
    graph_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    query: str
    query_id: str
    
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    
    # Summary metrics
    total_agents_invoked: int = 0
    total_chunks_retrieved: int = 0
    total_claims_made: int = 0
    total_duration_ms: float = 0.0
    
    def add_node(self, node: GraphNode) -> str:
        """Add node and return its ID"""
        self.nodes.append(node)
        return node.node_id
    
    def add_edge(self, edge: GraphEdge):
        """Add edge between existing nodes"""
        # Validate nodes exist
        node_ids = {n.node_id for n in self.nodes}
        if edge.source_node_id not in node_ids:
            raise ValueError(f"Source node {edge.source_node_id} not found")
        if edge.target_node_id not in node_ids:
            raise ValueError(f"Target node {edge.target_node_id} not found")
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to JSON-serializable dict"""
        return self.model_dump(mode="json")