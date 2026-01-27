"""REST API request/response payload types

These types define the contract for the REST API endpoints:
- /api/graph: Build graph from ICD-10-CM codes
- /api/node/{code}: Get node details
- /api/traverse/stream: Streaming traversal (uses server.events)
"""

from pydantic import BaseModel

from graph import GraphEdge, GraphNode


class GraphRequest(BaseModel):
    """Request body for building a graph from ICD-10-CM codes."""

    codes: list[str]


class GraphStats(BaseModel):
    """Statistics about a generated graph."""

    input_count: int
    node_count: int


class GraphResponse(BaseModel):
    """Response for /api/graph endpoint.

    Uses core.GraphNode and core.GraphEdge which serialize enums as strings.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: GraphStats
    invalid_codes: list[str] = []  # Codes that were filtered out (not in flat index)


class NodeDetailResponse(BaseModel):
    """Response for /api/node/{code} endpoint.

    Provides detailed information about a specific ICD-10-CM code.
    """

    code: str
    label: str
    depth: int
    parent: str | None
    metadata: dict[str, dict[str, str]]
    seven_chr_def: dict[str, str] | None


class TraversalRequest(BaseModel):
    """Request body for DFS traversal.

    Used by both /api/traverse and /api/traverse/stream endpoints.
    """

    clinical_note: str
    provider: str = "openai"  # "openai" | "cerebras" | "sambanova" | "anthropic" | "vertexai" | "other"
    api_key: str = ""
    model: str | None = None  # Optional model override
    selector: str = "llm"  # "llm" | "manual"
    max_tokens: int | None = None  # Optional max completion tokens
    temperature: float | None = None  # Optional temperature
    extra: dict[str, str] | None = None  # Provider-specific config (e.g., Vertex AI location/project_id)
    system_prompt: str | None = None  # Optional custom system prompt (uses default if None)
    scaffolded: bool = True  # True=tree traversal, False=zero-shot direct generation
    persist_cache: bool = True  # True=cache persists in SQLite, False=reset cache each request


class RewindRequest(BaseModel):
    """Request body for rewind traversal from a specific node.

    Used by /api/traverse/rewind endpoint to fork from a checkpoint
    and re-traverse with corrective feedback.
    """

    batch_id: str  # e.g., "E08.3|children" - identifies the checkpoint to fork from
    feedback: str  # User's corrective feedback text for the LLM
    clinical_note: str  # Original clinical note (needed to generate partition key for checkpoint lookup)
    existing_nodes: list[str] = []  # Node IDs already in graph (for lateral target parent lookup)
    provider: str = "openai"  # "openai" | "cerebras" | "sambanova" | "anthropic" | "vertexai" | "other"
    api_key: str = ""
    model: str | None = None  # Optional model override
    selector: str = "llm"  # "llm" | "manual"
    max_tokens: int | None = None  # Optional max completion tokens
    temperature: float | None = None  # Optional temperature
    extra: dict[str, str] | None = None  # Provider-specific config (e.g., Vertex AI location/project_id)
    system_prompt: str | None = None  # Optional custom system prompt (uses default if None)
    scaffolded: bool = True  # True=tree traversal, False=zero-shot direct generation
    persist_cache: bool = True  # True=cache persists in SQLite, False=reset cache each request
