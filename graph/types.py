"""Graph types for ICD-10-CM visualization

These types are the single source of truth for graph nodes and edges,
used across REST API, AG-UI streaming, and agent state.

Key feature: ConfigDict(use_enum_values=True) ensures enums serialize
as strings (e.g., "root") rather than enum objects, matching frontend expectations.
"""

from pydantic import BaseModel, ConfigDict

from .enums import EdgeType, NodeCategory


class GraphNode(BaseModel):
    """A node in the ICD-10-CM graph.

    Used for:
    - REST API responses (GraphResponse)
    - AG-UI streaming state snapshots
    - Internal agent state
    """

    model_config = ConfigDict(use_enum_values=True)

    id: str
    code: str
    label: str
    depth: int
    category: NodeCategory
    billable: bool = False


class GraphEdge(BaseModel):
    """An edge connecting nodes in the ICD-10-CM graph.

    Used for:
    - REST API responses (GraphResponse)
    - AG-UI streaming state snapshots
    - Internal agent state
    """

    model_config = ConfigDict(use_enum_values=True)

    source: str
    target: str
    edge_type: EdgeType
    rule: str | None = None  # e.g., "codeFirst", "codeAlso", "sevenChrDef"
