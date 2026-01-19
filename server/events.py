"""AG-UI protocol event types

These types implement the AG-UI streaming protocol for real-time graph updates:
- JsonPatchOp: RFC 6902 JSON Patch operations for incremental updates
- GraphState: Complete graph snapshot for STATE_SNAPSHOT events
- AGUIEvent: Main event wrapper for SSE streaming
"""

from typing import Any

from pydantic import BaseModel

from graph import GraphEdge, GraphNode

from .agui_enums import AGUIEventType


class JsonPatchOp(BaseModel):
    """RFC 6902 JSON Patch operation for incremental graph updates.

    Used in STATE_DELTA events to add/remove/update nodes and edges
    without sending the full graph state.
    """

    op: str  # "add" | "remove" | "replace"
    path: str
    value: dict[str, Any] | list[Any] | str | int | bool | None = None


class GraphState(BaseModel):
    """Graph state for STATE_SNAPSHOT events.

    Contains the complete current state of the graph, sent at the start
    of traversal and periodically during long operations.

    Uses core.GraphNode and core.GraphEdge which serialize enums as strings.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class AGUIEvent(BaseModel):
    """AG-UI protocol event for SSE streaming.

    Event types:
    - RUN_STARTED: Traversal begins
    - RUN_FINISHED: Traversal complete
    - STEP_STARTED: Batch processing begins
    - STEP_FINISHED: Batch complete with reasoning
    - STATE_SNAPSHOT: Full graph state
    - STATE_DELTA: Incremental updates via JSON Patch
    """

    type: AGUIEventType
    step_id: str | None = None
    state: GraphState | None = None  # For STATE_SNAPSHOT
    delta: list[JsonPatchOp] | None = None  # For STATE_DELTA
    metadata: dict[str, Any] | None = None
