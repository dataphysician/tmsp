"""AG-UI protocol event types

These types implement the AG-UI streaming protocol for real-time graph updates:
- JsonPatchOp: RFC 6902 JSON Patch operations for incremental updates
- GraphState: Complete graph snapshot for STATE_SNAPSHOT events
- AGUIEvent: Main event wrapper for SSE streaming
"""

from typing import Any, Literal

from pydantic import BaseModel

from graph import GraphEdge, GraphNode

from .agui_enums import AGUIEventType


class JsonPatchOp(BaseModel):
    """RFC 6902 JSON Patch operation for incremental graph updates.

    Used in STATE_DELTA events to add/remove/update nodes and edges
    without sending the full graph state.
    """

    op: Literal["add", "remove", "replace"]
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

    Implements AG-UI protocol v0.1 event envelope schema.

    Event types:
    - RUN_STARTED: Traversal begins (includes threadId, runId)
    - RUN_FINISHED: Traversal complete (includes threadId, runId)
    - RUN_ERROR: Traversal failed (includes threadId, runId, error message)
    - STEP_STARTED: Batch processing begins (includes stepName)
    - STEP_FINISHED: Batch complete with reasoning (includes stepName)
    - STATE_SNAPSHOT: Full graph state (includes snapshot)
    - STATE_DELTA: Incremental updates via JSON Patch (includes delta)

    Required fields by event type:
    - RUN_STARTED/RUN_FINISHED/RUN_ERROR: threadId, runId
    - STEP_STARTED/STEP_FINISHED: stepName
    - STATE_SNAPSHOT: snapshot
    - STATE_DELTA: delta
    """

    type: AGUIEventType

    # Run lifecycle fields (RUN_STARTED, RUN_FINISHED, RUN_ERROR)
    threadId: str | None = None
    runId: str | None = None
    parentRunId: str | None = None  # For rewind runs that fork from a previous run

    # Step lifecycle fields (STEP_STARTED, STEP_FINISHED)
    stepName: str | None = None

    # State fields
    snapshot: GraphState | None = None  # For STATE_SNAPSHOT
    delta: list[JsonPatchOp] | None = None  # For STATE_DELTA

    # Error field (RUN_ERROR)
    error: str | None = None

    # Additional metadata
    metadata: dict[str, Any] | None = None
