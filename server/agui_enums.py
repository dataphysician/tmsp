"""Server-specific enums

Contains enums used only by the server for SSE streaming protocol.
These are NOT needed for client-side (Pyodide) deployment.
"""

from enum import Enum


class AGUIEventType(str, Enum):
    """AG-UI protocol event types for SSE streaming.

    Implements AG-UI protocol v0.1 event types.
    Used by the server to stream real-time updates to the frontend
    via Server-Sent Events (SSE).

    Run lifecycle:
    - RUN_STARTED: Emitted when a traversal begins
    - RUN_FINISHED: Emitted when a traversal completes successfully
    - RUN_ERROR: Emitted when a traversal fails with an error

    Step lifecycle:
    - STEP_STARTED: Emitted when a batch processing step begins
    - STEP_FINISHED: Emitted when a batch processing step completes

    State updates:
    - STATE_SNAPSHOT: Full graph state (nodes + edges)
    - STATE_DELTA: Incremental updates via JSON Patch (RFC 6902)
    """

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
