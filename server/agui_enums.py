"""Server-specific enums

Contains enums used only by the server for SSE streaming protocol.
These are NOT needed for client-side (Pyodide) deployment.
"""

from enum import Enum


class AGUIEventType(str, Enum):
    """AG-UI protocol event types for SSE streaming.

    Used by the server to stream real-time updates to the frontend
    via Server-Sent Events (SSE).
    """

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
