"""Server-side components for TMSP

This package contains FastAPI server, AG-UI streaming events, and REST API payloads.
Uses the official ag-ui-protocol package for AG-UI event types and encoding.
Not needed for client-side Pyodide deployment.
"""

from .app import app
from .payloads import (
    GraphRequest,
    GraphResponse,
    GraphStats,
    NodeDetailResponse,
    TraversalRequest,
)

# Re-export AG-UI types from official package for convenience
from ag_ui.core import (
    EventType as AGUIEventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
    ReasoningStartEvent,
    ReasoningMessageStartEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningEndEvent,
    CustomEvent,
)
from ag_ui.encoder import EventEncoder

__all__ = [
    "app",
    "AGUIEventType",
    "RunStartedEvent",
    "RunFinishedEvent",
    "RunErrorEvent",
    "StepStartedEvent",
    "StepFinishedEvent",
    "StateSnapshotEvent",
    "StateDeltaEvent",
    "ReasoningStartEvent",
    "ReasoningMessageStartEvent",
    "ReasoningMessageContentEvent",
    "ReasoningMessageEndEvent",
    "ReasoningEndEvent",
    "CustomEvent",
    "EventEncoder",
    "GraphRequest",
    "GraphResponse",
    "GraphStats",
    "NodeDetailResponse",
    "TraversalRequest",
]
