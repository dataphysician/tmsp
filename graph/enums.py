"""Graph enums for the ICD-10-CM traversal system

Provides enums used for graph nodes, edges, and traversal status.
These are shared between server and client (Pyodide) deployments.
"""

from enum import Enum


class NodeCategory(str, Enum):
    """Category of a node in the ICD-10-CM graph."""

    ROOT = "root"
    FINALIZED = "finalized"
    ACTIVATOR = "activator"
    PLACEHOLDER = "placeholder"
    ANCESTOR = "ancestor"


class EdgeType(str, Enum):
    """Type of edge connecting nodes in the graph."""

    HIERARCHY = "hierarchy"
    LATERAL = "lateral"


class GraphStatus(str, Enum):
    """Status of the DFS traversal process."""

    IDLE = "idle"
    TRAVERSING = "traversing"
    COMPLETE = "complete"
    ERROR = "error"
