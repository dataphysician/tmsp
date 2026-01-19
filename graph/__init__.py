"""Graph package for ICD-10-CM visualization

This package provides:
- Graph types (GraphNode, GraphEdge) for nodes and edges
- Enums (NodeCategory, EdgeType, GraphStatus) for categorization
- Tree utilities for building and traversing ICD-10-CM hierarchies

This package is shared between server and client (Pyodide) deployments.
"""

from .enums import EdgeType, GraphStatus, NodeCategory
from .types import GraphEdge, GraphNode
from .trace_tree import (
    LATERAL_KEYS,
    VERTICAL_KEYS,
    build_graph,
    build_placeholder_chain,
    data,
    extract_seventh_char,
    find_nearest_anchor,
    get_activator_nodes,
    get_lateral_links,
    get_node_category,
    get_parent_code,
    get_placeholder_codes,
    get_seventh_char_def,
    resolve_code,
    trace_ancestors,
    trace_ancestors_batch,
    trace_with_nearest_anchor,
)

__all__ = [
    # Enums
    "NodeCategory",
    "EdgeType",
    "GraphStatus",
    # Graph types
    "GraphNode",
    "GraphEdge",
    # Tree utilities
    "LATERAL_KEYS",
    "VERTICAL_KEYS",
    "build_graph",
    "build_placeholder_chain",
    "data",
    "extract_seventh_char",
    "find_nearest_anchor",
    "get_activator_nodes",
    "get_lateral_links",
    "get_node_category",
    "get_parent_code",
    "get_placeholder_codes",
    "get_seventh_char_def",
    "resolve_code",
    "trace_ancestors",
    "trace_ancestors_batch",
    "trace_with_nearest_anchor",
]