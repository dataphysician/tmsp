"""FastAPI server for ICD-10-CM Tree Traversal visualization

Includes:
- REST API for graph building (verification)
- AG-UI streaming endpoint for Burr-based traversal
- SSE (Server-Sent Events) for real-time batch completion updates
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, TypedDict

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import types from consolidated modules
from graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeCategory,
    build_graph,
    data,
    extract_seventh_char,
    get_activator_nodes,
    get_node_category,
    get_parent_code,
    trace_seventh_char_def,
    resolve_code,
)

from ag_ui.core import (
    RunAgentInput,
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
from pydantic import BaseModel as _PydanticBaseModel


# AG-UI encoder instance (serializes events to SSE format with camelCase keys)
_agui_encoder = EventEncoder()


from ag_ui.core import BaseEvent as _BaseEvent, EventType as _EventType


def encode_event(event: _BaseEvent) -> str:
    """Encode an AG-UI event to SSE, injecting a millisecond timestamp."""
    return _agui_encoder.encode(
        event.model_copy(update={"timestamp": int(time.time() * 1000)})
    )


def _reasoning_message_start(message_id: str) -> ReasoningMessageStartEvent:
    """Create ReasoningMessageStartEvent with role='reasoning'.

    The TypeScript @ag-ui/core v0.0.45 schema requires role='reasoning',
    but the Python ag-ui-protocol v0.1.11 model restricts role to Literal['assistant'].
    Use model_construct() to bypass the Python validation.
    """
    return ReasoningMessageStartEvent.model_construct(
        type=_EventType.REASONING_MESSAGE_START,
        message_id=message_id,
        role="reasoning",
    )


def get_icd_parent(code: str) -> str | None:
    """Get the actual ICD-10 hierarchy parent of a code from the index."""
    node_info = data.get(code)
    if not node_info:
        return None
    parent = node_info.get("parent")
    if not parent or not isinstance(parent, dict):
        return None
    parent_codes = list(parent.keys())
    return parent_codes[0] if parent_codes else None


def is_hierarchy_edge(parent_id: str, node_id: str) -> bool:
    """Check if parent_id → node_id is a hierarchy or pseudohierarchy edge.

    Returns True for:
    - Actual ICD-10 parent-child relationships (from the index)
    - Placeholder/sevenChrDef paths where node_id is a code descendant
      of parent_id (e.g. T84.53 → T84.53X, T88.7X → T88.7XX)
    """
    # Direct parent match from the index
    actual_parent = get_icd_parent(node_id)
    if parent_id == actual_parent:
        return True
    # Code-prefix check covers placeholder nodes not in the index
    parent_norm = parent_id.replace(".", "")
    node_norm = node_id.replace(".", "")
    return node_norm.startswith(parent_norm) and len(node_norm) > len(parent_norm)


# Helper models for constructing graph state (kept from old events.py)
class JsonPatchOp(_PydanticBaseModel):
    """RFC 6902 JSON Patch operation for incremental graph updates."""
    op: Literal["add", "remove", "replace"]
    path: str
    value: dict[str, Any] | list[Any] | str | int | bool | None = None


class GraphState(_PydanticBaseModel):
    """Graph state for STATE_SNAPSHOT events."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
from .payloads import (
    GraphRequest,
    GraphResponse,
    GraphStats,
    NodeDetailResponse,
    RewindRequest,
    TraversalRequest,
)

# Zero-shot Burr app imports
from agent.zero_shot import (
    build_zero_shot_app,
    generate_zero_shot_cache_key,
    initialize_zero_shot_persister,
    ZERO_SHOT_PERSISTER,
)

# Scaffolded traversal cache key generator and state management
from agent.traversal import generate_traversal_cache_key, delete_persisted_state, PERSISTER, initialize_persister

# Track partition keys created during this server session (for selective cache clearing)
SESSION_PARTITION_KEYS: set[str] = set()

# Reference to current/last Burr app (for cancel endpoint and clear cache)
CURRENT_APP = None  # Type: Application | None (from burr.core)

# Reference to currently running async task (for cancel endpoint)
RUNNING_TASK = None  # Type: asyncio.Task | None

# ============================================================================
# Cache Version Tracking (Soft Invalidation)
# ============================================================================
# Maps config_hash -> version. When user invalidates, version increments.
# Key generation includes version, so old entries are orphaned (not deleted).
# Persisted to disk to survive server restarts (including hot reload).
CACHE_VERSIONS_FILE = Path(__file__).parent / "cache_versions.json"


def _load_cache_versions() -> dict[str, int]:
    """Load cache versions from disk."""
    if CACHE_VERSIONS_FILE.exists():
        try:
            return json.loads(CACHE_VERSIONS_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[CACHE] Warning: Failed to load cache_versions.json: {e}")
            return {}
    return {}


def _save_cache_versions() -> None:
    """Save cache versions to disk."""
    try:
        CACHE_VERSIONS_FILE.write_text(json.dumps(CACHE_VERSIONS))
    except OSError as e:
        print(f"[CACHE] Warning: Failed to save cache_versions.json: {e}")


CACHE_VERSIONS: dict[str, int] = _load_cache_versions()


def _compute_config_hash(
    clinical_note: str,
    provider: str,
    model: str,
    temperature: float,
    system_prompt: str | None,
    scaffolded: bool,
) -> str:
    """Compute a hash for cache version tracking.

    This identifies a unique configuration for version tracking purposes.
    """
    import hashlib
    key_str = (
        f"{provider}|{model}|{round(temperature, 2)}|"
        f"{(system_prompt or '').strip()}|{clinical_note.strip()}|"
        f"{'scaffolded' if scaffolded else 'zeroshot'}"
    )
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def get_cache_version(
    clinical_note: str,
    provider: str,
    model: str,
    temperature: float,
    system_prompt: str | None = None,
    scaffolded: bool = True,
) -> int:
    """Get current cache version for a configuration."""
    config_hash = _compute_config_hash(
        clinical_note, provider, model, temperature, system_prompt, scaffolded
    )
    version = CACHE_VERSIONS.get(config_hash, 0)
    print(f"[CACHE] get_cache_version hash={config_hash}, version={version}, model={model!r}, scaffolded={scaffolded}")
    return version


def increment_cache_version(
    clinical_note: str,
    provider: str,
    model: str,
    temperature: float,
    system_prompt: str | None = None,
    scaffolded: bool = True,
) -> int:
    """Increment and return new cache version for a configuration."""
    config_hash = _compute_config_hash(
        clinical_note, provider, model, temperature, system_prompt, scaffolded
    )
    current = CACHE_VERSIONS.get(config_hash, 0)
    new_version = current + 1
    CACHE_VERSIONS[config_hash] = new_version
    _save_cache_versions()  # Persist to disk
    print(f"[CACHE] increment_cache_version hash={config_hash}: {current} -> {new_version}, model={model!r}, scaffolded={scaffolded}")
    return new_version

# Helper to check if a node DIRECTLY has sevenChrDef metadata
# Note: This is for visual "activator" category only - NOT for sevenChrDef processing
# Nodes that inherit sevenChrDef from ancestors (self-activation) are NOT activators visually
def node_has_seven_chr_def(code: str) -> bool:
    """Check if a node directly has sevenChrDef metadata.

    Only returns True for nodes like T36, T88 that have sevenChrDef in their own metadata.
    Returns False for descendants like T36.1, T88.7 that inherit via self-activation.
    """
    if code not in data:
        return False
    entry = data[code]
    metadata = entry.get("metadata", {})
    return bool(metadata.get("sevenChrDef"))


def format_with_seventh_char(base_code: str, seventh_char: str) -> str:
    """Format ICD-10-CM code with 7th character and placeholder padding."""
    if "." in base_code:
        category, subcategory = base_code.split(".", 1)
    else:
        category = base_code[:3] if len(base_code) >= 3 else base_code
        subcategory = base_code[3:] if len(base_code) > 3 else ""
    padded_subcategory = subcategory.ljust(3, "X") + seventh_char
    return f"{category}.{padded_subcategory}"


# TypedDicts for type-safe graph building (matches GraphNode/GraphEdge pydantic models)
class NodeDict(TypedDict):
    id: str
    code: str
    label: str
    depth: int
    category: Literal["root", "finalized", "ancestor", "placeholder", "activator"]
    billable: bool

class EdgeDict(TypedDict):
    source: str
    target: str
    edge_type: Literal["hierarchy", "lateral"]
    rule: str | None


def build_graph_from_batch_data(
    batch_data: dict[str, dict],
    final_nodes: list[str],
    *,
    previous: tuple[list[NodeDict], list[EdgeDict], set[str], set[tuple[str, str]]] | None = None,
    only_batch_ids: set[str] | None = None,
) -> tuple[list[NodeDict], list[EdgeDict], set[str], set[tuple[str, str]]]:
    """Build graph nodes and edges from batch_data.

    Supports both full builds and incremental appends:
    - Full build (previous=None): Creates ROOT node, processes all batches.
    - Incremental (previous=state, only_batch_ids={id}): Appends only new
      batches to previous state. Dedup via seen_nodes/seen_edges is automatic.

    Args:
        batch_data: Dict mapping batch_id -> batch info with node_id, selected_ids, etc.
        final_nodes: List of finalized code IDs.
        previous: Optional prior state to append to (nodes, edges, seen_nodes, seen_edges).
        only_batch_ids: If set, only process these batch IDs (skip others).

    Returns:
        Tuple of (nodes_list, edges_list, seen_nodes_set, seen_edges_set)
    """
    if previous is not None:
        prev_nodes, prev_edges, prev_seen_nodes, prev_seen_edges = previous
        nodes = list(prev_nodes)
        edges = list(prev_edges)
        seen_nodes = set(prev_seen_nodes)
        seen_edges = set(prev_seen_edges)
    else:
        nodes = [{"id": "ROOT", "code": "ROOT", "label": "ROOT", "depth": 0, "category": "root", "billable": False}]
        edges = []
        seen_nodes = {"ROOT"}
        seen_edges = set()
    final_nodes_set = set(final_nodes)

    # Sort batches by depth to ensure parents are processed before children
    sorted_batches = sorted(
        [(bid, binfo) for bid, binfo in batch_data.items()],
        key=lambda x: (x[1].get("depth", 0), x[0])
    )

    for batch_id, batch_info in sorted_batches:
        if only_batch_ids is not None and batch_id not in only_batch_ids:
            continue
        node_id = batch_info.get("node_id")
        parent_id = batch_info.get("parent_id")
        depth = batch_info.get("depth", 0)
        candidates = batch_info.get("candidates", {})
        selected_ids = batch_info.get("selected_ids", [])
        seven_chr_authority = batch_info.get("seven_chr_authority")
        batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

        # Transform selected_ids for sevenChrDef batches
        if batch_type == "sevenChrDef" and node_id and selected_ids:
            selected_ids = [format_with_seventh_char(node_id, char) for char in selected_ids]

        # Handle sevenChrDef batches
        if batch_type == "sevenChrDef" and node_id:
            # Add base node if not seen
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                if node_id in data:
                    node_label = data[node_id].get("label", node_id)
                    node_depth = data[node_id].get("depth", depth)
                    node_category = "activator" if node_has_seven_chr_def(node_id) else "ancestor"
                else:
                    node_label = "Placeholder"
                    node_depth = depth
                    node_category = "placeholder"
                nodes.append({
                    "id": node_id, "code": node_id, "label": node_label,
                    "depth": node_depth, "category": node_category, "billable": False,
                })

            if parent_id and parent_id != node_id:
                edge_key = (parent_id, node_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    is_hier = is_hierarchy_edge(parent_id, node_id)
                    edges.append({
                        "source": parent_id, "target": node_id,
                        "edge_type": "hierarchy" if is_hier else "lateral",
                        "rule": None,
                    })

            # Create 7th character node if selected
            if selected_ids:
                seventh_char = selected_ids[0][-1] if selected_ids[0] else ""
                if "." in node_id:
                    base_category, base_subcategory = node_id.split(".", 1)
                else:
                    base_category = node_id[:3] if len(node_id) >= 3 else node_id
                    base_subcategory = node_id[3:] if len(node_id) > 3 else ""
                full_code = f"{base_category}.{base_subcategory.ljust(3, 'X')}{seventh_char}"

                # Transform candidates for label lookup
                transformed_candidates = {}
                for char, desc in candidates.items():
                    transformed_code = format_with_seventh_char(node_id, char)
                    transformed_candidates[transformed_code] = desc

                base_depth = data.get(node_id, {}).get("depth", depth)
                prev_node = node_id

                # Create placeholder nodes if base is shallower than depth 6
                if base_depth < 6:
                    if "." in node_id:
                        category, subcategory = node_id.split(".", 1)
                    else:
                        category = node_id[:3] if len(node_id) >= 3 else node_id
                        subcategory = node_id[3:] if len(node_id) > 3 else ""

                    current_sub = subcategory
                    for i in range(len(subcategory), 3):
                        current_sub += "X"
                        placeholder_code = f"{category}.{current_sub}"
                        placeholder_depth = base_depth + (i - len(subcategory) + 1)

                        if placeholder_code not in seen_nodes:
                            seen_nodes.add(placeholder_code)
                            nodes.append({
                                "id": placeholder_code, "code": placeholder_code,
                                "label": "Placeholder", "depth": placeholder_depth,
                                "category": "placeholder", "billable": False,
                            })

                        edge_key = (prev_node, placeholder_code)
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edges.append({
                                "source": prev_node, "target": placeholder_code,
                                "edge_type": "hierarchy", "rule": None,
                            })
                        prev_node = placeholder_code

                # Create depth-6 parent if needed
                if prev_node not in seen_nodes:
                    seen_nodes.add(prev_node)
                    nodes.append({
                        "id": prev_node, "code": prev_node,
                        "label": "Placeholder" if prev_node.endswith("X") else data.get(prev_node, {}).get("label", prev_node),
                        "depth": 6,
                        "category": "placeholder" if prev_node.endswith("X") else "ancestor",
                        "billable": False,
                    })

                # Create final 7th character node
                label = f"{seventh_char}: {transformed_candidates.get(full_code, seventh_char)}"
                if full_code not in seen_nodes:
                    seen_nodes.add(full_code)
                    is_finalized = full_code in final_nodes_set
                    nodes.append({
                        "id": full_code, "code": full_code, "label": label,
                        "depth": 7, "category": "finalized" if is_finalized else "ancestor", "billable": is_finalized,
                    })

                edge_key = (prev_node, full_code)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": prev_node, "target": full_code,
                        "edge_type": "lateral", "rule": "sevenChrDef",
                    })
            else:
                # Empty sevenChrDef selection — create placeholder for missing 7th char
                missing_code = f"{node_id}?"
                if missing_code not in seen_nodes:
                    seen_nodes.add(missing_code)
                    nodes.append({
                        "id": missing_code, "code": missing_code,
                        "label": "Missing Seventh Character Extension",
                        "depth": 7, "category": "placeholder", "billable": False,
                    })
                edge_key = (node_id, missing_code)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": node_id, "target": missing_code,
                        "edge_type": "lateral", "rule": "sevenChrDef",
                    })

        # Regular batches (children, codeFirst, codeAlso, etc.)
        elif batch_type != "sevenChrDef":
            # Add traversed node
            if node_id and node_id != "ROOT" and node_id not in seen_nodes:
                seen_nodes.add(node_id)
                if node_id in data:
                    node_label = data[node_id].get("label", node_id)
                    node_depth = data[node_id].get("depth", depth)
                else:
                    node_label = "Placeholder" if node_id.endswith("X") else node_id
                    node_depth = depth

                if node_has_seven_chr_def(node_id):
                    node_category = "activator"
                elif node_id not in data and node_id.endswith("X"):
                    node_category = "placeholder"
                elif node_id in final_nodes_set:
                    node_category = "finalized"
                else:
                    node_category = "ancestor"

                nodes.append({
                    "id": node_id, "code": node_id, "label": node_label,
                    "depth": node_depth, "category": node_category, "billable": node_id in final_nodes_set,
                })

            if node_id and parent_id and parent_id != node_id:
                edge_key = (parent_id, node_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    is_hier = is_hierarchy_edge(parent_id, node_id)
                    edges.append({
                        "source": parent_id, "target": node_id,
                        "edge_type": "hierarchy" if is_hier else "lateral",
                        "rule": None,
                    })

            # Add selected children
            for code in selected_ids:
                if code not in seen_nodes:
                    seen_nodes.add(code)
                    label = candidates.get(code, code)

                    if code in data:
                        code_depth = data[code].get("depth", depth + 1)
                    else:
                        code_depth = depth + 1

                    if node_has_seven_chr_def(code):
                        code_category = "activator"
                    elif code not in data and code.endswith("X"):
                        code_category = "placeholder"
                    elif code in final_nodes_set:
                        code_category = "finalized"
                    else:
                        code_category = "ancestor"

                    has_children = bool(data.get(code, {}).get("children"))
                    has_authority = seven_chr_authority is not None
                    is_activator = code_category == "activator"
                    is_finalized = code in final_nodes_set
                    billable = is_finalized or (not has_children and not has_authority and not is_activator)

                    nodes.append({
                        "id": code, "code": code, "label": label,
                        "depth": code_depth, "category": code_category, "billable": billable,
                    })

                # Add edge
                edge_source = node_id if node_id and node_id != "ROOT" else "ROOT"
                edge_key = (edge_source, code)

                if batch_type != "children":
                    edge_type = "lateral"
                    rule = batch_type
                else:
                    edge_type = "hierarchy"
                    rule = None

                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": edge_source, "target": code,
                        "edge_type": edge_type, "rule": rule,
                    })

                # For lateral edges, also create hierarchy edge from actual ICD-10 parent
                if edge_type == "lateral":
                    actual_parent = get_icd_parent(code)
                    if actual_parent and actual_parent in seen_nodes:
                        parent_edge_key = (actual_parent, code)
                        if parent_edge_key not in seen_edges:
                            seen_edges.add(parent_edge_key)
                            edges.append({
                                "source": actual_parent, "target": code,
                                "edge_type": "hierarchy", "rule": None,
                            })

    return nodes, edges, seen_nodes, seen_edges


def build_decisions_from_batch_data(batch_data: dict[str, dict]) -> list[dict]:
    """Build decision history from batch_data for RUN_FINISHED result.

    Shared by cached replay and rewind endpoints — eliminates duplication.

    Args:
        batch_data: Dict mapping batch_id -> batch info.

    Returns:
        List of decision dicts with transformed selected_ids and candidates.
    """
    all_decisions: list[dict] = []
    sorted_batches = sorted(batch_data.items(), key=lambda x: x[1].get("depth", 0))

    for batch_id, batch_info in sorted_batches:
        node_id = batch_info.get("node_id")
        candidates = batch_info.get("candidates", {})
        selected_ids = batch_info.get("selected_ids", [])
        reasoning = batch_info.get("reasoning", "")
        seven_chr_authority = batch_info.get("seven_chr_authority")
        depth = batch_info.get("depth", 0)
        batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

        # Transform selected_ids for sevenChrDef
        transformed_selected_ids = selected_ids
        if batch_type == "sevenChrDef" and node_id and selected_ids:
            transformed_selected_ids = [format_with_seventh_char(node_id, char) for char in selected_ids]

        # Transform candidates for sevenChrDef
        step_candidates = candidates
        if batch_type == "sevenChrDef" and node_id:
            step_candidates = {}
            for char, desc in candidates.items():
                transformed_code = format_with_seventh_char(node_id, char)
                step_candidates[transformed_code] = f"{char}: {desc}"

        # Compute selected_details
        selected_details: dict[str, dict] = {}
        for code in transformed_selected_ids:
            code_data = data.get(code, {})
            if node_has_seven_chr_def(code):
                cat = "activator"
            elif "X" in code:
                cat = "placeholder"
            else:
                cat = "ancestor"

            code_depth = code_data.get("depth", depth + 1) if code in data else depth + 1
            has_children = bool(code_data.get("children"))
            is_activator = cat == "activator"
            billable = not has_children and not (seven_chr_authority is not None) and not is_activator

            selected_details[code] = {
                "depth": code_depth,
                "category": cat,
                "billable": billable,
            }

        all_decisions.append({
            'batch_id': batch_id,
            'node_id': node_id,
            'batch_type': batch_type,
            'candidates': step_candidates,
            'selected_ids': transformed_selected_ids,
            'reasoning': reasoning,
            'selected_details': selected_details,
        })

    return all_decisions


# --- FastAPI App ---

app = FastAPI(
    title="ICD-10-CM Tree Traversal",
    description="Interactive visualization of ICD-10-CM code relationships",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from Vite build output
static_path = Path(__file__).parent.parent / "frontend" / "dist"
if static_path.exists():
    app.mount("/assets", StaticFiles(directory=static_path / "assets"), name="assets")


@app.get("/")
async def root() -> FileResponse:
    """Serve the main HTML page from Vite build."""
    index_path = Path(__file__).parent.parent / "frontend" / "dist" / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend not built. Run 'cd frontend && npm run build' first."
        )
    return FileResponse(index_path)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/graph", response_model=GraphResponse)
async def get_graph(request: GraphRequest) -> GraphResponse:
    """Build and return the graph data for the given ICD-10 codes."""
    codes = [code.strip().upper() for code in request.codes if code.strip()]

    if not codes:
        raise HTTPException(status_code=400, detail="No valid codes provided")

    # Filter out invalid codes instead of rejecting the entire request
    # This is important for zero-shot mode where LLM may predict non-existent codes
    valid_codes: list[str] = []
    invalid_codes: list[str] = []
    for code in codes:
        # Check if code exists directly or can be resolved (7th char codes)
        if code in data or resolve_code(code, data) is not None:
            valid_codes.append(code)
        else:
            invalid_codes.append(code)

    if invalid_codes:
        print(f"[/api/graph] Filtered out invalid codes: {', '.join(invalid_codes)}")

    if not valid_codes:
        # Return empty graph if no valid codes (instead of error)
        return GraphResponse(
            nodes=[],
            edges=[],
            stats=GraphStats(input_count=len(codes), node_count=0),
            invalid_codes=invalid_codes,
        )

    # Build the graph
    result = build_graph(valid_codes, data, show_all_paths=request.full_paths)

    # Extract sets for categorization
    leaves = result["leaves"]
    placeholders = result["placeholders"]
    activators = get_activator_nodes(result.get("seventh_char", {}))
    seventh_char = result.get("seventh_char", {})

    # Build parent->children map for depth calculation
    tree = result["tree"]
    child_to_parent: dict[str, str] = {}
    for parent, children_set in tree.items():
        for child in children_set:
            child_to_parent[child] = parent

    # Calculate depths for all nodes (including placeholders and 7th char codes)
    node_depths: dict[str, int] = {"ROOT": 0}

    def calc_depth(code: str) -> int:
        if code in node_depths:
            return node_depths[code]
        if code in data:
            depth = data[code].get("depth", 0)
            node_depths[code] = depth
            return depth
        # For placeholder/7th char codes, calculate from parent
        parent = child_to_parent.get(code)
        if parent:
            parent_depth = calc_depth(parent)
            node_depths[code] = parent_depth + 1
            return parent_depth + 1
        node_depths[code] = 0
        return 0

    # Pre-calculate all depths
    for code in result["nodes"]:
        calc_depth(code)

    # Build nodes list
    nodes: list[GraphNode] = []

    # Add ROOT node
    nodes.append(
        GraphNode(
            id="ROOT",
            code="ROOT",
            label="ICD-10-CM",
            depth=0,
            category=NodeCategory.ROOT,
        )
    )

    # Add all other nodes
    for code in result["nodes"]:
        entry = data.get(code, {})
        label = entry.get("label", "Placeholder" if code in placeholders else code)
        depth = node_depths.get(code, 0)

        # For depth 7 nodes, show only the 7th character key-value pair (e.g., "A: initial encounter")
        if depth == 7:
            seven_chr_def, _ = trace_seventh_char_def(code, data)
            if seven_chr_def:
                # 7th char is always the last character (4th position after the period)
                seventh_char_key = code[-1]
                if seventh_char_key in seven_chr_def:
                    label = f"{seventh_char_key}: {seven_chr_def[seventh_char_key]}"

        category_str = get_node_category(code, leaves, placeholders, activators)
        category = NodeCategory(category_str)

        # A node is billable only if it's a finalized code with no children in the ICD-10 hierarchy
        has_children = bool(entry.get("children", {}))
        billable = (category == NodeCategory.FINALIZED) and not has_children

        nodes.append(
            GraphNode(
                id=code,
                code=code,
                label=label,
                depth=depth,
                category=category,
                billable=billable,
            )
        )

    # Build edges list
    edges: list[GraphEdge] = []

    # Add edges from ROOT to root nodes
    for root_code in result["roots"]:
        edges.append(
            GraphEdge(
                source="ROOT",
                target=root_code,
                edge_type=EdgeType.HIERARCHY,
                rule=None,
            )
        )

    # Build lookup from lateral links for edge rules
    lateral_links = result.get("lateral_links", [])
    lateral_link_rules: dict[tuple[str, str], str] = {
        (src, tgt): rule for src, tgt, rule in lateral_links
    }

    # Add tree edges (parent-child relationships)
    # Lateral links ARE in tree for connectivity - mark them as LATERAL edge type
    seventh_char = result.get("seventh_char", {})
    tree = result["tree"]
    added_edges: set[tuple[str, str]] = set()

    for parent in tree:
        for child in tree[parent]:
            added_edges.add((parent, child))

            # Check if this is a lateral link (codeFirst, codeAlso, useAdditionalCode)
            if (parent, child) in lateral_link_rules:
                edges.append(
                    GraphEdge(
                        source=parent,
                        target=child,
                        edge_type=EdgeType.LATERAL,
                        rule=lateral_link_rules[(parent, child)],
                    )
                )
            # Check if this is a sevenChrDef edge
            # Only mark as sevenChrDef if this is the edge from depth-6 to depth-7
            # (the immediate parent of the 7th char code, not any ancestor)
            elif child in seventh_char:
                parent_depth = node_depths.get(parent, 0)
                child_depth = node_depths.get(child, 0)
                if parent_depth == 6 and child_depth == 7:
                    edges.append(
                        GraphEdge(
                            source=parent,
                            target=child,
                            edge_type=EdgeType.LATERAL,
                            rule="sevenChrDef",
                        )
                    )
                else:
                    # Regular hierarchy edge from ancestors to 7th char codes
                    edges.append(
                        GraphEdge(
                            source=parent,
                            target=child,
                            edge_type=EdgeType.HIERARCHY,
                            rule=None,
                        )
                    )
            else:
                edges.append(
                    GraphEdge(
                        source=parent,
                        target=child,
                        edge_type=EdgeType.HIERARCHY,
                        rule=None,
                    )
                )

    # Add remaining lateral link edges that aren't in tree
    # (annotation-only links where both nodes exist but aren't parent-child)
    for source, target, key in lateral_links:
        if (source, target) not in added_edges:
            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    edge_type=EdgeType.LATERAL,
                    rule=key,
                )
            )

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        stats=GraphStats(
            input_count=len(codes),
            node_count=len(nodes),
        ),
        invalid_codes=invalid_codes,
    )


@app.get("/api/node/{code}", response_model=NodeDetailResponse)
async def get_node_detail(code: str) -> NodeDetailResponse:
    """Get detailed information about a specific ICD-10 code."""
    code = code.strip().upper()

    if code == "ROOT":
        return NodeDetailResponse(
            code="ROOT",
            label="ICD-10-CM Classification System",
            depth=0,
            parent=None,
            metadata={},
            seven_chr_def=None,
        )

    if code not in data:
        raise HTTPException(status_code=404, detail=f"Code '{code}' not found")

    entry = data[code]
    label = entry.get("label", "")
    depth = entry.get("depth", 0)
    parent = get_parent_code(entry)

    # Extract metadata (filtering to relevant keys)
    raw_metadata = entry.get("metadata", {})
    metadata: dict[str, dict[str, str]] = {}

    for key in ("codeFirst", "codeAlso", "useAdditionalCode"):
        if key in raw_metadata and isinstance(raw_metadata[key], dict):
            metadata[key] = {k: str(v) for k, v in raw_metadata[key].items()}

    # Get sevenChrDef if present
    seven_chr_result = trace_seventh_char_def(code, data)
    seven_chr_def = seven_chr_result[0] if seven_chr_result else None

    return NodeDetailResponse(
        code=code,
        label=label,
        depth=depth,
        parent=parent,
        metadata=metadata,
        seven_chr_def=seven_chr_def,
    )


# --- Streaming Traversal Endpoint (Burr-based) ---

# TraversalRequest is imported from .payloads module


class BatchCompleteEvent(BaseModel):
    """Event emitted when a batch completes."""

    type: str = "batch_complete"
    batch_id: str
    node_id: str | None
    batch_type: str
    candidates: dict[str, str]
    selected_ids: list[str]
    reasoning: str


class TraversalCompleteEvent(BaseModel):
    """Event emitted when traversal completes."""

    type: str = "complete"
    final_nodes: list[str]
    batch_count: int


class DeleteCacheRequest(BaseModel):
    """Request to delete a cached traversal state."""

    clinical_note: str
    provider: str
    model: str
    temperature: float = 0.0
    scaffolded: bool = True  # True for tree traversal, False for zero-shot


class DeleteCacheResponse(BaseModel):
    """Response from cache deletion."""

    deleted: bool
    partition_key: str
    message: str


@app.post("/api/cache/delete", response_model=DeleteCacheResponse)
async def delete_cache_entry(request: DeleteCacheRequest):
    """Delete a specific cached traversal state.

    Use this when a traversal is cancelled or reset to ensure a fresh run.
    Generates the partition key from the same parameters used during traversal.
    """
    from agent.zero_shot import generate_zero_shot_cache_key

    if request.scaffolded:
        # Generate the same partition key used for scaffolded traversal
        partition_key = generate_traversal_cache_key(
            clinical_note=request.clinical_note,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
        )
        # Delete ALL entries (ROOT + subgraphs) for complete cache clear
        deleted = await delete_persisted_state(
            partition_key=partition_key,
            app_id=None,
            db_path="./cache.db",
            table_name="burr_state",
        )
    else:
        # Generate the same partition key used for zero-shot
        partition_key = generate_zero_shot_cache_key(
            clinical_note=request.clinical_note,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
        )
        deleted = await delete_persisted_state(
            partition_key=partition_key,
            app_id="zero_shot",
            db_path="./cache.db",
            table_name="zero_shot_state",
        )

    if deleted:
        return DeleteCacheResponse(
            deleted=True,
            partition_key=partition_key,
            message="Cache entry deleted successfully",
        )
    else:
        return DeleteCacheResponse(
            deleted=False,
            partition_key=partition_key,
            message="Cache entry not found or already deleted",
        )


class InvalidateCacheRequest(BaseModel):
    """Request to soft-invalidate cache for a configuration."""

    clinical_note: str
    provider: str
    model: str
    temperature: float = 0.0
    system_prompt: str | None = None
    scaffolded: bool = True


class InvalidateCacheResponse(BaseModel):
    """Response from cache invalidation."""

    success: bool
    new_version: int
    message: str


@app.post("/api/cache/invalidate", response_model=InvalidateCacheResponse)
async def invalidate_cache(request: InvalidateCacheRequest):
    """Hard-invalidate cache for a specific configuration.

    Increments the cache version AND deletes any existing cached data
    at the new version's key to guarantee a cache miss.
    """
    try:
        # Normalize parameters to match traversal endpoint
        model = request.model or ""
        temperature = request.temperature or 0.0

        new_version = increment_cache_version(
            clinical_note=request.clinical_note,
            provider=request.provider,
            model=model,
            temperature=temperature,
            system_prompt=request.system_prompt,
            scaffolded=request.scaffolded,
        )

        # Generate the cache key that will be used for the new version
        if request.scaffolded:
            partition_key = generate_traversal_cache_key(
                clinical_note=request.clinical_note,
                provider=request.provider,
                model=model,
                temperature=temperature,
                system_prompt=request.system_prompt,
                use_cache=True,
                cache_version=new_version,
            )
            table_name = "burr_state"
        else:
            partition_key = generate_zero_shot_cache_key(
                clinical_note=request.clinical_note,
                provider=request.provider,
                model=model,
                temperature=temperature,
                system_prompt=request.system_prompt,
                use_cache=True,
                cache_version=new_version,
            )
            table_name = "zero_shot_state"

        # Delete any existing cached data at the new version's key
        deleted = await delete_persisted_state(
            partition_key=partition_key,
            app_id=None,  # Delete ALL entries (ROOT + subgraphs)
            table_name=table_name,
        )

        action = "Deleted existing cache. " if deleted else ""
        return InvalidateCacheResponse(
            success=True,
            new_version=new_version,
            message=f"Cache invalidated. {action}Next run will use version {new_version}.",
        )
    except Exception as e:
        print(f"[CACHE] Error in invalidate_cache: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ClearAllCacheResponse(BaseModel):
    """Response from clearing session cache."""

    success: bool
    message: str
    entries_deleted: int


@app.post("/api/cache/clear-session", response_model=ClearAllCacheResponse)
async def clear_session_cache():
    """Clear cached state from this server session only.

    Only clears entries created since the server started, preserving older cached runs.
    Also clears the in-memory selector cache.
    """
    from candidate_selector.selector import clear_cache
    import aiosqlite

    total_deleted = 0

    # Clear in-memory selector cache
    clear_cache()

    # Only clear entries from this session
    if not SESSION_PARTITION_KEYS:
        return ClearAllCacheResponse(
            success=True,
            message="No session cache entries to clear",
            entries_deleted=0,
        )

    # Clear session entries from both SQLite tables
    try:
        async with aiosqlite.connect("./cache.db") as db:
            placeholders = ",".join("?" * len(SESSION_PARTITION_KEYS))
            keys = list(SESSION_PARTITION_KEYS)
            for table in ["burr_state", "zero_shot_state"]:
                cursor = await db.execute(
                    f"DELETE FROM {table} WHERE partition_key IN ({placeholders})",
                    keys,
                )
                total_deleted += cursor.rowcount
            await db.commit()
    except Exception as e:
        return ClearAllCacheResponse(
            success=False,
            message=f"Error clearing cache: {e}",
            entries_deleted=0,
        )

    # Clear the tracked session keys
    SESSION_PARTITION_KEYS.clear()

    return ClearAllCacheResponse(
        success=True,
        message="Session cache cleared successfully",
        entries_deleted=total_deleted,
    )


@app.post("/api/traverse/cancel")
async def cancel_traversal():
    """Cancel the currently running traversal.

    Uses task.cancel() to interrupt at the next await point.
    The CancelledError propagates up through Burr's internals and
    the task ends cleanly. Incomplete cache entries are deleted.
    """
    global RUNNING_TASK, CURRENT_APP

    if RUNNING_TASK is not None and not RUNNING_TASK.done():
        # Extract partition key BEFORE clearing CURRENT_APP
        partition_key = getattr(CURRENT_APP, '_partition_key', None) if CURRENT_APP else None

        RUNNING_TASK.cancel()
        print("[CANCEL] Task cancelled")

        # Wait for cancellation to complete
        try:
            await RUNNING_TASK
        except asyncio.CancelledError:
            pass  # Expected - task was cancelled
        except Exception as e:
            # Cleanup errors during cancellation (e.g., database already closed)
            print(f"[CANCEL] Task cleanup error (ignored): {type(e).__name__}: {e}")

        # Delete incomplete cache entry
        if partition_key:
            try:
                await delete_persisted_state(partition_key=partition_key, app_id=None)
                print(f"[CANCEL] Deleted incomplete cache for partition_key={partition_key}")
            except Exception as e:
                print(f"[CANCEL] Cache cleanup error (ignored): {e}")

        CURRENT_APP = None
        RUNNING_TASK = None
    else:
        print("[CANCEL] No running traversal to cancel")

    return {"cancelled": True}


@app.post("/api/cache/clear-last", response_model=ClearAllCacheResponse)
async def clear_last_cache():
    """Clear the most recent traversal run's cache.

    Uses the currently running app's partition key if available.
    """
    from candidate_selector.selector import clear_cache
    import aiosqlite

    global CURRENT_APP

    # Clear in-memory selector cache
    clear_cache()

    # Get partition key from running app if available
    partition_key = CURRENT_APP._partition_key if CURRENT_APP is not None else None

    if not partition_key:
        return ClearAllCacheResponse(
            success=True,
            message="No recent run to clear",
            entries_deleted=0,
        )

    total_deleted = 0

    try:
        async with aiosqlite.connect("./cache.db") as db:
            for table in ["burr_state", "zero_shot_state"]:
                cursor = await db.execute(
                    f"DELETE FROM {table} WHERE partition_key = ?",
                    (partition_key,),
                )
                total_deleted += cursor.rowcount
            await db.commit()
    except Exception as e:
        return ClearAllCacheResponse(
            success=False,
            message=f"Error clearing cache: {e}",
            entries_deleted=0,
        )

    # Remove from session keys
    SESSION_PARTITION_KEYS.discard(partition_key)

    return ClearAllCacheResponse(
        success=True,
        message="Last run cache cleared successfully",
        entries_deleted=total_deleted,
    )


@app.post("/api/cache/clear-all", response_model=ClearAllCacheResponse)
async def clear_all_cache():
    """Clear all cached state from both database tables.

    Clears everything including older cached runs from previous sessions.
    Also clears the in-memory selector cache.
    """
    from candidate_selector.selector import clear_cache
    import aiosqlite

    total_deleted = 0

    # Clear in-memory selector cache
    clear_cache()

    # Clear all entries from both SQLite tables
    try:
        async with aiosqlite.connect("./cache.db") as db:
            for table in ["burr_state", "zero_shot_state"]:
                cursor = await db.execute(f"DELETE FROM {table}")
                total_deleted += cursor.rowcount
            await db.commit()
    except Exception as e:
        return ClearAllCacheResponse(
            success=False,
            message=f"Error clearing cache: {e}",
            entries_deleted=0,
        )

    # Clear the tracked session keys too
    SESSION_PARTITION_KEYS.clear()

    return ClearAllCacheResponse(
        success=True,
        message="All cache cleared successfully",
        entries_deleted=total_deleted,
    )


@app.post("/api/traverse/stream")
async def stream_traversal(run_input: RunAgentInput, http_request: Request):
    """Run Burr-based DFS traversal with AG-UI streaming.

    Accepts standard AG-UI RunAgentInput with domain config in the `state` field.
    Uses AG-UI protocol over SSE with JSON Patch (RFC 6902) for incremental updates.

    Returns SSE stream with events:
    - RUN_STARTED: Traversal begins
    - STATE_SNAPSHOT: Initial graph with ROOT node
    - STEP_STARTED / STATE_DELTA / TEXT_MESSAGE / STEP_FINISHED: Per-batch updates
    - CUSTOM step_metadata: Domain-specific batch metadata
    - RUN_FINISHED: Traversal complete with final codes
    """
    # Extract domain config from AG-UI RunAgentInput.state
    state = run_input.state if isinstance(run_input.state, dict) else {}
    request = TraversalRequest(**state)
    # Import Burr components
    from agent import build_traversal_app, initialize_persister, cleanup_persister, set_batch_callback
    from candidate_selector.providers import create_config
    import candidate_selector.config as llm_config

    # Queue for streaming events
    event_queue: asyncio.Queue[str | None] = asyncio.Queue()

    # Incremental graph state — build_graph_from_batch_data accumulates here
    accumulated_batch_data: dict[str, dict] = {}
    graph_state: tuple[list[NodeDict], list[EdgeDict], set[str], set[tuple[str, str]]] | None = None
    seen_batches: set[str] = set()  # Track processed batches to avoid duplicate events

    # Batch callback to emit AG-UI events via STATE_SNAPSHOT
    def on_batch_complete(
        batch_id: str,
        node_id: str | None,
        parent_id: str | None,
        depth: int,
        candidates: dict[str, str],
        selected_ids: list[str],
        reasoning: str,
        seven_chr_authority: dict | None = None,
    ):
        nonlocal graph_state
        print(f"[CALLBACK] on_batch_complete called: batch_id={batch_id}, selected_ids={selected_ids}")

        # Skip duplicate batch callbacks
        if batch_id in seen_batches:
            print(f"[CALLBACK] SKIP DUPLICATE: batch_id={batch_id} already processed")
            return
        seen_batches.add(batch_id)
        batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

        # STEP_STARTED
        event_queue.put_nowait(encode_event(StepStartedEvent(step_name=batch_id)))

        # Accumulate batch data and build graph incrementally → STATE_SNAPSHOT
        accumulated_batch_data[batch_id] = {
            "node_id": node_id, "parent_id": parent_id, "depth": depth,
            "candidates": candidates, "selected_ids": selected_ids,
            "seven_chr_authority": seven_chr_authority,
        }
        graph_state = build_graph_from_batch_data(
            accumulated_batch_data, [],
            previous=graph_state, only_batch_ids={batch_id},
        )
        snapshot = GraphState(
            nodes=[GraphNode(**n) for n in graph_state[0]],
            edges=[GraphEdge(**e) for e in graph_state[1]],
        )
        event_queue.put_nowait(encode_event(StateSnapshotEvent(snapshot=snapshot.model_dump())))

        # Compute selected_details (independent of graph building)
        selected_details: dict[str, dict] = {}
        for code in selected_ids:
            code_data = data.get(code, {})
            if node_has_seven_chr_def(code):
                cat = "activator"
            elif "X" in code:
                cat = "placeholder"
            else:
                cat = "ancestor"
            code_depth = code_data.get("depth", depth + 1) if code in data else depth + 1
            has_children = bool(code_data.get("children"))
            is_activator = cat == "activator"
            billable = not has_children and not (seven_chr_authority is not None) and not is_activator
            selected_details[code] = {"depth": code_depth, "category": cat, "billable": billable}

        # REASONING events (AG-UI standard)
        is_error = reasoning.startswith("API Error:") if reasoning else False
        if reasoning:
            msg_id = str(uuid.uuid4())
            event_queue.put_nowait(encode_event(ReasoningStartEvent(message_id=msg_id)))
            event_queue.put_nowait(encode_event(_reasoning_message_start(msg_id)))
            event_queue.put_nowait(encode_event(ReasoningMessageContentEvent(message_id=msg_id, delta=reasoning)))
            event_queue.put_nowait(encode_event(ReasoningMessageEndEvent(message_id=msg_id)))
            event_queue.put_nowait(encode_event(ReasoningEndEvent(message_id=msg_id)))

        # STEP_FINISHED + CUSTOM step_metadata
        event_queue.put_nowait(encode_event(StepFinishedEvent(step_name=batch_id)))
        event_queue.put_nowait(encode_event(CustomEvent(name="step_metadata", value={
            "node_id": node_id, "batch_type": batch_type,
            "selected_ids": selected_ids, "candidates": candidates,
            "error": is_error, "selected_details": selected_details,
        })))

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate AG-UI SSE events."""
        global CURRENT_APP, RUNNING_TASK

        # Use client-provided IDs from RunAgentInput (AG-UI protocol compliance)
        run_id = run_input.run_id
        thread_id = run_input.thread_id

        try:
            # Configure LLM (use create_config for correct provider base URLs)
            # Pass temperature and max_tokens if provided, otherwise use provider defaults
            config_kwargs: dict = {}
            if request.temperature is not None:
                config_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                config_kwargs["max_completion_tokens"] = request.max_tokens

            # Debug: Print extra config for Vertex AI troubleshooting
            print(f"[SERVER] Request extra config: {request.extra}")
            print(f"[SERVER] Request scaffolded: {request.scaffolded}")

            llm_config.LLM_CONFIG = create_config(
                provider=request.provider,
                api_key=request.api_key,
                model=request.model,  # create_config uses provider default if None
                extra=request.extra,  # Provider-specific config (e.g., Vertex AI location/project_id)
                system_prompt=request.system_prompt,  # Custom system prompt (uses default if None)
                **config_kwargs,
            )
            print(f"[SERVER] LLM configured: provider={llm_config.LLM_CONFIG.provider}, "
                  f"model={llm_config.LLM_CONFIG.model}, base_url={llm_config.LLM_CONFIG.base_url}, "
                  f"temperature={llm_config.LLM_CONFIG.temperature}, max_tokens={llm_config.LLM_CONFIG.max_completion_tokens}")
            print(f"[SERVER] API key set: {'Yes' if llm_config.LLM_CONFIG.api_key else 'NO!'}")
            print(f"[SERVER] Extra config: {llm_config.LLM_CONFIG.extra}")
            print(f"[SERVER] Settings hash (for cache): {llm_config.LLM_CONFIG.settings_hash()}")
            print(f"[SERVER] Clinical note length: {len(request.clinical_note)} chars")
            print(f"[SERVER] Scaffolded mode: {request.scaffolded}")
            if request.system_prompt:
                print(f"[SERVER] Custom system prompt: {len(request.system_prompt)} chars")

            # Check cache BEFORE emitting RUN_STARTED
            # This allows frontend to know upfront if this is a cached replay
            #
            # IMPORTANT: For scaffolded mode, we call build_traversal_app() here (single source of truth)
            # rather than doing a separate cache check. This eliminates the dual cache check
            # issue where the early check and build_traversal_app() could disagree about cache status.
            is_cached = False
            scaffolded_burr_app = None  # Will be set for scaffolded mode
            scaffolded_partition_key = None

            if not request.scaffolded:
                # Zero-shot mode: check cache
                await initialize_zero_shot_persister(reset=False)
                # Look up current cache version for this configuration
                zs_cache_version = get_cache_version(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    system_prompt=request.system_prompt,
                    scaffolded=False,
                )
                zs_partition_key = generate_zero_shot_cache_key(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    max_completion_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                    use_cache=request.persist_cache,
                    cache_version=zs_cache_version,
                )
                if ZERO_SHOT_PERSISTER is not None:
                    cached_state = await ZERO_SHOT_PERSISTER.load(
                        partition_key=zs_partition_key,
                        app_id="zero_shot",
                    )
                    is_cached = (
                        cached_state is not None
                        and cached_state.get("state", {}).get("selected_codes") is not None
                    )
                print(f"[SERVER] Zero-shot cache check: partition_key={zs_partition_key}, cached={is_cached}")
            else:
                # Scaffolded mode: call build_traversal_app() directly (single source of truth for cache status)
                # This eliminates the dual cache check issue - build_traversal_app() is the authoritative check
                print(f"[SERVER] persist_cache={request.persist_cache} (use_cache for key generation)")
                # Look up current cache version for this configuration
                scaffolded_cache_version = get_cache_version(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    system_prompt=request.system_prompt,
                    scaffolded=True,
                )
                scaffolded_partition_key = generate_traversal_cache_key(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    system_prompt=request.system_prompt,
                    use_cache=request.persist_cache,
                    cache_version=scaffolded_cache_version,
                )
                SESSION_PARTITION_KEYS.add(scaffolded_partition_key)
                print(f"[SERVER] Scaffolded cache key: {scaffolded_partition_key}")

                # Initialize persister (always keep existing data)
                await initialize_persister(reset=False)

                # Set callback for streaming (needed before build_traversal_app for live traversal)
                set_batch_callback(on_batch_complete)
                print("[SERVER] Batch callback registered")

                # Build app - this is the SINGLE cache check (no separate early check)
                scaffolded_burr_app, is_cached = await build_traversal_app(
                    context=request.clinical_note,
                    default_selector=request.selector,
                    with_persistence=True,
                    partition_key=scaffolded_partition_key,
                )
                print(f"[SERVER] Scaffolded build_traversal_app result: cached={is_cached}")

                # Set CURRENT_APP for cancel endpoint to access
                CURRENT_APP = scaffolded_burr_app

            # RUN_STARTED + run_metadata custom event
            yield encode_event(RunStartedEvent(thread_id=thread_id, run_id=run_id, input=run_input))
            yield encode_event(CustomEvent(name="run_metadata", value={
                'clinical_note': request.clinical_note[:100], 'cached': is_cached,
            }))

            # STATE_SNAPSHOT with ROOT node
            initial_state = GraphState(
                nodes=[GraphNode(
                    id="ROOT",
                    code="ROOT",
                    label="ICD-10-CM",
                    depth=0,
                    category="root",
                    billable=False,
                )],
                edges=[],
            )
            yield encode_event(StateSnapshotEvent(snapshot=initial_state.model_dump()))

            # Branch based on scaffolded flag
            if not request.scaffolded:
                # Zero-shot mode: Burr app with SQLite caching
                print("[SERVER] Running in ZERO-SHOT mode (Burr app with caching)")

                # Initialize zero-shot persister (always keep existing data)
                await initialize_zero_shot_persister(reset=False)

                # Generate cache key from request parameters
                # Look up current cache version for this configuration
                zs_run_cache_version = get_cache_version(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    system_prompt=request.system_prompt,
                    scaffolded=False,
                )
                partition_key = generate_zero_shot_cache_key(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    max_completion_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                    use_cache=request.persist_cache,
                    cache_version=zs_run_cache_version,
                )
                SESSION_PARTITION_KEYS.add(partition_key)
                print(f"[SERVER] Zero-shot cache key: {partition_key}")

                # STEP_STARTED for zero-shot batch
                yield encode_event(StepStartedEvent(step_name='ROOT|zero-shot'))

                # Build zero-shot Burr app (checks cache)
                zs_app, was_cached = await build_zero_shot_app(
                    clinical_note=request.clinical_note,
                    partition_key=partition_key,
                )

                if was_cached:
                    # Cache hit - get results from cached state
                    print("[SERVER] Zero-shot CACHE HIT - using cached results")
                    selected_codes = zs_app.state.get("selected_codes", [])
                    reasoning = zs_app.state.get("reasoning", "")
                else:
                    # Cache miss - run the Burr app in background task for cancellation support
                    print("[SERVER] Zero-shot CACHE MISS - calling LLM")

                    async def run_zero_shot():
                        _, _, state = await zs_app.arun(halt_after=["finish"])
                        return state

                    zs_task = asyncio.create_task(run_zero_shot())

                    # Poll for client disconnection while waiting
                    while not zs_task.done():
                        if await http_request.is_disconnected():
                            print("[SERVER] Client disconnected during zero-shot - cancelling")
                            zs_task.cancel()
                            try:
                                await zs_task
                            except asyncio.CancelledError:
                                pass
                            return  # Exit generator
                        await asyncio.sleep(0.1)

                    final_state = await zs_task
                    selected_codes = final_state.get("selected_codes", [])
                    reasoning = final_state.get("reasoning", "")

                # Build STATE_DELTA ops for generated codes
                # Also build candidates and selected_details for STEP_FINISHED metadata
                ops: list[JsonPatchOp] = []
                candidates: dict[str, str] = {}
                selected_details: dict[str, dict] = {}

                for code in selected_codes:
                    # Handle 7th character codes (e.g., T84.53XD, T88.8XXA)
                    # These need special label handling: "base_label, 7th_char_meaning"
                    seventh_char = extract_seventh_char(code)
                    resolved_code = resolve_code(code, data)

                    if code in data:
                        # Code exists directly in data
                        label = data[code].get("label", code)
                        depth = data[code].get("depth", len(code.replace(".", "")))
                        children = data[code].get("children", [])
                        billable = len(children) == 0
                    elif seventh_char and resolved_code:
                        # 7th character code - build combined label
                        # Find the first non-placeholder ancestor for base label
                        base_code = resolved_code
                        base_label = ""
                        current = base_code
                        while current in data:
                            entry = data[current]
                            # Skip placeholder codes (those with X suffix and no meaningful label)
                            if not current.endswith("X") or entry.get("label", "").strip():
                                base_label = entry.get("label", "")
                                if base_label and not current.endswith("X"):
                                    break
                            parent = get_parent_code(entry)
                            if not parent or parent == "ROOT":
                                break
                            current = parent

                        # Get 7th char meaning from sevenChrDef
                        seventh_char_meaning = ""
                        seven_def_result = trace_seventh_char_def(code, data)
                        if seven_def_result:
                            seven_def, _ = seven_def_result
                            seventh_char_meaning = seven_def.get(seventh_char, "")

                        # Build combined label
                        if base_label and seventh_char_meaning:
                            label = f"{base_label}, {seventh_char_meaning}"
                        elif base_label:
                            label = base_label
                        elif seventh_char_meaning:
                            label = seventh_char_meaning
                        else:
                            label = code

                        # Depth for 7th char codes is always 7
                        depth = 7
                        # 7th char codes are always billable (leaf nodes)
                        billable = True
                    else:
                        # Code not in data and not a valid 7th char code
                        label = code
                        depth = len(code.replace(".", ""))
                        billable = True  # Assume billable if not in data

                    print(f"[ZERO-SHOT DEBUG] {code}: billable={billable}, label={label}")

                    # Store for STEP_FINISHED metadata
                    candidates[code] = label
                    selected_details[code] = {
                        "depth": depth,
                        "category": "finalized",
                        "billable": billable,
                    }

                    ops.append(JsonPatchOp(
                        op="add",
                        path="/nodes/-",
                        value={
                            "id": code,
                            "code": code,
                            "label": label,
                            "depth": depth,
                            "category": "finalized",
                            "billable": billable,
                        }
                    ))
                    # Edge from ROOT
                    ops.append(JsonPatchOp(
                        op="add",
                        path="/edges/-",
                        value={
                            "source": "ROOT",
                            "target": code,
                            "edge_type": "hierarchy",
                            "rule": None,
                        }
                    ))
                    # Add to finalized array
                    ops.append(JsonPatchOp(
                        op="add",
                        path="/finalized/-",
                        value=code,
                    ))

                if ops:
                    yield encode_event(StateDeltaEvent(delta=[op.model_dump() for op in ops]))

                # REASONING events for LLM reasoning, then STEP_FINISHED + CUSTOM
                is_error = reasoning.startswith("API Error:") if reasoning else False
                if reasoning:
                    msg_id = str(uuid.uuid4())
                    yield encode_event(ReasoningStartEvent(message_id=msg_id))
                    yield encode_event(_reasoning_message_start(msg_id))
                    yield encode_event(ReasoningMessageContentEvent(message_id=msg_id, delta=reasoning))
                    yield encode_event(ReasoningMessageEndEvent(message_id=msg_id))
                    yield encode_event(ReasoningEndEvent(message_id=msg_id))
                yield encode_event(StepFinishedEvent(step_name='ROOT|zero-shot'))
                yield encode_event(CustomEvent(name="step_metadata", value={
                    'node_id': 'ROOT', 'batch_type': 'zero-shot',
                    'selected_ids': selected_codes,
                    'candidates': candidates, 'error': is_error,
                    'selected_details': selected_details,
                }))

                # RUN_FINISHED or RUN_ERROR
                cache_status = "CACHED" if was_cached else "NEW"
                print(f"[SERVER] Zero-shot complete ({cache_status}): {len(selected_codes)} codes generated")

                if is_error:
                    # Emit RUN_ERROR for API errors
                    yield encode_event(RunErrorEvent(message=reasoning))
                else:
                    yield encode_event(RunFinishedEvent(
                        thread_id=thread_id, run_id=run_id,
                        result={
                            'final_nodes': selected_codes,
                            'batch_count': 1,
                            'mode': 'zero-shot',
                            'cached': was_cached,
                        },
                    ))
                return  # Exit generator - zero-shot mode complete

            # Scaffolded mode: Tree traversal with Burr
            # NOTE: build_traversal_app() was already called in the early check section above
            # This ensures RUN_STARTED has the accurate cache status (single source of truth)
            # Just use the already-built app and cache status
            burr_app = scaffolded_burr_app
            partition_key = scaffolded_partition_key

            if is_cached:
                # Cache hit! Emit STATE_SNAPSHOT with complete graph (AG-UI protocol aligned)
                # Instead of replaying 500+ individual events, send a single snapshot
                print("[SERVER] Scaffolded CACHE HIT - emitting STATE_SNAPSHOT")
                final_state = burr_app.state
                final_nodes = final_state.get("final_nodes", [])
                batch_data = final_state.get("batch_data", {})

                # Build complete graph and decisions using shared functions
                snapshot_nodes, snapshot_edges, _, _ = build_graph_from_batch_data(batch_data, final_nodes)
                all_decisions = build_decisions_from_batch_data(batch_data)

                # Emit single STATE_SNAPSHOT with complete graph
                snapshot_state = GraphState(
                    nodes=[GraphNode(**n) for n in snapshot_nodes],
                    edges=[GraphEdge(**e) for e in snapshot_edges]
                )
                yield encode_event(StateSnapshotEvent(snapshot=snapshot_state.model_dump()))

                from agent.actions import print_finished_summary
                print_finished_summary(batch_data, final_nodes, label="FINISHED! (CACHED)")

                # Emit RUN_FINISHED with final_nodes and all decisions
                print(f"[SERVER] Scaffolded complete (CACHED SNAPSHOT): {len(snapshot_nodes)} nodes, {len(snapshot_edges)} edges, {len(all_decisions)} decisions")
                yield encode_event(RunFinishedEvent(
                    thread_id=thread_id, run_id=run_id,
                    result={
                        'final_nodes': final_nodes,
                        'batch_count': len(batch_data),
                        'mode': 'scaffolded',
                        'cached': True,
                        'decisions': all_decisions,
                    },
                ))
                return  # Exit early for cache hit

            else:
                # Cache miss - run the traversal
                print("[SERVER] Scaffolded CACHE MISS - running traversal")

                # Run in background task so we can stream events
                async def run_app():
                    _, _, state = await burr_app.arun(halt_after=["finish"])
                    return state

                task = asyncio.create_task(run_app())
                RUNNING_TASK = task
                client_disconnected = False

                # Stream events while running
                while not task.done():
                    # Check for client disconnection
                    if await http_request.is_disconnected():
                        print("[SERVER] Client disconnected - cancelling traversal")
                        task.cancel()
                        client_disconnected = True
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        # Delete incomplete cache entry
                        if scaffolded_partition_key:
                            try:
                                await delete_persisted_state(partition_key=scaffolded_partition_key, app_id=None)
                                print(f"[SERVER] Deleted incomplete cache for partition_key={scaffolded_partition_key}")
                            except Exception as e:
                                print(f"[SERVER] Cache cleanup error (ignored): {e}")
                        break

                    try:
                        encoded = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        if encoded is not None:
                            yield encoded  # Already encoded by encode_event()
                    except asyncio.TimeoutError:
                        continue

                # If client disconnected, exit early
                if client_disconnected:
                    return

                # Get final state
                final_state = await task

                # Drain remaining events (already encoded strings)
                while not event_queue.empty():
                    encoded = event_queue.get_nowait()
                    if encoded is not None:
                        yield encoded

            # Final STATE_SNAPSHOT with finalized categories
            batch_data = final_state.get("batch_data", {})
            final_nodes_raw = final_state.get("final_nodes", [])
            final_nodes = list(dict.fromkeys(final_nodes_raw))  # Preserve order, remove duplicates

            final_graph_nodes, final_graph_edges, _, _ = build_graph_from_batch_data(batch_data, final_nodes)
            final_snapshot = GraphState(
                nodes=[GraphNode(**n) for n in final_graph_nodes],
                edges=[GraphEdge(**e) for e in final_graph_edges],
            )
            yield encode_event(StateSnapshotEvent(snapshot=final_snapshot.model_dump()))

            # RUN_FINISHED or RUN_ERROR - check if there was an LLM error
            root_reasoning = batch_data.get("ROOT", {}).get("reasoning", "")
            llm_error = root_reasoning if root_reasoning.startswith("API Error") else None

            cache_status = "CACHED" if is_cached else "NEW"
            if llm_error:
                error_type = llm_error.split("(", 1)[1].split(")", 1)[0] if "(" in llm_error and ")" in llm_error else "Unknown"
                print(f"[SERVER] Scaffolded complete ({cache_status}): final_nodes={final_nodes}, batch_count={len(batch_data)}, error={error_type}")
            else:
                print(f"[SERVER] Scaffolded complete ({cache_status}): final_nodes={final_nodes}, batch_count={len(batch_data)}")

            if llm_error:
                yield encode_event(RunErrorEvent(message=llm_error))
            else:
                yield encode_event(RunFinishedEvent(
                    thread_id=thread_id, run_id=run_id,
                    result={
                        'final_nodes': final_nodes,
                        'batch_count': len(batch_data),
                        'mode': 'scaffolded',
                        'cached': is_cached,
                    },
                ))
            print("[SERVER] Stream complete")

        except Exception as e:
            # Emit RUN_ERROR for unexpected exceptions
            yield encode_event(RunErrorEvent(message=str(e)))

        finally:
            # Cleanup
            RUNNING_TASK = None
            set_batch_callback(None)
            await cleanup_persister()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/traverse")
async def run_traversal_sync(request: TraversalRequest):
    """Run Burr-based DFS traversal (non-streaming).

    Returns the complete result after traversal finishes.
    For streaming updates, use /api/traverse/stream instead.
    """
    from agent import run_traversal as burr_run_traversal

    result = await burr_run_traversal(
        clinical_note=request.clinical_note,
        provider=request.provider,
        api_key=request.api_key,
        model=request.model,
        selector=request.selector,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        use_cache=request.persist_cache,
    )

    return result


@app.post("/api/traverse/rewind")
async def stream_rewind(run_input: RunAgentInput, http_request: Request):
    """Rewind traversal from a specific batch with feedback.

    Accepts standard AG-UI RunAgentInput with domain config in the `state` field.
    Uses Burr's fork pattern to branch from a checkpoint and inject corrective
    feedback. Streams updates via AG-UI SSE (same as /api/traverse/stream).

    The feedback is passed to the LLM selector as "CRITICAL ADDITIONAL GUIDANCE"
    that takes priority over the general clinical context.

    The backend is the authoritative source for pre-rewind state - it loads the
    cached state, prunes it to the rewind point, and emits a STATE_SNAPSHOT
    so the frontend doesn't need to carry or manipulate state.

    Returns SSE stream with events:
    - RUN_STARTED: Rewind begins (includes rewind_from batch_id)
    - STATE_SNAPSHOT: Pre-rewind graph state (pruned to rewind point)
    - STEP_STARTED / STATE_DELTA / TEXT_MESSAGE / STEP_FINISHED: Per-batch updates
    - CUSTOM step_metadata: Domain-specific batch metadata
    - RUN_FINISHED: Rewind complete with final codes
    """
    # Extract domain config from AG-UI RunAgentInput.state
    state = run_input.state if isinstance(run_input.state, dict) else {}
    request = RewindRequest(**state)
    from agent import retry_node, initialize_persister, cleanup_persister, set_batch_callback
    from agent.traversal import PARTITION_KEY, generate_traversal_cache_key, prune_state_for_rewind
    import agent.traversal as traversal_module
    from candidate_selector.providers import create_config
    import candidate_selector.config as llm_config

    # Queue for streaming events
    event_queue: asyncio.Queue[str | None] = asyncio.Queue()

    # Incremental graph state — build_graph_from_batch_data accumulates here
    accumulated_batch_data: dict[str, dict] = {}
    graph_state: tuple[list[NodeDict], list[EdgeDict], set[str], set[tuple[str, str]]] | None = None
    seen_batches: set[str] = set()  # Track processed batches to avoid duplicate events

    # Batch callback to emit AG-UI events via STATE_SNAPSHOT (same pattern as stream_traversal)
    def on_batch_complete(
        batch_id: str,
        node_id: str | None,
        parent_id: str | None,
        depth: int,
        candidates: dict[str, str],
        selected_ids: list[str],
        reasoning: str,
        seven_chr_authority: dict | None = None,
    ):
        nonlocal graph_state
        print(f"[REWIND CALLBACK] on_batch_complete: batch_id={batch_id}, selected_ids={selected_ids}")

        # Skip duplicate batch callbacks
        if batch_id in seen_batches:
            print(f"[REWIND CALLBACK] SKIP DUPLICATE: batch_id={batch_id} already processed")
            return
        seen_batches.add(batch_id)
        batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

        # STEP_STARTED
        event_queue.put_nowait(encode_event(StepStartedEvent(step_name=batch_id)))

        # Accumulate batch data and build graph incrementally → STATE_SNAPSHOT
        accumulated_batch_data[batch_id] = {
            "node_id": node_id, "parent_id": parent_id, "depth": depth,
            "candidates": candidates, "selected_ids": selected_ids,
            "seven_chr_authority": seven_chr_authority,
        }
        graph_state = build_graph_from_batch_data(
            accumulated_batch_data, [],
            previous=graph_state, only_batch_ids={batch_id},
        )
        snapshot = GraphState(
            nodes=[GraphNode(**n) for n in graph_state[0]],
            edges=[GraphEdge(**e) for e in graph_state[1]],
        )
        event_queue.put_nowait(encode_event(StateSnapshotEvent(snapshot=snapshot.model_dump())))

        # Compute selected_details (independent of graph building)
        selected_details: dict[str, dict] = {}
        for code in selected_ids:
            code_data = data.get(code, {})
            if node_has_seven_chr_def(code):
                cat = "activator"
            elif "X" in code:
                cat = "placeholder"
            else:
                cat = "ancestor"
            code_depth = code_data.get("depth", depth + 1) if code in data else depth + 1
            has_children = bool(code_data.get("children"))
            is_activator = cat == "activator"
            billable = not has_children and not (seven_chr_authority is not None) and not is_activator
            selected_details[code] = {"depth": code_depth, "category": cat, "billable": billable}

        # REASONING events (AG-UI standard)
        is_error = reasoning.startswith("API Error:") if reasoning else False
        if reasoning:
            msg_id = str(uuid.uuid4())
            event_queue.put_nowait(encode_event(ReasoningStartEvent(message_id=msg_id)))
            event_queue.put_nowait(encode_event(_reasoning_message_start(msg_id)))
            event_queue.put_nowait(encode_event(ReasoningMessageContentEvent(message_id=msg_id, delta=reasoning)))
            event_queue.put_nowait(encode_event(ReasoningMessageEndEvent(message_id=msg_id)))
            event_queue.put_nowait(encode_event(ReasoningEndEvent(message_id=msg_id)))

        # STEP_FINISHED + CUSTOM step_metadata
        event_queue.put_nowait(encode_event(StepFinishedEvent(step_name=batch_id)))
        event_queue.put_nowait(encode_event(CustomEvent(name="step_metadata", value={
            "node_id": node_id, "batch_type": batch_type,
            "selected_ids": selected_ids, "candidates": candidates,
            "error": is_error, "selected_details": selected_details,
        })))

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate AG-UI SSE events for rewind."""
        global RUNNING_TASK
        nonlocal graph_state

        # Use client-provided IDs from RunAgentInput (AG-UI protocol compliance)
        run_id = run_input.run_id
        thread_id = run_input.thread_id

        try:
            # Configure LLM
            config_kwargs: dict = {}
            if request.temperature is not None:
                config_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                config_kwargs["max_completion_tokens"] = request.max_tokens
            llm_config.LLM_CONFIG = create_config(
                provider=request.provider,
                api_key=request.api_key,
                model=request.model,
                extra=request.extra,
                system_prompt=request.system_prompt,  # Custom system prompt (uses default if None)
                **config_kwargs,
            )
            print(f"[REWIND] LLM configured: provider={llm_config.LLM_CONFIG.provider}, "
                  f"model={llm_config.LLM_CONFIG.model}")
            print(f"[REWIND] Rewinding from batch_id={request.batch_id}")
            print(f"[REWIND] Feedback: {request.feedback[:100]}..." if len(request.feedback) > 100 else f"[REWIND] Feedback: {request.feedback}")

            # Generate partition key from clinical note (must match original traversal)
            # Look up current cache version for this configuration
            rewind_cache_version = get_cache_version(
                clinical_note=request.clinical_note,
                provider=request.provider,
                model=request.model or "",
                temperature=request.temperature or 0.0,
                system_prompt=request.system_prompt,
                scaffolded=True,
            )
            partition_key = generate_traversal_cache_key(
                clinical_note=request.clinical_note,
                provider=request.provider,
                model=request.model or "",
                temperature=request.temperature or 0.0,
                system_prompt=request.system_prompt,
                cache_version=rewind_cache_version,
            )
            SESSION_PARTITION_KEYS.add(partition_key)
            # Set global PARTITION_KEY so retry_node can use it for fork
            traversal_module.PARTITION_KEY = partition_key
            print(f"[REWIND] Partition key set: {partition_key}")
            # RUN_STARTED with rewind metadata
            yield encode_event(RunStartedEvent(thread_id=thread_id, run_id=run_id, input=run_input))
            yield encode_event(CustomEvent(name="run_metadata", value={
                'rewind_from': request.batch_id, 'feedback': request.feedback[:100],
            }))

            # Initialize persister (always keep existing data for rewind to find fork point)
            await initialize_persister(reset=False)

            # Load cached state and build pre-rewind snapshot (backend is authoritative)
            # Use traversal_module.PERSISTER since initialize_persister() sets it after import
            cached_state = await traversal_module.PERSISTER.load(
                partition_key=partition_key,
                app_id="ROOT",
            )
            if cached_state is None:
                raise RuntimeError(
                    f"Cannot rewind: ROOT checkpoint not found for partition_key={partition_key}"
                )

            # Extract state dict
            state_obj = cached_state.get("state")
            if state_obj is None:
                raise RuntimeError("cached_state has no 'state' field")
            state_dict = state_obj.get_all() if hasattr(state_obj, 'get_all') else dict(state_obj)

            # Prune state to rewind point (removes descendants, keeps ancestors)
            pruned_state = prune_state_for_rewind(state_dict, request.batch_id)
            pruned_batch_data = pruned_state.get("batch_data", {})
            pruned_final_nodes = pruned_state.get("final_nodes", [])

            print(f"[REWIND] Pruned state: {len(pruned_batch_data)} batches, {len(pruned_final_nodes)} final nodes")

            # Build graph from pruned batch_data and initialize incremental state
            snapshot_nodes, snapshot_edges, snapshot_seen_nodes, snapshot_seen_edges = build_graph_from_batch_data(
                pruned_batch_data, pruned_final_nodes
            )
            # Initialize accumulated state for incremental builds during rewind
            accumulated_batch_data.update(pruned_batch_data)
            graph_state = (snapshot_nodes, snapshot_edges, snapshot_seen_nodes, snapshot_seen_edges)

            # Emit STATE_SNAPSHOT with pre-rewind graph
            snapshot_state = GraphState(
                nodes=[GraphNode(**n) for n in snapshot_nodes],
                edges=[GraphEdge(**e) for e in snapshot_edges]
            )
            yield encode_event(StateSnapshotEvent(snapshot=snapshot_state.model_dump()))
            print(f"[REWIND] Emitted STATE_SNAPSHOT: {len(snapshot_nodes)} nodes, {len(snapshot_edges)} edges")

            # Set callback for streaming
            set_batch_callback(on_batch_complete)
            print("[REWIND] Batch callback registered")

            # Run retry_node in background task so we can stream events
            # Uses explicit feedback injection (inject_feedback action)
            async def run_retry():
                try:
                    print(f"[REWIND] Starting retry_node...")
                    result = await retry_node(
                        batch_id=request.batch_id,
                        feedback=request.feedback,
                        selector=request.selector,
                    )
                    print(f"[REWIND] retry_node completed successfully")
                    return result
                except Exception as e:
                    print(f"[REWIND] retry_node raised exception: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            task = asyncio.create_task(run_retry())
            RUNNING_TASK = task
            client_disconnected = False

            # Stream events while running
            while not task.done():
                # Check for client disconnection
                if await http_request.is_disconnected():
                    print("[REWIND] Client disconnected - cancelling rewind")
                    task.cancel()
                    client_disconnected = True
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    break

                try:
                    encoded = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    if encoded is not None:
                        yield encoded  # Already encoded by encode_event()
                except asyncio.TimeoutError:
                    continue

            # If client disconnected, exit early
            if client_disconnected:
                return

            # Get final state
            final_state = await task

            # Drain remaining events (already encoded strings)
            while not event_queue.empty():
                encoded = event_queue.get_nowait()
                if encoded is not None:
                    yield encoded

            # Final STATE_SNAPSHOT with finalized categories
            batch_data = final_state.get("batch_data", {})
            final_nodes_raw = final_state.get("final_nodes", [])
            final_nodes = list(dict.fromkeys(final_nodes_raw))

            # Log DFS trace of revised batches
            from agent.actions import print_rewind_summary
            pruned_batch_ids = set(pruned_batch_data.keys())
            print_rewind_summary(batch_data, final_nodes, request.batch_id, pruned_batch_ids)

            final_graph_nodes, final_graph_edges, _, _ = build_graph_from_batch_data(batch_data, final_nodes)
            final_snapshot = GraphState(
                nodes=[GraphNode(**n) for n in final_graph_nodes],
                edges=[GraphEdge(**e) for e in final_graph_edges],
            )
            yield encode_event(StateSnapshotEvent(snapshot=final_snapshot.model_dump()))

            # Build decisions using shared helper
            all_decisions = build_decisions_from_batch_data(batch_data)

            # Check for LLM API errors in the rewind batch (same pattern as stream_traversal)
            rewind_batch_reasoning = batch_data.get(request.batch_id, {}).get("reasoning", "")
            llm_error = rewind_batch_reasoning if rewind_batch_reasoning.startswith("API Error") else None

            if llm_error:
                error_type = llm_error.split("(", 1)[1].split(")", 1)[0] if "(" in llm_error and ")" in llm_error else "Unknown"
                print(f"[REWIND] Complete: final_nodes={final_nodes}, error={error_type}")
            else:
                print(f"[REWIND] Complete: final_nodes={final_nodes}")

            if llm_error:
                yield encode_event(RunErrorEvent(message=llm_error))
            else:
                yield encode_event(RunFinishedEvent(
                    thread_id=thread_id, run_id=run_id,
                    result={
                        'final_nodes': final_nodes,
                        'batch_count': len(batch_data),
                        'rewind_from': request.batch_id,
                        'decisions': all_decisions,
                    },
                ))
            print("[REWIND] Stream complete")

        except Exception as e:
            print(f"[REWIND] Error: {e}")
            yield encode_event(RunErrorEvent(message=str(e)))

        finally:
            RUNNING_TASK = None
            set_batch_callback(None)
            await cleanup_persister()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
