"""FastAPI server for ICD-10-CM Tree Traversal visualization

Includes:
- REST API for graph building (verification)
- AG-UI streaming endpoint for Burr-based traversal
- SSE (Server-Sent Events) for real-time batch completion updates
"""

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
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
    get_seventh_char_def,
    resolve_code,
)

from .agui_enums import AGUIEventType
from .events import AGUIEvent, GraphState, JsonPatchOp
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

    # Build the graph using trace_tree
    result = build_graph(valid_codes, data)

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
            seven_chr_def, _ = get_seventh_char_def(code, data)
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
    seven_chr_result = get_seventh_char_def(code, data)
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
        deleted = await delete_persisted_state(
            partition_key=partition_key,
            app_id="ROOT",
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


@app.post("/api/traverse/stream")
async def stream_traversal(request: TraversalRequest):
    """Run Burr-based DFS traversal with AG-UI streaming.

    Uses AG-UI protocol over SSE with JSON Patch (RFC 6902) for incremental updates.

    Returns SSE stream with events:
    - RUN_STARTED: Traversal begins
    - STATE_SNAPSHOT: Initial graph with ROOT node
    - STEP_STARTED: Batch processing begins
    - STATE_DELTA: JSON Patch ops to add nodes/edges
    - STEP_FINISHED: Batch complete with reasoning
    - RUN_FINISHED: Traversal complete with final codes
    """
    # Import Burr components
    from agent import build_app, initialize_persister, cleanup_persister, set_batch_callback
    from candidate_selector.providers import create_config
    import candidate_selector.config as llm_config

    # Queue for streaming events
    event_queue: asyncio.Queue[AGUIEvent | None] = asyncio.Queue()

    # Track seen nodes to avoid duplicates - dict maps node_id -> index for updates
    seen_nodes: dict[str, int] = {"ROOT": 0}
    node_count = 1  # ROOT is index 0
    seen_edges: set[tuple[str, str]] = set()

    # Helper to format 7th character code (same logic as actions.py)
    def format_with_seventh_char(base_code: str, seventh_char: str) -> str:
        """Format ICD-10-CM code with 7th character and placeholder padding."""
        if "." in base_code:
            category, subcategory = base_code.split(".", 1)
        else:
            category = base_code[:3] if len(base_code) >= 3 else base_code
            subcategory = base_code[3:] if len(base_code) > 3 else ""
        padded_subcategory = subcategory.ljust(3, "X") + seventh_char
        return f"{category}.{padded_subcategory}"

    # Batch callback to emit AG-UI events
    def on_batch_complete(
        batch_id: str,
        node_id: str | None,
        parent_id: str | None,
        depth: int,
        candidates: dict[str, str],
        selected_ids: list[str],
        reasoning: str,
        seven_chr_authority: dict | None = None,  # {"batch_name": str, "resolution_pattern": str}
    ):
        nonlocal node_count
        print(f"[CALLBACK] on_batch_complete called: batch_id={batch_id}, selected_ids={selected_ids}")
        # Parse batch_type from batch_id
        batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

        # STEP_STARTED
        event_queue.put_nowait(AGUIEvent(
            type=AGUIEventType.STEP_STARTED,
            step_id=batch_id,
        ))

        # STATE_DELTA - add nodes and edges
        ops: list[JsonPatchOp] = []

        # Collect batch-specific nodes/edges for STEP_FINISHED (reconciliation)
        # These are collected regardless of deduplication to ensure complete data
        batch_nodes: list[dict] = []
        batch_edges: list[dict] = []

        # For sevenChrDef batches, first ensure the base placeholder node itself is in the graph
        # This handles cases like T84.53X|sevenChrDef where T84.53X is a synthetic placeholder
        if batch_type == "sevenChrDef" and node_id and node_id not in seen_nodes:
            # Add the placeholder/activator node itself
            seen_nodes[node_id] = node_count
            node_count += 1

            # Determine node properties
            if node_id in data:
                node_label = data[node_id].get("label", node_id)
                node_depth = data[node_id].get("depth", depth)
                node_category = "activator" if node_has_seven_chr_def(node_id) else "ancestor"
            else:
                # Synthetic placeholder node (not in ICD index)
                node_label = "Placeholder"
                node_depth = depth
                node_category = "placeholder"

            base_node_data = {
                "id": node_id,
                "code": node_id,
                "label": node_label,
                "depth": node_depth,
                "category": node_category,
                "billable": False,
            }
            ops.append(JsonPatchOp(op="add", path="/nodes/-", value=base_node_data))
            batch_nodes.append(base_node_data)

            # Add edge from parent to this node
            if parent_id and parent_id != node_id:
                edge_key = (parent_id, node_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    base_edge_data = {
                        "source": parent_id,
                        "target": node_id,
                        "edge_type": "hierarchy",
                        "rule": None,
                    }
                    ops.append(JsonPatchOp(op="add", path="/edges/-", value=base_edge_data))
                    batch_edges.append(base_edge_data)

        # Special handling for sevenChrDef batches:
        # - selected_ids contains full codes (e.g., ["T36.1X5A"]) - transformed by actions.py
        # - For codes at depth 3-5, create placeholder nodes (with X padding) until depth 6
        # - The 7th char node is added via lateral edge from the depth-6 node
        if batch_type == "sevenChrDef" and selected_ids and node_id:
            # Extract the 7th character from selected_ids (last char of full code)
            seventh_char = selected_ids[0][-1] if selected_ids[0] else ""

            # CRITICAL FIX: Compute full_code from node_id + 7th char
            # This ensures the code matches the batch's node, preventing cross-batch contamination
            if "." in node_id:
                base_category, base_subcategory = node_id.split(".", 1)
            else:
                base_category = node_id[:3] if len(node_id) >= 3 else node_id
                base_subcategory = node_id[3:] if len(node_id) > 3 else ""
            # Pad subcategory to 3 chars and add 7th character
            full_code = f"{base_category}.{base_subcategory.ljust(3, 'X')}{seventh_char}"

            # Get base code depth to determine if placeholder nodes are needed
            base_depth = data.get(node_id, {}).get("depth", depth)

            # Create placeholder nodes if base code is shorter than depth 6
            # ICD-10-CM format: XXX.XXXX where 7th char is position 4 after dot
            # Depth 3 = category (T36), depth 4 = 4-char (T36.1), depth 5 = 5-char (T36.1X)
            # depth 6 = 6-char (T36.1X5), depth 7 = 7-char (T36.1X5A)
            prev_node = node_id
            print(f"[SEVENCHRDEF] node_id={node_id}, full_code={full_code}, base_depth={base_depth}, seventh_char={seventh_char}")
            if base_depth < 6:
                # Parse base code to determine how many X's to add
                if "." in node_id:
                    category, subcategory = node_id.split(".", 1)
                else:
                    category = node_id[:3] if len(node_id) >= 3 else node_id
                    subcategory = node_id[3:] if len(node_id) > 3 else ""

                # Create placeholder nodes from current length to depth 6
                current_sub = subcategory
                print(f"[SEVENCHRDEF] Creating placeholders: subcategory={subcategory}, range({len(subcategory)}, 3)")
                for i in range(len(subcategory), 3):  # Pad to 3 chars (positions 4-6)
                    current_sub += "X"
                    placeholder_code = f"{category}.{current_sub}"
                    placeholder_depth = base_depth + (i - len(subcategory) + 1)
                    print(f"[SEVENCHRDEF] Creating placeholder: {placeholder_code} at depth {placeholder_depth}, prev_node={prev_node}")

                    # Always collect for batch data (reconciliation)
                    placeholder_node_data = {
                        "id": placeholder_code,
                        "code": placeholder_code,
                        "label": "Placeholder",
                        "depth": placeholder_depth,
                        "category": "placeholder",
                        "billable": False,
                    }
                    batch_nodes.append(placeholder_node_data)

                    # Edges TO placeholder nodes are hierarchy (not lateral)
                    # Placeholders can still open other parallel pathways
                    placeholder_edge_data = {
                        "source": prev_node,
                        "target": placeholder_code,
                        "edge_type": "hierarchy",
                        "rule": None,
                    }
                    batch_edges.append(placeholder_edge_data)

                    # STATE_DELTA ops (with deduplication for streaming)
                    if placeholder_code not in seen_nodes:
                        seen_nodes[placeholder_code] = node_count
                        node_count += 1
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/nodes/-",
                            value=placeholder_node_data,
                        ))

                    # Edge creation is OUTSIDE node check - edges need to be created
                    # even if node already exists (e.g., created by placeholder children batch)
                    edge_key = (prev_node, placeholder_code)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/edges/-",
                            value=placeholder_edge_data,
                        ))

                    prev_node = placeholder_code

            # Create the final 7th character node
            # candidates is keyed by full code (transformed by actions.py), e.g. {"T36.1X5A": "description"}
            label = f"{seventh_char}: {candidates.get(full_code, seventh_char)}"

            # Use prev_node as edge source - it's the depth-6 node computed from node_id
            # (either the original node_id if already depth 6, or the last placeholder created)
            # This is now consistent with full_code since both are derived from node_id
            correct_edge_source = prev_node

            # Ensure the parent node exists in the graph (create placeholder if needed)
            if correct_edge_source not in seen_nodes:
                seen_nodes[correct_edge_source] = node_count
                node_count += 1
                parent_depth = 6  # 7th char parent is always depth 6
                parent_node_data = {
                    "id": correct_edge_source,
                    "code": correct_edge_source,
                    "label": "Placeholder" if correct_edge_source.endswith("X") else data.get(correct_edge_source, {}).get("label", correct_edge_source),
                    "depth": parent_depth,
                    "category": "placeholder" if correct_edge_source.endswith("X") else "ancestor",
                    "billable": False,
                }
                ops.append(JsonPatchOp(op="add", path="/nodes/-", value=parent_node_data))
                batch_nodes.append(parent_node_data)

            # Always collect for batch data (reconciliation)
            final_node_data = {
                "id": full_code,
                "code": full_code,
                "label": label,
                "depth": 7,
                "category": "finalized",
                "billable": True,
            }
            batch_nodes.append(final_node_data)

            final_edge_data = {
                "source": correct_edge_source,
                "target": full_code,
                "edge_type": "lateral",
                "rule": "sevenChrDef",
            }
            batch_edges.append(final_edge_data)

            # STATE_DELTA ops (with deduplication for streaming)
            if full_code not in seen_nodes:
                seen_nodes[full_code] = node_count
                node_count += 1
                ops.append(JsonPatchOp(
                    op="add",
                    path="/nodes/-",
                    value=final_node_data,
                ))
                # Note: Don't add to /finalized/- here - authoritative count comes from RUN_FINISHED

                edge_key = (correct_edge_source, full_code)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    ops.append(JsonPatchOp(
                        op="add",
                        path="/edges/-",
                        value=final_edge_data,
                    ))
        else:
            # Regular batches (children, codeFirst, codeAlso, etc.)
            print(f"[BATCH] Processing {batch_id}: node_id={node_id}, selected_ids={selected_ids}")

            # Note: Empty children selection means the parent is a terminal node, but
            # we don't mark it as "finalized" here because terminal ≠ billable/submittable.
            # The authoritative final_nodes list comes from RUN_FINISHED.

            # For children batches, add the current node_id to the graph as a traversed node.
            # This ensures the traversal path is recorded, not just the selected children.
            # This is critical for:
            # - Placeholder batches spawned by SpawnSevenChr (T36.1X, T36.1XX, etc.)
            # - Regular nodes where LLM selected nothing but node was still traversed
            if (
                batch_type == "children"
                and node_id
                and node_id != "ROOT"
                and node_id not in seen_nodes
            ):
                seen_nodes[node_id] = node_count
                node_count += 1

                # Determine node properties
                if node_id in data:
                    node_label = data[node_id].get("label", node_id)
                    node_depth = data[node_id].get("depth", depth)
                else:
                    node_label = "Placeholder" if node_id.endswith("X") else node_id
                    node_depth = depth

                # Determine category
                if node_has_seven_chr_def(node_id):
                    node_category = "activator"
                elif node_id not in data and node_id.endswith("X"):
                    node_category = "placeholder"
                else:
                    node_category = "ancestor"

                traversed_node_data = {
                    "id": node_id,
                    "code": node_id,
                    "label": node_label,
                    "depth": node_depth,
                    "category": node_category,
                    "billable": False,  # Traversed nodes aren't billable (finalized ones are)
                }
                ops.append(JsonPatchOp(
                    op="add",
                    path="/nodes/-",
                    value=traversed_node_data,
                ))
                batch_nodes.append(traversed_node_data)

                # Add edge from parent to this node
                if parent_id and parent_id != node_id:
                    edge_key = (parent_id, node_id)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        traversed_edge_data = {
                            "source": parent_id,
                            "target": node_id,
                            "edge_type": "hierarchy",
                            "rule": None,
                        }
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/edges/-",
                            value=traversed_edge_data,
                        ))
                        batch_edges.append(traversed_edge_data)

                print(f"[TRAVERSED] Added traversed node: {node_id} at depth {node_depth} (category: {node_category})")

            for code in selected_ids:
                # Add node only if not already seen
                print(f"[DEDUP] code={code}: in_seen_nodes={code in seen_nodes}")
                if code not in seen_nodes:
                    seen_nodes[code] = node_count
                    node_count += 1
                    label = candidates.get(code, code)

                    # Always prefer actual ICD depth from data when available
                    # Fall back to traversal depth + 1 only if not in ICD index
                    if code in data:
                        node_depth = data[code].get("depth", depth + 1)
                    else:
                        node_depth = depth + 1

                    # Determine category (matches archive styling):
                    # - "activator" (blue border) for nodes with sevenChrDef metadata
                    # - "placeholder" (gray dashed) for synthetic codes not in index
                    # - "ancestor" for all other nodes
                    if node_has_seven_chr_def(code):
                        category = "activator"
                    elif code not in data and code.endswith("X"):
                        category = "placeholder"
                    else:
                        category = "ancestor"

                    # Billable only if:
                    # - Node has no children in ICD hierarchy
                    # - Node does NOT have a sevenChrDef authority (nodes with authority need 7th char)
                    # - Node itself does NOT have sevenChrDef (activator nodes need 7th char selection)
                    has_children = bool(data.get(code, {}).get("children"))
                    has_authority = seven_chr_authority is not None
                    is_activator = category == "activator"
                    # A node with sevenChrDef authority OR that is an activator is NOT billable
                    # because the actual billable code will be the 7th character extension
                    billable = not has_children and not has_authority and not is_activator

                    ops.append(JsonPatchOp(
                        op="add",
                        path="/nodes/-",
                        value={
                            "id": code,
                            "code": code,
                            "label": label,
                            "depth": node_depth,
                            "category": category,
                            "billable": billable,
                        }
                    ))

                # Add edge OUTSIDE the node check - allows multiple edges to same node
                # For children batches: source is the parent node
                # For lateral batches: source is the node with the metadata
                edge_source = node_id if node_id and node_id != "ROOT" else "ROOT"

                edge_key = (edge_source, code)

                # Determine edge type:
                # - "lateral" for non-children batches (codeFirst, codeAlso, etc.)
                # - "hierarchy" for regular children
                if batch_type != "children":
                    edge_type = "lateral"
                    rule = batch_type
                else:
                    edge_type = "hierarchy"
                    rule = None

                is_seen = edge_key in seen_edges
                print(f"[EDGE] {edge_source} -> {code}: batch_type={batch_type}, edge_type={edge_type}, seen={is_seen}")

                if not is_seen:
                    seen_edges.add(edge_key)
                    ops.append(JsonPatchOp(
                        op="add",
                        path="/edges/-",
                        value={
                            "source": edge_source,
                            "target": code,
                            "edge_type": edge_type,
                            "rule": rule,
                        }
                    ))

        # Log ops summary
        node_ops = [op for op in ops if op.path == "/nodes/-"]
        edge_ops = [op for op in ops if op.path == "/edges/-"]
        print(f"[OPS] {batch_id}: {len(node_ops)} nodes, {len(edge_ops)} edges")

        if ops:
            event_queue.put_nowait(AGUIEvent(
                type=AGUIEventType.STATE_DELTA,
                delta=ops,
            ))
            print(f"[STATE_DELTA] Sent {len(ops)} ops for {batch_id}")
        else:
            print(f"[STATE_DELTA] SKIPPED - no ops for {batch_id}")

        # Compute selected_details for reconciliation (O(1) dict lookups, <1ms)
        # This provides ICD domain data so frontend can derive graph structure
        selected_details: dict[str, dict] = {}
        for code in selected_ids:
            code_data = data.get(code, {})

            # Determine category
            if node_has_seven_chr_def(code):
                cat = "activator"
            elif "X" in code:
                cat = "placeholder"
            else:
                cat = "ancestor"

            # Always prefer actual ICD depth from data when available
            # Fall back to traversal depth + 1 only if not in ICD index
            if code in data:
                code_depth = code_data.get("depth", depth + 1)
            else:
                code_depth = depth + 1

            # Determine billable
            has_children = bool(code_data.get("children"))
            is_activator = cat == "activator"
            billable = not has_children and not (seven_chr_authority is not None) and not is_activator

            selected_details[code] = {
                "depth": code_depth,
                "category": cat,
                "billable": billable,
            }

        # STEP_FINISHED - include error flag if reasoning indicates API error
        is_error = reasoning.startswith("API Error:") if reasoning else False
        event_queue.put_nowait(AGUIEvent(
            type=AGUIEventType.STEP_FINISHED,
            step_id=batch_id,
            metadata={
                "node_id": node_id,
                "batch_type": batch_type,
                "selected_ids": selected_ids,
                "reasoning": reasoning,
                "candidates": candidates,
                "error": is_error,
                "selected_details": selected_details,  # NEW: ICD properties for reconciliation
            },
        ))

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate AG-UI SSE events."""
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
            # IMPORTANT: For scaffolded mode, we call build_app() here (single source of truth)
            # rather than doing a separate cache check. This eliminates the dual cache check
            # issue where the early check and build_app() could disagree about cache status.
            is_cached = False
            scaffolded_burr_app = None  # Will be set for scaffolded mode
            scaffolded_partition_key = None

            if not request.scaffolded:
                # Zero-shot mode: check cache
                await initialize_zero_shot_persister(reset=not request.persist_cache)
                zs_partition_key = generate_zero_shot_cache_key(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    max_completion_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
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
                # Scaffolded mode: call build_app() directly (single source of truth for cache status)
                # This eliminates the dual cache check issue - build_app() is the authoritative check
                scaffolded_partition_key = generate_traversal_cache_key(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    system_prompt=request.system_prompt,
                )
                print(f"[SERVER] Scaffolded cache key: {scaffolded_partition_key}")

                # Initialize persister
                await initialize_persister(reset=not request.persist_cache)

                # Set callback for streaming (needed before build_app for live traversal)
                set_batch_callback(on_batch_complete)
                print("[SERVER] Batch callback registered")

                # Build app - this is the SINGLE cache check (no separate early check)
                scaffolded_burr_app, is_cached = await build_app(
                    context=request.clinical_note,
                    default_selector=request.selector,
                    with_persistence=True,
                    partition_key=scaffolded_partition_key,
                )
                print(f"[SERVER] Scaffolded build_app result: cached={is_cached}")

            # RUN_STARTED with cache flag
            run_started_event = AGUIEvent(
                type=AGUIEventType.RUN_STARTED,
                metadata={'clinical_note': request.clinical_note[:100], 'cached': is_cached}
            )
            yield f"data: {run_started_event.model_dump_json()}\n\n"

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
            yield f"data: {AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, state=initial_state).model_dump_json()}\n\n"

            # Branch based on scaffolded flag
            if not request.scaffolded:
                # Zero-shot mode: Burr app with SQLite caching
                print("[SERVER] Running in ZERO-SHOT mode (Burr app with caching)")

                # Initialize zero-shot persister (reset if persist_cache=False)
                await initialize_zero_shot_persister(reset=not request.persist_cache)

                # Generate cache key from request parameters
                partition_key = generate_zero_shot_cache_key(
                    clinical_note=request.clinical_note,
                    provider=request.provider,
                    model=request.model or "",
                    temperature=request.temperature or 0.0,
                    max_completion_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                )
                print(f"[SERVER] Zero-shot cache key: {partition_key}")

                # STEP_STARTED for zero-shot batch
                yield f"data: {AGUIEvent(type=AGUIEventType.STEP_STARTED, step_id='ROOT|zero-shot').model_dump_json()}\n\n"

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
                    # Cache miss - run the Burr app
                    print("[SERVER] Zero-shot CACHE MISS - calling LLM")
                    _, _, final_state = await zs_app.arun(halt_after=["finish"])
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
                        seven_def_result = get_seventh_char_def(code, data)
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
                    yield f"data: {AGUIEvent(type=AGUIEventType.STATE_DELTA, delta=ops).model_dump_json()}\n\n"

                # STEP_FINISHED with populated candidates and selected_details
                is_error = reasoning.startswith("API Error:") if reasoning else False
                yield f"data: {AGUIEvent(type=AGUIEventType.STEP_FINISHED, step_id='ROOT|zero-shot', metadata={'node_id': 'ROOT', 'batch_type': 'zero-shot', 'selected_ids': selected_codes, 'reasoning': reasoning, 'candidates': candidates, 'error': is_error, 'selected_details': selected_details}).model_dump_json()}\n\n"

                # RUN_FINISHED
                run_finished_metadata = {
                    'final_nodes': selected_codes,
                    'batch_count': 1,
                    'mode': 'zero-shot',
                    'cached': was_cached,
                }
                if is_error:
                    run_finished_metadata['error'] = reasoning

                cache_status = "CACHED" if was_cached else "NEW"
                print(f"[SERVER] Zero-shot complete ({cache_status}): {len(selected_codes)} codes generated")
                yield f"data: {AGUIEvent(type=AGUIEventType.RUN_FINISHED, metadata=run_finished_metadata).model_dump_json()}\n\n"
                return  # Exit generator - zero-shot mode complete

            # Scaffolded mode: Tree traversal with Burr
            # NOTE: build_app() was already called in the early check section above
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

                # Build complete graph from cached state (single pass)
                # Use different variable names to avoid shadowing outer function's seen_nodes/seen_edges
                # (which are used by the live traversal path for finalization)
                snapshot_nodes: list[dict] = [{"id": "ROOT", "code": "ROOT", "label": "ROOT", "depth": 0, "category": "root", "billable": False}]
                snapshot_edges: list[dict] = []
                snapshot_seen_nodes: set[str] = {"ROOT"}
                snapshot_seen_edges: set[tuple[str, str]] = set()

                # Sort batches by depth to ensure parents are processed before children
                sorted_batches = sorted(
                    [(bid, binfo) for bid, binfo in batch_data.items()],
                    key=lambda x: (x[1].get("depth", 0), x[0])
                )

                print(f"[CACHE SNAPSHOT] Building snapshot from {len(sorted_batches)} batches")

                # Build all nodes and edges from cached batch data
                for batch_id, batch_info in sorted_batches:
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
                        if node_id not in snapshot_seen_nodes:
                            snapshot_seen_nodes.add(node_id)
                            if node_id in data:
                                node_label = data[node_id].get("label", node_id)
                                node_depth = data[node_id].get("depth", depth)
                                node_category = "activator" if node_has_seven_chr_def(node_id) else "ancestor"
                            else:
                                node_label = "Placeholder"
                                node_depth = depth
                                node_category = "placeholder"
                            snapshot_nodes.append({
                                "id": node_id, "code": node_id, "label": node_label,
                                "depth": node_depth, "category": node_category, "billable": False,
                            })

                        if parent_id and parent_id != node_id:
                            edge_key = (parent_id, node_id)
                            if edge_key not in snapshot_seen_edges:
                                snapshot_seen_edges.add(edge_key)
                                snapshot_edges.append({
                                    "source": parent_id, "target": node_id,
                                    "edge_type": "hierarchy", "rule": None,
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

                                    if placeholder_code not in snapshot_seen_nodes:
                                        snapshot_seen_nodes.add(placeholder_code)
                                        snapshot_nodes.append({
                                            "id": placeholder_code, "code": placeholder_code,
                                            "label": "Placeholder", "depth": placeholder_depth,
                                            "category": "placeholder", "billable": False,
                                        })

                                    edge_key = (prev_node, placeholder_code)
                                    if edge_key not in snapshot_seen_edges:
                                        snapshot_seen_edges.add(edge_key)
                                        snapshot_edges.append({
                                            "source": prev_node, "target": placeholder_code,
                                            "edge_type": "hierarchy", "rule": None,
                                        })
                                    prev_node = placeholder_code

                            # Create depth-6 parent if needed
                            if prev_node not in snapshot_seen_nodes:
                                snapshot_seen_nodes.add(prev_node)
                                snapshot_nodes.append({
                                    "id": prev_node, "code": prev_node,
                                    "label": "Placeholder" if prev_node.endswith("X") else data.get(prev_node, {}).get("label", prev_node),
                                    "depth": 6,
                                    "category": "placeholder" if prev_node.endswith("X") else "ancestor",
                                    "billable": False,
                                })

                            # Create final 7th character node
                            label = f"{seventh_char}: {transformed_candidates.get(full_code, seventh_char)}"
                            if full_code not in snapshot_seen_nodes:
                                snapshot_seen_nodes.add(full_code)
                                snapshot_nodes.append({
                                    "id": full_code, "code": full_code, "label": label,
                                    "depth": 7, "category": "finalized", "billable": True,
                                })

                            edge_key = (prev_node, full_code)
                            if edge_key not in snapshot_seen_edges:
                                snapshot_seen_edges.add(edge_key)
                                snapshot_edges.append({
                                    "source": prev_node, "target": full_code,
                                    "edge_type": "lateral", "rule": "sevenChrDef",
                                })

                    # Regular batches (children, codeFirst, codeAlso, etc.)
                    elif batch_type != "sevenChrDef":
                        # Add traversed node
                        if node_id and node_id != "ROOT" and node_id not in snapshot_seen_nodes:
                            snapshot_seen_nodes.add(node_id)
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
                            else:
                                node_category = "ancestor"

                            snapshot_nodes.append({
                                "id": node_id, "code": node_id, "label": node_label,
                                "depth": node_depth, "category": node_category, "billable": False,
                            })

                        if node_id and parent_id and parent_id != node_id:
                            edge_key = (parent_id, node_id)
                            if edge_key not in snapshot_seen_edges:
                                snapshot_seen_edges.add(edge_key)
                                snapshot_edges.append({
                                    "source": parent_id, "target": node_id,
                                    "edge_type": "hierarchy", "rule": None,
                                })

                        # Add selected children
                        for code in selected_ids:
                            if code not in snapshot_seen_nodes:
                                snapshot_seen_nodes.add(code)
                                label = candidates.get(code, code)

                                if code in data:
                                    code_depth = data[code].get("depth", depth + 1)
                                else:
                                    code_depth = depth + 1

                                if node_has_seven_chr_def(code):
                                    code_category = "activator"
                                elif code not in data and code.endswith("X"):
                                    code_category = "placeholder"
                                else:
                                    code_category = "ancestor"

                                has_children = bool(data.get(code, {}).get("children"))
                                has_authority = seven_chr_authority is not None
                                is_activator = code_category == "activator"
                                billable = not has_children and not has_authority and not is_activator

                                snapshot_nodes.append({
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

                            if edge_key not in snapshot_seen_edges:
                                snapshot_seen_edges.add(edge_key)
                                snapshot_edges.append({
                                    "source": edge_source, "target": code,
                                    "edge_type": edge_type, "rule": rule,
                                })

                # Build all decisions for benchmark comparison
                all_decisions: list[dict] = []
                for batch_id, batch_info in sorted_batches:
                    node_id = batch_info.get("node_id")
                    candidates = batch_info.get("candidates", {})
                    selected_ids = batch_info.get("selected_ids", [])
                    reasoning = batch_info.get("reasoning", "[CACHED]")
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

                        if code in data:
                            code_depth = code_data.get("depth", depth + 1)
                        else:
                            code_depth = depth + 1

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

                # Emit single STATE_SNAPSHOT with complete graph
                snapshot_state = GraphState(
                    nodes=[GraphNode(**n) for n in snapshot_nodes],
                    edges=[GraphEdge(**e) for e in snapshot_edges]
                )
                yield f"data: {AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, state=snapshot_state).model_dump_json()}\n\n"

                # Emit RUN_FINISHED with final_nodes and all decisions
                run_finished_metadata = {
                    'final_nodes': final_nodes,
                    'batch_count': len(batch_data),
                    'mode': 'scaffolded',
                    'cached': True,
                    'decisions': all_decisions,
                }
                print(f"[SERVER] Scaffolded complete (CACHED SNAPSHOT): {len(snapshot_nodes)} nodes, {len(snapshot_edges)} edges, {len(all_decisions)} decisions")
                yield f"data: {AGUIEvent(type=AGUIEventType.RUN_FINISHED, metadata=run_finished_metadata).model_dump_json()}\n\n"
                return  # Exit early for cache hit

            else:
                # Cache miss - run the traversal
                print("[SERVER] Scaffolded CACHE MISS - running traversal")

                # Run in background task so we can stream events
                async def run_app():
                    _, _, state = await burr_app.arun(halt_after=["finish"])
                    return state

                task = asyncio.create_task(run_app())

                # Stream events while running
                while not task.done():
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        if event is not None:
                            yield f"data: {event.model_dump_json()}\n\n"
                    except asyncio.TimeoutError:
                        continue

                # Get final state
                final_state = await task

                # Drain remaining events
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    if event is not None:
                        yield f"data: {event.model_dump_json()}\n\n"

            # Update finalized nodes in graph (deduplicate in case parallel batches reached same node)
            final_nodes_raw = final_state.get("final_nodes", [])
            final_nodes = list(dict.fromkeys(final_nodes_raw))  # Preserve order, remove duplicates

            finalize_ops: list[JsonPatchOp] = []
            for node in final_nodes:
                if node in seen_nodes:
                    node_index = seen_nodes[node]
                    # Update category to finalized (harmless if already finalized)
                    finalize_ops.append(JsonPatchOp(
                        op="replace",
                        path=f"/nodes/{node_index}/category",
                        value="finalized",
                    ))
                    # Add to finalized array
                    finalize_ops.append(JsonPatchOp(
                        op="add",
                        path="/finalized/-",
                        value=node,
                    ))

            if finalize_ops:
                yield f"data: {AGUIEvent(type=AGUIEventType.STATE_DELTA, delta=finalize_ops).model_dump_json()}\n\n"

            # RUN_FINISHED - check if there was an LLM error
            batch_data = final_state.get("batch_data", {})
            root_reasoning = batch_data.get("ROOT", {}).get("reasoning", "")
            llm_error = root_reasoning if root_reasoning.startswith("API Error:") else None

            run_finished_metadata = {
                'final_nodes': final_nodes,
                'batch_count': len(batch_data),
                'mode': 'scaffolded',
                'cached': is_cached,
            }
            if llm_error:
                run_finished_metadata['error'] = llm_error

            cache_status = "CACHED" if is_cached else "NEW"
            print(f"[SERVER] Scaffolded complete ({cache_status}): final_nodes={final_nodes}, batch_count={len(batch_data)}, error={bool(llm_error)}")
            yield f"data: {AGUIEvent(type=AGUIEventType.RUN_FINISHED, metadata=run_finished_metadata).model_dump_json()}\n\n"
            print("[SERVER] Stream complete")

        except Exception as e:
            error_event = AGUIEvent(
                type=AGUIEventType.RUN_FINISHED,
                metadata={"error": str(e)},
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

        finally:
            # Cleanup
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
async def stream_rewind(request: RewindRequest):
    """Rewind traversal from a specific batch with feedback.

    Uses Burr's fork pattern to branch from a checkpoint and inject corrective
    feedback. Streams updates via AG-UI SSE (same as /api/traverse/stream).

    The feedback is passed to the LLM selector as "CRITICAL ADDITIONAL GUIDANCE"
    that takes priority over the general clinical context.

    Returns SSE stream with events:
    - RUN_STARTED: Rewind begins (includes rewind_from batch_id)
    - STEP_STARTED: Batch processing begins
    - STATE_DELTA: JSON Patch ops to add nodes/edges
    - STEP_FINISHED: Batch complete with reasoning
    - RUN_FINISHED: Rewind complete with final codes
    """
    from agent import retry_node, initialize_persister, cleanup_persister, set_batch_callback
    from agent.traversal import PARTITION_KEY, generate_traversal_cache_key
    import agent.traversal as traversal_module
    from candidate_selector.providers import create_config
    import candidate_selector.config as llm_config

    # Queue for streaming events
    event_queue: asyncio.Queue[AGUIEvent | None] = asyncio.Queue()

    # Track seen nodes to avoid duplicates - dict maps node_id -> index for updates
    # Pre-populate with existing nodes from the graph (for lateral target parent lookup)
    seen_nodes: dict[str, int] = {"ROOT": 0}
    node_count = 1  # ROOT is index 0
    for existing_node in request.existing_nodes:
        if existing_node not in seen_nodes:
            seen_nodes[existing_node] = node_count
            node_count += 1
    seen_edges: set[tuple[str, str]] = set()

    # Helper to format 7th character code (same logic as actions.py)
    def format_with_seventh_char(base_code: str, seventh_char: str) -> str:
        """Format ICD-10-CM code with 7th character and placeholder padding."""
        if "." in base_code:
            category, subcategory = base_code.split(".", 1)
        else:
            category = base_code[:3] if len(base_code) >= 3 else base_code
            subcategory = base_code[3:] if len(base_code) > 3 else ""
        padded_subcategory = subcategory.ljust(3, "X") + seventh_char
        return f"{category}.{padded_subcategory}"

    # Batch callback to emit AG-UI events (same as stream_traversal)
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
        nonlocal node_count
        print(f"[REWIND CALLBACK] on_batch_complete: batch_id={batch_id}, selected_ids={selected_ids}")
        batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

        # STEP_STARTED
        event_queue.put_nowait(AGUIEvent(
            type=AGUIEventType.STEP_STARTED,
            step_id=batch_id,
        ))

        # STATE_DELTA - add nodes and edges
        ops: list[JsonPatchOp] = []
        batch_nodes: list[dict] = []
        batch_edges: list[dict] = []

        # Special handling for sevenChrDef batches
        if batch_type == "sevenChrDef" and selected_ids and node_id:
            # Extract the 7th character from selected_ids
            seventh_char = selected_ids[0][-1] if selected_ids[0] else ""

            # CRITICAL FIX: Compute full_code from node_id + 7th char
            # This ensures the code matches the batch's node
            if "." in node_id:
                base_category, base_subcategory = node_id.split(".", 1)
            else:
                base_category = node_id[:3] if len(node_id) >= 3 else node_id
                base_subcategory = node_id[3:] if len(node_id) > 3 else ""
            full_code = f"{base_category}.{base_subcategory.ljust(3, 'X')}{seventh_char}"

            base_depth = data.get(node_id, {}).get("depth", depth)
            prev_node = node_id

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

                    placeholder_node_data = {
                        "id": placeholder_code,
                        "code": placeholder_code,
                        "label": "Placeholder",
                        "depth": placeholder_depth,
                        "category": "placeholder",
                        "billable": False,
                    }
                    batch_nodes.append(placeholder_node_data)

                    placeholder_edge_data = {
                        "source": prev_node,
                        "target": placeholder_code,
                        "edge_type": "hierarchy",
                        "rule": None,
                    }
                    batch_edges.append(placeholder_edge_data)

                    if placeholder_code not in seen_nodes:
                        seen_nodes[placeholder_code] = node_count
                        node_count += 1
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/nodes/-",
                            value=placeholder_node_data,
                        ))

                    # Edge creation OUTSIDE node check
                    edge_key = (prev_node, placeholder_code)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/edges/-",
                            value=placeholder_edge_data,
                        ))

                    prev_node = placeholder_code

            label = f"{seventh_char}: {candidates.get(full_code, seventh_char)}"
            # Use prev_node as edge source - it's derived from node_id
            correct_edge_source = prev_node

            # Ensure the parent node exists in the graph (create placeholder if needed)
            if correct_edge_source not in seen_nodes:
                seen_nodes[correct_edge_source] = node_count
                node_count += 1
                parent_depth = 6  # 7th char parent is always depth 6
                parent_node_data = {
                    "id": correct_edge_source,
                    "code": correct_edge_source,
                    "label": "Placeholder" if correct_edge_source.endswith("X") else data.get(correct_edge_source, {}).get("label", correct_edge_source),
                    "depth": parent_depth,
                    "category": "placeholder" if correct_edge_source.endswith("X") else "ancestor",
                    "billable": False,
                }
                ops.append(JsonPatchOp(op="add", path="/nodes/-", value=parent_node_data))
                batch_nodes.append(parent_node_data)

            final_node_data = {
                "id": full_code,
                "code": full_code,
                "label": label,
                "depth": 7,
                "category": "finalized",
                "billable": True,
            }
            batch_nodes.append(final_node_data)

            final_edge_data = {
                "source": correct_edge_source,
                "target": full_code,
                "edge_type": "lateral",
                "rule": "sevenChrDef",
            }
            batch_edges.append(final_edge_data)

            if full_code not in seen_nodes:
                seen_nodes[full_code] = node_count
                node_count += 1
                ops.append(JsonPatchOp(
                    op="add",
                    path="/nodes/-",
                    value=final_node_data,
                ))

                edge_key = (correct_edge_source, full_code)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    ops.append(JsonPatchOp(
                        op="add",
                        path="/edges/-",
                        value=final_edge_data,
                    ))
        else:
            # Regular batches
            if (
                batch_type == "children"
                and node_id
                and node_id != "ROOT"
                and node_id not in seen_nodes
            ):
                seen_nodes[node_id] = node_count
                node_count += 1

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
                else:
                    node_category = "ancestor"

                traversed_node_data = {
                    "id": node_id,
                    "code": node_id,
                    "label": node_label,
                    "depth": node_depth,
                    "category": node_category,
                    "billable": False,
                }
                ops.append(JsonPatchOp(
                    op="add",
                    path="/nodes/-",
                    value=traversed_node_data,
                ))
                batch_nodes.append(traversed_node_data)

                if parent_id and parent_id != node_id:
                    edge_key = (parent_id, node_id)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        traversed_edge_data = {
                            "source": parent_id,
                            "target": node_id,
                            "edge_type": "hierarchy",
                            "rule": None,
                        }
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/edges/-",
                            value=traversed_edge_data,
                        ))
                        batch_edges.append(traversed_edge_data)

            for code in selected_ids:
                if batch_type != "children":
                    # Add the lateral target code itself
                    if code not in seen_nodes:
                        seen_nodes[code] = node_count
                        node_count += 1
                        label = candidates.get(code, code)

                        if code in data:
                            node_depth = data[code].get("depth", depth + 1)
                        else:
                            node_depth = depth + 1

                        if node_has_seven_chr_def(code):
                            category = "activator"
                        elif code not in data and code.endswith("X"):
                            category = "placeholder"
                        else:
                            category = "ancestor"

                        has_children = bool(data.get(code, {}).get("children"))
                        has_authority = seven_chr_authority is not None
                        is_activator = category == "activator"
                        billable = not has_children and not has_authority and not is_activator

                        ops.append(JsonPatchOp(
                            op="add",
                            path="/nodes/-",
                            value={
                                "id": code,
                                "code": code,
                                "label": label,
                                "depth": node_depth,
                                "category": category,
                                "billable": billable,
                            }
                        ))

                    # Create the lateral edge from source node to code
                    edge_source = node_id if node_id and node_id != "ROOT" else "ROOT"
                    lateral_edge_key = (edge_source, code)
                    if lateral_edge_key not in seen_edges:
                        seen_edges.add(lateral_edge_key)
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/edges/-",
                            value={
                                "source": edge_source,
                                "target": code,
                                "edge_type": "lateral",
                                "rule": batch_type,
                            }
                        ))

                else:
                    # Children batch - regular hierarchy handling
                    if code not in seen_nodes:
                        seen_nodes[code] = node_count
                        node_count += 1
                        label = candidates.get(code, code)

                        if code in data:
                            node_depth = data[code].get("depth", depth + 1)
                        else:
                            node_depth = depth + 1

                        if node_has_seven_chr_def(code):
                            category = "activator"
                        elif code not in data and code.endswith("X"):
                            category = "placeholder"
                        else:
                            category = "ancestor"

                        has_children = bool(data.get(code, {}).get("children"))
                        has_authority = seven_chr_authority is not None
                        is_activator = category == "activator"
                        billable = not has_children and not has_authority and not is_activator

                        ops.append(JsonPatchOp(
                            op="add",
                            path="/nodes/-",
                            value={
                                "id": code,
                                "code": code,
                                "label": label,
                                "depth": node_depth,
                                "category": category,
                                "billable": billable,
                            }
                        ))

                    edge_source = node_id if node_id and node_id != "ROOT" else "ROOT"
                    edge_key = (edge_source, code)

                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        ops.append(JsonPatchOp(
                            op="add",
                            path="/edges/-",
                            value={
                                "source": edge_source,
                                "target": code,
                                "edge_type": "hierarchy",
                                "rule": None,
                            }
                        ))

        if ops:
            event_queue.put_nowait(AGUIEvent(
                type=AGUIEventType.STATE_DELTA,
                delta=ops,
            ))

        # Compute selected_details for reconciliation
        selected_details: dict[str, dict] = {}
        for code in selected_ids:
            code_data = data.get(code, {})

            if node_has_seven_chr_def(code):
                cat = "activator"
            elif "X" in code:
                cat = "placeholder"
            else:
                cat = "ancestor"

            if code in data:
                code_depth = code_data.get("depth", depth + 1)
            else:
                code_depth = depth + 1

            has_children = bool(code_data.get("children"))
            is_activator = cat == "activator"
            billable = not has_children and not (seven_chr_authority is not None) and not is_activator

            selected_details[code] = {
                "depth": code_depth,
                "category": cat,
                "billable": billable,
            }

        is_error = reasoning.startswith("API Error:") if reasoning else False
        event_queue.put_nowait(AGUIEvent(
            type=AGUIEventType.STEP_FINISHED,
            step_id=batch_id,
            metadata={
                "node_id": node_id,
                "batch_type": batch_type,
                "selected_ids": selected_ids,
                "reasoning": reasoning,
                "candidates": candidates,
                "error": is_error,
                "selected_details": selected_details,
            },
        ))

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate AG-UI SSE events for rewind."""
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
            partition_key = generate_traversal_cache_key(
                clinical_note=request.clinical_note,
                provider=request.provider,
                model=request.model or "",
                temperature=request.temperature or 0.0,
                system_prompt=request.system_prompt,
            )
            # Set global PARTITION_KEY so retry_node can use it for fork
            traversal_module.PARTITION_KEY = partition_key
            print(f"[REWIND] Partition key set: {partition_key}")

            # RUN_STARTED with rewind metadata
            yield f"data: {AGUIEvent(type=AGUIEventType.RUN_STARTED, metadata={'rewind_from': request.batch_id, 'feedback': request.feedback[:100]}).model_dump_json()}\n\n"

            # Initialize persister (reset if persist_cache=False)
            # NOTE: If persist_cache=False, the database is reset and there's nothing to load!
            print(f"[REWIND] Initializing persister with reset={not request.persist_cache}")
            if not request.persist_cache:
                print("[REWIND] WARNING: persist_cache=False will reset database - rewind may fail!")
            await initialize_persister(reset=not request.persist_cache)

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

            # Stream events while running
            while not task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    if event is not None:
                        yield f"data: {event.model_dump_json()}\n\n"
                except asyncio.TimeoutError:
                    continue

            # Get final state
            final_state = await task

            # Drain remaining events
            while not event_queue.empty():
                event = event_queue.get_nowait()
                if event is not None:
                    yield f"data: {event.model_dump_json()}\n\n"

            # Update finalized nodes
            final_nodes_raw = final_state.get("final_nodes", [])
            final_nodes = list(dict.fromkeys(final_nodes_raw))

            finalize_ops: list[JsonPatchOp] = []
            for node in final_nodes:
                if node in seen_nodes:
                    node_index = seen_nodes[node]
                    finalize_ops.append(JsonPatchOp(
                        op="replace",
                        path=f"/nodes/{node_index}/category",
                        value="finalized",
                    ))
                    finalize_ops.append(JsonPatchOp(
                        op="add",
                        path="/finalized/-",
                        value=node,
                    ))

            if finalize_ops:
                yield f"data: {AGUIEvent(type=AGUIEventType.STATE_DELTA, delta=finalize_ops).model_dump_json()}\n\n"

            # RUN_FINISHED
            batch_data = final_state.get("batch_data", {})
            run_finished_metadata = {
                'final_nodes': final_nodes,
                'batch_count': len(batch_data),
                'rewind_from': request.batch_id,
            }

            print(f"[REWIND] Sending RUN_FINISHED: final_nodes={final_nodes}")
            yield f"data: {AGUIEvent(type=AGUIEventType.RUN_FINISHED, metadata=run_finished_metadata).model_dump_json()}\n\n"
            print("[REWIND] Stream complete")

        except Exception as e:
            print(f"[REWIND] Error: {e}")
            error_event = AGUIEvent(
                type=AGUIEventType.RUN_FINISHED,
                metadata={"error": str(e)},
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

        finally:
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
