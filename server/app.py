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

# Scaffolded traversal cache key generator
from agent.traversal import generate_traversal_cache_key

# Helper to check if a node has sevenChrDef metadata
def node_has_seven_chr_def(code: str) -> bool:
    """Check if a node has sevenChrDef metadata."""
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

    # Validate all codes exist in the ICD-10-CM index
    invalid_codes: list[str] = []
    for code in codes:
        # Check if code exists directly or can be resolved (7th char codes)
        if code not in data and resolve_code(code, data) is None:
            invalid_codes.append(code)

    if invalid_codes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ICD-10-CM codes: {', '.join(invalid_codes)}"
        )

    # Build the graph using trace_tree
    result = build_graph(codes, data)

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

    # Build set of lateral link edges
    lateral_links = result.get("lateral_links", [])
    lateral_edge_set = {(src, tgt) for src, tgt, _ in lateral_links}

    # Add tree edges (parent-child relationships)
    seventh_char = result.get("seventh_char", {})
    tree = result["tree"]

    for parent in tree:
        for child in tree[parent]:
            # Skip if this is a lateral link
            if (parent, child) in lateral_edge_set:
                continue

            # Check if this is a sevenChrDef edge
            if child in seventh_char:
                edges.append(
                    GraphEdge(
                        source=parent,
                        target=child,
                        edge_type=EdgeType.LATERAL,
                        rule="sevenChrDef",
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

    # Add lateral link edges
    for source, target, key in lateral_links:
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
        # - selected_ids already contains full codes (e.g., ["T36.1X5A"]) - transformed by actions.py
        # - For codes at depth 3-5, create placeholder nodes (with X padding) until depth 6
        # - The 7th char node is added via lateral edge from the depth-6 node
        if batch_type == "sevenChrDef" and selected_ids and node_id:
            # selected_ids[0] is already the full code (e.g., "T36.1X5A"), not just the character
            full_code = selected_ids[0]
            # Extract the 7th character for label lookup in candidates
            seventh_char = full_code[-1] if full_code else ""

            # Get base code depth to determine if placeholder nodes are needed
            base_depth = data.get(node_id, {}).get("depth", depth)

            # Create placeholder nodes if base code is shorter than depth 6
            # ICD-10-CM format: XXX.XXXX where 7th char is position 4 after dot
            # Depth 3 = category (T36), depth 4 = 4-char (T36.1), depth 5 = 5-char (T36.1X)
            # depth 6 = 6-char (T36.1X5), depth 7 = 7-char (T36.1X5A)
            prev_node = node_id
            if base_depth < 6:
                # Parse base code to determine how many X's to add
                if "." in node_id:
                    category, subcategory = node_id.split(".", 1)
                else:
                    category = node_id[:3] if len(node_id) >= 3 else node_id
                    subcategory = node_id[3:] if len(node_id) > 3 else ""

                # Create placeholder nodes from current length to depth 6
                current_sub = subcategory
                for i in range(len(subcategory), 3):  # Pad to 3 chars (positions 4-6)
                    current_sub += "X"
                    placeholder_code = f"{category}.{current_sub}"
                    placeholder_depth = base_depth + (i - len(subcategory) + 1)

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
                "source": prev_node,
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

                edge_key = (prev_node, full_code)
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
            print(f"[SERVER] Clinical note length: {len(request.clinical_note)} chars")
            print(f"[SERVER] Scaffolded mode: {request.scaffolded}")
            if request.system_prompt:
                print(f"[SERVER] Custom system prompt: {len(request.system_prompt)} chars")

            # RUN_STARTED
            yield f"data: {AGUIEvent(type=AGUIEventType.RUN_STARTED, metadata={'clinical_note': request.clinical_note[:100]}).model_dump_json()}\n\n"

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

                # Initialize zero-shot persister if needed
                if ZERO_SHOT_PERSISTER is None:
                    await initialize_zero_shot_persister()

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
            # Generate partition key for cross-run caching
            partition_key = generate_traversal_cache_key(
                clinical_note=request.clinical_note,
                provider=request.provider,
                model=request.model or "",
                temperature=request.temperature or 0.0,
                system_prompt=request.system_prompt,
            )
            print(f"[SERVER] Scaffolded cache key: {partition_key}")

            # Initialize persister (don't reset - preserve cache across runs)
            await initialize_persister(reset=False)

            # Set callback for streaming
            set_batch_callback(on_batch_complete)
            print("[SERVER] Batch callback registered")

            # Build app (checks cache and returns early if hit)
            burr_app, was_cached = await build_app(
                context=request.clinical_note,
                default_selector=request.selector,
                with_persistence=True,
                partition_key=partition_key,
            )

            if was_cached:
                # Cache hit! Replay cached batches to reconstruct full graph
                print("[SERVER] Scaffolded CACHE HIT - replaying cached batches")
                final_state = burr_app.state
                final_nodes = final_state.get("final_nodes", [])
                batch_data = final_state.get("batch_data", {})

                # Track seen nodes/edges for deduplication (same as live traversal)
                replay_seen_nodes: dict[str, int] = {"ROOT": 0}
                replay_seen_edges: set[tuple[str, str]] = set()
                replay_node_count = 1

                # Sort batches by depth to replay in correct order (parents before children)
                # Include ROOT batch - it must be replayed first for tree structure
                sorted_batches = sorted(
                    [(bid, binfo) for bid, binfo in batch_data.items()],
                    key=lambda x: (x[1].get("depth", 0), x[0])
                )

                print(f"[CACHE REPLAY] Replaying {len(sorted_batches)} batches")

                # Replay each batch with STEP_STARTED, STATE_DELTA, STEP_FINISHED
                for batch_id, batch_info in sorted_batches:
                    node_id = batch_info.get("node_id")
                    parent_id = batch_info.get("parent_id")
                    depth = batch_info.get("depth", 0)
                    candidates = batch_info.get("candidates", {})
                    selected_ids = batch_info.get("selected_ids", [])
                    reasoning = batch_info.get("reasoning", "[CACHED]")
                    seven_chr_authority = batch_info.get("seven_chr_authority")

                    # Parse batch_type from batch_id
                    batch_type = batch_id.rsplit("|", 1)[1] if "|" in batch_id else "children"

                    # Emit STEP_STARTED
                    yield f"data: {AGUIEvent(type=AGUIEventType.STEP_STARTED, step_id=batch_id).model_dump_json()}\n\n"

                    # Build STATE_DELTA ops (mirrors on_batch_complete logic)
                    ops: list[JsonPatchOp] = []

                    # Handle sevenChrDef batches
                    if batch_type == "sevenChrDef" and node_id and node_id not in replay_seen_nodes:
                        replay_seen_nodes[node_id] = replay_node_count
                        replay_node_count += 1

                        if node_id in data:
                            node_label = data[node_id].get("label", node_id)
                            node_depth = data[node_id].get("depth", depth)
                            node_category = "activator" if node_has_seven_chr_def(node_id) else "ancestor"
                        else:
                            node_label = "Placeholder"
                            node_depth = depth
                            node_category = "placeholder"

                        ops.append(JsonPatchOp(op="add", path="/nodes/-", value={
                            "id": node_id, "code": node_id, "label": node_label,
                            "depth": node_depth, "category": node_category, "billable": False,
                        }))

                        if parent_id and parent_id != node_id:
                            edge_key = (parent_id, node_id)
                            if edge_key not in replay_seen_edges:
                                replay_seen_edges.add(edge_key)
                                ops.append(JsonPatchOp(op="add", path="/edges/-", value={
                                    "source": parent_id, "target": node_id,
                                    "edge_type": "hierarchy", "rule": None,
                                }))

                    # Handle sevenChrDef final node creation
                    if batch_type == "sevenChrDef" and selected_ids and node_id:
                        full_code = selected_ids[0]
                        seventh_char = full_code[-1] if full_code else ""

                        base_depth = data.get(node_id, {}).get("depth", depth)
                        prev_node = node_id

                        # Create placeholder nodes if needed
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

                                if placeholder_code not in replay_seen_nodes:
                                    replay_seen_nodes[placeholder_code] = replay_node_count
                                    replay_node_count += 1
                                    ops.append(JsonPatchOp(op="add", path="/nodes/-", value={
                                        "id": placeholder_code, "code": placeholder_code,
                                        "label": "Placeholder", "depth": placeholder_depth,
                                        "category": "placeholder", "billable": False,
                                    }))

                                    edge_key = (prev_node, placeholder_code)
                                    if edge_key not in replay_seen_edges:
                                        replay_seen_edges.add(edge_key)
                                        ops.append(JsonPatchOp(op="add", path="/edges/-", value={
                                            "source": prev_node, "target": placeholder_code,
                                            "edge_type": "hierarchy", "rule": None,
                                        }))

                                prev_node = placeholder_code

                        # Create final 7th character node
                        label = f"{seventh_char}: {candidates.get(full_code, seventh_char)}"
                        if full_code not in replay_seen_nodes:
                            replay_seen_nodes[full_code] = replay_node_count
                            replay_node_count += 1
                            ops.append(JsonPatchOp(op="add", path="/nodes/-", value={
                                "id": full_code, "code": full_code, "label": label,
                                "depth": 7, "category": "finalized", "billable": True,
                            }))

                            edge_key = (prev_node, full_code)
                            if edge_key not in replay_seen_edges:
                                replay_seen_edges.add(edge_key)
                                ops.append(JsonPatchOp(op="add", path="/edges/-", value={
                                    "source": prev_node, "target": full_code,
                                    "edge_type": "lateral", "rule": "sevenChrDef",
                                }))

                    # Regular batches (children, codeFirst, codeAlso, etc.)
                    elif batch_type != "sevenChrDef":
                        # Add the traversed node itself
                        if node_id and node_id != "ROOT" and node_id not in replay_seen_nodes:
                            replay_seen_nodes[node_id] = replay_node_count
                            replay_node_count += 1

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

                            ops.append(JsonPatchOp(op="add", path="/nodes/-", value={
                                "id": node_id, "code": node_id, "label": node_label,
                                "depth": node_depth, "category": node_category, "billable": False,
                            }))

                            if parent_id and parent_id != node_id:
                                edge_key = (parent_id, node_id)
                                if edge_key not in replay_seen_edges:
                                    replay_seen_edges.add(edge_key)
                                    ops.append(JsonPatchOp(op="add", path="/edges/-", value={
                                        "source": parent_id, "target": node_id,
                                        "edge_type": "hierarchy", "rule": None,
                                    }))

                        # Add selected children
                        for code in selected_ids:
                            if code not in replay_seen_nodes:
                                replay_seen_nodes[code] = replay_node_count
                                replay_node_count += 1
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

                                ops.append(JsonPatchOp(op="add", path="/nodes/-", value={
                                    "id": code, "code": code, "label": label,
                                    "depth": code_depth, "category": code_category, "billable": billable,
                                }))

                            # Add edge
                            edge_source = node_id if node_id and node_id != "ROOT" else "ROOT"
                            edge_key = (edge_source, code)

                            if batch_type != "children":
                                edge_type = "lateral"
                                rule = batch_type
                            else:
                                edge_type = "hierarchy"
                                rule = None

                            if edge_key not in replay_seen_edges:
                                replay_seen_edges.add(edge_key)
                                ops.append(JsonPatchOp(op="add", path="/edges/-", value={
                                    "source": edge_source, "target": code,
                                    "edge_type": edge_type, "rule": rule,
                                }))

                    # Emit STATE_DELTA if we have ops
                    if ops:
                        yield f"data: {AGUIEvent(type=AGUIEventType.STATE_DELTA, delta=ops).model_dump_json()}\n\n"

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

                    # Emit STEP_FINISHED
                    yield f"data: {AGUIEvent(type=AGUIEventType.STEP_FINISHED, step_id=batch_id, metadata={'node_id': node_id, 'batch_type': batch_type, 'selected_ids': selected_ids, 'reasoning': reasoning, 'candidates': candidates, 'error': False, 'cached': True, 'selected_details': selected_details}).model_dump_json()}\n\n"

                # Skip the normal traversal flow - go straight to RUN_FINISHED
                run_finished_metadata = {
                    'final_nodes': final_nodes,
                    'batch_count': len(batch_data),
                    'mode': 'scaffolded',
                    'cached': True,
                }
                print(f"[SERVER] Scaffolded complete (CACHED): final_nodes={final_nodes}")
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
                'cached': was_cached,
            }
            if llm_error:
                run_finished_metadata['error'] = llm_error

            cache_status = "CACHED" if was_cached else "NEW"
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
            full_code = selected_ids[0]
            seventh_char = full_code[-1] if full_code else ""
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
                "source": prev_node,
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

                edge_key = (prev_node, full_code)
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

                if batch_type != "children":
                    edge_type = "lateral"
                    rule = batch_type
                else:
                    edge_type = "hierarchy"
                    rule = None

                if edge_key not in seen_edges:
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

            # RUN_STARTED with rewind metadata
            yield f"data: {AGUIEvent(type=AGUIEventType.RUN_STARTED, metadata={'rewind_from': request.batch_id, 'feedback': request.feedback[:100]}).model_dump_json()}\n\n"

            # Initialize persister (reuse existing DB, don't reset)
            await initialize_persister(reset=False)

            # Set callback for streaming
            set_batch_callback(on_batch_complete)
            print("[REWIND] Batch callback registered")

            # Build feedback_map with the feedback for this batch
            feedback_map = {request.batch_id: request.feedback}

            # Run retry_node in background task so we can stream events
            async def run_retry():
                return await retry_node(
                    batch_id=request.batch_id,
                    selector=request.selector,
                    feedback_map=feedback_map,
                )

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
