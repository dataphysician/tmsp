"""Burr actions for ICD-10-CM traversal

Core actions:
- load_node: Query ICD_INDEX, populate candidates dict
- select_candidates: Call selector, store selected_ids + reasoning
- finish_batch: Record termination, update final_nodes
- finish: Terminal state, display statistics
"""

from burr.core import State, action

from candidate_selector import SELECTOR_REGISTRY, llm_selector
from graph import data as ICD_INDEX


# ============================================================================
# Helper Functions
# ============================================================================


def format_with_seventh_char(base_code: str, seventh_char: str) -> str:
    """Format ICD-10-CM code with 7th character and placeholder padding.

    ICD-10-CM 7th character extension rules:
    - 7th character must be in the 7th position (excluding the dot)
    - If subcategory has fewer than 3 characters, pad with 'X' placeholders
    - Format: XXX.XXXX (3 chars + dot + up to 4 chars)

    Examples:
        format_with_seventh_char("S02.1", "A") -> "S02.1XXA"
        format_with_seventh_char("S02.11", "A") -> "S02.11XA"
        format_with_seventh_char("S02.111", "A") -> "S02.111A"
        format_with_seventh_char("S02", "A") -> "S02.XXXA"

    Args:
        base_code: Base ICD-10-CM code (e.g., "S02.1" or "S02.11")
        seventh_char: The 7th character to append (e.g., "A", "D", "S")

    Returns:
        Formatted ICD-10-CM code with 7th character
    """
    # Split on dot
    if "." in base_code:
        category, subcategory = base_code.split(".", 1)
    else:
        # No dot - treat first 3 chars as category, rest as subcategory
        category = base_code[:3] if len(base_code) >= 3 else base_code
        subcategory = base_code[3:] if len(base_code) > 3 else ""

    # Pad subcategory to 3 chars with X's, then add 7th char
    padded_subcategory = subcategory.ljust(3, "X") + seventh_char

    return f"{category}.{padded_subcategory}"


# ============================================================================
# Global Callback for Streaming
# ============================================================================

# Optional callback for streaming batch completions to UI
# Set this before running traversal to receive real-time updates
# Signature: (batch_id, node_id, parent_id, depth, candidates, selected_ids, reasoning)
BATCH_CALLBACK = None


def set_batch_callback(callback) -> None:
    """Set callback for batch completion events.

    The callback is invoked after each select_candidates completes.

    Args:
        callback: Function(batch_id, node_id, parent_id, depth, candidates, selected_ids, reasoning)
                  or None to disable
    """
    global BATCH_CALLBACK
    BATCH_CALLBACK = callback


# ============================================================================
# Actions
# ============================================================================


@action(reads=["batch_data"], writes=["batch_data"])
async def load_node(state: State) -> tuple[dict, State]:
    """SOURCE: Queries ICD-10-CM index to get candidates for this batch.

    Parses batch_id to extract node_id and batch_type (e.g., "B19.2|children"),
    queries ICD_INDEX[node_id] for the appropriate relationship type,
    and stores candidates dict in batch namespace.

    Batch types:
    - "children": Direct child nodes in ICD hierarchy
    - "codeFirst": Related codes that should be coded first
    - "codeAlso": Related codes that should also be coded
    - "useAdditionalCode": Additional codes to use with this code
    - "sevenChrDef": 7th character definitions (dict[char, description])

    For ROOT batch, uses initial node_id from batch_data.
    """
    batch_data = state.get("batch_data", {})
    current_batch_id = state.get("current_batch_id", "ROOT")

    if current_batch_id not in batch_data:
        batch_data[current_batch_id] = {}

    # Parse batch_id to extract node_id and batch_type
    if current_batch_id == "ROOT":
        # Special case: ROOT batch uses ROOT node
        node_id = "ROOT"
        batch_type = "children"
    elif "|" in current_batch_id:
        # Parse format: "B19.2|children" -> node_id="B19.2", batch_type="children"
        node_id, batch_type = current_batch_id.rsplit("|", 1)
    else:
        # Fallback for unexpected format
        node_id = batch_data[current_batch_id].get("node_id")
        batch_type = "children"

    # Query ICD-10-CM index based on batch_type
    candidates: dict[str, str] = {}

    if batch_type == "sevenChrDef":
        # Special handling for sevenChrDef: load from authority node
        authority = batch_data[current_batch_id].get("seven_chr_authority")
        authority_node_id = authority["batch_name"] if authority else None

        if authority_node_id and authority_node_id in ICD_INDEX:
            metadata = ICD_INDEX[authority_node_id].get("metadata", {})
            candidates = metadata.get("sevenChrDef", {})
            print(
                f"[{current_batch_id}] load_node: Loading sevenChrDef from "
                f"authority {authority_node_id} -> candidates={list(candidates.keys())}"
            )
        else:
            print(f"[{current_batch_id}] load_node: No sevenChrDef authority found")

    elif node_id in ICD_INDEX:
        node_data = ICD_INDEX[node_id]

        if batch_type == "children":
            candidates = node_data.get("children", {})
            # REVITALIZE: Log when X-ending node exists in index with real children
            if node_id.endswith("X") and candidates:
                print(
                    f"[{current_batch_id}] load_node: REVITALIZED {node_id} from index! "
                    f"Real children: {list(candidates.keys())[:5]}{'...' if len(candidates) > 5 else ''}"
                )
        elif batch_type in ["codeFirst", "codeAlso", "useAdditionalCode"]:
            candidates = node_data.get("metadata", {}).get(batch_type, {})
        else:
            candidates = {}

    else:
        # Node not in index - handle placeholder codes (e.g., T36.1X, T36.1XX)
        if batch_type == "children" and node_id.endswith("X"):
            candidates = {}
            print(
                f"[{current_batch_id}] load_node: Placeholder {node_id} -> empty candidates"
            )
        else:
            candidates = {}
            print(
                f"[{current_batch_id}] load_node: Node {node_id} not in index -> empty candidates"
            )

    # Log candidates for non-sevenChrDef batches (include X-ending nodes now)
    if batch_type != "sevenChrDef":
        print(
            f"[{current_batch_id}] load_node: node_id={node_id}, "
            f"batch_type={batch_type} -> candidates={list(candidates.keys())[:5]}"
            f"{'...' if len(candidates) > 5 else ''}"
        )

    batch_data[current_batch_id]["candidates"] = candidates
    batch_data[current_batch_id]["node_id"] = node_id

    return {}, state.update(batch_data=batch_data)


@action(reads=["batch_data"], writes=["pending_feedback", "current_batch_id"])
async def inject_feedback(
    state: State,
    batch_id: str,
    feedback: str,
) -> tuple[dict, State]:
    """Explicitly inject feedback before select_candidates.

    Makes feedback injection VISIBLE in the workflow trace, rather than
    hiding it in generic state manipulation. This is the entrypoint for
    rewind operations.

    Args:
        state: Current application state
        batch_id: The batch to inject feedback for
        feedback: The feedback string to inject

    Returns:
        (result_dict, new_state) tuple with pending_feedback set
    """
    print(f"\n{'=' * 60}")
    print("[INJECT_FEEDBACK] Action called!")
    print(f"  batch_id param: {batch_id}")
    print(f"  feedback param: {feedback[:100] if feedback else 'None'}{'...' if feedback and len(feedback) > 100 else ''}")

    batch_data = state.get("batch_data", {})
    print(f"  batch_data keys: {list(batch_data.keys())[:10]}...")

    if batch_id not in batch_data:
        print(f"  ERROR: batch_id '{batch_id}' not found!")
        raise KeyError(f"batch_id '{batch_id}' not found in batch_data. Available: {list(batch_data.keys())}")

    print(f"  Setting pending_feedback and current_batch_id in state")
    print(f"{'=' * 60}\n")

    return {}, state.update(
        pending_feedback=feedback,
        current_batch_id=batch_id,
    )


@action(reads=["batch_data", "context", "pending_feedback"], writes=["batch_data", "pending_feedback"])
async def select_candidates(
    state: State,
    selector: str = "llm",
) -> tuple[dict, State]:
    """SELECTOR: Picks subset from candidates using registry-based selector.

    Pure selection logic - delegates to selector functions conforming to SelectorProtocol.

    FEEDBACK SOURCE:
    - Reads from state["pending_feedback"] (set by inject_feedback action)
    - After use, pending_feedback is cleared (one-time consumption)

    SELECTION PHASE:
    - If no feedback found, selector runs normally (may use cache)
    - Uses SELECTOR_REGISTRY[selector] (default: "llm")
    - Reads context from state (guides all selectors)
    - Awaits async selector_fn(batch_id, context, candidates, feedback)

    Args:
        state: Current application state
        selector: String key for SELECTOR_REGISTRY (default: "llm")

    Returns:
        (result_dict, new_state) tuple with selected_ids
    """
    batch_data = state.get("batch_data", {})
    current_batch_id = state.get("current_batch_id", "ROOT")
    context = state.get("context", "")

    # Debug: Show state at entry
    pending_fb = state.get("pending_feedback")
    print(f"\n[SELECT_CANDIDATES] Entry - batch_id={current_batch_id}, pending_feedback={'set' if pending_fb else 'None'}")

    # Handle corrupted checkpoints
    if current_batch_id not in batch_data:
        if "ROOT" in batch_data:
            print(
                f"[WARNING] current_batch_id={current_batch_id} not in batch_data, "
                "using ROOT as fallback"
            )
            batch_data[current_batch_id] = batch_data["ROOT"].copy()
        else:
            raise KeyError(
                f"current_batch_id '{current_batch_id}' not found in batch_data. "
                f"Available: {list(batch_data.keys())}"
            )

    # SELECTOR RESOLUTION: Batch-level override > default_selector
    batch_selector = batch_data[current_batch_id].get("selector")
    default_selector = state.get("default_selector", "llm")
    active_selector = batch_selector if batch_selector is not None else default_selector

    print(
        f"[SELECTOR] batch={current_batch_id}, "
        f"batch_selector={batch_selector}, default={default_selector}, "
        f"active={active_selector}"
    )

    # EXPLICIT FEEDBACK: Read from dedicated pending_feedback state field
    # This is set by inject_feedback action during rewind operations
    candidates = batch_data[current_batch_id]["candidates"]
    feedback = state.get("pending_feedback")
    print(f"[SELECT_CANDIDATES] feedback from state: {repr(feedback)[:100] if feedback else 'None'}")

    # CACHE-FIRST: If no feedback and previous selection exists, reuse it
    if feedback is None:
        print(f"[SELECT_CANDIDATES] No feedback, checking cache...")
        previous_selection = batch_data[current_batch_id].get("selected_ids")
        if previous_selection is not None:
            print(f"[CACHE HIT] {current_batch_id} -> Reusing previous selection: {previous_selection}")

            # Still call callback to ensure nodes/edges are created for UI
            if BATCH_CALLBACK is not None:
                try:
                    node_id = batch_data[current_batch_id].get("node_id")
                    parent_id = batch_data[current_batch_id].get("parent_id")
                    depth = batch_data[current_batch_id].get("depth", 0)
                    reasoning = batch_data[current_batch_id].get("reasoning", "")
                    seven_chr_authority = batch_data[current_batch_id].get("seven_chr_authority")

                    # Prepare callback data - transform 7th char to full codes if needed
                    cb_candidates = candidates
                    cb_selected_ids = previous_selection

                    batch_type = current_batch_id.rsplit("|", 1)[1] if "|" in current_batch_id else "children"
                    if batch_type == "sevenChrDef" and node_id:
                        # Transform candidates with "X: description" format for display
                        cb_candidates = {}
                        for char, desc in candidates.items():
                            full_code = format_with_seventh_char(node_id, char)
                            cb_candidates[full_code] = f"{char}: {desc}"

                        # Transform selected_ids
                        if previous_selection:
                            cb_selected_ids = [format_with_seventh_char(node_id, char) for char in previous_selection]

                    BATCH_CALLBACK(
                        current_batch_id,
                        node_id,
                        parent_id,
                        depth,
                        cb_candidates,
                        cb_selected_ids,
                        reasoning,
                        seven_chr_authority,
                    )
                except Exception as e:
                    print(f"[CACHE CALLBACK ERROR] {current_batch_id}: {e}")

            return {"selected_ids": previous_selection}, state.update(
                batch_data=batch_data, pending_feedback=None  # Clear after check
            )

    print(f"[SELECT_CANDIDATES] batch_id={current_batch_id}, feedback='{feedback}'")

    # Registry lookup
    selector_fn = SELECTOR_REGISTRY.get(active_selector, llm_selector)

    print(
        f"[LLM SELECTION] {current_batch_id} -> Calling {active_selector} selector "
        f"with {len(candidates)} candidates"
    )
    if feedback:
        print(f"[LLM SELECTION] Using feedback: {feedback[:100]}...")

    # Execute selection
    result = await selector_fn(current_batch_id, context, candidates, feedback)

    # Unpack result - handle both tuple and list returns
    if isinstance(result, tuple):
        selected_ids, reasoning = result
    else:
        selected_ids = result
        reasoning = None

    print(
        f"[{current_batch_id}] select_candidates: "
        f"selector={active_selector}, "
        f"candidates={list(candidates.keys())[:5]} -> selected={selected_ids}"
    )

    batch_data[current_batch_id]["selected_ids"] = selected_ids

    # Store reasoning if present
    if reasoning:
        batch_data[current_batch_id]["reasoning"] = reasoning

    # CRITICAL: If selection is empty and no seven_chr_authority, check ancestry
    # and set authority if sevenChrDef exists. This ensures the transition to
    # spawn_seven_chr is taken instead of finish_batch for depth-6 nodes.
    batch_type = current_batch_id.rsplit("|", 1)[1] if "|" in current_batch_id else "children"
    if (not selected_ids and
        batch_type == "children" and
        batch_data[current_batch_id].get("seven_chr_authority") is None):
        node_id = batch_data[current_batch_id].get("node_id")
        if node_id:
            from graph import get_seventh_char_def
            seven_chr_result = get_seventh_char_def(node_id, ICD_INDEX)
            if seven_chr_result is not None:
                _, ancestor_with_def = seven_chr_result
                batch_data[current_batch_id]["seven_chr_authority"] = {
                    "batch_name": ancestor_with_def,
                    "resolution_pattern": "sevenChrDef"
                }
                print(f"[{current_batch_id}] SELF-ACTIVATE: {node_id} inherits sevenChrDef "
                      f"from ancestor {ancestor_with_def} - setting authority")

    # Call callback if registered (for live UI updates)
    if BATCH_CALLBACK is not None:
        try:
            node_id = batch_data[current_batch_id].get("node_id")
            parent_id = batch_data[current_batch_id].get("parent_id")
            depth = batch_data[current_batch_id].get("depth", 0)
            seven_chr_authority = batch_data[current_batch_id].get("seven_chr_authority")

            # Prepare callback data - transform 7th char to full codes if needed
            cb_candidates = candidates
            cb_selected_ids = selected_ids

            batch_type = current_batch_id.rsplit("|", 1)[1] if "|" in current_batch_id else "children"
            if batch_type == "sevenChrDef" and node_id:
                # Transform candidates with "X: description" format for display
                cb_candidates = {}
                for char, desc in candidates.items():
                    full_code = format_with_seventh_char(node_id, char)
                    cb_candidates[full_code] = f"{char}: {desc}"

                # Transform selected_ids
                if selected_ids:
                    cb_selected_ids = [format_with_seventh_char(node_id, char) for char in selected_ids]

            BATCH_CALLBACK(
                current_batch_id,
                node_id,
                parent_id,
                depth,
                cb_candidates,
                cb_selected_ids,
                reasoning or "",
                seven_chr_authority,
            )
        except Exception as e:
            print(f"[CALLBACK ERROR] {current_batch_id}: {e}")
            import traceback
            traceback.print_exc()

    return {"selected_ids": selected_ids}, state.update(
        batch_data=batch_data, pending_feedback=None  # Consumed - prevents re-use
    )


@action(reads=["batch_data", "final_nodes"], writes=["batch_data", "final_nodes"])
async def finish_batch(state: State) -> tuple[dict, State]:
    """Terminal node for each batch - records termination reason.

    Tracks why each batch finished:
    - Empty selection: Adds node_id to final_nodes list (leaf nodes)
    - Spawned children: Status set to selected_candidates
    - sevenChrDef batch: Formats code with 7th character and records

    Batch halts here, final state passed to reduce() for aggregation.
    """
    batch_data = state.get("batch_data", {})
    final_nodes = state.get("final_nodes", [])
    current_batch_id = state.get("current_batch_id", "ROOT")

    node_id = batch_data[current_batch_id].get("node_id")
    selected_ids = batch_data[current_batch_id].get("selected_ids", [])

    # Determine batch type
    batch_type = current_batch_id.rsplit("|", 1)[1] if "|" in current_batch_id else "children"

    # Special handling for sevenChrDef batches
    if batch_type == "sevenChrDef":
        if selected_ids:
            seventh_char = selected_ids[0]
            final_code = format_with_seventh_char(node_id, seventh_char)
            print(f"[{current_batch_id}] Extended {node_id} -> {final_code} (7th char: {seventh_char})")
        else:
            final_code = node_id
            print(f"[{current_batch_id}] No 7th char selected, using base code: {final_code}")

        # ALWAYS add the 7th char code to final_nodes
        # When a sevenChrDef is selected, that IS the final code - depth 7 codes are always terminal
        # regardless of whether the base node has children in the ICD index
        if final_code not in final_nodes:
            final_nodes.append(final_code)
            print(f"[{current_batch_id}] SEVENCHRDEF FINAL: Added {final_code} to final_nodes")
        else:
            print(f"[{current_batch_id}] SEVENCHRDEF FINAL: {final_code} already in final_nodes")

        batch_data[current_batch_id]["status"] = "completed_seven_chr"

    else:
        # Children or other batch types
        is_empty_selection = len(selected_ids) == 0

        if is_empty_selection:
            # When selection is empty, we've reached a terminal point.
            # Only report for children batches WITHOUT sevenChrDef authority.
            # When authority exists, spawn_seven_chr handles the final code (e.g., T36.1X5A).
            # For other batch types: don't add parent node (lateral batches don't define termination)
            has_seven_chr_authority = batch_data[current_batch_id].get("seven_chr_authority") is not None

            # SAFETY CHECK: Even if authority wasn't propagated, check if this node
            # has sevenChrDef in its ancestry. If so, it should NOT be finalized -
            # the 7th character is mandatory and spawn_seven_chr should handle it.
            # This prevents incorrect finalization at depth-6 nodes.
            # Also SET the authority so spawn_seven_chr transition will be taken.
            has_seven_chr_in_ancestry = False
            if not has_seven_chr_authority and node_id and batch_type == "children":
                from graph import get_seventh_char_def
                seven_chr_result = get_seventh_char_def(node_id, ICD_INDEX)
                if seven_chr_result is not None:
                    has_seven_chr_in_ancestry = True
                    _, ancestor_with_def = seven_chr_result
                    # Set the authority so spawn_seven_chr can use it
                    batch_data[current_batch_id]["seven_chr_authority"] = {
                        "batch_name": ancestor_with_def,
                        "resolution_pattern": "sevenChrDef"
                    }
                    print(f"[{current_batch_id}] SAFETY CHECK: {node_id} has sevenChrDef "
                          f"from ancestor {ancestor_with_def} - setting authority, NOT finalizing")

            should_report = batch_type == "children" and not has_seven_chr_authority and not has_seven_chr_in_ancestry

            if should_report and node_id is not None and node_id not in final_nodes:
                final_nodes.append(node_id)

            batch_data[current_batch_id]["status"] = "terminated_by_selector"

            if should_report:
                print(f"[{current_batch_id}] EMPTY SELECTION: Natural termination (leaf node)")
            elif has_seven_chr_authority:
                print(f"[{current_batch_id}] EMPTY SELECTION: sevenChrDef handled termination")
            elif has_seven_chr_in_ancestry:
                print(f"[{current_batch_id}] EMPTY SELECTION: sevenChrDef in ancestry - awaiting 7th char")
            else:
                print(f"[{current_batch_id}] EMPTY SELECTION: non-children batch, not reporting")

        else:
            batch_data[current_batch_id]["status"] = "selected_candidates"
            print(f"[{current_batch_id}] SELECTED CANDIDATES: Batch complete")

    print(
        f"[{current_batch_id}] finish_batch: node_id={node_id}, "
        f"selected={selected_ids}, status={batch_data[current_batch_id]['status']}"
    )

    return {}, state.update(batch_data=batch_data, final_nodes=final_nodes)


@action(reads=["batch_data", "final_nodes"], writes=[])
async def finish(state: State) -> tuple[dict, State]:
    """Terminal state - application ends here after all batches processed.

    Displays cumulative statistics from ICD-10-CM traversal.
    """
    print(f"\n{'=' * 60}")
    print("FINISHED!")

    batch_data = state.get("batch_data", {})
    final_nodes = state.get("final_nodes", [])

    print("\nSTATE SUMMARY:")
    print(f"Total batches executed: {len(batch_data)}")
    print(f"Final leaf nodes: {len(final_nodes)}")

    # Count by status
    spawned = [
        bid for bid, data in batch_data.items() if data.get("status") == "selected_candidates"
    ]
    terminated = [
        bid for bid, data in batch_data.items() if data.get("status") == "terminated_by_selector"
    ]
    print(f"Spawned children: {len(spawned)}")
    print(f"Terminated by selector: {len(terminated)}")

    print("\nBATCH DETAILS:")
    for batch_id, data in batch_data.items():
        print(
            f"  {batch_id}: node_id={data.get('node_id')}, "
            f"selected={data.get('selected_ids', [])}, "
            f"status={data.get('status', '_processing_')}"
        )

    if final_nodes:
        print("\nFINAL LEAF NODES (ICD codes):")
        for node_id in final_nodes:
            print(f"  {node_id}")

    print(f"{'=' * 60}\n")
    return {}, state
