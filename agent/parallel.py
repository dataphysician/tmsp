"""MapStates classes for parallel ICD-10-CM traversal

SpawnParallelBatches: Multi-batch fan-out for children + metadata relationships
SpawnSevenChr: Sequential 7th character selection at leaf nodes
"""

from typing import Any, Generator

from burr.core import ApplicationContext, GraphBuilder, State, expr
from burr.core.parallelism import MapStates, RunnableGraph, SubGraphTask, _cascade_adapter

from graph import get_seventh_char_def
from .actions import ICD_INDEX, finish_batch, load_node, select_candidates


# ============================================================================
# SpawnParallelBatches - Multi-Batch Fan-Out
# ============================================================================


class SpawnParallelBatches(MapStates):
    """Recursive parallel DFS batch processor.

    Key features:
    - Yields ALL children at once (parallel processing)
    - Each child runs full subgraph including recursive spawn_children
    - No explicit batch_queue (recursion handles traversal)
    - Dynamic current_batch_id (not hardcoded)

    State structure:
    - batch_data: All batch-specific data keyed by batch_id
    - final_nodes: List of terminal ICD codes

    Execution model:
    - Parallel: All siblings at each level run concurrently
    - Recursive: Each child can spawn more children via spawn_children action
    - DFS: Each branch explores to full depth before parent completes
    """

    def states(
        self, state: State, context: ApplicationContext, inputs: dict
    ) -> Generator[State, None, None]:
        """Yields ALL batches for parallel processing (multi-batch fan-out).

        For each selected node_id, creates up to 4 parallel batches:
        - {node_id}|children: Child nodes in ICD hierarchy
        - {node_id}|codeFirst: Related codes to code first (if exists)
        - {node_id}|codeAlso: Related codes to also code (if exists)
        - {node_id}|useAdditionalCode: Additional codes to use (if exists)
        """
        batch_data = state.get("batch_data", {})

        # Dynamic current_batch_id (critical for recursion!)
        current_batch_id = state.get("current_batch_id", "ROOT")
        selected_ids = batch_data[current_batch_id]["selected_ids"]

        if not selected_ids:
            print(f"\n{'=' * 60}")
            print(f"[{current_batch_id}] No children to spawn (selected_ids is empty)")
            print(f"{'=' * 60}\n")
            return  # Empty generator

        # Count total batches to spawn
        total_batches = 0
        batch_list: list[tuple[str, str]] = []

        # For each selected node, check which batch types exist
        for node_id in selected_ids:
            if node_id not in ICD_INDEX:
                continue

            node_data = ICD_INDEX[node_id]

            # Always spawn children batch
            batch_list.append((node_id, "children"))
            total_batches += 1

            # Check for metadata batches (exclude sevenChrDef - handled separately)
            metadata = node_data.get("metadata", {})
            for batch_type in ["codeFirst", "codeAlso", "useAdditionalCode"]:
                if metadata.get(batch_type):
                    batch_list.append((node_id, batch_type))
                    total_batches += 1

        print(f"\n{'=' * 60}")
        print(f"[{current_batch_id}] SPAWNING {total_batches} PARALLEL BATCHES")
        print(f"selected_ids: {selected_ids}")
        print(f"batch types: {batch_list}")
        print(f"{'=' * 60}\n")

        # Get current batch's sevenChrDef authority (if any)
        current_authority = batch_data[current_batch_id].get("seven_chr_authority")

        # Calculate parent depth for consistent child hierarchy
        if current_batch_id == "ROOT":
            parent_depth = 0
        elif "|" in current_batch_id:
            parent_node_id, _ = current_batch_id.rsplit("|", 1)
            parent_depth = ICD_INDEX.get(parent_node_id, {}).get("depth", 0)
        else:
            parent_depth = batch_data[current_batch_id].get("depth", 0)

        child_depth = parent_depth + 1

        # Yield ALL batches for parallel processing
        for node_id, batch_type in batch_list:
            child_batch_id = f"{node_id}|{batch_type}"

            # CYCLE DETECTION: Skip if batch already exists
            if child_batch_id in batch_data:
                print(
                    f"[{current_batch_id}] -> Skipping duplicate batch: "
                    f"{child_batch_id} (cycle detected)"
                )
                total_batches -= 1
                continue

            # Determine sevenChrDef authority for this child (paused task handoff)
            # For children batches: always determine authority fresh based on node's own hierarchy
            # For lateral batches: inherit from parent (they don't need sevenChrDef processing)
            if batch_type == "children":
                # Start fresh for children batches - don't inherit from parent
                # This fixes nodes reached via lateral jumps from different hierarchies
                child_authority = None

                # First check if the node itself has sevenChrDef
                if node_id in ICD_INDEX:
                    metadata = ICD_INDEX[node_id].get("metadata", {})
                    if metadata.get("sevenChrDef"):
                        child_authority = {
                            "batch_name": node_id,
                            "resolution_pattern": "sevenChrDef"
                        }
                        print(f"[7CHR PAUSED] Task created at {node_id} "
                              f"(will resume at descendant|sevenChrDef when depth 6→7)")

                # If no direct sevenChrDef, check ancestry (self-activation)
                # This handles nodes like T81.44 whose ancestor T81 has sevenChrDef
                if child_authority is None:
                    seven_chr_result = get_seventh_char_def(node_id, ICD_INDEX)
                    if seven_chr_result is not None:
                        _, ancestor_with_def = seven_chr_result
                        child_authority = {
                            "batch_name": ancestor_with_def,
                            "resolution_pattern": "sevenChrDef"
                        }
                        print(f"[7CHR SELF-ACTIVATE] {node_id} inherits paused task from ancestor {ancestor_with_def}")
            else:
                # Non-children batches (codeFirst, codeAlso, etc.) inherit from parent
                child_authority = current_authority

            # Determine parent_id for edge creation
            if current_batch_id == "ROOT":
                parent_node_id = "ROOT"
            elif "|" in current_batch_id:
                parent_node_id, _ = current_batch_id.rsplit("|", 1)
            else:
                parent_node_id = batch_data[current_batch_id].get("node_id", "ROOT")

            # Initialize batch_data for this batch
            child_batch_data = batch_data.copy()
            child_batch_data[child_batch_id] = {
                "node_id": node_id,
                "parent_id": parent_node_id,
                "seven_chr_authority": child_authority,
                "depth": child_depth,
            }

            # Log authority handoff
            if child_authority:
                authority_name = child_authority["batch_name"] if isinstance(child_authority, dict) else child_authority
                print(f"[7CHR HANDOFF] {child_batch_id} inherits paused task from {authority_name}")
            print(
                f"[{current_batch_id}] -> Spawning batch: {child_batch_id} "
                f"(node_id={node_id}, type={batch_type})"
            )

            # Yield batch state with FRESH final_nodes list
            yield state.update(
                current_batch_id=child_batch_id,
                batch_data=child_batch_data,
                final_nodes=[],
            )

    def action(self, state: State, inputs: dict) -> RunnableGraph:
        """Defines the recursive subgraph for each batch.

        Flow: load_node -> select_candidates -> (decision)
              - If empty selection -> finish_batch
              - Otherwise -> spawn_children (recursive!)
        """
        graph = (
            GraphBuilder()
            .with_actions(
                load_node=load_node,
                select_candidates=select_candidates,
                finish_batch=finish_batch,
                spawn_children=SpawnParallelBatches(),  # Recursive!
                spawn_seven_chr=SpawnSevenChr(),
            )
            .with_transitions(
                ("load_node", "select_candidates"),
                # Empty selection WITH sevenChrDef authority -> spawn sevenChrDef batch
                (
                    "select_candidates",
                    "spawn_seven_chr",
                    expr(
                        "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                        "batch_data[current_batch_id].get('seven_chr_authority') is not None and "
                        "'|children' in current_batch_id"
                    ),
                ),
                # Empty selection WITHOUT authority -> finish directly
                (
                    "select_candidates",
                    "finish_batch",
                    expr(
                        "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                        "(batch_data[current_batch_id].get('seven_chr_authority') is None or "
                        "'|children' not in current_batch_id)"
                    ),
                ),
                # Has selected children -> spawn children batches
                (
                    "select_candidates",
                    "spawn_children",
                    expr("len(batch_data[current_batch_id].get('selected_ids', [])) > 0"),
                ),
                # After spawning sevenChrDef, finish this batch
                ("spawn_seven_chr", "finish_batch"),
                # After spawning children, finish this batch
                ("spawn_children", "finish_batch"),
            )
            .build()
        )

        return RunnableGraph(graph=graph, entrypoint="load_node", halt_after=["finish_batch"])

    def tasks(
        self, state: State, context: ApplicationContext, inputs: dict[str, Any]
    ) -> Generator[SubGraphTask, None, None]:
        """Override to use batch_id as application_id for checkpointing."""
        action_subgraph = self.action(state, inputs)
        tracker = _cascade_adapter(self.tracker(), context.tracker)
        state_persister = _cascade_adapter(self.state_persister(), context.state_persister)
        state_initializer = _cascade_adapter(self.state_initializer(), context.state_initializer)

        # Detect retry context
        _app_id_check = "_retry" in context.app_id if hasattr(context, "app_id") else False
        try:
            _fm = state.get("feedback_map", {})
            _feedback_map_check = _fm and len(_fm) > 0
        except Exception:
            _feedback_map_check = False

        is_retry = _app_id_check or _feedback_map_check

        for substate in self.states(state, context, inputs):
            batch_id = substate.get("current_batch_id")
            app_id = batch_id

            is_cached = (
                substate.get("batch_data", {}).get(batch_id, {}).get("status")
                == "cached_duplicate"
            )

            # During retry: Disable persistence to avoid IntegrityError
            persister_for_batch = None if is_retry else (state_persister if not is_cached else None)

            yield SubGraphTask(
                graph=RunnableGraph.create(action_subgraph),
                inputs=inputs,
                state=substate,
                application_id=app_id,
                tracker=tracker,
                state_persister=persister_for_batch,
                state_initializer=state_initializer,
            )

    async def reduce(self, state: State, results) -> State:
        """Aggregates parallel batch results (async version for asyncio.gather)."""
        batch_data = state.get("batch_data", {})
        halted_by_exception = state.get("halted_by_exception", [])
        final_nodes = state.get("final_nodes", [])

        count = 0
        async for result_state in results:
            count += 1
            result_batch_data = result_state.get("batch_data", {})
            batch_data.update(result_batch_data)

            result_halted = result_state.get("halted_by_exception", [])
            result_finals = result_state.get("final_nodes", [])
            halted_by_exception.extend(result_halted)
            final_nodes.extend(result_finals)

        print(f"\n{'=' * 60}")
        print("PARALLEL BATCH LEVEL COMPLETED")
        print(f"Processing {count} parallel batches")
        print(f"Total halted by exception: {len(halted_by_exception)}")
        print(f"Total final nodes: {len(final_nodes)}")
        print(f"{'=' * 60}\n")

        return state.update(
            batch_data=batch_data,
            halted_by_exception=halted_by_exception,
            final_nodes=final_nodes,
        )

    def is_async(self) -> bool:
        """Tell Burr to use asyncio.gather() instead of ThreadPoolExecutor."""
        return True

    @property
    def reads(self) -> list[str]:
        return ["batch_data"]

    @property
    def writes(self) -> list[str]:
        return ["batch_data", "halted_by_exception", "final_nodes"]

    def state_persister(self, **kwargs):
        """Disable persistence for parallel branches.

        Parallel batch persistence causes SQLite UNIQUE constraint conflicts.
        Cross-run caching is handled at the ROOT app level in build_app().
        """
        return None

    def state_initializer(self, **kwargs):
        """Disable state loading for parallel branches.

        State initialization for parallel batches is handled via the states() generator.
        Cross-run caching is handled at the ROOT app level in build_app().
        """
        return None


# ============================================================================
# SpawnSevenChr - 7th Character Selection
# ============================================================================


class SpawnSevenChr(MapStates):
    """Spawns a single sevenChrDef pseudo-batch for terminal children nodes.

    This runs AFTER children batch terminates (empty selection), creating
    a sequential dependency: children exploration -> 7th char selection.

    The sevenChrDef batch follows the same lifecycle:
    - load_node: Loads 7th char candidates from authority node
    - select_candidates: Picks 7th character (0-1)
    - finish_batch: Formats and records final code with 7th char
    """

    def states(
        self, state: State, context: ApplicationContext, inputs: dict
    ) -> Generator[State, None, None]:
        """Yields either a placeholder children batch or sevenChrDef batch.

        Logic uses CODE STRUCTURE instead of tracked depth (which can desync):
        1. Extract subcategory from code (chars after the dot)
        2. If subcategory < 3 chars AND no real children → placeholder needed
        3. If subcategory >= 3 chars OR has real children → spawn sevenChrDef

        ICD-10-CM rule: 7th character must be in 7th position.
        Subcategory must have 3 chars before 7th char can be added.
        """
        batch_data = state.get("batch_data", {})
        current_batch_id = state.get("current_batch_id", "ROOT")

        base_node_id = batch_data[current_batch_id].get("node_id")
        authority = batch_data[current_batch_id].get("seven_chr_authority")

        # Extract batch_name from authority dict
        authority_node_id = authority["batch_name"] if authority else None

        if not authority_node_id:
            print(f"\n{'=' * 60}")
            print(f"[{current_batch_id}] No sevenChrDef authority - skipping")
            print(f"{'=' * 60}\n")
            return

        # Check if node exists in ICD_INDEX with real children
        # If so, LLM already saw them and opted out - don't create artificial placeholder
        node_has_real_children = (
            base_node_id in ICD_INDEX and
            bool(ICD_INDEX[base_node_id].get("children"))
        )

        # CRITICAL FIX: Use CODE STRUCTURE instead of tracked depth
        # Extract subcategory (chars after the dot) to determine placeholder need
        if "." in base_node_id:
            subcategory = base_node_id.split(".", 1)[1]
        else:
            # For codes without dot (shouldn't happen for sevenChrDef candidates)
            subcategory = base_node_id[3:] if len(base_node_id) > 3 else ""

        # ICD-10-CM: subcategory needs 3 chars before 7th char can be added
        needs_placeholder = len(subcategory) < 3 and not node_has_real_children

        if needs_placeholder:
            # Subcategory too short - create placeholder
            placeholder_id = base_node_id + "X"
            placeholder_batch_id = f"{placeholder_id}|children"

            # CYCLE DETECTION
            if placeholder_batch_id in batch_data:
                print(f"\n{'=' * 60}")
                print(
                    f"[{current_batch_id}] -> Skipping duplicate placeholder batch: "
                    f"{placeholder_batch_id}"
                )
                print(f"{'=' * 60}\n")
                return

            # Calculate depth from code structure (more reliable than tracked depth)
            # depth = category(3) + subcategory_length + new_X(1)
            # Note: dot doesn't count in ICD depth
            placeholder_depth = 4 + len(subcategory)

            child_batch_data = batch_data.copy()
            child_batch_data[placeholder_batch_id] = {
                "node_id": placeholder_id,
                "parent_id": base_node_id,
                "seven_chr_authority": authority,  # Pass full authority dict
                "depth": placeholder_depth,
            }

            print(f"\n{'=' * 60}")
            print(f"[7CHR HANDOFF] {placeholder_batch_id} inherits paused task from {authority_node_id}")
            print(f"Placeholder node: {placeholder_id}")
            print(f"Subcategory: {subcategory} (len={len(subcategory)}, needs 3)")
            print(f"{'=' * 60}\n")

            yield state.update(
                current_batch_id=placeholder_batch_id,
                batch_data=child_batch_data,
                final_nodes=[],
            )
        else:
            # Subcategory has 3+ chars OR node has real children - spawn sevenChrDef
            seven_chr_batch_id = f"{base_node_id}|sevenChrDef"

            # CYCLE DETECTION - Only skip if sevenChrDef batch was fully completed
            if seven_chr_batch_id in batch_data:
                existing = batch_data[seven_chr_batch_id]
                # Only skip if this specific sevenChrDef batch completed its processing
                if existing.get("status") == "completed_seven_chr":
                    print(f"\n{'=' * 60}")
                    print(
                        f"[{current_batch_id}] -> Skipping already-completed "
                        f"sevenChrDef batch: {seven_chr_batch_id}"
                    )
                    print(f"{'=' * 60}\n")
                    return
                # Batch exists but not completed - allow this path to continue
                print(f"\n{'=' * 60}")
                print(
                    f"[{current_batch_id}] -> Batch {seven_chr_batch_id} exists but "
                    f"not completed (status={existing.get('status')}) - proceeding"
                )
                print(f"{'=' * 60}\n")

            child_batch_data = batch_data.copy()
            child_batch_data[seven_chr_batch_id] = {
                "node_id": base_node_id,
                "parent_id": base_node_id,
                "seven_chr_authority": authority,  # Pass full authority dict
                "depth": batch_data[current_batch_id].get("depth", 0),
            }

            print(f"\n{'=' * 60}")
            print(f"[7CHR RESUME] {current_batch_id} at depth 6→7")
            print(f"-> spawning {seven_chr_batch_id} (task from {authority_node_id})")
            print(f"Base node: {base_node_id}")
            print(f"Subcategory: {subcategory} (len={len(subcategory)}, ready)")
            print(f"{'=' * 60}\n")

            yield state.update(
                current_batch_id=seven_chr_batch_id,
                batch_data=child_batch_data,
                final_nodes=[],
            )

    def action(self, state: State, inputs: dict) -> RunnableGraph:
        """Defines subgraph for placeholder or sevenChrDef batch.

        For placeholder |children batches, includes full transition logic
        to recursively spawn more placeholders or sevenChrDef via spawn_seven_chr.
        """
        graph = (
            GraphBuilder()
            .with_actions(
                load_node=load_node,
                select_candidates=select_candidates,
                finish_batch=finish_batch,
                spawn_children=SpawnParallelBatches(),
                spawn_seven_chr=SpawnSevenChr(),  # Recursive for placeholder spawning
            )
            .with_transitions(
                ("load_node", "select_candidates"),
                # Empty + authority + |children → spawn_seven_chr (handles placeholder or sevenChrDef)
                (
                    "select_candidates",
                    "spawn_seven_chr",
                    expr(
                        "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                        "batch_data[current_batch_id].get('seven_chr_authority') is not None and "
                        "'|children' in current_batch_id"
                    ),
                ),
                # Empty + no authority OR non-children batch → finish
                (
                    "select_candidates",
                    "finish_batch",
                    expr(
                        "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                        "(batch_data[current_batch_id].get('seven_chr_authority') is None or "
                        "'|children' not in current_batch_id)"
                    ),
                ),
                # Has selections → spawn_children
                (
                    "select_candidates",
                    "spawn_children",
                    expr("len(batch_data[current_batch_id].get('selected_ids', [])) > 0"),
                ),
                ("spawn_seven_chr", "finish_batch"),
                ("spawn_children", "finish_batch"),
            )
            .build()
        )

        return RunnableGraph(graph=graph, entrypoint="load_node", halt_after=["finish_batch"])

    def tasks(
        self, state: State, context: ApplicationContext, inputs: dict
    ) -> Generator[SubGraphTask, None, None]:
        """Override to use batch_id as application_id."""
        action_subgraph = self.action(state, inputs)
        tracker = _cascade_adapter(self.tracker(), context.tracker)
        state_persister = _cascade_adapter(self.state_persister(), context.state_persister)
        state_initializer = _cascade_adapter(self.state_initializer(), context.state_initializer)

        # Detect retry context
        _app_id_check = "_retry" in context.app_id if hasattr(context, "app_id") else False
        try:
            _fm = state.get("feedback_map", {})
            _feedback_map_check = _fm and len(_fm) > 0
        except Exception:
            _feedback_map_check = False

        is_retry = _app_id_check or _feedback_map_check

        for substate in self.states(state, context, inputs):
            batch_id = substate.get("current_batch_id")
            app_id = batch_id

            is_cached = (
                substate.get("batch_data", {}).get(batch_id, {}).get("status")
                == "cached_duplicate"
            )

            persister_for_batch = None if is_retry else (state_persister if not is_cached else None)

            yield SubGraphTask(
                graph=RunnableGraph.create(action_subgraph),
                inputs=inputs,
                state=substate,
                application_id=app_id,
                tracker=tracker,
                state_persister=persister_for_batch,
                state_initializer=state_initializer,
            )

    async def reduce(self, state: State, results) -> State:
        """Merge sevenChrDef batch result."""
        batch_data = state.get("batch_data", {})
        final_nodes = state.get("final_nodes", [])

        async for result_state in results:
            result_batch_data = result_state.get("batch_data", {})
            batch_data.update(result_batch_data)

            result_finals = result_state.get("final_nodes", [])
            final_nodes.extend(result_finals)

        print(f"\n{'=' * 60}")
        print("SEVENCHRONDEF BATCH COMPLETED")
        print(f"{'=' * 60}\n")

        return state.update(batch_data=batch_data, final_nodes=final_nodes)

    def is_async(self) -> bool:
        return True

    @property
    def reads(self) -> list[str]:
        return ["batch_data"]

    @property
    def writes(self) -> list[str]:
        return ["batch_data", "final_nodes"]

    def state_persister(self, **kwargs):
        """Disable persistence for SevenChr branches."""
        return None

    def state_initializer(self, **kwargs):
        """Disable state loading for SevenChr branches."""
        return None
