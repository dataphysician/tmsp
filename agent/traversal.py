"""Main Burr application for ICD-10-CM traversal

Provides:
- build_app(): Creates Burr application with optional persistence
- retry_node(): Retry from checkpoint with feedback
- generate_traversal_cache_key(): Generate cache key for cross-run caching

Usage:
    from agent.traversal import build_app, generate_traversal_cache_key
    from candidate_selector import configure_llm, LLM_CONFIG

    # Configure LLM (required for candidate_selector)
    import candidate_selector.config as llm_config
    llm_config.LLM_CONFIG = configure_llm(
        provider="openai",
        api_key="sk-..."
    )

    # Generate cache key for cross-run caching
    partition_key = generate_traversal_cache_key(
        clinical_note="Patient with type 2 diabetes...",
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
    )

    # Build and run (uses cache if available)
    app = await build_app(
        context="Patient with type 2 diabetes...",
        default_selector="llm",
        partition_key=partition_key,
    )
    _, _, final_state = await app.arun(halt_after=["finish"])
"""

import hashlib
import os

from burr.core import ApplicationBuilder, State, expr
from burr.integrations.persisters.b_aiosqlite import AsyncSQLitePersister

from .actions import finish, finish_batch, inject_feedback, load_node, select_candidates, set_batch_callback
from .parallel import SpawnParallelBatches, SpawnSevenChr


# ============================================================================
# Global Persister and Partition Key
# ============================================================================

# Initialized via initialize_persister() because await cannot be used at module level
PERSISTER: AsyncSQLitePersister | None = None

# Global partition key for cross-run caching (set by server before building app)
PARTITION_KEY: str | None = None


def generate_traversal_cache_key(
    clinical_note: str,
    provider: str,
    model: str | None,
    temperature: float,
    system_prompt: str | None = None,
) -> str:
    """Generate a deterministic cache key for scaffolded traversal.

    The key is based on all inputs that affect the LLM responses during traversal.
    Different clinical notes or model configurations produce different keys,
    ensuring cache isolation between runs.

    Args:
        clinical_note: The clinical context string
        provider: LLM provider (e.g., "openai", "anthropic")
        model: Model name (None uses provider default)
        temperature: Temperature setting
        system_prompt: Custom system prompt (if any)

    Returns:
        A hash string to use as partition_key (e.g., "scaffolded_a1b2c3d4e5f6g7h8")
    """
    # Normalize inputs
    normalized = {
        "clinical_note": clinical_note.strip(),
        "provider": provider,
        "model": model or "",
        "temperature": round(temperature, 2),  # Avoid float precision issues
        "system_prompt": (system_prompt or "").strip(),
    }

    # Create deterministic string representation
    key_str = (
        f"{normalized['provider']}|{normalized['model']}|"
        f"{normalized['temperature']}|"
        f"{normalized['system_prompt']}|{normalized['clinical_note']}"
    )

    # Hash for shorter key
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

    return f"scaffolded_{key_hash}"


async def initialize_persister(
    db_path: str = "./cache.db",
    table_name: str = "burr_state",
    reset: bool = False,
) -> AsyncSQLitePersister:
    """Initialize the global async SQLite persister.

    Args:
        db_path: Path to SQLite database file
        table_name: Table name for state storage
        reset: If True, delete existing database before creating

    Returns:
        Initialized AsyncSQLitePersister
    """
    global PERSISTER

    # Cleanup existing persister before reset to avoid orphaned connections
    if PERSISTER is not None:
        await PERSISTER.cleanup()
        PERSISTER = None

    if reset and os.path.exists(db_path):
        os.remove(db_path)

    PERSISTER = await AsyncSQLitePersister.from_values(
        db_path=db_path,
        table_name=table_name,
    )

    await PERSISTER.initialize()
    return PERSISTER


async def cleanup_persister() -> None:
    """Cleanup persister connection."""
    global PERSISTER
    if PERSISTER is not None:
        await PERSISTER.cleanup()
        PERSISTER = None


async def delete_persisted_state(
    partition_key: str,
    app_id: str = "ROOT",
    db_path: str = "./cache.db",
    table_name: str = "burr_state",
) -> bool:
    """Delete a specific persisted state entry.

    Use this when a traversal is cancelled or reset to ensure a fresh run.

    Args:
        partition_key: The partition key of the entry to delete
        app_id: Application identifier (default: "ROOT")
        db_path: Path to SQLite database file
        table_name: Table name for state storage

    Returns:
        True if entry was deleted, False if not found or error
    """
    import aiosqlite

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                f"DELETE FROM {table_name} WHERE partition_key = ? AND app_id = ?",
                (partition_key, app_id),
            )
            await db.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                print(f"[PERSISTER] Deleted entry: partition_key={partition_key}, app_id={app_id}")
            return deleted
    except Exception as e:
        print(f"[PERSISTER] Error deleting entry: {e}")
        return False


# ============================================================================
# State Pruning for Rewind
# ============================================================================


def get_descendant_batch_ids(batch_id: str, batch_data: dict[str, dict]) -> set[str]:
    """Find all batch_ids that are descendants of the given batch.

    Uses batch_data to trace the tree structure via selected_ids.

    Args:
        batch_id: The batch to find descendants for
        batch_data: Dict mapping batch_id to batch info

    Returns:
        Set of descendant batch_ids (not including the input batch_id)
    """
    descendants: set[str] = set()

    # Get the selected_ids from the rewind batch
    rewind_batch_data = batch_data.get(batch_id, {})
    selected_ids = rewind_batch_data.get("selected_ids", [])

    # BFS to collect all descendant batch_ids
    queue = list(selected_ids)

    while queue:
        node_id = queue.pop(0)

        # Check all batch types for this node
        for btype in ["children", "codeFirst", "codeAlso", "useAdditionalCode", "sevenChrDef"]:
            child_batch_id = f"{node_id}|{btype}"
            if child_batch_id in batch_data and child_batch_id not in descendants:
                descendants.add(child_batch_id)
                child_selected = batch_data[child_batch_id].get("selected_ids", [])
                queue.extend(child_selected)

    return descendants


def prune_state_for_rewind(state_dict: dict, rewind_batch_id: str) -> dict:
    """Prepare state for rewind by removing descendant batch data and final_nodes.

    This ensures that when we rewind to a batch:
    1. The rewind batch's selection is cleared (will be re-computed with feedback)
    2. All descendant batches are removed
    3. Final nodes from descendants are removed

    Args:
        state_dict: The full state dict from checkpoint
        rewind_batch_id: The batch_id to rewind to

    Returns:
        New state dict with pruned data
    """
    batch_data = state_dict.get("batch_data", {}).copy()
    final_nodes = list(state_dict.get("final_nodes", []))

    # Find all descendant batch_ids
    descendants = get_descendant_batch_ids(rewind_batch_id, batch_data)

    # Collect final_nodes from descendant batches
    descendant_finals: set[str] = set()
    for desc_batch_id in descendants:
        desc_batch = batch_data.get(desc_batch_id, {})
        node_id = desc_batch.get("node_id")
        if node_id:
            descendant_finals.add(node_id)
        for sel in desc_batch.get("selected_ids", []):
            descendant_finals.add(sel)

        # For sevenChrDef batches, also remove the full 7th char code
        # The final_nodes entry is format_with_seventh_char(node_id, selected_ids[0])
        batch_type = desc_batch_id.rsplit("|", 1)[1] if "|" in desc_batch_id else "children"
        if batch_type == "sevenChrDef" and node_id:
            selected_ids = desc_batch.get("selected_ids", [])
            if selected_ids:
                # Reconstruct the full code that was added to final_nodes
                from agent.actions import format_with_seventh_char
                full_code = format_with_seventh_char(node_id, selected_ids[0])
                descendant_finals.add(full_code)

    # Clear the rewind batch's selection (will be re-computed with feedback)
    # Also track the rewind batch's parent node - it should lose finalized status
    # because the rewind might add children (making it no longer a leaf)
    rewind_node_id = None
    if rewind_batch_id in batch_data:
        rewind_node_id = batch_data[rewind_batch_id].get("node_id")

        # For sevenChrDef rewind batch, also remove its full code from final_nodes
        rewind_batch_type = rewind_batch_id.rsplit("|", 1)[1] if "|" in rewind_batch_id else "children"
        if rewind_batch_type == "sevenChrDef" and rewind_node_id:
            rewind_selected = batch_data[rewind_batch_id].get("selected_ids", [])
            if rewind_selected:
                from agent.actions import format_with_seventh_char
                rewind_full_code = format_with_seventh_char(rewind_node_id, rewind_selected[0])
                descendant_finals.add(rewind_full_code)

        batch_data[rewind_batch_id] = {
            k: v for k, v in batch_data[rewind_batch_id].items()
            if k not in ("selected_ids", "reasoning", "status")
        }

    # Remove descendant batches entirely
    for desc_batch_id in descendants:
        batch_data.pop(desc_batch_id, None)

    # Remove descendant final_nodes from the pruned list
    # For children batch rewinds, also remove the rewind node itself because
    # it may gain children after rewind (making it no longer a leaf/finalized)
    # For lateral batch rewinds, keep the rewind node - lateral links don't affect leaf status
    nodes_to_remove = descendant_finals
    if rewind_node_id and rewind_batch_id in batch_data:
        rewind_batch_type = batch_data[rewind_batch_id].get("batch_type", "children")
        if rewind_batch_type == "children":
            nodes_to_remove = descendant_finals | {rewind_node_id}
    pruned_finals = [fn for fn in final_nodes if fn not in nodes_to_remove]

    return {
        **state_dict,
        "batch_data": batch_data,
        "final_nodes": pruned_finals,
    }


# ============================================================================
# Application Builder
# ============================================================================


async def build_app(
    app_id: str = "ROOT",
    with_persistence: bool = True,
    context: str = "",
    default_selector: str = "llm",
    partition_key: str | None = None,
) -> tuple:
    """Build the main Burr application for ICD-10-CM traversal.

    Uses async/await for WASM compatibility. Burr auto-detects async actions
    and uses asyncio.gather() for parallelism.

    Supports cross-run caching via partition_key. When a partition_key is
    provided and a completed run exists in the cache, returns the cached
    state without re-running the LLM.

    Args:
        app_id: Application identifier (default: "ROOT")
        with_persistence: Whether to enable SQLite state persistence
        context: Clinical context string to guide LLM selections
        default_selector: Default selector type - "llm" or "manual"
        partition_key: Cache partition key (from generate_traversal_cache_key)

    Returns:
        Tuple of (app, cached: bool)
        - If cached=True: app.state contains the cached results
        - If cached=False: caller should run app.arun(halt_after=["finish"])

    Example:
        # Configure LLM first
        import candidate_selector.config as llm_config
        llm_config.LLM_CONFIG = configure_llm(
            provider="openai",
            api_key="sk-..."
        )

        # Generate partition key for caching
        partition_key = generate_traversal_cache_key(
            clinical_note="Patient with diabetes...",
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
        )

        # Build and run
        app, cached = await build_app(
            context="Patient with diabetes presenting with hyperglycemia...",
            default_selector="llm",
            partition_key=partition_key,
        )

        if not cached:
            _, _, final_state = await app.arun(halt_after=["finish"])
        else:
            final_state = app.state

        # Get results
        final_codes = final_state.get("final_nodes")
    """
    global PARTITION_KEY
    PARTITION_KEY = partition_key

    # Check for cached result if partition_key is provided
    if partition_key and with_persistence and PERSISTER is not None:
        cached_state = await PERSISTER.load(
            partition_key=partition_key,
            app_id=app_id,
        )

        if cached_state is not None:
            # Cache hit! Check if the run completed (has final_nodes)
            state_dict = cached_state.get("state", {})
            final_nodes = state_dict.get("final_nodes", [])

            if final_nodes:  # Non-empty final_nodes indicates completed run
                print(f"\n{'=' * 60}")
                print("[BUILD_APP] CACHE HIT!")
                print(f"  partition_key: {partition_key}")
                print(f"  cached final_nodes: {len(final_nodes)} codes")
                print(f"{'=' * 60}\n")

                # Build a minimal app with the cached state for consistent API
                app = await (
                    ApplicationBuilder()
                    .with_actions(
                        load_node=load_node,
                        select_candidates=select_candidates,
                        spawn_parallel=SpawnParallelBatches(),
                        spawn_seven_chr=SpawnSevenChr(),
                        finish=finish,
                    )
                    .with_transitions(
                        ("load_node", "select_candidates"),
                        ("select_candidates", "finish"),
                    )
                    .with_entrypoint("load_node")
                    .with_state(**state_dict)
                    .with_identifiers(app_id=app_id, partition_key=partition_key)
                    .abuild()
                )

                return app, True  # cached=True

    # Cache miss - build full app
    print(f"\n{'=' * 60}")
    print("[BUILD_APP] Initializing app with:")
    print(f"  app_id: {app_id}")
    print(f"  context length: {len(context)} chars")
    print(f"  context preview: {context[:100]}..." if context else "  context: <EMPTY>")
    print(f"  default_selector: {default_selector}")
    print(f"  with_persistence: {with_persistence}")
    print(f"  partition_key: {partition_key}")
    print(f"{'=' * 60}\n")

    builder = (
        ApplicationBuilder()
        .with_actions(
            load_node=load_node,
            select_candidates=select_candidates,
            spawn_parallel=SpawnParallelBatches(),
            spawn_seven_chr=SpawnSevenChr(),
            finish=finish,
        )
        .with_transitions(
            # Cache hit -> exit early
            (
                "load_node",
                "finish",
                expr("batch_data.get(current_batch_id, {}).get('status') == 'cached_duplicate'"),
            ),
            # Normal flow
            ("load_node", "select_candidates"),
            # Empty selection WITH authority -> spawn sevenChrDef
            (
                "select_candidates",
                "spawn_seven_chr",
                expr(
                    "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                    "batch_data[current_batch_id].get('seven_chr_authority') is not None"
                ),
            ),
            # Empty selection WITHOUT authority -> finish
            (
                "select_candidates",
                "finish",
                expr(
                    "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                    "batch_data[current_batch_id].get('seven_chr_authority') is None"
                ),
            ),
            # Has selections -> spawn parallel
            (
                "select_candidates",
                "spawn_parallel",
                expr("len(batch_data[current_batch_id].get('selected_ids', [])) > 0"),
            ),
            # After sevenChrDef -> finish
            ("spawn_seven_chr", "finish"),
            # After parallel -> finish
            ("spawn_parallel", "finish"),
        )
        .with_entrypoint("load_node")
        .with_state(
            current_batch_id=app_id,
            batch_data={app_id: {"node_id": "ROOT", "parent_id": None, "seven_chr_authority": None, "depth": 0}},
            context=context,
            default_selector=default_selector,
            final_nodes=[],
            pending_feedback=None,  # Set by inject_feedback during rewind
        )
    )

    if with_persistence and PERSISTER is not None:
        builder = builder.with_state_persister(PERSISTER)
        if partition_key:
            builder = builder.with_identifiers(app_id=app_id, partition_key=partition_key)
        else:
            builder = builder.with_identifiers(app_id=app_id)

    return await builder.abuild(), False  # cached=False


# ============================================================================
# Retry Function
# ============================================================================


async def retry_node(
    batch_id: str,
    feedback: str,
    selector: str | None = None,
) -> State:
    """Retry from a checkpoint with EXPLICIT feedback injection.

    Workflow: inject_feedback → select_candidates → spawn_parallel → finish

    The inject_feedback action makes feedback visible in execution trace,
    providing clear data flow instead of hidden state manipulation.

    Args:
        batch_id: Batch to retry from ("ROOT" or "B19.2|children" etc.)
        feedback: Feedback string for the target batch
        selector: Optional selector override ("llm", "manual")

    Returns:
        Final state from retry execution

    Example:
        final_state = await retry_node(
            batch_id="E08.3|children",
            feedback="select E08.32 - this is the correct code for diabetic retinopathy",
            selector="llm",
        )
    """
    if PERSISTER is None:
        raise RuntimeError(
            "PERSISTER not initialized. Call initialize_persister() first."
        )

    print(f"\n{'=' * 60}")
    print(f"RETRY NODE (Explicit Feedback Injection): {batch_id}")
    print(f"Loading state from: app_id=ROOT, partition_key={PARTITION_KEY}")
    print(f"Entrypoint: inject_feedback")
    print(f"Selector: {selector if selector else 'default (llm)'}")
    print(f"Feedback: {feedback[:100]}{'...' if len(feedback) > 100 else ''}")
    print(f"{'=' * 60}\n")

    # Load the ROOT checkpoint state directly (more reliable than fork for cached traversals)
    cached_state = await PERSISTER.load(
        partition_key=PARTITION_KEY,
        app_id="ROOT",
    )

    if cached_state is None:
        raise RuntimeError(
            f"Cannot rewind: ROOT checkpoint not found for partition_key={PARTITION_KEY}"
        )

    # Extract state and sequence_id from PersistedStateData
    # The "state" field is a Burr State object
    state_obj = cached_state.get("state")
    if state_obj is None:
        raise RuntimeError("cached_state has no 'state' field")

    # Get the original sequence_id for later cache update
    # Burr's load() returns highest sequence_id, we need to save at sequence_id+1
    original_sequence_id = cached_state.get("sequence_id", 0)
    print(f"[RETRY] Original sequence_id: {original_sequence_id}")

    # Get the dict representation for validation and modification
    state_dict = state_obj.get_all() if hasattr(state_obj, 'get_all') else dict(state_obj)
    print(f"[RETRY] state_dict keys: {list(state_dict.keys())}")

    batch_data = state_dict.get("batch_data", {})
    print(f"[RETRY] Loaded state with {len(batch_data)} batches: {list(batch_data.keys())[:10]}...")

    if batch_id not in batch_data:
        raise KeyError(
            f"Batch '{batch_id}' not found in cached state. "
            f"Available batches: {list(batch_data.keys())}"
        )

    # CRITICAL FIX 1: Prune descendant state before re-running
    # This removes:
    # - All descendant batches from batch_data
    # - All final_nodes that came from descendants
    # - The selected_ids/reasoning/status from the rewind batch itself
    pruned_state_dict = prune_state_for_rewind(state_dict, batch_id)

    original_batches = len(state_dict.get("batch_data", {}))
    pruned_batches = len(pruned_state_dict.get("batch_data", {}))
    print(f"[RETRY] Pruned {original_batches - pruned_batches} descendant batches")

    # DON'T inject feedback into state here!
    # The inject_feedback action will do it explicitly, making the data flow visible
    # in the execution trace.

    # CRITICAL: Remove Burr's internal state tracking keys!
    # These cause Burr to think the app already ran and skip the entrypoint.
    # Keys like __PRIOR_STEP, __SEQUENCE_ID tell Burr where execution left off.
    burr_internal_keys = [k for k in pruned_state_dict.keys() if k.startswith("__")]
    for key in burr_internal_keys:
        del pruned_state_dict[key]
    # Also remove halted_by_exception if present
    pruned_state_dict.pop("halted_by_exception", None)
    print(f"[RETRY] Removed Burr internal keys: {burr_internal_keys}")

    # Set current_batch_id in state for consistency
    pruned_state_dict["current_batch_id"] = batch_id
    # Ensure pending_feedback is None (inject_feedback will set it)
    pruned_state_dict["pending_feedback"] = None
    # Clear old feedback_map if present (we use pending_feedback now)
    pruned_state_dict["feedback_map"] = {}

    # Build retry application with PRUNED state
    # Entrypoint is inject_feedback which makes feedback injection VISIBLE
    # NOTE: The retry app MUST include finish_batch because sevenChrDef codes
    # need to be finalized after the rewind selection. Without finish_batch,
    # the new 7th char code won't be added to final_nodes.
    retry_app = await (
        ApplicationBuilder()
        .with_actions(
            inject_feedback=inject_feedback,  # Explicit feedback injection action
            load_node=load_node,
            select_candidates=select_candidates,
            spawn_parallel=SpawnParallelBatches(),
            spawn_seven_chr=SpawnSevenChr(),
            finish_batch=finish_batch,  # Required for sevenChrDef finalization
            finish=finish,
        )
        .with_transitions(
            # Rewind starts with explicit feedback injection
            ("inject_feedback", "select_candidates"),
            # Normal transitions from here
            ("load_node", "select_candidates"),
            # Empty selection WITH authority -> spawn sevenChrDef
            (
                "select_candidates",
                "spawn_seven_chr",
                expr(
                    "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                    "batch_data[current_batch_id].get('seven_chr_authority') is not None"
                ),
            ),
            # Empty selection WITHOUT authority -> finish_batch
            (
                "select_candidates",
                "finish_batch",
                expr(
                    "len(batch_data[current_batch_id].get('selected_ids', [])) == 0 and "
                    "batch_data[current_batch_id].get('seven_chr_authority') is None"
                ),
            ),
            # Has selections -> spawn parallel
            (
                "select_candidates",
                "spawn_parallel",
                expr("len(batch_data[current_batch_id].get('selected_ids', [])) > 0"),
            ),
            # After sevenChrDef -> finish_batch
            ("spawn_seven_chr", "finish_batch"),
            # After parallel -> finish_batch
            ("spawn_parallel", "finish_batch"),
            # After finish_batch -> finish
            ("finish_batch", "finish"),
        )
        .with_entrypoint("inject_feedback")  # Start with explicit feedback injection
        .with_state(**pruned_state_dict)
        .with_identifiers(app_id=f"rewind-{batch_id}", partition_key=PARTITION_KEY)
        .abuild()
    )

    # Build inputs dict for inject_feedback action
    inputs: dict[str, str] = {
        "batch_id": batch_id,
        "feedback": feedback,
    }
    if selector is not None:
        inputs["selector"] = selector

    print(f"[RETRY] About to run app with inputs: {list(inputs.keys())}")
    print(f"[RETRY] inputs['batch_id'] = {inputs['batch_id']}")
    print(f"[RETRY] inputs['feedback'] = {inputs['feedback'][:100]}..." if len(inputs.get('feedback', '')) > 100 else f"[RETRY] inputs['feedback'] = {inputs.get('feedback')}")

    # Run - starts at inject_feedback which sets pending_feedback in state
    try:
        _, _, final_state = await retry_app.arun(halt_after=["finish"], inputs=inputs)
        print(f"[RETRY] App completed successfully")
    except Exception as e:
        print(f"[RETRY] App raised exception: {e}")
        import traceback
        traceback.print_exc()
        raise

    # CRITICAL FIX 3: Update ROOT checkpoint with rewind results
    # This ensures subsequent cache hits return the updated state
    # Without this, any request that triggers cache lookup would get old results
    #
    # Burr's SQLite persister uses (partition_key, app_id, sequence_id) as primary key.
    # load() returns the HIGHEST sequence_id entry. We must save with sequence_id > original
    # to ensure our update becomes the "latest" and is returned by subsequent load() calls.
    if PERSISTER is not None and PARTITION_KEY is not None:
        try:
            # Get the state dict to save
            final_state_dict = final_state.get_all() if hasattr(final_state, 'get_all') else dict(final_state)

            # pending_feedback is already None (consumed by select_candidates)
            # No need to clear - the explicit action design handles this automatically

            # Use sequence_id + 1 to ensure this becomes the "latest" checkpoint
            # Burr's load() orders by sequence_id DESC and returns first result
            new_sequence_id = original_sequence_id + 1

            await PERSISTER.save(
                partition_key=PARTITION_KEY,
                app_id="ROOT",
                sequence_id=new_sequence_id,
                position="finish",
                state=State(final_state_dict),
                status="completed",
            )
            print(f"[RETRY] Updated ROOT checkpoint with rewind results (sequence_id={new_sequence_id})")
            print(f"[RETRY] New final_nodes count: {len(final_state_dict.get('final_nodes', []))}")
        except Exception as e:
            print(f"[RETRY] Warning: Failed to update ROOT checkpoint: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the rewind - the results are still valid for this session

    print(f"\n{'=' * 60}")
    print(f"RETRY COMPLETED: {batch_id}")
    print(f"{'=' * 60}\n")

    return final_state


# ============================================================================
# Main Entry Point
# ============================================================================


async def run_traversal(
    clinical_note: str,
    provider: str = "openai",
    api_key: str = "",
    model: str | None = None,
    selector: str = "llm",
    with_persistence: bool = True,
    on_batch_complete=None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    use_cache: bool = False,
) -> dict:
    """High-level API to run ICD-10-CM traversal.

    Convenience function that handles:
    1. LLM configuration
    2. Persister initialization
    3. Callback setup
    4. Application building and execution
    5. Cross-run caching (when use_cache=True)

    Args:
        clinical_note: Clinical context text
        provider: LLM provider ("openai", "cerebras", "sambanova", "anthropic", "other")
        api_key: API key for provider
        model: Optional model override
        selector: Selector type ("llm" or "manual")
        with_persistence: Enable SQLite checkpointing
        on_batch_complete: Optional callback for streaming
        temperature: Optional temperature override
        max_tokens: Optional max completion tokens override
        system_prompt: Optional custom system prompt
        use_cache: If True, enable cross-run caching (don't reset DB)

    Returns:
        Dict with final_nodes, batch_data, and cached (bool)
    """
    # Configure LLM
    from candidate_selector.providers import create_config
    import candidate_selector.config as llm_config

    config_kwargs: dict = {}
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if max_tokens is not None:
        config_kwargs["max_completion_tokens"] = max_tokens

    llm_config.LLM_CONFIG = create_config(
        provider=provider,
        api_key=api_key,
        model=model,
        **config_kwargs,
    )

    # Generate partition key for caching
    partition_key = None
    if use_cache:
        partition_key = generate_traversal_cache_key(
            clinical_note=clinical_note,
            provider=provider,
            model=model,
            temperature=temperature or 0.0,
            system_prompt=system_prompt,
        )

    # Initialize persister if needed (don't reset if using cache)
    if with_persistence:
        await initialize_persister(reset=not use_cache)

    # Set callback
    if on_batch_complete is not None:
        set_batch_callback(on_batch_complete)

    try:
        # Build app (may return cached result)
        app, cached = await build_app(
            context=clinical_note,
            default_selector=selector,
            with_persistence=with_persistence,
            partition_key=partition_key,
        )

        if cached:
            # Cache hit - use existing state
            final_state = app.state
        else:
            # Cache miss - run the traversal
            _, _, final_state = await app.arun(halt_after=["finish"])

        return {
            "final_nodes": final_state.get("final_nodes", []),
            "batch_data": final_state.get("batch_data", {}),
            "cached": cached,
        }

    finally:
        if with_persistence:
            await cleanup_persister()


# ============================================================================
# CLI Main
# ============================================================================


async def main():
    """Example CLI usage with cross-run caching."""
    clinical_note = "Patient with type 2 diabetes presenting with hyperglycemia"

    # Configure LLM
    from candidate_selector import configure_llm
    import candidate_selector.config as llm_config

    llm_config.LLM_CONFIG = configure_llm(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    # Generate partition key for caching
    partition_key = generate_traversal_cache_key(
        clinical_note=clinical_note,
        provider="openai",
        model=llm_config.LLM_CONFIG.model,
        temperature=llm_config.LLM_CONFIG.temperature,
    )

    # Initialize persister (don't reset -> preserve cache)
    await initialize_persister(reset=True)

    try:
        # Build app (may return cached result)
        app, cached = await build_app(
            context=clinical_note,
            default_selector="llm",
            with_persistence=True,
            partition_key=partition_key,
        )

        if cached:
            # Cache hit - use existing state
            final_state = app.state
            print("\n[CACHE HIT] Using cached results")
        else:
            # Cache miss - run the traversal
            _, _, final_state = await app.arun(halt_after=["finish"])
            print("\n[CACHE MISS] Traversal completed")

        print(f"\nFinal codes: {final_state.get('final_nodes', [])}")

    finally:
        await cleanup_persister()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
