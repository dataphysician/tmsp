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

from .actions import finish, finish_batch, load_node, select_candidates, set_batch_callback
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


# ============================================================================
# Application Builder
# ============================================================================


async def build_app(
    app_id: str = "ROOT",
    with_persistence: bool = True,
    context: str = "",
    default_selector: str = "llm",
    feedback_map: dict[str, str] | None = None,
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
        feedback_map: Optional dict of batch_id -> feedback for corrections
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
            feedback_map=feedback_map or {},
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
    selector: str | None = None,
    feedback_map: dict[str, str] | None = None,
) -> State:
    """Retry from a checkpoint with optional selector override and feedback.

    Uses fork pattern: creates NEW execution history branching from checkpoint.
    feedback_map is passed as inputs and merged into state's feedback_map dict.

    Args:
        batch_id: Batch to retry from ("ROOT" or "B.n3" etc.)
        selector: Optional selector override ("llm", "manual")
        feedback_map: Optional dict mapping batch_id to feedback string

    Returns:
        Final state from retry execution

    Example:
        final_state = await retry_node(
            batch_id="ROOT",
            selector="llm",
            feedback_map={
                "E08.3|children": "select E08.32",
                "E08.32|children": "select E08.321",
            }
        )
    """
    if PERSISTER is None:
        raise RuntimeError(
            "PERSISTER not initialized. Call initialize_persister() first."
        )

    print(f"\n{'=' * 60}")
    print(f"RETRY NODE: {batch_id}")
    print(f"Fork from: app_id={batch_id}, sequence_id=0")
    print(f"Entrypoint: select_candidates")
    print(f"Selector: {selector if selector else 'default (llm)'}")
    print(f"Feedback map: {feedback_map if feedback_map else 'none'}")
    print(f"{'=' * 60}\n")

    # Build retry application with fork pattern
    retry_app = await (
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
        .with_state_persister(PERSISTER)
        .initialize_from(
            PERSISTER,
            resume_at_next_action=False,
            default_entrypoint="select_candidates",
            default_state={},
            fork_from_app_id=batch_id,
            fork_from_sequence_id=1,
            fork_from_partition_key=PARTITION_KEY,
        )
        .abuild()
    )

    # Build inputs dict
    inputs: dict[str, str | dict[str, str]] = {"current_batch_id": batch_id}
    if selector is not None:
        inputs["selector"] = selector
    if feedback_map is not None:
        inputs["feedback_map"] = feedback_map

    # Run
    _, _, final_state = await retry_app.arun(halt_after=["finish"], inputs=inputs)

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

    # Initialize persister (don't reset - preserve cache)
    await initialize_persister(reset=False)

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
