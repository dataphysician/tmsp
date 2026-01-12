"""Main Burr application for ICD-10-CM traversal.

Provides:
- build_app(): Creates Burr application with optional persistence
- retry_node(): Retry from checkpoint with feedback

Usage:
    from agent.traversal import build_app
    from candidate_selector import configure_llm, LLM_CONFIG

    # Configure LLM (required for candidate_selector)
    import candidate_selector.config as llm_config
    llm_config.LLM_CONFIG = configure_llm(
        provider="openai",
        api_key="sk-..."
    )

    # Build and run
    app = await build_app(
        context="Patient with type 2 diabetes...",
        default_selector="llm"
    )
    _, _, final_state = await app.arun(halt_after=["finish"])
"""

import os

from burr.core import ApplicationBuilder, State, expr
from burr.integrations.persisters.b_aiosqlite import AsyncSQLitePersister

from .actions import finish, finish_batch, load_node, select_candidates, set_batch_callback
from .parallel import SpawnParallelBatches, SpawnSevenChr


# ============================================================================
# Global Persister
# ============================================================================

# Initialized via initialize_persister() because await cannot be used at module level
PERSISTER: AsyncSQLitePersister | None = None


async def initialize_persister(
    db_path: str = "./burr_checkpoint.db",
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
):
    """Build the main Burr application for ICD-10-CM traversal.

    Uses async/await for WASM compatibility. Burr auto-detects async actions
    and uses asyncio.gather() for parallelism.

    Args:
        app_id: Application identifier (default: "ROOT")
        with_persistence: Whether to enable SQLite state persistence
        context: Clinical context string to guide LLM selections
        default_selector: Default selector type - "llm" or "manual"
        feedback_map: Optional dict of batch_id -> feedback for corrections

    Returns:
        Burr Application instance (async-enabled)

    Example:
        # Configure LLM first
        import candidate_selector.config as llm_config
        llm_config.LLM_CONFIG = configure_llm(
            provider="openai",
            api_key="sk-..."
        )

        # Build and run
        app = await build_app(
            context="Patient with diabetes presenting with hyperglycemia...",
            default_selector="llm"
        )
        _, _, final_state = await app.arun(halt_after=["finish"])

        # Get results
        final_codes = final_state.get("final_nodes")
    """
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
        builder = builder.with_state_persister(PERSISTER).with_identifiers(app_id=app_id)

    print(f"\n{'=' * 60}")
    print("[BUILD_APP] Initializing app with:")
    print(f"  app_id: {app_id}")
    print(f"  context length: {len(context)} chars")
    print(f"  context preview: {context[:100]}..." if context else "  context: <EMPTY>")
    print(f"  default_selector: {default_selector}")
    print(f"  with_persistence: {with_persistence}")
    print(f"{'=' * 60}\n")

    return await builder.abuild()


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
            fork_from_partition_key=None,
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
) -> dict:
    """High-level API to run ICD-10-CM traversal.

    Convenience function that handles:
    1. LLM configuration
    2. Persister initialization
    3. Callback setup
    4. Application building and execution

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

    Returns:
        Dict with final_nodes and batch_data
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

    # Initialize persister if needed
    if with_persistence:
        await initialize_persister(reset=True)

    # Set callback
    if on_batch_complete is not None:
        set_batch_callback(on_batch_complete)

    try:
        # Build and run
        app = await build_app(
            context=clinical_note,
            default_selector=selector,
            with_persistence=with_persistence,
        )

        _, _, final_state = await app.arun(halt_after=["finish"])

        return {
            "final_nodes": final_state.get("final_nodes", []),
            "batch_data": final_state.get("batch_data", {}),
        }

    finally:
        if with_persistence:
            await cleanup_persister()


# ============================================================================
# CLI Main
# ============================================================================


async def main():
    """Example CLI usage."""
    # Initialize persister
    await initialize_persister(reset=True)

    try:
        # Configure LLM (example with manual selector for testing)
        from candidate_selector import configure_llm
        import candidate_selector.config as llm_config

        llm_config.LLM_CONFIG = configure_llm(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

        # Build app
        app = await build_app(
            context="Patient with type 2 diabetes presenting with hyperglycemia",
            default_selector="llm",
            with_persistence=True,
        )

        # Run
        _, _, final_state = await app.arun(halt_after=["finish"])

        print(f"\nFinal codes: {final_state.get('final_nodes', [])}")

    finally:
        await cleanup_persister()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
