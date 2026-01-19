"""Zero-shot Burr application for ICD-10-CM code generation

Provides a Burr-based zero-shot code generator with:
- SQLite persistence for caching (same as scaffolded traversal)
- Automatic cache hit detection via partition_key
- Consistent architecture with scaffolded mode

Usage:
    from agent.zero_shot import build_zero_shot_app, generate_zero_shot_cache_key

    # Generate cache key from inputs
    partition_key = generate_zero_shot_cache_key(
        clinical_note="Patient with...",
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
        system_prompt=None
    )

    # Build and run (will use cache if available)
    app, cached = await build_zero_shot_app(
        clinical_note="Patient with...",
        partition_key=partition_key,
    )

    if cached:
        # Cache hit - get results from state
        final_state = app.state
    else:
        # Cache miss - run the app
        _, _, final_state = await app.arun(halt_after=["finish"])

    codes = final_state.get("selected_codes", [])
    reasoning = final_state.get("reasoning", "")
"""

import hashlib

from burr.core import ApplicationBuilder, State, action
from burr.integrations.persisters.b_aiosqlite import AsyncSQLitePersister

from candidate_selector.selector import zero_shot_selector


# ============================================================================
# Global Persister (shared with scaffolded traversal)
# ============================================================================

ZERO_SHOT_PERSISTER: AsyncSQLitePersister | None = None


async def initialize_zero_shot_persister(
    db_path: str = "./cache.db",
    table_name: str = "zero_shot_state",
) -> AsyncSQLitePersister:
    """Initialize the zero-shot SQLite persister.

    Uses a separate database from scaffolded traversal to avoid conflicts.

    Args:
        db_path: Path to SQLite database file
        table_name: Table name for state storage

    Returns:
        Initialized AsyncSQLitePersister
    """
    global ZERO_SHOT_PERSISTER

    # Don't reset - we want to keep the cache!
    ZERO_SHOT_PERSISTER = await AsyncSQLitePersister.from_values(
        db_path=db_path,
        table_name=table_name,
    )

    await ZERO_SHOT_PERSISTER.initialize()
    return ZERO_SHOT_PERSISTER


async def cleanup_zero_shot_persister() -> None:
    """Cleanup persister connection."""
    global ZERO_SHOT_PERSISTER
    if ZERO_SHOT_PERSISTER is not None:
        await ZERO_SHOT_PERSISTER.cleanup()
        ZERO_SHOT_PERSISTER = None


# ============================================================================
# Cache Key Generation
# ============================================================================


def generate_zero_shot_cache_key(
    clinical_note: str,
    provider: str,
    model: str,
    temperature: float,
    max_completion_tokens: int | None = None,
    system_prompt: str | None = None,
) -> str:
    """Generate a deterministic cache key for zero-shot requests.

    The key is based on all inputs that affect the LLM response.

    Args:
        clinical_note: The clinical note text
        provider: LLM provider (e.g., "openai", "anthropic")
        model: Model name
        temperature: Temperature setting
        max_completion_tokens: Max completion tokens setting
        system_prompt: Custom system prompt (if any)

    Returns:
        A hash string to use as partition_key
    """
    # Normalize inputs
    normalized = {
        "clinical_note": clinical_note.strip(),
        "provider": provider,
        "model": model,
        "temperature": round(temperature, 2),  # Avoid float precision issues
        "max_completion_tokens": max_completion_tokens or 0,
        "system_prompt": (system_prompt or "").strip(),
    }

    # Create deterministic string representation
    key_str = (
        f"{normalized['provider']}|{normalized['model']}|"
        f"{normalized['temperature']}|{normalized['max_completion_tokens']}|"
        f"{normalized['system_prompt']}|{normalized['clinical_note']}"
    )

    # Hash for shorter key
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

    return f"zeroshot_{key_hash}"


# ============================================================================
# Burr Actions
# ============================================================================


@action(reads=["clinical_note"], writes=["selected_codes", "reasoning"])
async def call_llm(state: State) -> State:
    """Call the LLM to generate ICD-10 codes.

    This is the main action that invokes the zero-shot selector.
    """
    clinical_note = state["clinical_note"]

    print(f"[ZERO-SHOT BURR] Calling LLM for clinical note ({len(clinical_note)} chars)")

    # Call the existing zero_shot_selector
    selected_codes, reasoning = await zero_shot_selector(
        clinical_note=clinical_note,
        config=None,  # Uses global LLM_CONFIG
    )

    print(f"[ZERO-SHOT BURR] LLM returned {len(selected_codes)} codes")

    return state.update(
        selected_codes=selected_codes,
        reasoning=reasoning,
    )


@action(reads=["selected_codes", "reasoning"], writes=[])
def finish(state: State) -> State:
    """Terminal state - no-op, just marks completion."""
    codes = state.get("selected_codes", [])
    print(f"[ZERO-SHOT BURR] Finished with {len(codes)} codes")
    return state


# ============================================================================
# Application Builder
# ============================================================================


async def build_zero_shot_app(
    clinical_note: str,
    partition_key: str,
    app_id: str = "zero_shot",
) -> tuple:
    """Build the zero-shot Burr application.

    Checks for cached results first. If found, returns the cached state
    without running the LLM.

    Args:
        clinical_note: The clinical note to analyze
        partition_key: Cache key (from generate_zero_shot_cache_key)
        app_id: Application ID (default: "zero_shot")

    Returns:
        Tuple of (app, cached: bool)
        - If cached=True: app.state contains the cached results
        - If cached=False: caller should run app.arun(halt_after=["finish"])
    """
    if ZERO_SHOT_PERSISTER is None:
        raise RuntimeError(
            "Zero-shot persister not initialized. "
            "Call initialize_zero_shot_persister() first."
        )

    # Check for cached result
    cached_state = await ZERO_SHOT_PERSISTER.load(
        partition_key=partition_key,
        app_id=app_id,
    )

    if cached_state is not None:
        # Cache hit! Check if the run completed (has selected_codes)
        state_dict = cached_state.get("state", {})
        if "selected_codes" in state_dict and state_dict["selected_codes"] is not None:
            print(f"[ZERO-SHOT BURR] Cache HIT for partition_key={partition_key}")
            print(f"[ZERO-SHOT BURR] Cached codes: {len(state_dict.get('selected_codes', []))} codes")

            # Build a minimal app with the cached state
            # This allows the caller to access state consistently
            app = await (
                ApplicationBuilder()
                .with_actions(
                    call_llm=call_llm,
                    finish=finish,
                )
                .with_transitions(
                    ("call_llm", "finish"),
                )
                .with_entrypoint("call_llm")
                .with_state(**state_dict)
                .with_identifiers(app_id=app_id, partition_key=partition_key)
                .abuild()
            )

            return app, True  # cached=True

    # Cache miss - build app to run
    print(f"[ZERO-SHOT BURR] Cache MISS for partition_key={partition_key}")

    app = await (
        ApplicationBuilder()
        .with_actions(
            call_llm=call_llm,
            finish=finish,
        )
        .with_transitions(
            ("call_llm", "finish"),
        )
        .with_entrypoint("call_llm")
        .with_state(
            clinical_note=clinical_note,
            selected_codes=None,  # Will be set by call_llm
            reasoning=None,  # Will be set by call_llm
        )
        .with_state_persister(ZERO_SHOT_PERSISTER)
        .with_identifiers(app_id=app_id, partition_key=partition_key)
        .abuild()
    )

    return app, False  # cached=False


# ============================================================================
# High-level API
# ============================================================================


async def run_zero_shot(
    clinical_note: str,
    provider: str,
    model: str,
    temperature: float,
    system_prompt: str | None = None,
) -> tuple[list[str], str, bool]:
    """High-level API to run zero-shot code generation with caching.

    Handles persister initialization, cache checking, and execution.

    Args:
        clinical_note: The clinical note to analyze
        provider: LLM provider
        model: Model name
        temperature: Temperature setting
        system_prompt: Optional custom system prompt

    Returns:
        Tuple of (selected_codes, reasoning, was_cached)
    """
    # Initialize persister if needed
    if ZERO_SHOT_PERSISTER is None:
        await initialize_zero_shot_persister()

    # Generate cache key
    partition_key = generate_zero_shot_cache_key(
        clinical_note=clinical_note,
        provider=provider,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    # Build app (checks cache)
    app, cached = await build_zero_shot_app(
        clinical_note=clinical_note,
        partition_key=partition_key,
    )

    if cached:
        # Cache hit - return cached results
        return (
            app.state.get("selected_codes", []),
            app.state.get("reasoning", ""),
            True,
        )

    # Cache miss - run the app
    _, _, final_state = await app.arun(halt_after=["finish"])

    return (
        final_state.get("selected_codes", []),
        final_state.get("reasoning", ""),
        False,
    )
