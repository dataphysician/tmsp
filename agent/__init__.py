"""Agent orchestration module for ICD-10-CM traversal

Provides Burr-based orchestration replacing the previous Pydantic-AI implementation.
Supports multi-batch parallel fan-out for children + metadata relationships,
7th character handling, and SQLite persistence for retry/resume.

## Benchmark Streaming Architecture

The benchmark feature streams traversal events to the frontend via SSE (Server-Sent Events).
For cached traversals, special handling is required to prevent UI blocking:

### Live Traversal Flow
1. Server emits events as they occur (1-3 second intervals between batches)
2. Frontend processes each event immediately via `onEvent()` callback
3. React state updates incrementally, graph renders progressively

### Cached Replay Flow
1. Server detects cached state and sets `metadata.cached = true` in RUN_STARTED
2. Frontend (`api.ts`) detects cached flag and accumulates ALL events
3. On RUN_FINISHED, accumulated events are processed with EVENT LOOP YIELDING:
   - Process 10 events, then `await new Promise(resolve => setTimeout(resolve, 0))`
   - This prevents "Page Unresponsive" errors for 150+ cached events
4. Graph renders once with complete data, ensuring correct positioning

### Why Event Loop Yielding Matters
Without yielding, 150+ cached events processed synchronously block the main thread,
triggering Chrome's "Page Unresponsive" detection. Yielding every 10 events allows
the browser to process pending UI updates and handle user interactions.

See CLAUDE.md "Critical: Cached Traversal Node Positioning" for full implementation details.

## Example usage

    from agent import build_traversal_app, run_traversal, initialize_persister
    from candidate_selector import configure_llm
    import candidate_selector.config as llm_config

    # Configure LLM
    llm_config.LLM_CONFIG = configure_llm(
        provider="openai",
        api_key="sk-..."
    )

    # Option 1: High-level API
    result = await run_traversal(
        clinical_note="Patient with type 2 diabetes...",
        provider="openai",
        api_key="sk-..."
    )
    print(result["final_nodes"])

    # Option 2: Low-level API with more control
    await initialize_persister()
    app = await build_traversal_app(
        context="Patient with type 2 diabetes...",
        default_selector="llm"
    )
    _, _, state = await app.arun(halt_after=["finish"])
"""

from .actions import (
    BATCH_CALLBACK,
    ICD_INDEX,
    finish,
    finish_batch,
    format_with_seventh_char,
    inject_feedback,
    load_node,
    select_candidates,
    set_batch_callback,
)
from .parallel import SpawnParallelBatches, SpawnSevenChr
from .state_types import CandidateDecision, DecisionPoint, TraversalState
from .traversal import (
    PARTITION_KEY,
    PERSISTER,
    build_traversal_app,
    cleanup_persister,
    delete_persisted_state,
    generate_traversal_cache_key,
    get_descendant_batch_ids,
    initialize_persister,
    prune_state_for_rewind,
    retry_node,
    run_traversal,
)
from .zero_shot import (
    ZERO_SHOT_PERSISTER,
    build_zero_shot_app,
    cleanup_zero_shot_persister,
    generate_zero_shot_cache_key,
    initialize_zero_shot_persister,
    run_zero_shot,
)

__all__ = [
    # State Types
    "CandidateDecision",
    "DecisionPoint",
    "TraversalState",
    # Actions
    "load_node",
    "select_candidates",
    "inject_feedback",
    "finish_batch",
    "finish",
    "format_with_seventh_char",
    "set_batch_callback",
    "BATCH_CALLBACK",
    "ICD_INDEX",
    # Parallel
    "SpawnParallelBatches",
    "SpawnSevenChr",
    # Traversal
    "build_traversal_app",
    "retry_node",
    "run_traversal",
    "initialize_persister",
    "cleanup_persister",
    "delete_persisted_state",
    "generate_traversal_cache_key",
    "get_descendant_batch_ids",
    "prune_state_for_rewind",
    "PERSISTER",
    "PARTITION_KEY",
    # Zero-shot
    "build_zero_shot_app",
    "run_zero_shot",
    "generate_zero_shot_cache_key",
    "initialize_zero_shot_persister",
    "cleanup_zero_shot_persister",
    "ZERO_SHOT_PERSISTER",
]
