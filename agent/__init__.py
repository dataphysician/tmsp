"""Agent orchestration module for ICD-10-CM traversal

Provides Burr-based orchestration replacing the previous Pydantic-AI implementation.
Supports multi-batch parallel fan-out for children + metadata relationships,
7th character handling, and SQLite persistence for retry/resume.

Example usage:
    from agent import build_app, run_traversal, initialize_persister
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
    app = await build_app(
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
    load_node,
    select_candidates,
    set_batch_callback,
)
from .parallel import SpawnParallelBatches, SpawnSevenChr
from .state_types import CandidateDecision, DecisionPoint, TraversalState
from .traversal import (
    PARTITION_KEY,
    PERSISTER,
    build_app,
    cleanup_persister,
    generate_traversal_cache_key,
    initialize_persister,
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
    "build_app",
    "retry_node",
    "run_traversal",
    "initialize_persister",
    "cleanup_persister",
    "generate_traversal_cache_key",
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
