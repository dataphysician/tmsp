"""Agent orchestration module for ICD-10-CM traversal.

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
    PERSISTER,
    build_app,
    cleanup_persister,
    initialize_persister,
    retry_node,
    run_traversal,
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
    "PERSISTER",
]
