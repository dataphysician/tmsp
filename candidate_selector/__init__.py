"""Candidate Selector module for ICD-10-CM code selection

Provides extensible LLM-based and manual selector functions for use with
Burr orchestration of ICD-10-CM hierarchy traversal.

Supported providers:
- OpenAI (gpt-4o, gpt-4o-mini)
- Cerebras (llama3.1-8b)
- SambaNova (Meta-Llama-3.1-8B-Instruct)
- Custom OpenAI-compatible endpoints

Example usage:
    from candidate_selector import LLM_CONFIG, configure_llm, llm_selector

    # Configure provider
    LLM_CONFIG = configure_llm(
        provider="openai",
        api_key="sk-...",
        model="gpt-4o"
    )

    # Use selector in Burr action
    selected, reasoning = await llm_selector(
        batch_id="E11|children",
        context="Patient with type 2 diabetes...",
        candidates={"E11.0": "Type 2 diabetes with hyperosmolarity", ...}
    )
"""

from .config import LLM_CONFIG, LLMConfig, configure_llm
from .providers import (
    PROVIDER_MODELS,
    PROVIDER_URLS,
    cerebras_config,
    create_config,
    openai_config,
    sambanova_config,
    uses_strict_mode,
)
from .selector import (
    SELECTOR_CACHE,
    SELECTOR_REGISTRY,
    CodeSelectionResult,
    SelectorProtocol,
    clear_cache,
    get_selector,
    llm_selector,
    manual_selector,
)

__all__ = [
    # Config
    "LLM_CONFIG",
    "LLMConfig",
    "configure_llm",
    # Providers
    "PROVIDER_URLS",
    "PROVIDER_MODELS",
    "openai_config",
    "cerebras_config",
    "sambanova_config",
    "create_config",
    "uses_strict_mode",
    # Selector
    "SelectorProtocol",
    "CodeSelectionResult",
    "SELECTOR_CACHE",
    "SELECTOR_REGISTRY",
    "llm_selector",
    "manual_selector",
    "get_selector",
    "clear_cache",
]
