"""LLM Configuration for ICD-10-CM code selection.

Supports OpenAI-compatible APIs: OpenAI, Cerebras, SambaNova, and custom endpoints.
"""

import hashlib
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM-based selector using OpenAI-compatible APIs.

    Supports OpenAI, Cerebras, SambaNova, and custom endpoints.

    Attributes:
        api_key: API authentication key
        base_url: API base URL (e.g., "https://api.openai.com/v1")
        model: Model identifier (e.g., "gpt-4o", "llama3.1-8b")
        temperature: Sampling temperature (0-2, lower = more deterministic)
        max_completion_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        provider: Provider name for logging/caching ("openai", "cerebras", "sambanova")
    """

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_completion_tokens: int = 8000
    timeout: float = 180.0
    provider: str = "openai"

    def settings_hash(self) -> str:
        """Return hash of settings that affect LLM output for cache invalidation."""
        settings = f"{self.provider}|{self.model}|{self.temperature}|{self.base_url}"
        return hashlib.md5(settings.encode()).hexdigest()[:8]


def configure_llm(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_completion_tokens: int = 8000,
    timeout: float = 180.0,
    provider: str = "openai",
    **kwargs,
) -> LLMConfig:
    """Configure LLM client with OpenAI-compatible API.

    All parameters default to OpenAI values.
    Provider parameter kept for metadata/caching only.

    Args:
        api_key: API authentication key (required)
        base_url: API base URL (default: OpenAI)
        model: Model identifier (default: gpt-4o)
        temperature: Sampling temperature (default: 0.1)
        max_completion_tokens: Maximum tokens to generate (default: 8000)
        timeout: Request timeout in seconds (default: 180.0)
        provider: Provider name for logging/caching (default: "openai")
        **kwargs: Additional parameters (ignored for forward compatibility)

    Returns:
        LLMConfig: Configured LLM configuration object

    Examples:
        # OpenAI
        config = configure_llm(api_key="sk-...")

        # Cerebras
        config = configure_llm(
            provider="cerebras",
            api_key="csk-...",
            base_url="https://api.cerebras.ai/v1",
            model="llama3.1-8b"
        )

        # SambaNova
        config = configure_llm(
            provider="sambanova",
            api_key="...",
            base_url="https://api.sambanova.ai/v1",
            model="Meta-Llama-3.1-8B-Instruct"
        )
    """
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        provider=provider,
    )


# Global LLM configuration - must be set before using llm_selector
LLM_CONFIG: LLMConfig | None = None
