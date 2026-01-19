"""LLM Configuration for ICD-10-CM code selection

Supports OpenAI-compatible APIs: OpenAI, Cerebras, SambaNova, Anthropic, Vertex AI, and custom endpoints.
"""

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM-based selector using OpenAI-compatible APIs.

    Supports OpenAI, Cerebras, SambaNova, Anthropic, Vertex AI, and custom endpoints.

    Attributes:
        api_key: API authentication key (for Vertex AI, this is the access token)
        base_url: API base URL (e.g., "https://api.openai.com/v1")
        model: Model identifier (e.g., "gpt-4o", "gemini-3-flash-preview")
        temperature: Sampling temperature (0-2, lower = more deterministic)
        max_completion_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        provider: Provider name for logging/caching
        extra: Provider-specific configuration (e.g., {"location": "us-central1", "project_id": "my-project"} for Vertex AI)
        system_prompt: Custom system prompt (uses default LLM_SYSTEM_PROMPT if None)
    """

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_completion_tokens: int = 8000
    timeout: float = 180.0
    provider: str = "openai"
    extra: dict[str, str] | None = field(default=None)
    system_prompt: str | None = field(default=None)

    def settings_hash(self) -> str:
        """Return hash of settings that affect LLM output for cache invalidation."""
        settings = f"{self.provider}|{self.model}|{self.temperature}|{self.base_url}"
        if self.extra:
            settings += f"|{json.dumps(self.extra, sort_keys=True)}"
        if self.system_prompt:
            settings += f"|{self.system_prompt[:100]}"  # Include first 100 chars of custom prompt
        return hashlib.md5(settings.encode()).hexdigest()[:8]


def configure_llm(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_completion_tokens: int = 8000,
    timeout: float = 180.0,
    provider: str = "openai",
    extra: dict[str, str] | None = None,
    system_prompt: str | None = None,
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
        extra: Provider-specific configuration (e.g., Vertex AI location/project_id)
        system_prompt: Custom system prompt (uses default LLM_SYSTEM_PROMPT if None)
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

        # Vertex AI
        config = configure_llm(
            provider="vertexai",
            api_key="<access_token>",
            model="gemini-3-flash-preview",
            extra={"location": "us-central1", "project_id": "my-project"}
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
        extra=extra,
        system_prompt=system_prompt,
    )


# Global LLM configuration - must be set before using llm_selector
LLM_CONFIG: LLMConfig | None = None
