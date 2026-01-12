"""Provider presets for OpenAI-compatible LLM APIs.

Each provider has specific configurations:
- OpenAI: strict=true (100% schema accuracy)
- Cerebras: strict=true (5000 char limit, no recursion)
- SambaNova: strict=false (best-effort compliance)
"""

from .config import LLMConfig


# Provider base URLs
PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "other": "",  # User must provide base URL
}

# Default models per provider
PROVIDER_MODELS = {
    "openai": "gpt-4o-mini",
    "cerebras": "gpt-oss-120b",
    "sambanova": "Meta-Llama-3.1-8B-Instruct",
    "anthropic": "claude-opus-4-5",
    "other": "",  # User must provide model name
}

# Providers that support strict mode for structured outputs
STRICT_MODE_PROVIDERS = {"openai", "cerebras"}

# Providers that use non-OpenAI-compatible API format
ANTHROPIC_STYLE_PROVIDERS = {"anthropic"}


def openai_config(
    api_key: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_completion_tokens: int = 8000,
    timeout: float = 180.0,
) -> LLMConfig:
    """Create OpenAI configuration.

    Args:
        api_key: OpenAI API key (sk-...)
        model: Model identifier (default: gpt-4o)
        temperature: Sampling temperature (default: 0.1)
        max_completion_tokens: Max tokens to generate (default: 8000)
        timeout: Request timeout in seconds (default: 180.0)

    Returns:
        LLMConfig configured for OpenAI
    """
    return LLMConfig(
        api_key=api_key,
        base_url=PROVIDER_URLS["openai"],
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        provider="openai",
    )


def cerebras_config(
    api_key: str,
    model: str = "llama3.1-8b",
    temperature: float = 0.1,
    max_completion_tokens: int = 8000,
    timeout: float = 180.0,
) -> LLMConfig:
    """Create Cerebras configuration.

    Note: Cerebras has a 5000 character schema limit and no recursive schemas.

    Args:
        api_key: Cerebras API key (csk-...)
        model: Model identifier (default: llama3.1-8b)
        temperature: Sampling temperature (default: 0.1)
        max_completion_tokens: Max tokens to generate (default: 8000)
        timeout: Request timeout in seconds (default: 180.0)

    Returns:
        LLMConfig configured for Cerebras
    """
    return LLMConfig(
        api_key=api_key,
        base_url=PROVIDER_URLS["cerebras"],
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        provider="cerebras",
    )


def sambanova_config(
    api_key: str,
    model: str = "Meta-Llama-3.1-8B-Instruct",
    temperature: float = 0.1,
    max_completion_tokens: int = 8000,
    timeout: float = 180.0,
) -> LLMConfig:
    """Create SambaNova configuration.

    Note: SambaNova uses best-effort schema compliance (strict=false).

    Args:
        api_key: SambaNova API key
        model: Model identifier (default: Meta-Llama-3.1-8B-Instruct)
        temperature: Sampling temperature (default: 0.1)
        max_completion_tokens: Max tokens to generate (default: 8000)
        timeout: Request timeout in seconds (default: 180.0)

    Returns:
        LLMConfig configured for SambaNova
    """
    return LLMConfig(
        api_key=api_key,
        base_url=PROVIDER_URLS["sambanova"],
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        provider="sambanova",
    )


def anthropic_config(
    api_key: str,
    model: str = "claude-opus-4-5",
    temperature: float = 0.0,
    max_completion_tokens: int = 64000,
    timeout: float = 180.0,
) -> LLMConfig:
    """Create Anthropic configuration.

    Note: Anthropic uses a different API format than OpenAI:
    - System message goes in separate 'system' parameter, not in messages array
    - Uses 'x-api-key' header instead of 'Authorization: Bearer'
    - Uses 'anthropic-version' header

    Args:
        api_key: Anthropic API key (sk-ant-...)
        model: Model identifier (default: claude-opus-4-5)
        temperature: Sampling temperature (default: 0.0 for deterministic output)
        max_completion_tokens: Max tokens to generate (default: 64000)
        timeout: Request timeout in seconds (default: 180.0)

    Returns:
        LLMConfig configured for Anthropic
    """
    return LLMConfig(
        api_key=api_key,
        base_url=PROVIDER_URLS["anthropic"],
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        provider="anthropic",
    )


def other_config(
    api_key: str,
    model: str,
    base_url: str = "",
    temperature: float = 0.1,
    max_completion_tokens: int = 8000,
    timeout: float = 180.0,
) -> LLMConfig:
    """Create configuration for other OpenAI-compatible providers.

    Use this for providers not explicitly supported. Assumes OpenAI-compatible API.

    Args:
        api_key: API authentication key
        model: Model identifier (required)
        base_url: API base URL (required for other providers)
        temperature: Sampling temperature (default: 0.1)
        max_completion_tokens: Max tokens to generate (default: 8000)
        timeout: Request timeout in seconds (default: 180.0)

    Returns:
        LLMConfig configured for OpenAI-compatible API
    """
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        provider="other",
    )


def create_config(
    provider: str,
    api_key: str,
    model: str | None = None,
    **kwargs,
) -> LLMConfig:
    """Create LLM configuration for a named provider.

    Factory function that dispatches to provider-specific config functions.

    Args:
        provider: Provider name ("openai", "cerebras", "sambanova")
        api_key: API authentication key
        model: Optional model override (uses provider default if None)
        **kwargs: Additional config options (temperature, timeout, etc.)

    Returns:
        LLMConfig for the specified provider

    Raises:
        ValueError: If provider is unknown

    Example:
        config = create_config("openai", api_key="sk-...", model="gpt-4o-mini")
    """
    provider = provider.lower()

    if provider not in PROVIDER_URLS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: {', '.join(PROVIDER_URLS.keys())}"
        )

    # Use provider default model if not specified
    if model is None:
        model = PROVIDER_MODELS[provider]

    config_funcs = {
        "openai": openai_config,
        "cerebras": cerebras_config,
        "sambanova": sambanova_config,
        "anthropic": anthropic_config,
        "other": other_config,
    }

    return config_funcs[provider](api_key=api_key, model=model, **kwargs)


def uses_strict_mode(provider: str) -> bool:
    """Check if provider supports strict mode for structured outputs.

    Args:
        provider: Provider name

    Returns:
        True if provider supports strict=true in response_format
    """
    return provider.lower() in STRICT_MODE_PROVIDERS


def uses_anthropic_api(provider: str) -> bool:
    """Check if provider uses Anthropic-style API format.

    Anthropic API differs from OpenAI in:
    - System message in separate 'system' parameter
    - Uses 'x-api-key' header instead of 'Authorization: Bearer'
    - Requires 'anthropic-version' header
    - Messages array only contains user/assistant roles

    Args:
        provider: Provider name

    Returns:
        True if provider uses Anthropic-style API
    """
    return provider.lower() in ANTHROPIC_STYLE_PROVIDERS
