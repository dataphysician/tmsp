"""Async selector functions for ICD-10-CM code selection.

Provides llm_selector (API-based) and manual_selector (interactive CLI).
"""

import hashlib
import json
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from . import config as llm_config
from .config import LLMConfig
from .providers import uses_strict_mode, uses_anthropic_api


# ============================================================================
# Selector Protocol
# ============================================================================


class SelectorProtocol(Protocol):
    """Protocol for selector functions that conform to expected signature.

    All selectors must accept:
    - batch_id: Unique identifier for current batch (e.g., "B19.2|children")
    - context: Guiding string from state for selector decisions
    - candidates: Dict of {node_id: label} representing available choices
    - feedback: Optional feedback string for selector adjustment

    And must return:
    - tuple[list[str], str]: (selected node_ids, reasoning)
    """

    async def __call__(
        self,
        batch_id: str,
        context: str,
        candidates: dict[str, str],
        feedback: str | None,
    ) -> tuple[list[str], str]: ...


# ============================================================================
# Structured Output Model
# ============================================================================


class CodeSelectionResult(BaseModel):
    """ICD-10 code selection result with structured output.

    Used with LLM response_format (json_schema) for guaranteed structure.
    LLM returns reasoning first (for chain-of-thought), then selected codes.

    Field order matters: reasoning first enables better auto-regressive generation.
    """

    reasoning: str = Field(
        description=(
            "Brief explanation for why these codes were selected or not selected "
            "(1-3 sentences). Generate this first to inform code selection."
        )
    )
    selected_codes: list[str] = Field(
        description=(
            "List of selected ICD-10 code IDs (0 to N codes). "
            "Return empty list if no codes are clinically appropriate."
        )
    )


# ============================================================================
# Cache
# ============================================================================


# Selector cache for consistent selections across parallel branches
# Stores tuple[list[str], str] for (selected_codes, reasoning)
SELECTOR_CACHE: dict[str, tuple[list[str], str]] = {}


def clear_cache() -> None:
    """Clear the selector cache."""
    SELECTOR_CACHE.clear()


# ============================================================================
# System Prompt
# ============================================================================


LLM_SYSTEM_PROMPT = """You are an expert ICD-10-CM medical coding assistant. Your task is to select the most clinically relevant ICD-10-CM codes (or chapters) from a list of candidates based on the provided clinical context.

You are helping traverse the ICD-10-CM hierarchy. When the CURRENT CODE is "ROOT", you are selecting which chapter(s) to explore. Otherwise, you are selecting specific codes or subcategories.

RULES:
1. Select any number (0 to N) of codes/chapters that are clinically relevant to the context
2. Consider the relationship type (children, codeFirst, codeAlso, useAdditionalCode):
   - useAdditionalCode: These codes SHOULD be selected if the condition is present, regardless of whether it's acute or chronic
   - codeFirst/codeAlso: Consider coding guidelines and clinical relevance
3. Prioritize clinical accuracy and specificity
4. If no codes are clinically relevant, return an empty list
5. IMPORTANT: When at ROOT level, select the chapter(s) that contain relevant codes for the clinical context
6. IMPORTANT: When user provides ADDITIONAL GUIDANCE or feedback, prioritize that information over general clinical context
7. Think step-by-step: First generate reasoning, then select codes based on that reasoning
8. Return your response as a JSON object with "reasoning" field FIRST, then "selected_codes" field

RESPONSE FORMAT:
You must return a valid JSON object with this exact structure (reasoning FIRST):
{"reasoning": "brief explanation", "selected_codes": ["code1", "code2", ...]}

Examples:
- Multiple selections: {"reasoning": "Selected specific hepatitis C codes based on chronic liver disease context.", "selected_codes": ["B19.20", "B19.21", "B19.22"]}
- Single selection: {"reasoning": "Type 2 diabetes with hyperglycemia matches the clinical presentation.", "selected_codes": ["E11.65"]}
- Selecting chapter: {"reasoning": "Endocrine disorders chapter relevant for diabetes context.", "selected_codes": ["Chapter_04"]}
- No selections: {"reasoning": "No infectious disease codes are clinically appropriate for this metabolic condition.", "selected_codes": []}

Do not include any text outside the JSON object."""


# ============================================================================
# Helper Functions
# ============================================================================


def _build_user_prompt(
    batch_id: str,
    context: str,
    candidates: dict[str, str],
    feedback: str | None,
) -> str:
    """Build user prompt with clinical context and candidate codes.

    Args:
        batch_id: Batch identifier (e.g., "B19.2|children")
        context: Clinical context string from state
        candidates: Dict of {node_id: label}
        feedback: Optional additional guidance

    Returns:
        Formatted prompt string with context and "code - label" format
    """
    # Parse batch_id to extract node and relationship
    if "|" in batch_id:
        node_id, relationship = batch_id.rsplit("|", 1)
    else:
        node_id = batch_id
        relationship = "children"

    # Format candidates as "code - label" lines
    candidates_text = "\n".join(
        f"{code} - {label}" for code, label in candidates.items()
    )

    # Build prompt with feedback FIRST if provided (highest priority)
    prompt_parts = []

    # FEEDBACK FIRST - most important information
    if feedback:
        prompt_parts.append(
            f"CRITICAL ADDITIONAL GUIDANCE (MUST PRIORITIZE):\n{feedback}\n"
        )

    if context:
        prompt_parts.append(f"CLINICAL CONTEXT:\n{context}\n")

    prompt_parts.extend(
        [
            f"CURRENT CODE: {node_id}",
            f"RELATIONSHIP TYPE: {relationship}\n",
            f"AVAILABLE CANDIDATES:\n{candidates_text}",
        ]
    )

    prompt_parts.append(
        "\nSelect clinically appropriate codes (0 to N codes). "
        "Return empty list if none are appropriate."
    )

    return "\n".join(prompt_parts)


def _sanitize_schema_for_strict_mode(schema: dict) -> dict:
    """Add additionalProperties: false for OpenAI/Cerebras strict mode.

    OpenAI strict mode requires all objects to have additionalProperties=false.
    Pydantic's model_json_schema() doesn't always include this recursively.

    Args:
        schema: JSON schema dict from Pydantic

    Returns:
        Sanitized schema with additionalProperties: false on all objects
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            if isinstance(value, (dict, list)):
                _sanitize_schema_for_strict_mode(value)
    elif isinstance(schema, list):
        for item in schema:
            _sanitize_schema_for_strict_mode(item)
    return schema


async def _call_anthropic_api_structured(
    system_prompt: str,
    user_messages: list[dict[str, str]],
    config: LLMConfig,
) -> tuple[list[str], str]:
    """Make async Anthropic API call with structured outputs.

    Anthropic API differs from OpenAI:
    - System message in separate 'system' parameter
    - Uses 'x-api-key' header instead of 'Authorization: Bearer'
    - Requires 'anthropic-version' header
    - Requires 'anthropic-beta' header for structured outputs
    - Uses 'output_format' instead of 'response_format'
    - Endpoint is /messages instead of /chat/completions

    Args:
        system_prompt: System prompt (separate from messages)
        user_messages: User/assistant messages only
        config: LLM configuration

    Returns:
        Tuple of (selected_codes, reasoning)

    Raises:
        Exception: On API errors, validation failures, or timeouts
    """
    # Generate JSON schema from Pydantic model
    schema = CodeSelectionResult.model_json_schema()

    # Sanitize schema for Anthropic (requires additionalProperties: false)
    schema = _sanitize_schema_for_strict_mode(schema)

    # Prepare timeout configuration
    timeout = httpx.Timeout(
        connect=5.0,
        read=config.timeout,
        write=5.0,
        pool=5.0,
    )

    # Anthropic-specific headers (including beta header for structured outputs)
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "structured-outputs-2025-11-13",
        "Content-Type": "application/json",
    }

    # Prepare Anthropic payload with structured output format
    payload: dict = {
        "model": config.model,
        "max_tokens": config.max_completion_tokens,
        "system": system_prompt,
        "messages": user_messages,
        "output_format": {
            "type": "json_schema",
            "schema": schema,
        },
    }

    # Only add temperature if not 1.0 (Anthropic default)
    if config.temperature != 1.0:
        payload["temperature"] = config.temperature

    endpoint = f"{config.base_url.rstrip('/')}/messages"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()

            # Anthropic response structure: {"content": [{"type": "text", "text": "..."}]}
            content_blocks = result.get("content", [])
            if not content_blocks:
                raise Exception("Empty response from Anthropic API")

            # Get text from first text block
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content = block.get("text", "")
                    break

            if not content:
                raise Exception("No text content in Anthropic response")

            # Parse JSON string
            parsed = json.loads(content)

            # Validate with Pydantic
            validated = CodeSelectionResult.model_validate(parsed)

            # Return selected codes and reasoning
            return (validated.selected_codes, validated.reasoning)

    except httpx.TimeoutException as e:
        raise Exception(f"Anthropic API timeout after {config.timeout}s: {e!s}")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text if e.response else "No response"
        raise Exception(f"Anthropic API HTTP {e.response.status_code}: {error_detail}")
    except httpx.RequestError as e:
        raise Exception(f"Anthropic API network error: {e!s}")
    except json.JSONDecodeError as e:
        raise Exception(f"Anthropic response not valid JSON: {e!s}")
    except Exception as e:
        raise Exception(f"Anthropic API call failed: {e!s}")


async def _call_llm_api_structured(
    messages: list[dict[str, str]],
    config: LLMConfig,
) -> tuple[list[str], str]:
    """Make async LLM API call with structured outputs (Pydantic response_format).

    Uses CodeSelectionResult Pydantic model for guaranteed JSON structure.

    Provider differences:
    - OpenAI: strict=true (100% accuracy, requires schema sanitization)
    - Cerebras: strict=true (5000 char limit, no recursion)
    - SambaNova: strict=false (best-effort compliance)
    - Anthropic: Uses separate API format (dispatched to _call_anthropic_api_structured)

    Args:
        messages: Chat messages (system + user)
        config: LLM configuration

    Returns:
        Tuple of (selected_codes, reasoning)

    Raises:
        Exception: On API errors, validation failures, or timeouts
    """
    # Dispatch to Anthropic API if using Anthropic provider
    if uses_anthropic_api(config.provider):
        # Extract system message and user messages
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)
        return await _call_anthropic_api_structured(system_prompt, user_messages, config)

    # OpenAI-compatible API flow
    # Generate JSON schema from Pydantic model
    schema = CodeSelectionResult.model_json_schema()

    # Determine strict mode based on provider
    use_strict = uses_strict_mode(config.provider)

    # Sanitize schema for strict mode
    if use_strict:
        schema = _sanitize_schema_for_strict_mode(schema)

    # Prepare timeout configuration
    timeout = httpx.Timeout(
        connect=5.0,
        read=config.timeout,
        write=5.0,
        pool=5.0,
    )

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    # Prepare payload with response_format
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "code_selection_result",
                "strict": use_strict,
                "schema": schema,
            },
        },
    }

    endpoint = f"{config.base_url.rstrip('/')}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()

            # Extract content (contains JSON string with structured output)
            content = result["choices"][0]["message"]["content"]

            # Parse JSON string
            parsed = json.loads(content)

            # Validate with Pydantic
            validated = CodeSelectionResult.model_validate(parsed)

            # Return selected codes and reasoning
            return (validated.selected_codes, validated.reasoning)

    except httpx.TimeoutException as e:
        raise Exception(f"LLM API timeout after {config.timeout}s: {e!s}")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text if e.response else "No response"
        raise Exception(f"LLM API HTTP {e.response.status_code}: {error_detail}")
    except httpx.RequestError as e:
        raise Exception(f"LLM API network error: {e!s}")
    except json.JSONDecodeError as e:
        raise Exception(f"LLM response not valid JSON: {e!s}")
    except Exception as e:
        raise Exception(f"LLM API call failed: {e!s}")


# ============================================================================
# Selector Functions
# ============================================================================


async def llm_selector(
    batch_id: str,
    context: str,
    candidates: dict[str, str],
    feedback: str | None = None,
) -> tuple[list[str], str]:
    """LLM-based candidate selection using OpenAI-compatible APIs with structured outputs.

    Makes async API calls to LLM (OpenAI, Cerebras, SambaNova, or custom)
    to select 0 to N most clinically relevant ICD-10-CM codes from candidates.

    Features:
    - Structured outputs with Pydantic CodeSelectionResult model
    - Async httpx API calls for non-blocking execution
    - Provider-specific strict mode
    - Caching to reduce API costs and ensure consistency
    - Graceful error handling with empty selection

    Args:
        batch_id: Unique batch identifier (e.g., "B19.2|children")
        context: Clinical context string from state
        candidates: Dict of {node_id: label} representing available choices
        feedback: Optional feedback string for LLM adjustment

    Returns:
        Tuple of (selected_codes, reasoning)

    Raises:
        ValueError: If LLM_CONFIG is not set
    """
    if not candidates:
        return ([], "No candidates available")

    # Get LLM configuration
    config = llm_config.LLM_CONFIG

    # Check if API key is configured
    if not config or not config.api_key:
        raise ValueError(
            "LLM API key is not configured. "
            "Set candidate_selector.config.LLM_CONFIG before running traversal."
        )

    # Create cache key
    node_id = batch_id.rsplit("|", 1)[0] if "|" in batch_id else batch_id
    candidate_keys = tuple(sorted(candidates.keys()))
    context_hash = hashlib.md5(context.encode()).hexdigest()[:8] if context else "nocontext"
    candidate_hash = hashlib.md5(str(candidate_keys).encode()).hexdigest()[:8]
    feedback_part = feedback[:50] if feedback else ""
    cache_key = f"{node_id}|{context_hash}|{candidate_hash}|{feedback_part}|{config.settings_hash()}"

    # Check cache first
    if cache_key in SELECTOR_CACHE:
        cached_codes, cached_reasoning = SELECTOR_CACHE[cache_key]
        print(f"[LLM SELECTOR CACHE HIT] {node_id} -> {cached_codes}")
        return (cached_codes, cached_reasoning)

    print(f"[LLM SELECTOR] Provider: {config.provider}, Model: {config.model}, Base URL: {config.base_url}")
    print(f"[LLM SELECTOR] Batch: {batch_id}, Candidates: {len(candidates)}")
    if feedback:
        print(f"[LLM SELECTOR] Feedback provided: '{feedback[:100]}...'")

    try:
        # Build messages
        user_prompt = _build_user_prompt(batch_id, context, candidates, feedback)

        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # If feedback provided, add emphasis message
        if feedback:
            emphasis_message = (
                "CRITICAL OVERRIDE INSTRUCTION:\n\n"
                f"The user has provided specific guidance that MUST take priority:\n\n"
                f"{feedback}\n\n"
                "You MUST incorporate this guidance into your code selection."
            )
            messages.append({"role": "user", "content": emphasis_message})

        # Call LLM API with structured outputs
        selected, reasoning = await _call_llm_api_structured(messages, config)

        # Validate selections against actual candidates
        valid_selected = [code for code in selected if code in candidates]

        if len(valid_selected) != len(selected):
            invalid = set(selected) - set(valid_selected)
            print(f"[LLM SELECTOR WARNING] Invalid codes filtered: {invalid}")

        print(f"[LLM SELECTOR] Selected: {valid_selected}")
        print(f"[LLM SELECTOR] Reasoning: {reasoning}")

        # Cache the result
        SELECTOR_CACHE[cache_key] = (valid_selected, reasoning)

        return (valid_selected, reasoning)

    except Exception as e:
        import traceback
        error_message = str(e)
        print(f"[LLM SELECTOR ERROR] {error_message}")
        print("[LLM SELECTOR TRACEBACK]")
        traceback.print_exc()
        print("[LLM SELECTOR] Returning empty selection")

        reasoning = f"API Error: {error_message}. Returned empty selection."
        return ([], reasoning)


async def manual_selector(
    batch_id: str,
    context: str,
    candidates: dict[str, str],
    feedback: str | None = None,
) -> tuple[list[str], str]:
    """Interactive manual/human-in-loop selection using input() prompts.

    Displays candidates and prompts user to select via comma-separated indices or IDs.
    Also prompts for reasoning text to document the selection decision.

    Args:
        batch_id: Unique batch identifier (e.g., "B19.2|children")
        context: Clinical context string to guide manual review
        candidates: Dict of {node_id: label} representing available choices
        feedback: Optional feedback string (if provided, parses as selection)

    Returns:
        Tuple of (selected_ids, reasoning)
    """
    if not candidates:
        return ([], "No candidates available")

    # If feedback is provided (retry scenario), parse it instead of interactive prompt
    if feedback:
        print(f"\n[MANUAL SELECTOR] Using feedback for batch: {batch_id}")
        print(f"[MANUAL SELECTOR] Feedback: {feedback}")
        try:
            # Parse feedback as comma-separated node_ids
            parts = [p.strip() for p in feedback.split(",")]
            selected = [p for p in parts if p in candidates]

            if selected:
                print(f"[MANUAL SELECTOR] Parsed selection: {selected}")
                return (selected, f"Manual selection from feedback: {feedback}")
        except Exception as e:
            print(f"[MANUAL SELECTOR] Error parsing feedback: {e}")

        return ([], f"Invalid feedback: {feedback}")

    # Interactive selection mode
    print(f"\n{'=' * 70}")
    print("MANUAL SELECTION REQUIRED")
    print(f"{'=' * 70}")
    print(f"Batch ID: {batch_id}")
    print(f"Context: {context[:200]}{'...' if len(context) > 200 else ''}")
    print(f"\nAvailable Candidates ({len(candidates)}):")
    print("-" * 70)

    # Display candidates with index numbers
    candidate_list = list(candidates.items())
    for idx, (code, description) in enumerate(candidate_list, 1):
        print(f"  {idx:2d}. {code:12s} - {description}")

    print("-" * 70)

    # Prompt for selection
    print("\nEnter selection:")
    print("  - By index: '1,3,5' (comma-separated numbers)")
    print("  - By ID: 'A00.0,A00.1' (comma-separated codes)")
    print("  - Leave empty to skip all candidates")

    try:
        selection_input = input("\nYour selection: ").strip()

        if not selection_input:
            # Empty selection
            reasoning_input = input("Reasoning for skipping: ").strip()
            return ([], reasoning_input or "Manual decision: skip all candidates")

        # Try parsing as indices first
        selected_ids = []
        try:
            # Check if input looks like indices (numbers)
            if all(part.strip().isdigit() for part in selection_input.split(",")):
                indices = [int(x.strip()) - 1 for x in selection_input.split(",")]
                selected_ids = [
                    candidate_list[i][0]
                    for i in indices
                    if 0 <= i < len(candidate_list)
                ]
            else:
                # Parse as IDs
                parts = [p.strip() for p in selection_input.split(",")]
                selected_ids = [p for p in parts if p in candidates]
        except (ValueError, IndexError) as e:
            print(f"Invalid input format: {e}")
            return ([], f"Error: Invalid selection format - {selection_input}")

        # Prompt for reasoning
        print(f"\nSelected {len(selected_ids)} candidate(s): {selected_ids}")
        reasoning_input = input("Reasoning for this selection: ").strip()

        return (
            selected_ids,
            reasoning_input or f"Manual selection: {', '.join(selected_ids)}",
        )

    except (EOFError, KeyboardInterrupt):
        print("\n[MANUAL SELECTOR] Selection interrupted")
        return ([], "Selection interrupted by user")
    except Exception as e:
        print(f"[MANUAL SELECTOR] Error during selection: {e}")
        return ([], f"Error during manual selection: {e}")


# ============================================================================
# Selector Registry
# ============================================================================


SELECTOR_REGISTRY: dict[str, SelectorProtocol] = {
    "llm": llm_selector,
    "manual": manual_selector,
}


def get_selector(name: str) -> SelectorProtocol:
    """Get a selector function by name.

    Args:
        name: Selector name ("llm" or "manual")

    Returns:
        Selector function conforming to SelectorProtocol

    Raises:
        ValueError: If selector name is unknown
    """
    if name not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selector: {name}. "
            f"Available: {', '.join(SELECTOR_REGISTRY.keys())}"
        )
    return SELECTOR_REGISTRY[name]
