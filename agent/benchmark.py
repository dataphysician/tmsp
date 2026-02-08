"""Headless benchmark for ICD-10-CM code comparison.

Compares expected codes against LLM traversal results, computing
code-level and path-level metrics.

Example:
    from agent.benchmark import run_benchmark

    code_metrics, path_metrics = await run_benchmark(
        clinical_note="Patient with type 2 diabetes...",
        expected_codes=["E11.9", "E11.65"],
        provider="openai",
        api_key="sk-...",
    )
    print(f"Recall: {code_metrics['recall']:.2%}")
    print(f"Path recall: {path_metrics['recall']:.2%}")
    print(f"Missed codes: {code_metrics['missed']}")
    print(f"Missed nodes: {path_metrics['missed']}")
"""

from agent import run_traversal, run_zero_shot
from graph import build_graph, data as icd_index, resolve_code


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def shares_ancestry(code_a: str, code_b: str) -> bool:
    """Fast prefix check for potential ancestry relationship."""
    norm_a = code_a.replace(".", "")
    norm_b = code_b.replace(".", "")
    return norm_a.startswith(norm_b) or norm_b.startswith(norm_a)


def _get_depth(code: str, index: dict[str, dict]) -> int | None:
    """Get depth for any code, including 7th-char codes not in the index."""
    if code in index:
        return index[code]["depth"]
    resolved = resolve_code(code, index)
    if resolved is not None:
        return index[resolved]["depth"] + 1
    return None


def get_ancestry_relationship(
    code_a: str,
    code_b: str,
    index: dict[str, dict],
) -> tuple[str, int]:
    """Determine ancestry relationship between two codes.

    Returns:
        ("exact", 0)                - same code
        ("ancestor", depth_diff)    - code_a is ancestor of code_b
        ("descendant", depth_diff)  - code_a is descendant of code_b
        ("unrelated", 0)            - no ancestry relationship
    """
    depth_a = _get_depth(code_a, index)
    depth_b = _get_depth(code_b, index)

    if depth_a is None or depth_b is None or not shares_ancestry(code_a, code_b):
        return ("unrelated", 0)

    depth_diff = depth_a - depth_b

    match depth_diff:
        case 0:
            return ("exact", 0)
        case d if d < 0:
            return ("ancestor", abs(d))
        case _:
            return ("descendant", depth_diff)


# ─────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────


def compare_results(
    expected_codes: set[str],
    traversed_codes: set[str],
    expected_nodes: set[str],
    traversed_nodes: set[str],
    index: dict[str, dict],
) -> tuple[dict, dict]:
    """Compare expected vs traversed at both code and path levels.

    Undershoot/overshoot is computed once from finalized codes only
    (no sense comparing intermediate nodes for depth deviation).

    Returns:
        (code_metrics, path_metrics) tuple with identical keys:
        exact, missed, other, undershoot, overshoot, recall, precision
    """
    # --- Finalized code comparison ---
    exact = expected_codes & traversed_codes
    missed_codes = expected_codes - traversed_codes
    other = traversed_codes - expected_codes

    undershoot: dict[str, dict] = {}
    overshoot: dict[str, dict] = {}

    for exp in missed_codes:
        for trav in other:
            rel, depth_diff = get_ancestry_relationship(trav, exp, index)
            match rel:
                case "ancestor":
                    undershoot[exp] = {"undershot": trav, "depth_diff": depth_diff}
                    break
                case "descendant":
                    overshoot[exp] = {"overshot": trav, "depth_diff": depth_diff}
                    break

    code_metrics = {
        "exact": exact,
        "missed": missed_codes,
        "other": other,
        "undershoot": undershoot,
        "overshoot": overshoot,
        "recall": len(exact) / len(expected_codes) if expected_codes else 0.0,
        "precision": len(exact) / len(traversed_codes) if traversed_codes else 0.0,
    }

    # --- Full path comparison (same keys, node-level sets) ---
    path_exact = expected_nodes & traversed_nodes
    path_missed = expected_nodes - traversed_nodes
    path_other = traversed_nodes - expected_nodes

    path_metrics = {
        "exact": path_exact,
        "missed": path_missed,
        "other": path_other,
        "undershoot": undershoot,
        "overshoot": overshoot,
        "recall": len(path_exact) / len(expected_nodes) if expected_nodes else 0.0,
        "precision": len(path_exact) / len(traversed_nodes) if traversed_nodes else 0.0,
    }

    return code_metrics, path_metrics


# ─────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────


async def run_benchmark(
    clinical_note: str,
    expected_codes: list[str],
    provider: str,
    api_key: str = "",
    model: str | None = None,
    temperature: float = 0.0,
    scaffolded: bool = True,
    **kwargs,
) -> tuple[dict, dict]:
    """Run benchmark comparison.

    Args:
        clinical_note: Clinical note to process
        expected_codes: Expected finalized codes (ground truth)
        provider: LLM provider
        api_key: API key for provider
        model: Model name (required for zero-shot)
        temperature: Temperature setting
        scaffolded: True for scaffolded traversal, False for zero-shot
        **kwargs: Additional arguments:
            selector: Selector type (scaffolded only, default "llm")
            system_prompt: Custom system prompt
            max_tokens: Max completion tokens (scaffolded only)
            use_cache: Enable caching (default True)

    Returns:
        (code_metrics, path_metrics) tuple - both have same keys:
        exact, missed, other, undershoot, overshoot, recall, precision
    """
    if scaffolded:
        result = await run_traversal(
            clinical_note=clinical_note,
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            selector=kwargs.get("selector", "llm"),
            system_prompt=kwargs.get("system_prompt"),
            max_tokens=kwargs.get("max_tokens"),
            use_cache=kwargs.get("use_cache", True),
        )
        traversed_codes = set(result["final_nodes"])
        batch_data = result.get("batch_data", {})
        traversed_nodes = {
            bd["node_id"]
            for bd in batch_data.values()
            if bd.get("node_id")
        }

    else:
        # Configure LLM (run_zero_shot doesn't do this internally)
        from candidate_selector.providers import create_config
        import candidate_selector.config as llm_config

        config_kwargs: dict = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if kwargs.get("max_tokens") is not None:
            config_kwargs["max_completion_tokens"] = kwargs["max_tokens"]

        llm_config.LLM_CONFIG = create_config(
            provider=provider,
            api_key=api_key,
            model=model,
            system_prompt=kwargs.get("system_prompt"),
            **config_kwargs,
        )

        codes, reasoning, cached = await run_zero_shot(
            clinical_note=clinical_note,
            provider=provider,
            model=model or llm_config.LLM_CONFIG.model,
            temperature=temperature,
            system_prompt=kwargs.get("system_prompt"),
        )
        traversed_codes = set(codes)

        # Infer ancestry paths from finalized codes
        traversed_graph = build_graph(list(traversed_codes), index=icd_index)
        traversed_nodes = traversed_graph["nodes"]

    # Build expected graph for path comparison
    expected_graph = build_graph(expected_codes, index=icd_index)
    expected_nodes = expected_graph["nodes"]

    return compare_results(
        expected_codes=set(expected_codes),
        traversed_codes=traversed_codes,
        expected_nodes=expected_nodes,
        traversed_nodes=traversed_nodes,
        index=icd_index,
    )
