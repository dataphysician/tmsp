#!/usr/bin/env python3
"""Generate TypeScript types from Pydantic models.

This script exports JSON schemas from the consolidated Pydantic models
and generates TypeScript interfaces for the frontend.

Usage:
    python scripts/generate_types.py

Output:
    frontend/src/lib/generated/types.ts
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import EdgeType, GraphEdge, GraphNode, GraphStatus, NodeCategory
from server import AGUIEvent, AGUIEventType, GraphState, JsonPatchOp
from server import GraphRequest, GraphResponse, GraphStats, NodeDetailResponse, TraversalRequest
from agent.state_types import CandidateDecision, DecisionPoint, TraversalState


def json_type_to_typescript(json_type: str | list[str], format: str | None = None) -> str:
    """Convert JSON Schema type to TypeScript type."""
    if isinstance(json_type, list):
        # Handle union types like ["string", "null"]
        ts_types = [json_type_to_typescript(t) for t in json_type]
        return " | ".join(ts_types)

    type_map = {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "null": "null",
        "object": "Record<string, unknown>",
        "array": "unknown[]",
    }
    return type_map.get(json_type, "unknown")


def schema_to_typescript(
    name: str,
    schema: dict[str, Any],
    definitions: dict[str, Any],
    indent: str = "  ",
) -> str:
    """Convert a JSON Schema to TypeScript interface."""
    lines: list[str] = []

    # Handle enum types
    if "enum" in schema:
        values = schema["enum"]
        if all(isinstance(v, str) for v in values):
            union = " | ".join(f"'{v}'" for v in values)
            lines.append(f"export type {name} = {union};")
            return "\n".join(lines)

    # Handle object types (interfaces)
    if schema.get("type") == "object" or "properties" in schema:
        lines.append(f"export interface {name} {{")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            ts_type = resolve_type(prop_schema, definitions)
            optional = "" if prop_name in required else "?"
            lines.append(f"{indent}{prop_name}{optional}: {ts_type};")

        lines.append("}")
        return "\n".join(lines)

    # Handle simple types
    ts_type = resolve_type(schema, definitions)
    lines.append(f"export type {name} = {ts_type};")
    return "\n".join(lines)


def resolve_type(schema: dict[str, Any], definitions: dict[str, Any]) -> str:
    """Resolve a JSON Schema type to TypeScript."""
    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/$defs/"):
            type_name = ref.split("/")[-1]
            return type_name
        return "unknown"

    # Handle allOf (used for refs with additional constraints)
    if "allOf" in schema:
        types = [resolve_type(s, definitions) for s in schema["allOf"]]
        return types[0] if len(types) == 1 else f"({' & '.join(types)})"

    # Handle anyOf (union types)
    if "anyOf" in schema:
        types = [resolve_type(s, definitions) for s in schema["anyOf"]]
        # Filter out null for optional handling
        non_null = [t for t in types if t != "null"]
        has_null = "null" in types
        if len(non_null) == 1:
            return f"{non_null[0]} | null" if has_null else non_null[0]
        return " | ".join(types)

    # Handle array types
    if schema.get("type") == "array":
        items = schema.get("items", {})
        item_type = resolve_type(items, definitions)
        return f"{item_type}[]"

    # Handle object types with additionalProperties (dict/Record)
    if schema.get("type") == "object":
        additional_props = schema.get("additionalProperties")
        if additional_props is True:
            return "Record<string, unknown>"
        elif isinstance(additional_props, dict):
            value_type = resolve_type(additional_props, definitions)
            return f"Record<string, {value_type}>"
        return "Record<string, unknown>"

    # Handle const (literal types)
    if "const" in schema:
        const = schema["const"]
        if isinstance(const, str):
            return f"'{const}'"
        return str(const)

    # Handle enum
    if "enum" in schema:
        values = schema["enum"]
        if all(isinstance(v, str) for v in values):
            return " | ".join(f"'{v}'" for v in values)
        return " | ".join(str(v) for v in values)

    # Handle basic types
    json_type = schema.get("type")
    if json_type:
        return json_type_to_typescript(json_type, schema.get("format"))

    return "unknown"


def generate_typescript(models: list[type], output_path: Path) -> None:
    """Generate TypeScript types from Pydantic models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all definitions and schemas
    all_defs: dict[str, dict] = {}
    model_schemas: list[tuple[str, dict]] = []

    for model in models:
        schema = model.model_json_schema()
        name = model.__name__

        # Collect definitions
        if "$defs" in schema:
            all_defs.update(schema["$defs"])

        model_schemas.append((name, schema))

    # Generate TypeScript
    lines: list[str] = [
        "/**",
        " * AUTO-GENERATED TypeScript types from Python Pydantic models.",
        " * Do not edit manually - regenerate using: python scripts/generate_types.py",
        " */",
        "",
    ]

    # Generate enum types first (from definitions)
    generated_types: set[str] = set()

    for def_name, def_schema in all_defs.items():
        if "enum" in def_schema and def_name not in generated_types:
            ts_code = schema_to_typescript(def_name, def_schema, all_defs)
            lines.append(ts_code)
            lines.append("")
            generated_types.add(def_name)

    # Generate interface types
    for name, schema in model_schemas:
        if name not in generated_types:
            ts_code = schema_to_typescript(name, schema, all_defs)
            lines.append(ts_code)
            lines.append("")
            generated_types.add(name)

    output_path.write_text("\n".join(lines))
    print(f"Generated {output_path}")


def main() -> None:
    """Main entry point."""
    # Models to export
    models = [
        # Core types
        GraphNode,
        GraphEdge,
        # Agent types
        CandidateDecision,
        DecisionPoint,
        TraversalState,
        # AG-UI types
        JsonPatchOp,
        GraphState,
        AGUIEvent,
        # API types
        GraphRequest,
        GraphStats,
        GraphResponse,
        TraversalRequest,
        NodeDetailResponse,
    ]

    # Output path
    output_path = Path(__file__).parent.parent / "frontend" / "src" / "lib" / "generated" / "types.ts"

    generate_typescript(models, output_path)
    print(f"\nGenerated {len(models)} TypeScript types")


if __name__ == "__main__":
    main()
