"""ICD-10-CM tree building and graph construction utilities.

This module provides:
- ICD-10-CM data loading from the flat index
- Ancestor tracing for code hierarchy
- Lateral link resolution (codeFirst, codeAlso, useAdditionalCode)
- 7th character handling and placeholder chain building
- Graph construction with nearest-anchor provenance
"""

from ast import literal_eval
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "static" / "icd10cm.txt"

with open(DATA_PATH, "r") as f:
    data: dict[str, dict] = literal_eval(f.read())


def get_parent_code(entry: dict) -> str | None:
    """Extract the parent code from an entry's parent dict."""
    parent = entry.get("parent")
    if not parent or not isinstance(parent, dict):
        return None
    # Parent dict has a single key which is the parent code
    parent_codes = list(parent.keys())
    return parent_codes[0] if parent_codes else None


def resolve_code(code: str, index: dict[str, dict]) -> str | None:
    """Resolve a code to its base form if it has a 7th character extension.

    Handles cases like:
    - T36.1X5D -> T36.1X5 (drop 7th char)
    - T84.53XD -> T84.53 (X is placeholder, D is 7th char)
    - V29.9XXS -> V29.9 (drop 7th char and X placeholders)

    Returns the resolved code if found, None if not resolvable.
    """
    if code in index:
        return code

    # Try dropping last character (7th character extension like A, D, S)
    if len(code) > 1:
        base = code[:-1]
        if base in index:
            return base
        # Try dropping X placeholders iteratively
        while base.endswith("X"):
            base = base[:-1]
            if base in index:
                return base

    return None


def build_placeholder_chain(code: str, index: dict[str, dict]) -> list[str]:
    """Build the chain of placeholder codes from a 7th char code to its base.

    For V29.9XXS, returns: ['V29.9XX', 'V29.9X', 'V29.9']
    For T36.1X5D, returns: ['T36.1X5']
    For codes without placeholders, returns: [base] if different from code

    Returns list of intermediate codes from 7th char code to base (inclusive).
    """
    if code in index:
        return []

    chain: list[str] = []
    current = code

    # Drop the 7th character first
    if len(current) > 1:
        current = current[:-1]
        if current in index:
            chain.append(current)
            return chain
        chain.append(current)

        # Now drop X placeholders one at a time
        while current.endswith("X"):
            current = current[:-1]
            if current in index:
                chain.append(current)
                return chain
            chain.append(current)

    return chain


def get_placeholder_codes(code: str, index: dict[str, dict]) -> set[str]:
    """Get all placeholder codes (ending in X) in the chain from code to base."""
    chain = build_placeholder_chain(code, index)
    return {c for c in chain if c.endswith("X")}


def trace_ancestors(code: str, index: dict[str, dict] = data) -> list[str]:
    """Trace all ancestors of a code from immediate parent to highest ancestor.

    Args:
        code: The ICD-10-CM code to trace
        index: The flat index dict (defaults to loaded data)

    Returns:
        List of ancestor codes from immediate parent to ROOT (exclusive).
        Includes placeholder chain for 7th char codes (e.g., V29.9XXS -> V29.9XX -> V29.9X -> V29.9)
    """
    # Resolve code to base form if it has 7th character extension
    resolved = resolve_code(code, index)
    if resolved is None:
        return []

    ancestors: list[str] = []
    current = resolved

    # If the code was resolved to a different base, include the placeholder chain
    if resolved != code:
        placeholder_chain = build_placeholder_chain(code, index)
        ancestors.extend(placeholder_chain)

    while current in index:
        parent_code = get_parent_code(index[current])
        if parent_code is None or parent_code == "ROOT":
            break
        if parent_code not in index:
            break
        ancestors.append(parent_code)
        current = parent_code

    return ancestors


def trace_ancestors_batch(codes: list[str], index: dict[str, dict] = data) -> list[list[str]]:
    """Trace ancestors for multiple codes.

    Args:
        codes: List of ICD-10-CM codes to trace
        index: The flat index dict (defaults to loaded data)

    Returns:
        List of ancestor lists, one per input code
    """
    return [trace_ancestors(code, index) for code in codes]


# Lateral link keys - create anchors to other codes in the batch
LATERAL_KEYS = ("useAdditionalCode", "codeFirst", "codeAlso")

# Vertical chain key - applies only to direct parent-child ancestry
VERTICAL_KEYS = ("sevenChrDef",)


def get_lateral_links(code: str, index: dict[str, dict] = data) -> set[str]:
    """Extract all codes linked via lateral metadata keys from a code's metadata."""
    if code not in index:
        return set()

    metadata = index[code].get("metadata", {})
    linked: set[str] = set()

    for key in LATERAL_KEYS:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, dict):
                linked.update(value.keys())

    return linked


def get_seventh_char_def(
    code: str, index: dict[str, dict] = data
) -> tuple[dict[str, str], str] | None:
    """Find sevenChrDef in the direct parent-child ancestry of a code.

    Returns (sevenChrDef dict, source_node) if found in any ancestor, None otherwise.
    """
    resolved = resolve_code(code, index)
    if resolved is None:
        return None

    # Check the resolved code and all its ancestors
    current = resolved
    while current in index:
        metadata = index[current].get("metadata", {})
        if "sevenChrDef" in metadata:
            return (metadata["sevenChrDef"], current)

        parent_code = get_parent_code(index[current])
        if parent_code is None or parent_code == "ROOT":
            break
        if parent_code not in index:
            break
        current = parent_code

    return None


def extract_seventh_char(code: str) -> str | None:
    """Extract the 7th character from a code if present.

    The 7th character is the last character for codes like:
    - T36.1X5D (7th char = D)
    - T84.53XD (7th char = D, X is placeholder)
    """
    if len(code) < 7:
        return None

    # Count significant positions (excluding X placeholders before 7th char)
    # Format: XXX.XXYZ where Z is 7th char
    if "." in code:
        parts = code.split(".")
        if len(parts) == 2 and len(parts[1]) >= 4:
            return code[-1]

    return None


def find_nearest_anchor(
    codes: list[str],
    index: dict[str, dict] = data,
) -> dict[str, tuple[str, str, str] | None]:
    """Find the nearest anchor for each code based on lateral metadata links.

    For each code, checks if any other code's ancestry contains a node that
    links to this code (or an ancestor of this code) via LATERAL_KEYS.

    Supports transitive anchoring: if metadata links to an ancestor of an
    input code, the input code anchors at that ancestor.

    Note: sevenChrDef is NOT used for anchoring - it only applies to the
    direct parent-child chain.

    Args:
        codes: List of codes to analyze
        index: The flat index dict

    Returns:
        dict mapping code -> (anchor_code, metadata_key, source_node) or None
        - anchor_code: the node where the input code's ancestry is truncated
        - metadata_key: the lateral key (useAdditionalCode, codeFirst, codeAlso)
        - source_node: the node that has the metadata linking to anchor_code
    """
    code_set = set(codes)
    anchors: dict[str, tuple[str, str, str] | None] = {c: None for c in codes}

    # Pre-compute full ancestry for each input code (as sets for fast lookup)
    code_ancestors: dict[str, list[str]] = {}
    code_ancestor_sets: dict[str, set[str]] = {}
    for code in codes:
        ancestors = trace_ancestors(code, index)
        code_ancestors[code] = ancestors
        code_ancestor_sets[code] = set(ancestors)

    def update_anchor(target_code: str, anchor_code: str, key: str, source_node: str) -> None:
        """Update anchor if this one is deeper (nearer)."""
        current = anchors[target_code]
        anchor_depth = index.get(anchor_code, {}).get("depth", 0)

        if current is None:
            anchors[target_code] = (anchor_code, key, source_node)
        else:
            existing_depth = index.get(current[0], {}).get("depth", 0)
            if anchor_depth > existing_depth:
                anchors[target_code] = (anchor_code, key, source_node)

    # For each code, trace its ancestry and check metadata links
    for code in codes:
        chain = [code] + code_ancestors[code]

        for node in chain:
            if node not in index:
                continue

            metadata = index[node].get("metadata", {})
            for key in LATERAL_KEYS:
                if key not in metadata:
                    continue

                linked_codes = metadata[key]
                if not isinstance(linked_codes, dict):
                    continue

                for linked in linked_codes:
                    # Case 1: Direct match - linked code is in input batch
                    if linked in code_set and linked != code:
                        update_anchor(linked, node, key, node)

                    # Case 2: Transitive - linked code is an ancestor of an input code
                    for other_code in code_set:
                        if other_code == code:
                            continue
                        if linked in code_ancestor_sets[other_code]:
                            # linked is an ancestor of other_code
                            # other_code should anchor at linked, source is node
                            update_anchor(other_code, linked, key, node)

    return anchors


def trace_with_nearest_anchor(
    codes: list[str],
    index: dict[str, dict] = data,
) -> dict[str, dict]:
    """Trace ancestors for a batch of codes with Nearest-Anchor Provenance.

    For codes linked via LATERAL_KEYS from another code's ancestry, the
    provenance stops at the anchor point instead of tracing to ROOT.

    Supports transitive anchoring: if the anchor is an ancestor of the input
    code, the chain is truncated at that ancestor (keeping the path from
    input code to anchor).

    Args:
        codes: List of codes to trace
        index: The flat index dict

    Returns:
        dict mapping each code to:
        - ancestors: list of ancestor codes (truncated at anchor if applicable)
        - anchor: (anchor_code, metadata_key) or None
        - visited: set of all codes in the chain
    """
    anchors = find_nearest_anchor(codes, index)
    result: dict[str, dict] = {}

    for code in codes:
        full_ancestors = trace_ancestors(code, index)
        anchor = anchors[code]

        if anchor is None:
            # No anchor - full trace to ROOT
            ancestors = full_ancestors
        else:
            # Truncate at anchor point
            anchor_code = anchor[0]
            if anchor_code in full_ancestors:
                # Anchor is in the ancestry chain - truncate there
                idx = full_ancestors.index(anchor_code)
                ancestors = full_ancestors[: idx + 1]  # Include anchor
            else:
                # Anchor is external (direct match case)
                ancestors = [anchor_code]

        visited = {code} | set(ancestors)

        result[code] = {
            "ancestors": ancestors,
            "anchor": anchor,
            "visited": visited,
        }

    return result


def build_graph(
    codes: list[str],
    index: dict[str, dict] = data,
) -> dict:
    """Find the minimal set of nodes to cover all paths from ROOT to end codes.

    Uses Nearest-Anchor Provenance to reduce paths where lateral links exist.
    Tracks sevenChrDef for leaf nodes in their direct parent-child ancestry.

    Args:
        codes: List of end codes
        index: The flat index dict

    Returns:
        dict with:
        - nodes: set of all required nodes
        - tree: dict mapping parent -> set of children
        - roots: set of top-level nodes (connect to ROOT)
        - leaves: the original input codes
        - anchored: dict of code -> (anchor_code, key, source_node)
        - lateral_links: list of (source_node, anchor_code, key) for visualization
        - seventh_char: dict of code -> (char, meaning, source_node) for codes with 7th char
        - placeholders: set of placeholder codes (ending in X)
    """
    provenance = trace_with_nearest_anchor(codes, index)

    all_nodes: set[str] = set()
    tree: dict[str, set[str]] = {}
    roots: set[str] = set()
    anchored: dict[str, tuple[str, str, str]] = {}
    lateral_links: list[tuple[str, str, str]] = []  # (source, anchor, key)
    seventh_char: dict[str, tuple[str, str, str]] = {}
    placeholders: set[str] = set()

    for code in codes:
        info = provenance[code]
        ancestors = info["ancestors"]
        anchor = info["anchor"]

        # Add code and all ancestors to node set
        all_nodes.add(code)
        all_nodes.update(ancestors)

        if anchor:
            anchored[code] = anchor
            anchor_code, key, source_node = anchor
            # Track lateral link for visualization
            if anchor_code == source_node:
                # Direct match - link from source to the anchored code
                lateral_links.append((source_node, code, key))
            else:
                # Transitive match - link from source to the linked ancestor
                lateral_links.append((source_node, anchor_code, key))

        # Check for 7th character in leaf nodes (all input codes)
        char = extract_seventh_char(code)
        if char:
            result = get_seventh_char_def(code, index)
            if result:
                seven_def, source_node = result
                if char in seven_def:
                    seventh_char[code] = (char, seven_def[char], source_node)
            # Collect placeholder codes from the chain
            placeholders.update(get_placeholder_codes(code, index))

        # Build tree edges (child -> parent relationship, but we store parent -> children)
        chain = [code] + ancestors
        for i in range(len(chain) - 1):
            child, parent = chain[i], chain[i + 1]
            if parent not in tree:
                tree[parent] = set()
            tree[parent].add(child)

        # Identify root of this chain (only for non-anchored codes)
        # Anchored codes attach to existing nodes, not to ROOT
        if anchor is None:
            if ancestors:
                root = ancestors[-1]
            else:
                root = code
            roots.add(root)

    # Deduplicate lateral links
    lateral_links = list(set(lateral_links))

    return {
        "nodes": all_nodes,
        "tree": tree,
        "roots": roots,
        "leaves": set(codes),
        "anchored": anchored,
        "lateral_links": lateral_links,
        "seventh_char": seventh_char,
        "placeholders": placeholders,
        "count": len(all_nodes),
    }


def get_node_category(
    code: str,
    leaves: set[str],
    placeholders: set[str],
    activators: set[str],
) -> str:
    """Determine the category of a node for frontend styling.

    Args:
        code: The ICD-10-CM code
        leaves: Set of input/finalized codes
        placeholders: Set of placeholder codes (ending in X)
        activators: Set of codes with sevenChrDef metadata

    Returns:
        Category string: "root", "finalized", "placeholder", "activator", or "ancestor"
    """
    if code == "ROOT":
        return "root"
    if code in leaves:
        return "finalized"
    if code in placeholders:
        return "placeholder"
    if code in activators:
        return "activator"
    return "ancestor"


def get_activator_nodes(seventh_char: dict[str, tuple[str, str, str]]) -> set[str]:
    """Extract the set of nodes that define sevenChrDef.

    Args:
        seventh_char: Dict from build_graph result mapping code -> (char, meaning, source_node)

    Returns:
        Set of source nodes that contain sevenChrDef definitions
    """
    return {source_node for _, _, source_node in seventh_char.values()}
