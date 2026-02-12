"""ICD-10-CM tree building and graph construction utilities

This module provides:
- ICD-10-CM data loading from the flat index
- Ancestor tracing for code hierarchy
- Lateral link resolution (codeFirst, codeAlso, useAdditionalCode)
- 7th character handling and placeholder chain building
- Graph construction with nearest-anchor provenance
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "static" / "icd10cm.json"

with open(DATA_PATH, "r") as f:
    data: dict[str, dict] = json.load(f)


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
    - T88.XXXA -> T88 (drop 7th char, X placeholders, and trailing dot)

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
        # Handle trailing dot after stripping all X's (e.g., T88. -> T88)
        if base.endswith("."):
            base = base[:-1]
            if base in index:
                return base

    return None


def build_placeholder_chain(code: str, index: dict[str, dict]) -> list[str]:
    """Build the chain of placeholder codes from a 7th char code to its base.

    For V29.9XXS, returns: ['V29.9XX', 'V29.9X', 'V29.9']
    For T36.1X5D, returns: ['T36.1X5']
    For T88.XXXA, returns: ['T88.XXX', 'T88.XX', 'T88.X', 'T88']
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
            # Don't append trailing dot forms (e.g., T88.)
            if not current.endswith("."):
                chain.append(current)

        # Handle trailing dot after stripping all X's (e.g., T88. -> T88)
        if current.endswith("."):
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


# Lateral link keys - create anchors to other codes in the batch
LATERAL_KEYS = ("useAdditionalCode", "codeFirst", "codeAlso")

# Vertical chain key - applies only to direct parent-child ancestry
VERTICAL_KEYS = ("sevenChrDef",)


def trace_seventh_char_def(
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
    """Find the nearest anchor for each code using a global 'lowest link' strategy.

    Algorithm:
    1. Trace ancestry for all input codes.
    2. Identify ALL potential lateral links between any node in Chain A and any node in Chain B.
    3. Sort all potential links by 'closeness' (depth of the anchor node + depth of source node).
       We want the 'lowest' link (deepest in the tree).
    4. Assign anchors greedily from the sorted list, ensuring no cycles are created.
    5. Once a code is anchored, ignore higher links for that code (truncation).

    Args:
        codes: List of codes to analyze
        index: The flat index dict

    Returns:
        dict mapping code -> (anchor_code, metadata_key, source_node) or None
    """
    code_set = set(codes)
    anchors: dict[str, tuple[str, str, str] | None] = {c: None for c in codes}
    
    # 1. Trace ancestries
    code_ancestors: dict[str, list[str]] = {}
    code_ancestor_sets: dict[str, set[str]] = {}
    
    for code in codes:
        ancestors = trace_ancestors(code, index)
        # Store full chain: [code, parent, grandparent, ...]
        code_ancestors[code] = [code] + ancestors
        code_ancestor_sets[code] = set(code_ancestors[code])

    # 2. Find all candidate links
    # Candidate: (target_code, anchor_code, key, source_node, weight)
    # We want to anchor 'target_code' onto 'anchor_code'.
    # 'source_node' is the node in target_code's chain that has the metadata link.
    candidates = []

    for target_code in codes:
        chain = code_ancestors[target_code]
        for source_node in chain:
            if source_node not in index:
                continue

            metadata = index[source_node].get("metadata", {})
            for key in LATERAL_KEYS:
                if key not in metadata:
                    continue

                linked_codes = metadata[key]
                # Normalize to dict keys iterator if needed
                if isinstance(linked_codes, dict):
                    linked_targets = linked_codes.keys()
                else:
                    linked_targets = linked_codes

                for link_target in linked_targets:
                    # check if this link_target points to the ancestry of ANY OTHER input code
                    # If link_target is in ancestry of 'dest_code', then 'target_code' anchors to 'dest_code's chain' at 'link_target'.

                    for potential_anchor_root in codes:
                        if potential_anchor_root == target_code:
                            continue

                        # CRITICAL: Check if link_target is actually in this code's ancestry
                        if link_target not in code_ancestor_sets[potential_anchor_root]:
                            continue

                        # We found a valid link! link_target is in the ancestry of potential_anchor_root
                        # So target_code (via source_node) -> link_target (in potential_anchor_root's chain)
                        # EXACT NODE ANCHORING:
                        # We always anchor the specific node being referenced (link_target) to the source node.
                        # This allows ancestors (like I13) to be anchored, carrying their subtrees (I13.0) with them.
                        # Universal Rule: Parent = Source. Child = Linked Node.

                        c_target = link_target  # The node being moved/anchored
                        c_anchor = source_node  # The new parent
                        c_root_provider = potential_anchor_root  # The chain this node belongs to

                        # Calculate Weight: Higher is "lower/deeper" in the tree.
                        anchor_depth = index.get(c_anchor, {}).get("depth", 0)
                        source_depth = index.get(c_target, {}).get("depth", 0)

                        weight = (anchor_depth * 100) + source_depth

                        candidates.append({
                            "target_code": c_target,
                            "anchor_code": c_anchor,
                            "anchor_root": c_root_provider,
                            "key": key,
                            "source_node": source_node,
                            "weight": weight,
                            "anchor_depth": anchor_depth,
                            "target_depth": index.get(c_target, {}).get("depth", 0)
                        })

    # Sort DESCENDING by weight (deepest links first), then deterministic tie-breaker
    # This ensures leaf-adjacent links are established first, cleaving their ancestries
    # before those ancestors can participate as sources for other links
    candidates.sort(key=lambda x: (x["weight"], x["target_code"], x["anchor_code"]), reverse=True)

    # Dynamic Graph Rewiring Implementation
    # 1. Build Initial Parent Map from Chains
    parent = {}
    valid_nodes = set()
    
    for code in codes:
        chain = code_ancestors[code]
        valid_nodes.update(chain) # All chain nodes initially valid
        for i in range(len(chain) - 1):
            child = chain[i]
            par = chain[i+1]
            if child not in parent: 
                 parent[child] = par

    # Track original parents to allow upward tracing for invalidation
    original_parent = parent.copy()

    def invalidate_ancestors(node):
        # Trace UP from node using ORIGINAL parents.
        # Mark nodes as invalid (meaning they can no longer be SOURCES for new links).
        # This prevents cycles like N18 -> N18.9 -> I13.0 -> I13 -> N18.
        curr = node
        while curr in original_parent:
            par = original_parent[curr]
            valid_nodes.discard(par)
            curr = par

    def is_descendant(p, node):
        # Check if 'node' is an ancestor of 'p' (i.e. p is descendant of node) using CURRENT parents
        curr = p
        visited = set()
        while curr in parent:
            if curr == node: return True
            if curr in visited: break
            visited.add(curr)
            curr = parent[curr]
        return curr == node

    cleaved_nodes = set()
    anchors = {} 

    for cand in candidates:
        t = cand["target_code"] 
        s = cand["anchor_code"] 
        
        # Check Validity of S and T
        # T (Target) must persist, so we don't strictly check if T is in valid_nodes?
        # Actually user said "For every node that was targetted... remove ancestors".
        # If T was already invalidated, can it be targeted? 
        # T is an ancestor of X, and X moved. T is gone.
        # So T should effectively be dead. But maybe we can salvage it?
        # Let's trust valid_nodes.
             
        if s not in valid_nodes:
             continue # Source is Invalid/Pruned

        if s in cleaved_nodes:
            continue # Source was previously moved, so it can't steal others? (Heuristic)
            
        if t in cleaved_nodes:
             continue # Target already moved. Deepest link wins.

        if is_descendant(s, t):
             continue # Cycle prevention
             
        # Apply Move
        parent[t] = s
        cleaved_nodes.add(t)
        
        # Invalidate Old Ancestors of T (cleave them away)
        invalidate_ancestors(t)
        
        anchors[t] = (s, cand["key"], cand["source_node"])

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

    # Helper to get parent from index (uses same logic as get_parent_code)
    def get_parent_from_index(node_code: str) -> str | None:
        if node_code == "ROOT":
            return None
        node_info = index.get(node_code)
        if not node_info:
            return None
        parent = node_info.get("parent")
        if not parent or not isinstance(parent, dict):
            return None
        parent_codes = list(parent.keys())
        parent_code = parent_codes[0] if parent_codes else None
        # Don't return ROOT - it's a virtual node added by the graph builder
        if parent_code == "ROOT":
            return None
        return parent_code

    for code in codes:
        new_chain: list[str] = []
        visited_path_nodes: set[str] = set()
        chain_lateral_links: list[tuple[str, str, str]] = []

        # Resolve 7th char codes to their base form
        resolved = resolve_code(code, index)
        if resolved is None:
            result[code] = {
                "ancestors": [],
                "anchor": None,
                "visited": {code},
                "lateral_links": [],
            }
            continue

        # Handle 7th char codes: add placeholder chain first
        if resolved != code:
            # For 7th char codes, mark the original code as visited
            visited_path_nodes.add(code)
            placeholder_chain = build_placeholder_chain(code, index)
            new_chain.extend(placeholder_chain)
            visited_path_nodes.update(placeholder_chain)

        # Start walking from the resolved code
        curr: str | None = resolved

        while curr:
            # If already visited, get parent and continue (handles 7th char resolved code)
            if curr in visited_path_nodes:
                curr = get_parent_from_index(curr)
                continue

            visited_path_nodes.add(curr)

            # Don't add the input code to ancestors list
            if curr != code:
                new_chain.append(curr)

            # Determine next parent
            if curr in anchors:
                # Anchored! Jump to anchor source
                anchor_target, key, source_node = anchors[curr]
                chain_lateral_links.append((source_node, curr, key))
                curr = anchor_target
            else:
                curr = get_parent_from_index(curr)

        result[code] = {
            "ancestors": new_chain,
            "anchor": anchors.get(code),
            "visited": visited_path_nodes,
            "lateral_links": chain_lateral_links,
        }

    return result


# ============================================================================
# Build Graph Helper Methods
# ============================================================================


def _collect_seventh_char_info(
    code: str,
    index: dict[str, dict],
) -> tuple[tuple[str, str, str] | None, set[str]]:
    """Collect 7th char info and placeholder codes for a code.

    Returns:
        (seventh_char_entry or None, placeholder_codes)
    """
    char = extract_seventh_char(code)
    if not char:
        return None, set()

    result = trace_seventh_char_def(code, index)
    if result:
        seven_def, source_node = result
        if char in seven_def:
            return (char, seven_def[char], source_node), get_placeholder_codes(code, index)
    return None, get_placeholder_codes(code, index)


def _build_tree_edges(tree: dict[str, set[str]], chain: list[str]) -> None:
    """Add edges to tree from a code->ancestors chain (mutates tree)."""
    for i in range(len(chain) - 1):
        child, parent = chain[i], chain[i + 1]
        if parent not in tree:
            tree[parent] = set()
        tree[parent].add(child)


def _collect_lateral_links_from_nodes(
    all_nodes: set[str],
    index: dict[str, dict],
) -> list[tuple[str, str, str]]:
    """Scan all nodes for lateral links (used in show_all_paths mode)."""
    lateral_links = []
    for node in all_nodes:
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
                if linked in all_nodes:
                    lateral_links.append((node, linked, key))
    return list(set(lateral_links))


# ============================================================================
# Build Graph
# ============================================================================


def build_graph(
    codes: list[str],
    index: dict[str, dict] = data,
    show_all_paths: bool = False,
) -> dict:
    """Build graph from ROOT to target codes.

    Args:
        codes: List of end codes
        index: The flat index dict
        show_all_paths: If False (default), use nearest-anchor provenance
                        for minimal graph. If True, show complete paths
                        to ROOT for all codes.

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
    all_nodes: set[str] = set()
    tree: dict[str, set[str]] = {}
    roots: set[str] = set()
    anchored: dict[str, tuple[str, str, str]] = {}
    lateral_links: list[tuple[str, str, str]] = []
    seventh_char: dict[str, tuple[str, str, str]] = {}
    placeholders: set[str] = set()

    # Compute provenance once for all codes (batch mode enables cross-code lateral discovery)
    provenance = None if show_all_paths else trace_with_nearest_anchor(codes, index)

    for code in codes:
        if show_all_paths:
            # Full paths mode: simple ancestry trace
            ancestors = trace_ancestors(code, index)
            chain_lateral_links: list[tuple[str, str, str]] = []
            anchor = None
        else:
            # Minimal mode: use nearest-anchor provenance
            info = provenance[code]
            ancestors = info["ancestors"]
            anchor = info["anchor"]
            chain_lateral_links = info.get("lateral_links", [])

        # Add code and ancestors to node set
        all_nodes.add(code)
        all_nodes.update(ancestors)

        # Record anchor (minimal mode only)
        if anchor:
            anchored[code] = anchor

        # Collect lateral links from provenance (minimal mode only)
        for src, tgt, key in chain_lateral_links:
            lateral_links.append((src, tgt, key))

        # Collect 7th char info
        seventh_entry, placeholder_codes = _collect_seventh_char_info(code, index)
        if seventh_entry:
            seventh_char[code] = seventh_entry
        placeholders.update(placeholder_codes)

        # Build tree edges
        chain = [code] + ancestors
        _build_tree_edges(tree, chain)

        # Identify root
        if show_all_paths or not chain_lateral_links:
            if ancestors:
                roots.add(ancestors[-1])
            else:
                roots.add(code)

    # Collect lateral links (different approach per mode)
    if show_all_paths:
        lateral_links = _collect_lateral_links_from_nodes(all_nodes, index)
    else:
        # Filter to only show when both endpoints in graph
        lateral_links = [
            (src, tgt, key)
            for src, tgt, key in set(lateral_links)
            if src in all_nodes and tgt in all_nodes
        ]

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