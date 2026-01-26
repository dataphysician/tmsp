from pathlib import Path
from ast import literal_eval
import sys

# Load data
DATA_PATH = Path("static/icd10cm.txt")
print(f"Loading {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    data = literal_eval(f.read())

LATERAL_KEYS = ("useAdditionalCode", "codeFirst", "codeAlso")

def get_parent_code(entry):
    parent = entry.get("parent")
    if not parent or not isinstance(parent, dict):
        return None
    parent_codes = list(parent.keys())
    return parent_codes[0] if parent_codes else None

def trace_ancestors(code, index):
    if code not in index:
        return []
    ancestors = []
    current = code
    while current in index:
        parent = get_parent_code(index[current])
        if not parent or parent == "ROOT":
            break
        if parent not in index:
            break
        # Primitive check for 7th char to avoid complex import
        ancestors.append(parent)
        current = parent
    return ancestors

def find_nearest_anchor(
    codes: list[str],
    index: dict[str, dict] = data,
) -> dict[str, tuple[str, str, str] | None]:
    code_set = set(codes)
    anchors = {c: None for c in codes}
    
    code_ancestors = {}
    code_ancestor_sets = {}
    
    for code in codes:
        ancestors = trace_ancestors(code, index)
        code_ancestors[code] = [code] + ancestors
        code_ancestor_sets[code] = set(code_ancestors[code])

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
                if isinstance(linked_codes, dict):
                    linked_targets = linked_codes.keys()
                else:
                    linked_targets = linked_codes

                for link_target in linked_targets:
                    for potential_anchor_root in codes:
                        if potential_anchor_root == target_code:
                            continue
                        
                        if link_target in code_ancestor_sets[potential_anchor_root]:
                            # EXACT NODE ANCHORING:
                            # We always anchor the specific node being referenced (link_target) to the source node.
                            # This allows ancestors (like I13) to be anchored, carrying their subtrees (I13.0) with them.
                            # Universal Rule: Parent = Source. Child = Linked Node.
                            
                            c_target = link_target  # The node being moved/anchored
                            c_anchor = source_node  # The new parent
                            c_root_provider = potential_anchor_root # The chain this node belongs to

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
                                "anchor_depth": anchor_depth
                            })

    # Sort descending by weight, then deterministic tie-breaker
    print("\nCandidates (Sorted by Weight):")
    for c in candidates:
        print(f"  {c['target_code']} -> {c['anchor_code']} (via {c['source_node']} {c['key']}, Root={c['anchor_root']}) Weight={c['weight']}")

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

    def get_ancestry_path(node):
        path = []
        curr = node
        while curr in parent:
            path.append(curr)
            curr = parent[curr]
            if curr in path: # Cycle safety
                break
        path.append(curr) # Add root
        return path

    def invalidate_ancestors(node):
        # Trace UP from node using ORIGINAL parents.
        # Mark nodes as invalid.
        # Stop if we hit a node that is clearly shared?
        # User said "remove all the ancestors from the list (use a copied list)".
        # Implying for THAT specific chain.
        # But here we have a global graph.
        # If N18 is ancestor of N18.9.
        # And N18.9 anchors to I13.0.
        # N18 is no longer needed *for N18.9*.
        # Is N18 needed for anything else?
        # If N18 is not an input code, and no other input code uses it.
        # Then N18 becomes a "zombie".
        # We can mark it invalid.
        curr = node
        while curr in original_parent:
            par = original_parent[curr]
            # Don't invalidate if par is an Input Code? (User didn't say that).
            # But par is an ancestor.
            # Only invalidate if logic dictates.
            # Let's simple remove from valid_nodes.
            if par in valid_nodes:
                # Need reference counting? 
                # Or just pessimistic invalidation?
                # User: "remove all the ancestors".
                # If N18 is removed.
                pass
            valid_nodes.discard(par)
            curr = par

    def is_descendant(p, node):
        # Check if 'node' is an ancestor of 'p' (i.e. p is descendant of node)
        curr = p
        visited = set()
        while curr in parent:
            if curr == node: return True
            if curr in visited: break
            visited.add(curr)
            curr = parent[curr]
        return curr == node

    candidates.sort(key=lambda x: (x["weight"], x["target_code"], x["anchor_code"]), reverse=True)

    cleaved_nodes = set()
    anchors = {} 

    print("\nProcessing Links (Dynamic Rewiring + Invalidation):")
    for cand in candidates:
        t = cand["target_code"] 
        s = cand["anchor_code"] 
        
        # Check Validity of S and T
        if t not in valid_nodes:
             # Allowed? "For every node that was targetted".
             # If T was invalidated (it was an ancestor of a moved node?).
             # No, invalidate_ancestors goes UP. T is the bottom.
             # T should remain valid.
             pass
             
        if s not in valid_nodes:
             print(f"  SKIPPED: {t} -> {s} (Source {s} is Invalid/Pruned)")
             continue

        if s in cleaved_nodes:
            print(f"  SKIPPED: {t} -> {s} (Source {s} was previously cleaved)")
            continue
            
        if t in cleaved_nodes:
             print(f"  SKIPPED: {t} -> {s} (Target {t} is already cleaved)")
             continue

        if is_descendant(s, t):
             print(f"  SKIPPED: {t} -> {s} (Cycle)")
             continue
             
        # Apply Move
        print(f"  ACCEPTED: {t} -> {s} (Reparenting {t} to {s})")
        parent[t] = s
        cleaved_nodes.add(t)
        
        # Invalidate Old Ancestors of T
        invalidate_ancestors(t)
        
        anchors[t] = (s, cand["key"], cand["source_node"])

    return anchors

result = find_nearest_anchor(['I13.0', 'N18.9', 'I50.9'], data)
print("\nFinal Anchors:")
for k, v in result.items():
    print(f"  {k}: {v}")
