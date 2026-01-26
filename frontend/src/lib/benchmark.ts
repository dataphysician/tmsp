/**
 * Benchmark utility functions for comparing expected vs traversed ICD-10-CM codes.
 *
 * The comparison is based on endpoint relationships between finalized codes:
 * - Matched: Same code (exact match)
 * - Undershoot: Traversed finalized at ancestor of expected
 * - Overshoot: Traversed finalized at descendant of expected
 * - Missed: Expected not matched by any traversed
 * - Other: Traversed not related to any expected (hidden from graph, shown in report)
 */

import type {
  GraphNode,
  GraphEdge,
  BenchmarkGraphNode,
  BenchmarkNodeStatus,
  BenchmarkMetrics,
  ExpectedCodeOutcome,
  OvershootMarker,
  EdgeMissMarker,
} from './types';

/**
 * Build a map of code -> set of ancestor codes from graph edges.
 * Considers hierarchy edges AND ALL lateral edges (sevenChrDef, codeFirst,
 * codeAlso, useAdditionalCode) which all represent parent-child relationships.
 */
export function buildAncestorMap(edges: GraphEdge[]): Map<string, Set<string>> {
  // Build parent map from hierarchy edges AND all lateral edges
  // All lateral edges represent ancestry relationships for benchmark comparison
  const parentMap = new Map<string, string>();
  for (const edge of edges) {
    const isHierarchy = edge.edge_type === 'hierarchy';
    const isLateral = edge.edge_type === 'lateral';

    if (isHierarchy || isLateral) {
      // edge.source is parent, edge.target is child
      parentMap.set(edge.target, edge.source);
    }
  }

  // Build ancestor set for each node
  const ancestorMap = new Map<string, Set<string>>();
  for (const [child] of parentMap) {
    const ancestors = new Set<string>();
    let current = child;
    while (parentMap.has(current)) {
      const parent = parentMap.get(current)!;
      ancestors.add(parent);
      current = parent;
    }
    ancestorMap.set(child, ancestors);
  }

  return ancestorMap;
}

/**
 * Check if codeA is an ancestor of codeB using the ancestor map.
 */
export function isAncestor(
  codeA: string,
  codeB: string,
  ancestorMap: Map<string, Set<string>>
): boolean {
  const ancestors = ancestorMap.get(codeB);
  return ancestors?.has(codeA) ?? false;
}

/**
 * Compare expected finalized codes against traversed finalized codes.
 *
 * For each expected code, determines the outcome based on endpoint relationships.
 * Also identifies "other" codes (traversed codes unrelated to any expected - hidden from graph).
 *
 * Special handling for sevenChrDef codes:
 * - If expected is a depth-7 code (e.g., T36.1X5D) and traversal finalized at its
 *   immediate sevenChrDef parent (T36.1X5), this is "missed" not "undershoot"
 * - Rationale: The sevenChrDef batch was presented to the LLM, so it saw the options
 *   but made the wrong choice (or chose to stop). This is a missed decision.
 */
export function compareFinalizedCodes(
  expectedCodes: Set<string>,
  expectedAncestorMap: Map<string, Set<string>>,
  traversedCodes: Set<string>,
  traversedAncestorMap: Map<string, Set<string>>,
  expectedEdges?: GraphEdge[]
): { outcomes: ExpectedCodeOutcome[]; otherCodes: string[] } {
  const outcomes: ExpectedCodeOutcome[] = [];
  const matchedTraversed = new Set<string>();

  // Build sevenChrDef parent map if edges provided
  const sevenChrDefParentMap = expectedEdges
    ? buildSevenChrDefParentMap(expectedEdges)
    : new Map<string, string>();

  // Build lateral parent map for all lateral edges (codeFirst, codeAlso, useAdditionalCode)
  const lateralParentMap = expectedEdges
    ? buildLateralParentMap(expectedEdges)
    : new Map<string, { parent: string; rule: string }>();

  for (const expected of expectedCodes) {
    // Check exact match first (MATCHED status)
    if (traversedCodes.has(expected)) {
      outcomes.push({
        expectedCode: expected,
        status: 'exact',
        relatedCode: expected,
      });
      matchedTraversed.add(expected);
      continue;
    }

    // Check if expected code was reached via ANY lateral edge (non-sevenChrDef)
    // If the lateral source was finalized, treat as exact match for lateral targets
    const lateralInfo = lateralParentMap.get(expected);
    if (lateralInfo && lateralInfo.rule !== 'sevenChrDef' && traversedCodes.has(lateralInfo.parent)) {
      // The lateral source was finalized - treat as exact match for lateral targets
      outcomes.push({
        expectedCode: expected,
        status: 'exact',
        relatedCode: lateralInfo.parent,
      });
      matchedTraversed.add(lateralInfo.parent);
      continue;
    }

    // Check if this is a sevenChrDef child where the parent was traversed
    // If so, the sevenChrDef batch was presented - this is "missed" not "undershoot"
    const sevenChrDefParent = sevenChrDefParentMap.get(expected);
    if (sevenChrDefParent && traversedCodes.has(sevenChrDefParent)) {
      // The depth-6 parent was finalized, meaning the sevenChrDef batch was presented
      // but the correct 7th character wasn't selected. This is a missed decision.
      outcomes.push({
        expectedCode: expected,
        status: 'missed',
        relatedCode: sevenChrDefParent,
      });
      matchedTraversed.add(sevenChrDefParent);
      continue;
    }

    // Check undershoot: traversed code is ancestor of expected (but not sevenChrDef parent)
    const expectedAncestors = expectedAncestorMap.get(expected) ?? new Set();
    let foundUndershoot: string | null = null;
    for (const traversed of traversedCodes) {
      if (expectedAncestors.has(traversed)) {
        foundUndershoot = traversed;
        matchedTraversed.add(traversed);
        break;
      }
    }
    if (foundUndershoot) {
      outcomes.push({
        expectedCode: expected,
        status: 'undershoot',
        relatedCode: foundUndershoot,
      });
      continue;
    }

    // Check overshoot: expected is ancestor of traversed code
    let foundOvershoot: string | null = null;
    for (const traversed of traversedCodes) {
      const traversedAncestors = traversedAncestorMap.get(traversed) ?? new Set();
      if (traversedAncestors.has(expected)) {
        foundOvershoot = traversed;
        matchedTraversed.add(traversed);
        break;
      }
    }
    if (foundOvershoot) {
      outcomes.push({
        expectedCode: expected,
        status: 'overshoot',
        relatedCode: foundOvershoot,
      });
      continue;
    }

    // Missed: no relationship found
    outcomes.push({
      expectedCode: expected,
      status: 'missed',
      relatedCode: null,
    });
  }

  // Other: traversed codes not matched to any expected (hidden from graph, shown in report)
  // Also filter out codes that are ancestors of other traversed codes (redundant)
  const unmatchedCodes = [...traversedCodes].filter((t) => !matchedTraversed.has(t));

  // Build set of all ancestors of unmatched codes
  const ancestorsOfUnmatched = new Set<string>();
  for (const code of unmatchedCodes) {
    const ancestors = traversedAncestorMap.get(code);
    if (ancestors) {
      for (const ancestor of ancestors) {
        ancestorsOfUnmatched.add(ancestor);
      }
    }
  }

  // Filter out codes that are ancestors of other unmatched codes
  const otherCodes = unmatchedCodes.filter((t) => !ancestorsOfUnmatched.has(t));

  return { outcomes, otherCodes };
}

/**
 * Compute aggregate benchmark metrics from comparison results.
 *
 * @param expectedCodes - User-defined expected finalized codes
 * @param traversedCodes - Finalized codes from traversal
 * @param expectedNodeIds - All node IDs in expected trajectory
 * @param traversedNodeIds - All node IDs visited during traversal
 * @param outcomes - Per-code comparison outcomes
 * @param otherCodes - Traversed codes unrelated to expected
 */
export function computeBenchmarkMetrics(
  expectedCodes: Set<string>,
  traversedCodes: Set<string>,
  expectedNodeIds: Set<string>,
  traversedNodeIds: Set<string>,
  outcomes: ExpectedCodeOutcome[],
  otherCodes: string[]
): BenchmarkMetrics {
  const exactCount = outcomes.filter((o) => o.status === 'exact').length;
  const undershootCount = outcomes.filter((o) => o.status === 'undershoot').length;
  const overshootCount = outcomes.filter((o) => o.status === 'overshoot').length;
  const missedCount = outcomes.filter((o) => o.status === 'missed').length;

  // Traversal Recall: What fraction of the expected trajectory was visited?
  // Note: Does NOT penalize valid alternative paths that reach the same endpoints.
  const expectedNodesTraversed = [...expectedNodeIds].filter(id => traversedNodeIds.has(id)).length;
  const traversalRecall = expectedNodeIds.size > 0 ? expectedNodesTraversed / expectedNodeIds.size : 0;

  // Final Codes Recall: What fraction of expected endpoints were exactly matched?
  // Penalizes undershoots, overshoots, and misses (only exact matches count).
  const finalCodesRecall = expectedCodes.size > 0 ? exactCount / expectedCodes.size : 0;

  return {
    expectedCount: expectedCodes.size,
    traversedCount: traversedCodes.size,
    expectedNodesCount: expectedNodeIds.size,
    traversedNodesCount: traversedNodeIds.size,
    exactCount,
    undershootCount,
    overshootCount,
    missedCount,
    otherCount: otherCodes.length,
    traversalRecall,
    finalCodesRecall,
    outcomes,
    otherCodes,
  };
}

/**
 * Build parent map from edges (hierarchy + all lateral edges).
 * Used for ancestor expansion. Includes all lateral edges (sevenChrDef,
 * codeFirst, codeAlso, useAdditionalCode) to properly track lateral pathways.
 */
function buildParentMap(edges: GraphEdge[]): Map<string, string> {
  const parentMap = new Map<string, string>();
  for (const edge of edges) {
    if (edge.edge_type === 'hierarchy' || edge.edge_type === 'lateral') {
      parentMap.set(String(edge.target), String(edge.source));
    }
  }
  return parentMap;
}

/**
 * Build hierarchy-only ancestor map for overshoot detection.
 * Overshoot means "went deeper in the HIERARCHY than expected".
 * Lateral targets should NOT be considered overshoots - they are separate pathways.
 */
function buildHierarchyAncestorMap(edges: GraphEdge[]): Map<string, Set<string>> {
  // Build parent map from hierarchy edges ONLY
  const parentMap = new Map<string, string>();
  for (const edge of edges) {
    if (edge.edge_type === 'hierarchy') {
      parentMap.set(String(edge.target), String(edge.source));
    }
  }

  // Build ancestor set for each node
  const ancestorMap = new Map<string, Set<string>>();
  for (const [child] of parentMap) {
    const ancestors = new Set<string>();
    let current = child;
    while (parentMap.has(current)) {
      const parent = parentMap.get(current)!;
      ancestors.add(parent);
      current = parent;
    }
    ancestorMap.set(child, ancestors);
  }

  return ancestorMap;
}

/**
 * Build sevenChrDef parent map: child -> parent.
 * Used to detect when a traversed node is the immediate sevenChrDef parent
 * of an expected code, which means the sevenChrDef batch was presented.
 */
function buildSevenChrDefParentMap(edges: GraphEdge[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const edge of edges) {
    if (edge.edge_type === 'lateral' && edge.rule === 'sevenChrDef') {
      map.set(String(edge.target), String(edge.source));
    }
  }
  return map;
}

/**
 * Build lateral parent map for all lateral edges: child -> { parent, rule }.
 * Used to detect when an expected code was reached via any lateral edge
 * (codeFirst, codeAlso, useAdditionalCode) and its source was finalized.
 */
function buildLateralParentMap(edges: GraphEdge[]): Map<string, { parent: string; rule: string }> {
  const map = new Map<string, { parent: string; rule: string }>();
  for (const edge of edges) {
    if (edge.edge_type === 'lateral' && edge.rule) {
      map.set(String(edge.target), { parent: String(edge.source), rule: edge.rule });
    }
  }
  return map;
}

/**
 * Compute all benchmark visualization data in a single pass.
 *
 * Logic:
 * 1. traversedSet = streamedNodes ∪ finalizedCodes ∪ ancestors(finalizedCodes)
 * 2. Node status: matched > undershoot > traversed > expected
 * 3. Missed edges: source traversed, target not
 * 4. Overshoots: finalized descendant of expected leaf
 */
export function computeBenchmarkVisualization(
  expectedNodes: GraphNode[],
  expectedEdges: GraphEdge[],
  streamedNodeIds: Set<string>,
  finalizedCodes: Set<string>,
  expectedLeaves: Set<string>,
  traversedEdges: GraphEdge[] = [],
  _traversedNodes: GraphNode[] = []  // Kept for API compatibility; graph view only shows expected nodes
): {
  nodes: BenchmarkGraphNode[];
  edges: GraphEdge[];
  missedEdgeMarkers: EdgeMissMarker[];
  overshootMarkers: OvershootMarker[];
  traversedSet: Set<string>;
} {
  // Build parent map for ancestor expansion from BOTH expected and traversed edges
  // This ensures finalized codes that aren't in expected graph (like T36.1X5A when
  // expected was T36.1X5D) still have their ancestors added to traversedSet
  const expectedParentMap = buildParentMap(expectedEdges);
  const traversedParentMap = buildParentMap(traversedEdges);
  const combinedParentMap = new Map([...expectedParentMap, ...traversedParentMap]);

  // Build ancestor map for undershoot detection (includes all edges including lateral)
  const ancestorMap = buildAncestorMap(expectedEdges);

  // Build hierarchy-only ancestor map for overshoot detection
  // Overshoot = went deeper in HIERARCHY than expected
  // Lateral targets should NOT be considered overshoots - they are separate pathways
  const hierarchyAncestorMap = buildHierarchyAncestorMap(expectedEdges);

  // Build sevenChrDef parent map for special handling
  // If expected is depth-7 and its sevenChrDef parent was finalized, that's NOT undershoot
  // because the sevenChrDef batch WAS presented (it IS the children batch for depth-6→7 hop)
  const sevenChrDefParentMap = buildSevenChrDefParentMap(expectedEdges);

  // 1. Build traversedSet = streamedNodes ∪ finalizedCodes ∪ ancestors(all traversed nodes)
  const traversedSet = new Set([...streamedNodeIds, ...finalizedCodes]);

  // Expand ancestors for ALL traversed nodes (not just finalized codes)
  // This ensures intermediate nodes like T84.53, T84.53X are marked traversed
  // even if the backend didn't explicitly send them as nodes
  const nodesToExpand = new Set([...streamedNodeIds, ...finalizedCodes]);
  for (const code of nodesToExpand) {
    let current = code;
    // Use combinedParentMap first (includes traversed edges), fallback to expected
    while (combinedParentMap.has(current) || expectedParentMap.has(current)) {
      const parent = combinedParentMap.get(current) ?? expectedParentMap.get(current);
      if (!parent || parent === 'ROOT') break;
      traversedSet.add(parent);
      current = parent;
    }
  }

  // 2. Compute node statuses
  const nodes: BenchmarkGraphNode[] = expectedNodes.map((node) => {
    const nodeId = node.id;
    const isFinalized = finalizedCodes.has(nodeId);
    const isExpectedLeaf = expectedLeaves.has(nodeId);

    let benchmarkStatus: BenchmarkNodeStatus;

    // Priority: matched > undershoot > traversed > expected
    if (isFinalized && isExpectedLeaf) {
      benchmarkStatus = 'matched';
    } else if (isFinalized) {
      // Check if finalized at ancestor of expected leaf → undershoot
      // EXCEPTION: If this node is the sevenChrDef parent of an expected leaf,
      // that's NOT undershoot - the sevenChrDef batch was presented (it IS the
      // children batch for depth-6→7 transitions), so it's a missed decision
      let isUndershoot = false;
      for (const leaf of expectedLeaves) {
        if (ancestorMap.get(leaf)?.has(nodeId)) {
          // Check if this is the sevenChrDef parent of the leaf
          const sevenChrDefParent = sevenChrDefParentMap.get(leaf);
          if (sevenChrDefParent === nodeId) {
            // This is the sevenChrDef parent - not undershoot, the batch was presented
            continue;
          }
          isUndershoot = true;
          break;
        }
      }
      benchmarkStatus = isUndershoot ? 'undershoot' : 'traversed';
    } else if (traversedSet.has(nodeId)) {
      benchmarkStatus = 'traversed';
    } else {
      benchmarkStatus = 'expected';
    }

    return { ...node, benchmarkStatus };
  });

  // NOTE: We intentionally do NOT add traversed nodes that aren't in the expected graph.
  // The graph view only shows the expected graph with benchmark status overlays.
  // Overshoot codes are visualized via overshootMarkers (arrows from expected leaves).
  // Nodes outside the expected graph appear in the report view via combinedNodes, not here.

  // 3. Compute missed edge markers
  // For each edge: source traversed AND target NOT traversed AND no descendant of target traversed → red X
  // This ensures we only mark actual divergence points, not intermediate nodes that were skipped
  // but where the traversal continued to a descendant.
  const missedEdgeMarkers: EdgeMissMarker[] = [];

  // Build descendant map from ancestor map (reverse lookup)
  const expectedDescendantsMap = new Map<string, Set<string>>();
  for (const node of expectedNodes) {
    expectedDescendantsMap.set(node.id, new Set());
  }
  // For each node, add it to all its ancestors' descendant sets
  for (const node of expectedNodes) {
    const ancestors = ancestorMap.get(node.id);
    if (ancestors) {
      for (const ancestor of ancestors) {
        const descSet = expectedDescendantsMap.get(ancestor);
        if (descSet) {
          descSet.add(node.id);
        }
      }
    }
  }

  for (const edge of expectedEdges) {
    const src = String(edge.source);
    const tgt = String(edge.target);

    // DEFENSIVE CHECK: If target is finalized, it cannot be missed
    // This handles any edge case where traversedSet construction might differ
    if (finalizedCodes.has(tgt)) {
      continue;
    }

    const sourceTraversed = src === 'ROOT' || traversedSet.has(src);
    const targetTraversed = traversedSet.has(tgt);

    if (sourceTraversed && !targetTraversed) {
      // Check if any descendant of target was traversed
      // If so, the traversal continued past this point (just via a different intermediate path)
      const descendants = expectedDescendantsMap.get(tgt) || new Set();
      const anyDescendantTraversed = [...descendants].some(d => traversedSet.has(d));

      if (!anyDescendantTraversed) {
        missedEdgeMarkers.push({
          edgeSource: src,
          edgeTarget: tgt,
          missedCode: tgt,
        });
      }
    }
  }

  // 4. Compute overshoot markers
  // For each finalized: if HIERARCHY descendant of expected leaf → overshoot
  // Use hierarchyAncestorMap to exclude lateral targets (they are separate pathways, not overshoots)
  const overshootMarkers: OvershootMarker[] = [];
  for (const finalized of finalizedCodes) {
    const hierarchyAncestors = hierarchyAncestorMap.get(finalized) ?? new Set();
    for (const leaf of expectedLeaves) {
      if (hierarchyAncestors.has(leaf)) {
        overshootMarkers.push({
          sourceNode: leaf,
          targetCode: finalized,
          depth: calculateDepthFromCode(finalized),
        });
        break;
      }
    }
  }

  return { nodes, edges: expectedEdges, missedEdgeMarkers, overshootMarkers, traversedSet };
}

/**
 * Calculate ICD depth from code structure.
 * Chapter_* = 1, Ranges (X##-X##) = 2, Codes = character count without dots
 */
function calculateDepthFromCode(code: string): number {
  if (code.startsWith('Chapter_')) return 1;
  if (code.includes('-')) return 2;
  return code.replace(/\./g, '').length;
}

/**
 * Phase 1: Update nodes from 'expected' to 'traversed' based on selected_ids.
 * Called during streaming (STEP_FINISHED events) for real-time feedback.
 *
 * Only updates nodes that are:
 * 1. Currently 'expected' status
 * 2. In the traversed codes (nodeId + selected_ids from this event)
 * 3. Part of the expected graph (verified via expectedNodeIds)
 *
 * Note: nodeId is included because it represents the node being processed
 * in the current batch - it has been traversed to reach this point.
 * This is important for:
 * - Placeholder nodes (T84.53X) which are nodeId in sevenChrDef batches
 * - Intermediate nodes which are nodeId in children batches
 */
export function updateTraversedNodes(
  currentNodes: BenchmarkGraphNode[],
  selectedIds: string[],
  expectedNodeIds: Set<string>,
  nodeId?: string
): BenchmarkGraphNode[] {
  // Include both selectedIds and nodeId in traversed set
  const traversedSet = new Set(selectedIds);
  if (nodeId && nodeId !== 'ROOT') {
    traversedSet.add(nodeId);
  }

  return currentNodes.map((node) => {
    // Handle both 'expected' status AND idle/undefined status (from reset)
    const isColorable = node.benchmarkStatus === 'expected' || node.benchmarkStatus === undefined;
    if (
      isColorable &&
      traversedSet.has(node.id) &&
      expectedNodeIds.has(node.id)
    ) {
      return { ...node, benchmarkStatus: 'traversed' as const };
    }
    return node;
  });
}

/**
 * Phase 2: Compute final statuses and markers based on finalized code comparison.
 * Called only at RUN_FINISHED.
 *
 * @param nodes - Current nodes (already have 'traversed' status from Phase 1)
 * @param expectedEdges - Expected graph edges (for ancestor/descendant computation)
 * @param streamedNodeIds - All node IDs selected during traversal (from Phase 1)
 * @param finalizedCodes - Codes finalized by benchmark traversal
 * @param expectedLeaves - User-specified expected finalized codes
 * @param traversedEdges - Edges from traversal (for parent map expansion)
 */
export function computeFinalizedComparison(
  nodes: BenchmarkGraphNode[],
  expectedEdges: GraphEdge[],
  streamedNodeIds: Set<string>,
  finalizedCodes: Set<string>,
  expectedLeaves: Set<string>,
  traversedEdges: GraphEdge[] = [],
  options: { finalizedOnlyMode?: boolean } = {}
): {
  nodes: BenchmarkGraphNode[];
  overshootMarkers: OvershootMarker[];
  missedEdgeMarkers: EdgeMissMarker[];
  traversedSet: Set<string>;
} {
  const { finalizedOnlyMode = false } = options;

  // Build nodeMap for O(1) lookups (avoid O(n) nodes.find() calls)
  const nodeMap = new Map<string, BenchmarkGraphNode>();
  for (const node of nodes) {
    nodeMap.set(node.id, node);
  }

  // Build parent maps for ancestor expansion
  const expectedParentMap = buildParentMap(expectedEdges);
  const traversedParentMap = buildParentMap(traversedEdges);
  const combinedParentMap = new Map([...expectedParentMap, ...traversedParentMap]);

  // Build ancestor map for undershoot detection (includes all edges including lateral)
  const ancestorMap = buildAncestorMap(expectedEdges);

  // Build hierarchy-only ancestor map for overshoot detection
  // Overshoot = went deeper in HIERARCHY than expected
  // Lateral targets should NOT be considered overshoots - they are separate pathways
  const hierarchyAncestorMap = buildHierarchyAncestorMap(expectedEdges);

  // Build sevenChrDef parent map for special handling
  // If expected is depth-7 and its sevenChrDef parent was finalized, that's "missed" not "undershoot"
  // because the sevenChrDef batch WAS presented (it IS the children batch for depth-6→7 hop)
  const sevenChrDefParentMap = buildSevenChrDefParentMap(expectedEdges);

  // Build traversedSet
  // In finalizedOnlyMode (zero-shot without infer precursors): only finalized codes, no ancestor expansion
  // Otherwise: streamedNodes ∪ finalizedCodes ∪ ancestors
  let traversedSet: Set<string>;
  if (finalizedOnlyMode) {
    // Only the finalized codes themselves - no intermediate "traversed" nodes
    traversedSet = new Set(finalizedCodes);
  } else {
    traversedSet = new Set([...streamedNodeIds, ...finalizedCodes]);
    const nodesToExpand = new Set([...streamedNodeIds, ...finalizedCodes]);
    for (const code of nodesToExpand) {
      let current = code;
      while (combinedParentMap.has(current) || expectedParentMap.has(current)) {
        const parent = combinedParentMap.get(current) ?? expectedParentMap.get(current);
        if (!parent || parent === 'ROOT') break;
        traversedSet.add(parent);
        current = parent;
      }
    }
  }

  // First pass: identify undershoot nodes (stopping points)
  // For each missed expected leaf, find its DEEPEST finalized ancestor
  // Only that deepest ancestor is the "undershoot" - other ancestors are just "traversed"
  //
  // EXCEPTION: If the expected leaf has a sevenChrDef parent that was finalized,
  // that's NOT undershoot - it's "missed" because the sevenChrDef batch was presented
  // (the sevenChrDef batch IS the children batch for depth-6→7 transitions)
  const undershootNodes = new Set<string>();
  for (const leaf of expectedLeaves) {
    if (!finalizedCodes.has(leaf)) {
      // Check if this is a sevenChrDef child where the parent was finalized
      // If so, skip undershoot - the sevenChrDef batch was presented, this is "missed"
      const sevenChrDefParent = sevenChrDefParentMap.get(leaf);
      if (sevenChrDefParent && finalizedCodes.has(sevenChrDefParent)) {
        // The depth-6 parent was finalized, sevenChrDef batch was presented
        // This is a missed decision, not undershoot - skip adding to undershootNodes
        continue;
      }

      // This leaf was missed - find deepest finalized ancestor (the stopping point)
      const ancestors = ancestorMap.get(leaf) || new Set();
      let deepestFinalized: string | null = null;
      let deepestDepth = -1;
      for (const ancestor of ancestors) {
        if (finalizedCodes.has(ancestor)) {
          // Get depth from node using O(1) map lookup
          const ancestorNode = nodeMap.get(ancestor);
          const depth = ancestorNode?.depth ?? 0;
          if (depth > deepestDepth) {
            deepestDepth = depth;
            deepestFinalized = ancestor;
          }
        }
      }
      if (deepestFinalized) {
        undershootNodes.add(deepestFinalized);
      }
    }
  }

  // Compute final node statuses
  // In finalizedOnlyMode: matched > undershoot > expected (no "traversed" status)
  // Otherwise: matched > undershoot > traversed > expected
  const updatedNodes = nodes.map((node) => {
    const nodeId = node.id;
    const isExpectedLeaf = expectedLeaves.has(nodeId);
    const isBenchmarkFinalized = finalizedCodes.has(nodeId);

    // Matched: expected leaf that was finalized
    if (isExpectedLeaf && isBenchmarkFinalized) {
      return { ...node, benchmarkStatus: 'matched' as const };
    }

    // Undershoot: the deepest finalized ancestor of a missed expected leaf
    if (undershootNodes.has(nodeId)) {
      return { ...node, benchmarkStatus: 'undershoot' as const };
    }

    // In finalizedOnlyMode, skip "traversed" status - only endpoint comparison matters
    if (!finalizedOnlyMode) {
      // Traversed: finalized nodes on correct path, or any node in traversedSet
      if (isBenchmarkFinalized || traversedSet.has(nodeId)) {
        return { ...node, benchmarkStatus: 'traversed' as const };
      }
    }

    // Expected: nodes not yet visited
    // Always reset to 'expected' to ensure stale statuses from previous runs are cleared.
    // This is critical for cached replays where the RAF to reset nodes is cancelled,
    // leaving benchmarkCombinedNodesRef with stale statuses from the previous run.
    return { ...node, benchmarkStatus: 'expected' as const };
  });

  // Build descendant map from ancestor map (for missed edge detection)
  // Single pass: for each node, add it to all its ancestors' descendant sets
  const expectedDescendantsMap = new Map<string, Set<string>>();
  for (const node of nodes) {
    const ancestors = ancestorMap.get(node.id);
    if (ancestors) {
      for (const ancestor of ancestors) {
        if (!expectedDescendantsMap.has(ancestor)) {
          expectedDescendantsMap.set(ancestor, new Set());
        }
        expectedDescendantsMap.get(ancestor)!.add(node.id);
      }
    }
  }

  // Compute missed edge markers
  // Edge is missed if: source ∈ traversedSet AND target ∉ traversedSet
  // AND no descendant of target ∈ traversedSet
  // DEFENSIVE: Never mark finalized codes as missed (they are by definition traversed)
  const missedEdgeMarkers: EdgeMissMarker[] = [];
  for (const edge of expectedEdges) {
    const src = String(edge.source);
    const tgt = String(edge.target);

    // DEFENSIVE CHECK: If target is finalized, it cannot be missed
    // This handles any edge case where traversedSet construction might differ
    if (finalizedCodes.has(tgt)) {
      continue;
    }

    const sourceTraversed = src === 'ROOT' || traversedSet.has(src);
    const targetTraversed = traversedSet.has(tgt);

    if (sourceTraversed && !targetTraversed) {
      const descendants = expectedDescendantsMap.get(tgt) || new Set();
      const anyDescendantTraversed = [...descendants].some((d) => traversedSet.has(d));

      if (!anyDescendantTraversed) {
        missedEdgeMarkers.push({
          edgeSource: src,
          edgeTarget: tgt,
          missedCode: tgt,
        });
      }
    }
  }

  // Compute overshoot markers
  // Finalized code is overshoot if it's a HIERARCHY descendant of an expected leaf
  // Use hierarchyAncestorMap to exclude lateral targets (they are separate pathways, not overshoots)
  const overshootMarkers: OvershootMarker[] = [];
  for (const finalized of finalizedCodes) {
    const hierarchyAncestors = hierarchyAncestorMap.get(finalized) ?? new Set();
    for (const leaf of expectedLeaves) {
      if (hierarchyAncestors.has(leaf)) {
        overshootMarkers.push({
          sourceNode: leaf,
          targetCode: finalized,
          depth: calculateDepthFromCode(finalized),
        });
        break;
      }
    }
  }

  return { nodes: updatedNodes, overshootMarkers, missedEdgeMarkers, traversedSet };
}

/**
 * Initialize expected graph nodes with 'expected' status.
 */
export function initializeExpectedNodes(nodes: GraphNode[]): BenchmarkGraphNode[] {
  return nodes.map((node) => ({
    ...node,
    benchmarkStatus: 'expected' as BenchmarkNodeStatus,
  }));
}

/**
 * Reset nodes to expected state for re-running benchmark.
 * All nodes start as 'expected' (black dashed outline), then flip to 'traversed'
 * (green solid outline) as streaming progresses via streamingTraversedIds.
 */
export function resetNodesToIdle(nodes: BenchmarkGraphNode[]): BenchmarkGraphNode[] {
  return nodes.map((node) => ({
    ...node,
    benchmarkStatus: 'expected' as const,
  }));
}

/**
 * Extract finalized codes from graph nodes.
 */
export function extractFinalizedCodes(nodes: GraphNode[]): Set<string> {
  return new Set(
    nodes.filter((n) => n.category === 'finalized').map((n) => n.id)
  );
}
