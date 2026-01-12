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
 * Considers hierarchy edges AND sevenChrDef lateral edges (which represent
 * parent-child relationships for 7th character codes).
 */
export function buildAncestorMap(edges: GraphEdge[]): Map<string, Set<string>> {
  // Build parent map from hierarchy edges AND sevenChrDef lateral edges
  // sevenChrDef edges are lateral but still represent ancestry (6th char -> 7th char)
  const parentMap = new Map<string, string>();
  for (const edge of edges) {
    const isHierarchy = edge.edge_type === 'hierarchy';
    const isSevenChrDef = edge.edge_type === 'lateral' && edge.rule === 'sevenChrDef';

    if (isHierarchy || isSevenChrDef) {
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
 */
export function compareFinalizedCodes(
  expectedCodes: Set<string>,
  expectedAncestorMap: Map<string, Set<string>>,
  traversedCodes: Set<string>,
  traversedAncestorMap: Map<string, Set<string>>
): { outcomes: ExpectedCodeOutcome[]; otherCodes: string[] } {
  const outcomes: ExpectedCodeOutcome[] = [];
  const matchedTraversed = new Set<string>();

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

    // Check undershoot: traversed code is ancestor of expected
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

  // Traversal Accuracy: How well did traversal cover the expected trajectory?
  // (Expected nodes that were traversed) / (Total expected nodes)
  const expectedNodesTraversed = [...expectedNodeIds].filter(id => traversedNodeIds.has(id)).length;
  const traversalAccuracy = expectedNodeIds.size > 0 ? expectedNodesTraversed / expectedNodeIds.size : 0;

  // Final Codes Accuracy: How accurate were finalization decisions?
  // exact / expected (penalizes over/under shoots and misses)
  const finalCodesAccuracy = expectedCodes.size > 0 ? exactCount / expectedCodes.size : 0;

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
    traversalAccuracy,
    finalCodesAccuracy,
    outcomes,
    otherCodes,
  };
}

/**
 * Build parent map from edges (hierarchy + sevenChrDef only).
 * Used for ancestor expansion.
 */
function buildParentMap(edges: GraphEdge[]): Map<string, string> {
  const parentMap = new Map<string, string>();
  for (const edge of edges) {
    if (edge.edge_type === 'hierarchy' ||
        (edge.edge_type === 'lateral' && edge.rule === 'sevenChrDef')) {
      parentMap.set(String(edge.target), String(edge.source));
    }
  }
  return parentMap;
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

  // Build ancestor map for undershoot/overshoot detection (expected graph only)
  const ancestorMap = buildAncestorMap(expectedEdges);

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
      let isUndershoot = false;
      for (const leaf of expectedLeaves) {
        if (ancestorMap.get(leaf)?.has(nodeId)) {
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
  // For each finalized: if descendant of expected leaf → overshoot
  const overshootMarkers: OvershootMarker[] = [];
  for (const finalized of finalizedCodes) {
    const ancestors = ancestorMap.get(finalized) ?? new Set();
    for (const leaf of expectedLeaves) {
      if (ancestors.has(leaf)) {
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
 * Initialize expected graph nodes with 'expected' status.
 */
export function initializeExpectedNodes(nodes: GraphNode[]): BenchmarkGraphNode[] {
  return nodes.map((node) => ({
    ...node,
    benchmarkStatus: 'expected' as BenchmarkNodeStatus,
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

