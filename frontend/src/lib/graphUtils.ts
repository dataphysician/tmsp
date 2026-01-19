import type { GraphNode, GraphEdge } from './types';

/**
 * Merge nodes by ID, with incoming nodes overwriting existing ones.
 * Used for two-stage rendering (Stream + Reconcile) where STEP_FINISHED
 * has authoritative depth from selected_details.
 */
export function mergeById(existing: GraphNode[], incoming: GraphNode[]): GraphNode[] {
  const map = new Map(existing.map(n => [n.id, n]));
  for (const node of incoming) {
    map.set(node.id, node);
  }
  return [...map.values()];
}

/**
 * Merge edges by source|target key, with incoming edges overwriting existing ones.
 */
export function mergeByKey(existing: GraphEdge[], incoming: GraphEdge[]): GraphEdge[] {
  const key = (e: GraphEdge) => `${e.source}|${e.target}`;
  const map = new Map(existing.map(e => [key(e), e]));
  for (const edge of incoming) {
    map.set(key(edge), edge);
  }
  return [...map.values()];
}

/**
 * Calculate ICD depth from code structure.
 * - Chapter_* = 1
 * - Ranges (X##-X##) = 2
 * - Codes = character count without dots
 */
export function calculateDepthFromCode(code: string): number {
  if (code.startsWith('Chapter_')) return 1;
  if (code.includes('-')) return 2; // Range like I20-I25
  // For actual codes: count characters excluding dots
  return code.replace(/\./g, '').length;
}

/**
 * Parse code input string into an array of normalized codes.
 * Splits by comma, newline, tab, or space and normalizes to uppercase.
 */
export function parseCodeInput(input: string): string[] {
  return input
    .split(/[,\n\t\s]+/)
    .map(c => c.trim().toUpperCase())
    .filter(c => c.length > 0);
}

/**
 * Get all descendant node IDs from a given node using BFS traversal.
 * Used for removing descendants when rewinding from a node.
 *
 * @param nodeId - The node to find descendants of
 * @param edges - The graph edges to traverse
 * @returns Set of all descendant node IDs (not including the source node)
 */
export function getDescendantNodeIds(nodeId: string, edges: GraphEdge[]): Set<string> {
  // Build parent -> children map
  const childMap = new Map<string, string[]>();
  for (const edge of edges) {
    const src = edge.source as string;
    const tgt = edge.target as string;
    if (!childMap.has(src)) {
      childMap.set(src, []);
    }
    childMap.get(src)!.push(tgt);
  }

  // BFS to collect all descendants
  const descendants = new Set<string>();
  const queue = childMap.get(nodeId) || [];

  while (queue.length > 0) {
    const current = queue.shift()!;
    if (descendants.has(current)) continue;
    descendants.add(current);
    const children = childMap.get(current) || [];
    queue.push(...children);
  }

  return descendants;
}

/**
 * Extract batch type from decision's current_label.
 * E.g., "children batch" -> "children"
 *
 * @param currentLabel - The decision point's current_label
 * @returns The batch type (e.g., "children", "codeFirst", etc.)
 */
export function extractBatchType(currentLabel: string): string {
  // Format is "{batchType} batch"
  const match = currentLabel.match(/^(\w+)\s+batch$/);
  return match ? match[1] : 'children';
}