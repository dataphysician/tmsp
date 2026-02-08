/**
 * Node styling utilities for GraphViewer.
 *
 * Centralized functions for determining node fill, stroke, and other visual properties
 * based on node state (finalized, activator, placeholder, benchmark status, etc.)
 */

import type { GraphNode, BenchmarkGraphNode, DecisionPoint } from './types';

// ============================================================================
// Node Type Classification
// ============================================================================

/** Check if node is a leaf node (billable, finalized category, or in finalized codes list) */
function isLeafNode(node: GraphNode, finalizedCodes: Set<string>): boolean {
  return node.billable || node.category === 'finalized' || finalizedCodes.has(node.code);
}

/** Check if node should be styled as finalized (ONLY submitted codes, not billable ancestors) */
export function isFinalizedNode(node: GraphNode, finalizedCodes: Set<string>): boolean {
  return finalizedCodes.has(node.code);
}

/** Check if node is an activator (has sevenChrDef metadata) */
function isActivatorNode(node: GraphNode): boolean {
  return node.category === 'activator';
}

/** Check if node is a placeholder (synthetic X-padded code) */
function isPlaceholderNode(node: GraphNode): boolean {
  return node.category === 'placeholder';
}

/** Check if node is a sevenChrDef finalized node (depth 7 seventh character code) */
export function isSevenChrDefFinalizedNode(node: GraphNode, finalizedCodes?: Set<string>): boolean {
  const code = node.code || node.id;
  // Check if the code is directly in finalized codes
  if (finalizedCodes && code && finalizedCodes.has(code)) return true;
  // For depth 7 nodes (sevenChrDef targets), also check category
  // This handles zero-shot mode where depth 7 nodes are the finalized endpoints
  if (node.depth === 7 && node.category === 'finalized') return true;
  return false;
}

// ============================================================================
// Standard Mode Node Styling
// ============================================================================
// Priority: activator (blue) > placeholder (dashed gray) > finalized/depth7 (green) > ancestor (dark)
// Styling aligned with benchmark mode for visual consistency

export function getNodeFill(node: GraphNode, finalizedCodes: Set<string>): string {
  if (isActivatorNode(node)) return '#ffffff';
  if (isSevenChrDefFinalizedNode(node, finalizedCodes)) return '#dcfce7';  // Match benchmark 'matched' fill
  if (isFinalizedNode(node, finalizedCodes)) return '#dcfce7';  // Match benchmark 'matched' fill
  if (isPlaceholderNode(node)) return '#ffffff';
  return '#ffffff';
}

export function getNodeStroke(node: GraphNode, finalizedCodes: Set<string>): string {
  if (isActivatorNode(node)) return '#3b82f6';
  if (isSevenChrDefFinalizedNode(node, finalizedCodes)) return '#16a34a';  // Match benchmark green
  if (isFinalizedNode(node, finalizedCodes)) return '#16a34a';  // Match benchmark green
  if (isPlaceholderNode(node)) return '#94a3b8';
  return '#1e293b';  // Match benchmark 'expected' stroke
}

export function getNodeStrokeWidth(node: GraphNode, finalizedCodes: Set<string>): number {
  if (isActivatorNode(node)) return 2;
  if (isSevenChrDefFinalizedNode(node, finalizedCodes)) return 4.5;
  if (isFinalizedNode(node, finalizedCodes)) return 4.5;
  if (isPlaceholderNode(node)) return 1.5;
  return 1.5;
}

// ============================================================================
// Overlay Colors
// ============================================================================

export function getOverlayColors(
  node: GraphNode,
  finalizedCodes: Set<string>
): { bgColor: string; borderColor: string } {
  const activator = isActivatorNode(node);
  const placeholder = isPlaceholderNode(node);
  const finalized = !placeholder && isFinalizedNode(node, finalizedCodes);

  // Aligned with benchmark mode colors for consistency
  const bgColor = activator ? 'rgba(239, 246, 255, 0.98)' :
    placeholder ? 'rgba(248, 250, 252, 0.98)' :
      finalized ? 'rgba(220, 252, 231, 0.98)' :  // Match benchmark 'matched' fill (#dcfce7)
        'rgba(255, 255, 255, 0.98)';

  const borderColor = activator ? '#2563eb' :
    placeholder ? '#94a3b8' :
      finalized ? '#16a34a' :  // Match benchmark green
        '#1e293b';  // Match benchmark 'expected' stroke

  return { bgColor, borderColor };
}

// ============================================================================
// Batch/Decision Helpers
// ============================================================================

/** Normalize batch label to lowercase name without " batch" suffix */
export function normalizeBatchName(label: string | undefined): string {
  return (label || '').replace(' batch', '').toLowerCase();
}

/** Check if a decision should be included for overlay display */
export function shouldIncludeDecision(
  dec: DecisionPoint,
  node: GraphNode,
  finalizedCodes: Set<string>,
  nodesWithSevenChrDefChildren: Set<string>
): boolean {
  const batchName = normalizeBatchName(dec.current_label);
  const leaf = isLeafNode(node, finalizedCodes);
  const placeholderOrActivator = isPlaceholderNode(node) || isActivatorNode(node);

  const hasReasoning = dec.candidates.some(c => c.reasoning && c.reasoning.trim().length > 0);
  const hasCandidates = dec.candidates.length > 0;

  if ((leaf || placeholderOrActivator) && batchName === 'children') {
    if (hasReasoning || hasCandidates) {
      return true;
    }
    return false;
  }

  if (nodesWithSevenChrDefChildren.has(node.id) && batchName === 'children') {
    return false;
  }

  return true;
}

// ============================================================================
// Benchmark Mode Node Styling
// ============================================================================

/** Get effective benchmark status, considering streaming traversed IDs */
export function getEffectiveBenchmarkStatus(
  node: BenchmarkGraphNode,
  streamingTraversedIds?: Set<string>
): BenchmarkGraphNode['benchmarkStatus'] {
  if (streamingTraversedIds && node.benchmarkStatus === 'expected' && streamingTraversedIds.has(node.id)) {
    return 'traversed';
  }
  return node.benchmarkStatus;
}

export function getBenchmarkNodeFill(
  node: BenchmarkGraphNode,
  streamingTraversedIds?: Set<string>
): string {
  const status = getEffectiveBenchmarkStatus(node, streamingTraversedIds);
  switch (status) {
    case 'expected':
      return '#ffffff';
    case 'traversed':
      return '#ffffff';
    case 'matched':
      return '#dcfce7';
    case 'undershoot':
      return '#fef3c7';
    default:
      return '#ffffff';
  }
}

export function getBenchmarkNodeStroke(
  node: BenchmarkGraphNode,
  streamingTraversedIds?: Set<string>
): string {
  const status = getEffectiveBenchmarkStatus(node, streamingTraversedIds);
  switch (status) {
    case 'expected':
      return '#1e293b';
    case 'traversed':
      return '#16a34a';
    case 'matched':
      return '#16a34a';
    case 'undershoot':
      return '#16a34a';
    default:
      return '#1e293b';
  }
}

export function getBenchmarkNodeStrokeWidth(
  node: BenchmarkGraphNode,
  streamingTraversedIds?: Set<string>
): number {
  const status = getEffectiveBenchmarkStatus(node, streamingTraversedIds);
  switch (status) {
    case 'matched':
      return 4.5;
    case 'traversed':
    case 'undershoot':
      return 4;
    default:
      return 1.5;
  }
}

export function getBenchmarkNodeStrokeDasharray(
  node: BenchmarkGraphNode,
  streamingTraversedIds?: Set<string>
): string | null {
  const status = getEffectiveBenchmarkStatus(node, streamingTraversedIds);
  return status === 'expected' ? '4,2' : null;
}
