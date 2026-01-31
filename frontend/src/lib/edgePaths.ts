/**
 * Edge path creation utilities for GraphViewer.
 *
 * Functions for creating SVG path strings for hierarchy and lateral edges.
 */

import type { GraphEdge } from './types';

const ARROW_PADDING_STRAIGHT = 6;
const ARROW_PADDING_CURVED = 8;

/**
 * Create straight edge path for hierarchy edges.
 * Path goes from bottom center of source to top center of target.
 */
export function createEdgePath(
  edge: GraphEdge,
  positions: Map<string, { x: number; y: number }>,
  nodeHeight: number
): string {
  const src = positions.get(String(edge.source));
  const tgt = positions.get(String(edge.target));

  if (!src || !tgt) return '';

  const x1 = src.x;
  const y1 = src.y + nodeHeight / 2;
  const x2 = tgt.x;
  const y2 = tgt.y - nodeHeight / 2 - ARROW_PADDING_STRAIGHT;

  return `M${x1},${y1} L${x2},${y2}`;
}

/**
 * Create curved edge path for lateral edges (quadratic Bezier).
 * Path arcs upward to avoid overlapping with hierarchy edges.
 */
export function createCurvedEdgePath(
  edge: GraphEdge,
  positions: Map<string, { x: number; y: number }>,
  nodeHeight: number
): string {
  const src = positions.get(String(edge.source));
  const tgt = positions.get(String(edge.target));

  if (!src || !tgt) return '';

  const x1 = src.x;
  const y1 = src.y + nodeHeight / 2;
  const x2 = tgt.x;
  const y2 = tgt.y - nodeHeight / 2 - ARROW_PADDING_CURVED;

  // Control point for quadratic Bezier curve
  const dx = x2 - x1;
  const cx = x1 + dx / 2;
  const cy = Math.min(y1, y2) - Math.abs(dx) * 0.2 - 20;

  return `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`;
}

/**
 * Determine if an edge should use curved path (lateral edges).
 */
export function shouldUseCurvedPath(edge: GraphEdge): boolean {
  return edge.edge_type === 'lateral';
}

/**
 * Create appropriate edge path based on edge type.
 */
export function createPath(
  edge: GraphEdge,
  positions: Map<string, { x: number; y: number }>,
  nodeHeight: number
): string {
  if (shouldUseCurvedPath(edge)) {
    return createCurvedEdgePath(edge, positions, nodeHeight);
  }
  return createEdgePath(edge, positions, nodeHeight);
}

/**
 * Get a point along a straight edge path at parameter t (0-1).
 * Returns coordinates near the arrowhead when t is close to 1.
 */
export function getPointOnStraightEdge(
  srcPos: { x: number; y: number },
  tgtPos: { x: number; y: number },
  nodeHeight: number,
  t: number
): { x: number; y: number } {
  const x1 = srcPos.x;
  const y1 = srcPos.y + nodeHeight / 2;
  const x2 = tgtPos.x;
  const y2 = tgtPos.y - nodeHeight / 2 - ARROW_PADDING_STRAIGHT;

  return {
    x: x1 + t * (x2 - x1),
    y: y1 + t * (y2 - y1)
  };
}

/**
 * Get a point along a curved (Bezier) edge path at parameter t (0-1).
 * Uses quadratic Bezier formula: B(t) = (1-t)²P0 + 2(1-t)t*P1 + t²P2
 */
export function getPointOnCurvedEdge(
  srcPos: { x: number; y: number },
  tgtPos: { x: number; y: number },
  nodeHeight: number,
  t: number
): { x: number; y: number } {
  const x1 = srcPos.x;
  const y1 = srcPos.y + nodeHeight / 2;
  const x2 = tgtPos.x;
  const y2 = tgtPos.y - nodeHeight / 2 - ARROW_PADDING_CURVED;

  // Control point for quadratic Bezier
  const dx = x2 - x1;
  const cx = x1 + dx / 2;
  const cy = Math.min(y1, y2) - Math.abs(dx) * 0.2 - 20;

  // Quadratic Bezier formula
  const oneMinusT = 1 - t;
  return {
    x: oneMinusT * oneMinusT * x1 + 2 * oneMinusT * t * cx + t * t * x2,
    y: oneMinusT * oneMinusT * y1 + 2 * oneMinusT * t * cy + t * t * y2
  };
}
