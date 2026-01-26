/**
 * Overlay rendering utilities for GraphViewer.
 *
 * Provides functions for rendering expanded node overlays with batch panels,
 * decision information, and feedback buttons.
 */

import * as d3 from 'd3';
import type { GraphNode, GraphEdge, DecisionPoint, BenchmarkGraphNode } from './types';
import { wrapText } from './textUtils';
import { normalizeBatchName, shouldIncludeDecision, shouldShowBatch, getOverlayColors } from './nodeStyles';

/** Context passed to overlay rendering functions */
export interface OverlayContext {
  /** D3 selection for the overlay group */
  overlayGroup: d3.Selection<SVGGElement, unknown, null, undefined>;
  /** Node positions map */
  positions: Map<string, { x: number; y: number }>;
  /** All nodes in the graph */
  nodeMap: Map<string, GraphNode>;
  /** Decision history */
  decisions: DecisionPoint[];
  /** Set of finalized codes */
  finalizedCodesSet: Set<string>;
  /** Nodes with sevenChrDef children */
  nodesWithSevenChrDefChildren: Set<string>;
  /** Map from sevenChrDef target to parent */
  sevenChrDefParentMap: Map<string, string>;
  /** Hierarchy edges for ancestor lookup */
  hierarchyEdges: GraphEdge[];
  /** Expected leaves in benchmark mode */
  expectedLeaves: Set<string>;
  /** Whether in benchmark mode */
  benchmarkMode: boolean;
  /** Whether rewind is allowed */
  allowRewind: boolean;
  /** Ref for pinned node ID */
  pinnedNodeIdRef: React.MutableRefObject<string | null>;
  /** Ref for hide timeout */
  hideTimeoutRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>;
  /** Ref for active overlay node */
  activeOverlayNodeRef: React.MutableRefObject<string | null>;
  /** Ref for last interaction time */
  lastInteractionTime: React.MutableRefObject<number>;
  /** Callback for rewind click */
  onNodeRewindClick?: (nodeId: string, batchType?: string, feedback?: string) => void;
  /** Callback to hide overlay */
  hideExpandedNode: () => void;
  /** Callback to cancel hide timeout */
  cancelHideTimeout: () => void;
  /** All children map (for basic overlay children section) */
  allChildren: Map<string, string[]>;
  /** SevenChrDef edges (for basic overlay) */
  sevenChrDefEdges: GraphEdge[];
  /** Other lateral edges (for basic overlay) */
  otherLateralEdges: GraphEdge[];
}

/** Constants for overlay layout */
const OVERLAY_PADDING = 14;
const LINE_HEIGHT = 15;
const MAX_CHARS_PER_LINE = 45;
const PANEL_WIDTH = 340;
const PANEL_GAP = 16;
const MAX_PANELS = 3;

/**
 * Main entry point for showing node overlay.
 * Coordinates between batch panels and regular overlays.
 */
export function showNodeOverlay(
  node: GraphNode,
  ctx: OverlayContext,
  cancelShowTimeout: () => void
): void {
  // Cancel any pending show/hide timeouts
  cancelShowTimeout();
  ctx.cancelHideTimeout();

  // Track which node's overlay is active
  ctx.activeOverlayNodeRef.current = node.id;

  // Remove any existing expanded overlays
  ctx.overlayGroup.selectAll('*').remove();

  // Try batch panels first (for nodes with decisions)
  if (renderBatchPanelsOverlay(node, ctx)) {
    return;
  }

  // Fall back to regular overlay
  renderRegularOverlay(node, ctx);
}

/**
 * Get the full label for a node, combining ancestor label for activator nodes.
 */
function getFullLabel(
  node: GraphNode,
  sevenChrDefParentMap: Map<string, string>,
  nodeMap: Map<string, GraphNode>,
  hierarchyEdges: GraphEdge[]
): string {
  let fullLabel = node.label;
  if (node.category === 'activator' || node.depth === 7) {
    const ancestorLabel = getActivatorAncestorLabel(node.id, sevenChrDefParentMap, nodeMap, hierarchyEdges);
    const labelValue = node.label.includes(': ') ? node.label.split(': ').slice(1).join(': ') : node.label;
    if (ancestorLabel) {
      fullLabel = `${ancestorLabel}, ${labelValue}`;
    }
  }
  return fullLabel;
}

/**
 * Get ancestor label for activator nodes.
 */
function getActivatorAncestorLabel(
  nodeId: string,
  sevenChrDefParentMap: Map<string, string>,
  nodeMap: Map<string, GraphNode>,
  hierarchyEdges: GraphEdge[]
): string {
  const parentId = sevenChrDefParentMap.get(nodeId);
  if (!parentId) return '';

  let currentId = parentId;
  while (currentId && currentId !== 'ROOT') {
    const node = nodeMap.get(currentId);
    if (node && node.category !== 'placeholder') {
      return node.label;
    }
    const parentEdge = hierarchyEdges.find(e => String(e.target) === currentId);
    if (parentEdge) {
      currentId = String(parentEdge.source);
    } else {
      break;
    }
  }
  return '';
}

/**
 * Get the display code for a node.
 * For sevenChrDef targets, combines parent code + 7th character.
 */
function getDisplayCode(
  node: GraphNode,
  sevenChrDefParentMap: Map<string, string>,
  nodeMap: Map<string, GraphNode>
): string {
  if (node.depth === 7) {
    return node.code;
  }
  const parentId = sevenChrDefParentMap.get(node.id);
  if (parentId) {
    const parentNode = nodeMap.get(parentId);
    if (parentNode) {
      return parentNode.code + node.code;
    }
  }
  return node.code;
}

/**
 * Calculate panel data for decisions to display.
 */
interface PanelData {
  decision: DecisionPoint;
  batchName: string;
  labelLines: string[];
  selectedCandidates: DecisionPoint['candidates'];
  selectedItems: { code: string; labelLines: string[] }[];
  reasoningLines: string[];
  showBatch: boolean;
  contentHeight: number;
}

function calculatePanelData(
  nodeDecisions: DecisionPoint[],
  fullLabel: string,
  node: GraphNode,
  finalizedCodesSet: Set<string>,
  nodesWithSevenChrDefChildren: Set<string>,
  allowRewind: boolean,
  benchmarkMode: boolean,
  onNodeRewindClick?: (nodeId: string, batchType?: string, feedback?: string) => void
): PanelData[] {
  const panelData: PanelData[] = [];

  nodeDecisions.slice(0, MAX_PANELS).forEach((decision) => {
    const batchName = normalizeBatchName(decision.current_label);
    const labelLines = wrapText(fullLabel, MAX_CHARS_PER_LINE);
    const selectedCandidates = decision.candidates.filter(c => c.selected);
    const showBatch = shouldShowBatch(batchName, node, finalizedCodesSet, nodesWithSevenChrDefChildren);

    const selectedItems: { code: string; labelLines: string[] }[] = [];
    selectedCandidates.forEach((candidate) => {
      const labelMaxChars = MAX_CHARS_PER_LINE - 4;
      selectedItems.push({
        code: candidate.code,
        labelLines: wrapText(candidate.label, labelMaxChars),
      });
    });

    const firstSelected = selectedCandidates[0];
    const reasoningLines = firstSelected?.reasoning
      ? wrapText(firstSelected.reasoning, MAX_CHARS_PER_LINE - 2)
      : [];

    // Calculate content height
    let contentHeight = OVERLAY_PADDING;
    contentHeight += 16; // Code + (Billable) row
    contentHeight += 8; // Gap after code line
    contentHeight += labelLines.length * LINE_HEIGHT;

    if (showBatch) {
      contentHeight += 18; // Gap + batch name header
      contentHeight += 6; // Gap after header

      if (selectedCandidates.length > 0) {
        selectedItems.forEach((item, idx) => {
          contentHeight += LINE_HEIGHT; // Code line
          contentHeight += item.labelLines.length * LINE_HEIGHT;
          if (idx < selectedItems.length - 1) {
            contentHeight += 6;
          }
        });
      } else {
        contentHeight += LINE_HEIGHT; // "None Selected"
      }

      if (reasoningLines.length > 0) {
        contentHeight += 16; // Gap before "Reasoning:" header
        contentHeight += LINE_HEIGHT; // "Reasoning:" header
        contentHeight += reasoningLines.length * LINE_HEIGHT;
      }

      if (allowRewind && !benchmarkMode && node.id !== 'ROOT' && onNodeRewindClick) {
        contentHeight += 12; // Gap before button
        contentHeight += 26; // Button height
      }
    }

    contentHeight += OVERLAY_PADDING; // Bottom padding

    panelData.push({
      decision,
      batchName,
      labelLines,
      selectedCandidates,
      selectedItems,
      reasoningLines,
      showBatch,
      contentHeight,
    });
  });

  return panelData;
}

/**
 * Render a batch panel for a decision.
 */
function renderBatchPanel(
  panel: d3.Selection<SVGGElement, unknown, null, undefined>,
  data: PanelData,
  node: GraphNode,
  panelWidth: number,
  panelHeight: number,
  displayCode: string,
  ctx: OverlayContext
): void {
  const billableText = node.billable ? '(Billable)' : '(Non-Billable)';

  // Background
  panel.append('rect')
    .attr('class', 'batch-panel-bg')
    .attr('width', panelWidth)
    .attr('height', panelHeight)
    .attr('rx', 6)
    .attr('fill', 'rgba(255, 255, 255, 0.98)')
    .attr('stroke', '#e2e8f0')
    .attr('stroke-width', 1.5)
    .attr('filter', 'drop-shadow(0 2px 6px rgba(0, 0, 0, 0.1))');

  let yPos = OVERLAY_PADDING + 14;

  // Code line
  const codeLineText = panel.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', 13);

  codeLineText.append('tspan')
    .attr('font-weight', 700)
    .attr('font-family', 'ui-monospace, monospace')
    .attr('fill', '#0f172a')
    .text(displayCode);

  codeLineText.append('tspan')
    .attr('font-weight', 500)
    .attr('fill', node.billable ? '#15803d' : '#64748b')
    .text(` ${billableText}`);

  // Label lines
  yPos += 10;
  data.labelLines.forEach((line) => {
    yPos += LINE_HEIGHT;
    panel.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('fill', '#334155')
      .text(line);
  });

  if (!data.showBatch) return;

  // Batch name header
  yPos += 20;
  panel.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', 11)
    .attr('font-weight', 600)
    .attr('fill', '#7c3aed')
    .text(`${data.batchName}:`);

  yPos += 6;

  if (data.selectedCandidates.length > 0) {
    data.selectedItems.forEach((item, itemIdx) => {
      yPos += LINE_HEIGHT;
      panel.append('text')
        .attr('x', OVERLAY_PADDING + 8)
        .attr('y', yPos)
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('font-family', 'ui-monospace, monospace')
        .attr('fill', '#1e293b')
        .text(item.code);

      item.labelLines.forEach((line) => {
        yPos += LINE_HEIGHT;
        panel.append('text')
          .attr('x', OVERLAY_PADDING + 16)
          .attr('y', yPos)
          .attr('font-size', 10)
          .attr('fill', '#475569')
          .text(line);
      });

      if (itemIdx < data.selectedItems.length - 1) {
        yPos += 4;
      }
    });
  } else {
    yPos += LINE_HEIGHT;
    panel.append('text')
      .attr('x', OVERLAY_PADDING + 8)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('font-style', 'italic')
      .attr('fill', '#94a3b8')
      .text('None Selected');
  }

  // Reasoning section
  if (data.reasoningLines.length > 0) {
    yPos += 18;
    panel.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('fill', '#64748b')
      .text('Reasoning:');

    data.reasoningLines.forEach((line) => {
      yPos += LINE_HEIGHT;
      panel.append('text')
        .attr('x', OVERLAY_PADDING + 8)
        .attr('y', yPos)
        .attr('font-size', 10)
        .attr('font-style', 'italic')
        .attr('fill', '#64748b')
        .text(line);
    });
  }

  // Investigate Batch button
  if (ctx.allowRewind && !ctx.benchmarkMode && node.id !== 'ROOT' && ctx.onNodeRewindClick) {
    const buttonWidth = 120;
    const buttonHeight = 24;
    const batchType = data.decision.current_label.match(/^(\w+)\s+batch$/)?.[1] || 'children';

    yPos += 12;

    const feedbackBtn = panel.append('g')
      .attr('class', 'feedback-button')
      .attr('transform', `translate(${OVERLAY_PADDING}, ${yPos})`)
      .style('cursor', 'pointer')
      .style('pointer-events', 'auto');

    feedbackBtn.append('rect')
      .attr('width', buttonWidth)
      .attr('height', buttonHeight)
      .attr('rx', 4)
      .attr('fill', '#7c3aed')
      .style('pointer-events', 'auto')
      .style('cursor', 'pointer')
      .on('mousedown', (event: MouseEvent) => event.stopPropagation())
      .on('mouseup', (event: MouseEvent) => event.stopPropagation())
      .on('click', (event: MouseEvent) => {
        event.stopPropagation();
        event.preventDefault();
        ctx.lastInteractionTime.current = Date.now();
        ctx.onNodeRewindClick?.(node.id, batchType, '');
      });

    feedbackBtn.append('text')
      .attr('x', buttonWidth / 2)
      .attr('y', buttonHeight / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .style('pointer-events', 'none')
      .style('user-select', 'none')
      .text('Investigate Batch');

    feedbackBtn.on('mouseenter', function () {
      d3.select(this).select('rect').attr('fill', '#6d28d9');
    });
    feedbackBtn.on('mouseleave', function () {
      d3.select(this).select('rect').attr('fill', '#7c3aed');
    });
  }
}

/**
 * Render batch panels overlay for a node with multiple decisions.
 */
export function renderBatchPanelsOverlay(
  node: GraphNode,
  ctx: OverlayContext
): boolean {
  const pos = ctx.positions.get(node.id);
  if (!pos) return false;

  const benchmarkNode = node as BenchmarkGraphNode;
  const isTraversedInBenchmark = ctx.benchmarkMode &&
    benchmarkNode.benchmarkStatus &&
    benchmarkNode.benchmarkStatus !== 'expected';

  // Find ALL decisions for this node
  const nodeDecisionsRaw = ctx.decisions
    ? ctx.decisions.filter(dec => dec.current_node === node.id)
    : [];

  // Deduplicate panels
  const nodeDecisions = nodeDecisionsRaw.reduce((acc, dec) => {
    const normalizedLabel = normalizeBatchName(dec.current_label);
    const normalizedCodes = (dec.selected_codes || [])
      .map(c => c.replace(/[^A-Za-z0-9.]/g, ''))
      .sort()
      .join(',');
    const key = `${normalizedLabel}:${normalizedCodes}`;
    if (!acc.seen.has(key)) {
      acc.seen.add(key);
      acc.result.push(dec);
    }
    return acc;
  }, { seen: new Set<string>(), result: [] as typeof nodeDecisionsRaw }).result
    .filter(dec => shouldIncludeDecision(dec, node, ctx.finalizedCodesSet, ctx.nodesWithSevenChrDefChildren));

  const shouldShowBatchPanels = (isTraversedInBenchmark && nodeDecisions.length > 0) ||
    (!ctx.benchmarkMode && nodeDecisions.length > 0);

  if (!shouldShowBatchPanels) return false;

  const fullLabel = getFullLabel(node, ctx.sevenChrDefParentMap, ctx.nodeMap, ctx.hierarchyEdges);
  const displayCode = getDisplayCode(node, ctx.sevenChrDefParentMap, ctx.nodeMap);

  const panelData = calculatePanelData(
    nodeDecisions,
    fullLabel,
    node,
    ctx.finalizedCodesSet,
    ctx.nodesWithSevenChrDefChildren,
    ctx.allowRewind,
    ctx.benchmarkMode,
    ctx.onNodeRewindClick
  );

  if (panelData.length === 0) return false;

  // Calculate dimensions
  const maxPanels = Math.min(panelData.length, MAX_PANELS);
  const panelHeight = Math.max(...panelData.map(p => p.contentHeight), 140);
  const totalWidth = maxPanels * PANEL_WIDTH + (maxPanels - 1) * PANEL_GAP;
  const startX = pos.x - totalWidth / 2;
  const startY = pos.y - panelHeight / 2;

  // Render each panel
  panelData.forEach((data, idx) => {
    const panelX = startX + idx * (PANEL_WIDTH + PANEL_GAP);
    const panel = ctx.overlayGroup.append('g')
      .attr('class', 'batch-panel')
      .attr('transform', `translate(${panelX}, ${startY})`)
      .on('click', (event: MouseEvent) => event.stopPropagation())
      .on('mousedown', (event: MouseEvent) => event.stopPropagation())
      .on('mouseenter', () => ctx.cancelHideTimeout())
      .on('mouseleave', () => {
        if (!ctx.pinnedNodeIdRef.current) {
          ctx.hideTimeoutRef.current = setTimeout(() => ctx.hideExpandedNode(), 500);
        }
      });

    renderBatchPanel(panel, data, node, PANEL_WIDTH, panelHeight, displayCode, ctx);
  });

  // Show indicator if more panels exist
  if (nodeDecisions.length > maxPanels) {
    const indicatorX = startX + totalWidth + 10;
    ctx.overlayGroup.append('text')
      .attr('x', indicatorX)
      .attr('y', startY + panelHeight / 2)
      .attr('font-size', 12)
      .attr('fill', '#7c3aed')
      .attr('font-weight', 600)
      .text(`+${nodeDecisions.length - maxPanels}`);
  }

  return true;
}

/**
 * Render a regular (single) overlay for a node.
 */
export function renderRegularOverlay(
  node: GraphNode,
  ctx: OverlayContext
): void {
  const pos = ctx.positions.get(node.id);
  if (!pos) return;

  const fullLabel = getFullLabel(node, ctx.sevenChrDefParentMap, ctx.nodeMap, ctx.hierarchyEdges);
  const displayCode = getDisplayCode(node, ctx.sevenChrDefParentMap, ctx.nodeMap);
  const billableText = node.billable ? '(Billable)' : '(Non-Billable)';
  const labelLines = wrapText(fullLabel, MAX_CHARS_PER_LINE);

  // Find decision for this node (traverse mode)
  const nodeDecision = !ctx.benchmarkMode && ctx.decisions
    ? ctx.decisions.find(dec => {
      if (dec.current_node !== node.id) return false;
      return shouldIncludeDecision(dec, node, ctx.finalizedCodesSet, ctx.nodesWithSevenChrDefChildren);
    })
    : null;

  // If we have a decision, render with batch info
  if (nodeDecision) {
    const batchName = normalizeBatchName(nodeDecision.current_label);
    const selectedCandidates = nodeDecision.candidates.filter(c => c.selected);

    const selectedItems: { code: string; labelLines: string[] }[] = [];
    selectedCandidates.forEach((candidate) => {
      selectedItems.push({
        code: candidate.code,
        labelLines: wrapText(candidate.label, MAX_CHARS_PER_LINE - 4),
      });
    });

    const firstSelected = selectedCandidates[0];
    const reasoningLines = firstSelected?.reasoning
      ? wrapText(firstSelected.reasoning, MAX_CHARS_PER_LINE - 2)
      : [];

    // Calculate content height
    let contentHeight = OVERLAY_PADDING;
    contentHeight += 16;
    contentHeight += 8;
    contentHeight += labelLines.length * LINE_HEIGHT;
    contentHeight += 18;
    contentHeight += 6;

    selectedItems.forEach((item, idx) => {
      contentHeight += LINE_HEIGHT;
      contentHeight += item.labelLines.length * LINE_HEIGHT;
      if (idx < selectedItems.length - 1) {
        contentHeight += 6;
      }
    });

    if (reasoningLines.length > 0) {
      contentHeight += 16;
      contentHeight += LINE_HEIGHT;
      contentHeight += reasoningLines.length * LINE_HEIGHT;
    }

    if (ctx.allowRewind && !ctx.benchmarkMode && node.id !== 'ROOT' && ctx.onNodeRewindClick) {
      contentHeight += 12;
      contentHeight += 26;
    }

    contentHeight += OVERLAY_PADDING;

    const overlayWidth = 340;
    const overlayHeight = Math.max(contentHeight, 140);
    const overlayX = pos.x - overlayWidth / 2;
    const overlayY = pos.y - overlayHeight / 2;

    const { bgColor, borderColor } = getOverlayColors(node, ctx.finalizedCodesSet);

    const overlay = ctx.overlayGroup.append('g')
      .attr('class', `expanded-node node-${node.category}`)
      .attr('transform', `translate(${overlayX}, ${overlayY})`)
      .on('click', (event: MouseEvent) => event.stopPropagation())
      .on('mousedown', (event: MouseEvent) => event.stopPropagation())
      .on('mouseenter', () => ctx.cancelHideTimeout())
      .on('mouseleave', () => {
        if (!ctx.pinnedNodeIdRef.current) {
          ctx.hideTimeoutRef.current = setTimeout(() => ctx.hideExpandedNode(), 500);
        }
      });

    overlay.append('rect')
      .attr('width', overlayWidth)
      .attr('height', overlayHeight)
      .attr('fill', 'rgba(255,255,255,0.01)')
      .style('pointer-events', 'all');

    overlay.append('rect')
      .attr('class', 'expanded-bg')
      .attr('width', overlayWidth)
      .attr('height', overlayHeight)
      .attr('rx', 6)
      .attr('ry', 6)
      .attr('fill', bgColor)
      .attr('stroke', borderColor)
      .attr('stroke-width', 2)
      .attr('filter', 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.15))');

    let yPos = OVERLAY_PADDING + 14;

    // Code line
    const codeLineText = overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 13);

    codeLineText.append('tspan')
      .attr('font-weight', 700)
      .attr('font-family', 'ui-monospace, monospace')
      .attr('fill', '#0f172a')
      .text(displayCode);

    codeLineText.append('tspan')
      .attr('font-weight', 500)
      .attr('fill', node.billable ? '#15803d' : '#64748b')
      .text(` ${billableText}`);

    // Label lines
    yPos += 10;
    labelLines.forEach((line) => {
      yPos += LINE_HEIGHT;
      overlay.append('text')
        .attr('x', OVERLAY_PADDING)
        .attr('y', yPos)
        .attr('font-size', 11)
        .attr('fill', '#334155')
        .text(line);
    });

    // Batch name header
    yPos += 20;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('fill', '#7c3aed')
      .text(`${batchName}:`);

    yPos += 6;
    selectedItems.forEach((item, itemIdx) => {
      yPos += LINE_HEIGHT;
      overlay.append('text')
        .attr('x', OVERLAY_PADDING + 8)
        .attr('y', yPos)
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('font-family', 'ui-monospace, monospace')
        .attr('fill', '#1e293b')
        .text(item.code);

      item.labelLines.forEach((line) => {
        yPos += LINE_HEIGHT;
        overlay.append('text')
          .attr('x', OVERLAY_PADDING + 16)
          .attr('y', yPos)
          .attr('font-size', 10)
          .attr('fill', '#475569')
          .text(line);
      });

      if (itemIdx < selectedItems.length - 1) {
        yPos += 4;
      }
    });

    // Reasoning section
    if (reasoningLines.length > 0) {
      yPos += 18;
      overlay.append('text')
        .attr('x', OVERLAY_PADDING)
        .attr('y', yPos)
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('fill', '#64748b')
        .text('Reasoning:');

      reasoningLines.forEach((line) => {
        yPos += LINE_HEIGHT;
        overlay.append('text')
          .attr('x', OVERLAY_PADDING + 8)
          .attr('y', yPos)
          .attr('font-size', 10)
          .attr('font-style', 'italic')
          .attr('fill', '#64748b')
          .text(line);
      });
    }

    // Investigate Batch button
    if (ctx.allowRewind && !ctx.benchmarkMode && node.id !== 'ROOT' && ctx.onNodeRewindClick) {
      const buttonWidth = 120;
      const buttonHeight = 24;
      const batchType = nodeDecision.current_label.match(/^(\w+)\s+batch$/)?.[1] || 'children';

      yPos += 12;

      const feedbackBtn = overlay.append('g')
        .attr('class', 'feedback-button')
        .attr('transform', `translate(${OVERLAY_PADDING}, ${yPos})`)
        .style('cursor', 'pointer')
        .style('pointer-events', 'auto');

      feedbackBtn.append('rect')
        .attr('width', buttonWidth)
        .attr('height', buttonHeight)
        .attr('rx', 4)
        .attr('fill', '#7c3aed')
        .style('pointer-events', 'auto')
        .style('cursor', 'pointer')
        .on('mousedown', (event: MouseEvent) => event.stopPropagation())
        .on('mouseup', (event: MouseEvent) => event.stopPropagation())
        .on('click', (event: MouseEvent) => {
          event.stopPropagation();
          event.preventDefault();
          ctx.lastInteractionTime.current = Date.now();
          ctx.onNodeRewindClick?.(node.id, batchType, '');
        });

      feedbackBtn.append('text')
        .attr('x', buttonWidth / 2)
        .attr('y', buttonHeight / 2 + 4)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .style('pointer-events', 'none')
        .style('user-select', 'none')
        .text('Investigate Batch');

      feedbackBtn.on('mouseenter', function () {
        d3.select(this).select('rect').attr('fill', '#6d28d9');
      });
      feedbackBtn.on('mouseleave', function () {
        d3.select(this).select('rect').attr('fill', '#7c3aed');
      });
    }

    return;
  }

  // Benchmark expected overlay
  const benchmarkExpected = ctx.benchmarkMode &&
    (node as BenchmarkGraphNode).benchmarkStatus === 'expected';

  if (benchmarkExpected) {
    renderBenchmarkExpectedOverlay(node, pos, labelLines, displayCode, ctx);
    return;
  }

  // Basic overlay (no decision)
  renderBasicOverlay(node, pos, labelLines, displayCode, ctx);
}

/**
 * Render overlay for benchmark expected nodes.
 */
function renderBenchmarkExpectedOverlay(
  node: GraphNode,
  pos: { x: number; y: number },
  labelLines: string[],
  displayCode: string,
  ctx: OverlayContext
): void {
  const isExpectedLeaf = ctx.expectedLeaves.has(node.id);
  const billableText = node.billable ? '(Billable)' : '(Non-Billable)';

  const longestLabelLine = labelLines.reduce((a, b) => a.length > b.length ? a : b, '');
  const estimatedCodeWidth = displayCode.length * 11 + 100;
  const estimatedLabelWidth = longestLabelLine.length * 8;
  const minWidth = 280;
  const maxWidth = 420;
  const overlayWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(estimatedCodeWidth, estimatedLabelWidth) + OVERLAY_PADDING * 2));

  let contentHeight = OVERLAY_PADDING;
  contentHeight += 18;
  contentHeight += 10;
  contentHeight += labelLines.length * LINE_HEIGHT;
  if (isExpectedLeaf) {
    contentHeight += 22;
  }
  contentHeight += OVERLAY_PADDING;
  const overlayHeight = contentHeight;

  const overlayX = pos.x - overlayWidth / 2;
  const overlayY = pos.y - overlayHeight / 2;

  const bgColor = 'rgba(255, 255, 255, 0.98)';
  const borderColor = '#1e293b';

  const overlay = ctx.overlayGroup.append('g')
    .attr('class', `expanded-node node-${node.category}`)
    .attr('transform', `translate(${overlayX}, ${overlayY})`)
    .on('click', (event: MouseEvent) => event.stopPropagation())
    .on('mousedown', (event: MouseEvent) => event.stopPropagation())
    .on('mouseenter', () => ctx.cancelHideTimeout())
    .on('mouseleave', () => {
      if (!ctx.pinnedNodeIdRef.current) {
        ctx.hideTimeoutRef.current = setTimeout(() => ctx.hideExpandedNode(), 500);
      }
    });

  overlay.append('rect')
    .attr('width', overlayWidth)
    .attr('height', overlayHeight)
    .attr('fill', 'rgba(255,255,255,0.01)')
    .style('pointer-events', 'all');

  overlay.append('rect')
    .attr('class', 'expanded-bg')
    .attr('width', overlayWidth)
    .attr('height', overlayHeight)
    .attr('rx', 6)
    .attr('ry', 6)
    .attr('fill', bgColor)
    .attr('stroke', borderColor)
    .attr('stroke-width', 2)
    .attr('filter', 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.15))');

  let yPos = OVERLAY_PADDING + 14;

  const codeLineText = overlay.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', 13);

  codeLineText.append('tspan')
    .attr('font-weight', 700)
    .attr('font-family', 'ui-monospace, monospace')
    .attr('fill', '#0f172a')
    .text(displayCode);

  codeLineText.append('tspan')
    .attr('font-weight', 500)
    .attr('fill', node.billable ? '#15803d' : '#64748b')
    .text(` ${billableText}`);

  yPos += 10;
  labelLines.forEach((line) => {
    yPos += LINE_HEIGHT;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('fill', '#334155')
      .text(line);
  });

  if (isExpectedLeaf) {
    yPos += 20;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 12)
      .attr('fill', '#334155')
      .text('🏁 Expected Final Code');
  }
}

/**
 * Render basic overlay (no decision, not benchmark expected).
 * Includes children section showing child nodes and their rules.
 */
function renderBasicOverlay(
  node: GraphNode,
  pos: { x: number; y: number },
  labelLines: string[],
  displayCode: string,
  ctx: OverlayContext
): void {
  const billableText = node.billable ? '(Billable)' : '(Non-Billable)';

  // Get children codes for "Children:" section
  const childIds = ctx.allChildren.get(node.id) || [];
  const nextCodes: { code: string; rule?: string }[] = [];
  for (const childId of childIds) {
    const childNode = ctx.nodeMap.get(childId);
    if (childNode) {
      const lateralEdge = [...ctx.sevenChrDefEdges, ...ctx.otherLateralEdges].find(
        e => String(e.source) === node.id && String(e.target) === childId
      );
      nextCodes.push({ code: childNode.code, rule: lateralEdge?.rule ?? undefined });
    }
  }

  const longestLabelLine = labelLines.reduce((a, b) => a.length > b.length ? a : b, '');
  const estimatedCodeWidth = displayCode.length * 11 + 100;
  const estimatedLabelWidth = longestLabelLine.length * 8;
  const minWidth = 280;
  const maxWidth = 420;
  const overlayWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(estimatedCodeWidth, estimatedLabelWidth) + OVERLAY_PADDING * 2));

  // Calculate height including children section
  let contentHeight = OVERLAY_PADDING;
  contentHeight += 18; // Code line
  contentHeight += 10; // Gap after code
  contentHeight += labelLines.length * LINE_HEIGHT; // Label lines
  if (nextCodes.length > 0) {
    contentHeight += 16; // Gap + header
    contentHeight += Math.min(nextCodes.length, 4) * LINE_HEIGHT; // Children items
    if (nextCodes.length > 4) contentHeight += LINE_HEIGHT; // "+N more"
  }
  contentHeight += OVERLAY_PADDING;
  const overlayHeight = contentHeight;

  const overlayX = pos.x - overlayWidth / 2;
  const overlayY = pos.y - overlayHeight / 2;

  const { bgColor, borderColor } = getOverlayColors(node, ctx.finalizedCodesSet);

  const overlay = ctx.overlayGroup.append('g')
    .attr('class', `expanded-node node-${node.category}`)
    .attr('transform', `translate(${overlayX}, ${overlayY})`)
    .on('click', (event: MouseEvent) => event.stopPropagation())
    .on('mousedown', (event: MouseEvent) => event.stopPropagation())
    .on('mouseenter', () => ctx.cancelHideTimeout())
    .on('mouseleave', () => {
      if (!ctx.pinnedNodeIdRef.current) {
        ctx.hideTimeoutRef.current = setTimeout(() => ctx.hideExpandedNode(), 500);
      }
    });

  overlay.append('rect')
    .attr('width', overlayWidth)
    .attr('height', overlayHeight)
    .attr('fill', 'rgba(255,255,255,0.01)')
    .style('pointer-events', 'all');

  overlay.append('rect')
    .attr('class', 'expanded-bg')
    .attr('width', overlayWidth)
    .attr('height', overlayHeight)
    .attr('rx', 6)
    .attr('ry', 6)
    .attr('fill', bgColor)
    .attr('stroke', borderColor)
    .attr('stroke-width', 2)
    .attr('filter', 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.15))');

  let yPos = OVERLAY_PADDING + 14;

  const codeLineText = overlay.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', 13);

  codeLineText.append('tspan')
    .attr('font-weight', 700)
    .attr('font-family', 'ui-monospace, monospace')
    .attr('fill', '#0f172a')
    .text(displayCode);

  codeLineText.append('tspan')
    .attr('font-weight', 500)
    .attr('fill', node.billable ? '#15803d' : '#64748b')
    .text(` ${billableText}`);

  yPos += 10;
  labelLines.forEach((line) => {
    yPos += LINE_HEIGHT;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('fill', '#334155')
      .text(line);
  });

  // Children codes section
  if (nextCodes.length > 0) {
    yPos += 18;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('fill', '#64748b')
      .text('Children:');

    const displayCount = Math.min(nextCodes.length, 4);
    nextCodes.slice(0, displayCount).forEach((nc) => {
      yPos += LINE_HEIGHT;
      const nextText = overlay.append('text')
        .attr('x', OVERLAY_PADDING + 8)
        .attr('y', yPos)
        .attr('font-size', 11)
        .attr('fill', '#475569');

      nextText.append('tspan').text(nc.code);
      if (nc.rule) {
        nextText.append('tspan')
          .attr('fill', '#ea580c')
          .attr('font-weight', 500)
          .text(` (${nc.rule})`);
      }
    });

    if (nextCodes.length > 4) {
      yPos += LINE_HEIGHT;
      overlay.append('text')
        .attr('x', OVERLAY_PADDING + 8)
        .attr('y', yPos)
        .attr('font-size', 11)
        .attr('fill', '#94a3b8')
        .attr('font-style', 'italic')
        .text(`+${nextCodes.length - 4} more...`);
    }
  }
}
