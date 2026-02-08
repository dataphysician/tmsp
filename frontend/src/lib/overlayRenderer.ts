/**
 * Overlay rendering utilities for GraphViewer.
 *
 * Provides functions for rendering expanded node overlays with batch panels,
 * decision information, and feedback buttons.
 */

import * as d3 from 'd3';
import type { GraphNode, GraphEdge, DecisionPoint, BenchmarkGraphNode } from './types';
import { wrapText } from './textUtils';
import { normalizeBatchName, shouldIncludeDecision, getOverlayColors } from './nodeStyles';

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
  /** Map from child node ID to parent node ID (hierarchy edges only) - O(1) lookup */
  hierarchyParentMap: Map<string, string>;
  /** Map from "source|target" to lateral edge - O(1) lookup */
  lateralEdgeMap: Map<string, GraphEdge>;
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
  /** Viewport bounds for constraining overlay position */
  viewportBounds?: ViewportBounds;
  /** Function to get current d3 zoom transform (dynamic, not stale) */
  getTransform?: () => d3.ZoomTransform;
}

/** Viewport bounds for overlay positioning constraints */
interface ViewportBounds {
  width: number;
  height: number;
  topMargin: number;
  bottomMargin: number;
}

/** Constants for overlay layout (2x scale) */
const OVERLAY_PADDING = 28;
const LINE_HEIGHT = 30;
const MAX_CHARS_PER_LINE = 75;
const PANEL_WIDTH = 840;
const ROOT_PANEL_WIDTH = 1600; // ROOT gets much wider panel - more chars per line = shorter content = larger scale
const PANEL_GAP = 32;
const MAX_PANELS = 3;

/** Font size constants (2x scale) */
const CODE_FONT_SIZE = 26;
const LABEL_FONT_SIZE = 22;
const BATCH_HEADER_FONT_SIZE = 22;
const SELECTED_CODE_FONT_SIZE = 22;
const SELECTED_LABEL_FONT_SIZE = 20;
const REASONING_FONT_SIZE = 20;
const BUTTON_FONT_SIZE = 22;
const BUTTON_WIDTH = 240;
const BUTTON_HEIGHT = 48;

/** Edge padding for viewport constraints */
const EDGE_PADDING = 10;
/** Extra padding for multi-panel overlays */
const MULTI_PANEL_EDGE_PADDING = 20;

/**
 * Constrain overlay position to stay within viewport bounds.
 *
 * Overlays are sized relative to the visible SCREEN area, not the SVG zoom level.
 * This means:
 * - When zoomed out: overlays scale UP in SVG coordinates to stay readable on screen
 * - When zoomed in: overlays scale DOWN in SVG coordinates to fit the screen
 * - Always constrained to fit within the visible viewport
 *
 * @param nodeX - Node center X position in SVG coordinates
 * @param nodeY - Node center Y position in SVG coordinates
 * @param overlayWidth - Width of the overlay (in base/unscaled units)
 * @param overlayHeight - Height of the overlay (in base/unscaled units)
 * @param viewportBounds - Viewport dimensions and bottom margin
 * @param getTransform - Function to get current zoom transform
 * @param strictContain - If true, strictly contain within viewport (for multi-panel)
 *                        If false, prefer staying close to node horizontally (for single panel)
 * @returns Position and scale factor for the overlay
 */
function constrainOverlayPosition(
  nodeX: number,
  nodeY: number,
  overlayWidth: number,
  overlayHeight: number,
  viewportBounds: ViewportBounds | undefined,
  getTransform: (() => d3.ZoomTransform) | undefined,
  strictContain: boolean = false,
  isSimpleOverlay: boolean = false
): { x: number; y: number; scaleFactor: number } {
  // Default scale factor (no compensation)
  let scaleFactor = 1;

  // If no viewport bounds or transform getter, return centered position
  if (!viewportBounds || !getTransform) {
    const overlayX = nodeX - overlayWidth / 2;
    const overlayY = nodeY - overlayHeight / 2;
    return { x: overlayX, y: overlayY, scaleFactor };
  }

  // Get current transform (dynamic, not stale)
  const currentTransform = getTransform();

  // Use larger padding for multi-panel overlays (in screen pixels)
  const screenPadding = strictContain ? MULTI_PANEL_EDGE_PADDING : EDGE_PADDING;
  // Convert screen padding to SVG units for consistent appearance regardless of zoom
  const svgPadding = screenPadding / currentTransform.k;

  // Convert viewport bounds to SVG coordinate space
  // Account for topMargin in svgMinY so overlays don't overlap top UI
  const svgMinX = -currentTransform.x / currentTransform.k;
  const svgMinY = (viewportBounds.topMargin - currentTransform.y) / currentTransform.k;
  const svgMaxX = (viewportBounds.width - currentTransform.x) / currentTransform.k;
  const svgMaxY = (viewportBounds.height - viewportBounds.bottomMargin - currentTransform.y) / currentTransform.k;

  // Calculate available space in SVG coordinates (with zoom-compensated padding)
  const svgAvailableWidth = svgMaxX - svgMinX - svgPadding * 2;
  const svgAvailableHeight = svgMaxY - svgMinY - svgPadding * 2;

  // Fixed screen size: compensate for zoom so overlay appears constant size on screen
  // Simple overlays (basic info only) use smaller scale (~11% of screen area)
  // Full overlays (batch panels, reasoning) use larger scale to fill viewport
  const targetScreenScale = isSimpleOverlay ? 1.0 : 3.0;
  const targetScale = targetScreenScale / currentTransform.k;

  // Ensure overlay fits within viewport - no overflow allowed
  // Height constraint is strict to prevent vertical overflow
  const fitWidthScale = svgAvailableWidth / overlayWidth;
  const fitHeightScale = svgAvailableHeight / overlayHeight;
  const maxFitScale = Math.min(fitWidthScale, fitHeightScale);

  // Use target scale, but constrain to fit viewport
  scaleFactor = Math.min(targetScale, maxFitScale);

  // Apply final scale factor to dimensions
  const scaledWidth = overlayWidth * scaleFactor;
  const scaledHeight = overlayHeight * scaleFactor;
  const padding = svgPadding;

  // Start centered on node with scaled dimensions
  let overlayX = nodeX - scaledWidth / 2;
  let overlayY = nodeY - scaledHeight / 2;

  if (strictContain) {
    // For multi-panel: strictly contain within viewport with extra padding
    if (scaledWidth > svgAvailableWidth) {
      overlayX = svgMinX + padding;
    } else {
      if (overlayX < svgMinX + padding) {
        overlayX = svgMinX + padding;
      } else if (overlayX + scaledWidth > svgMaxX - padding) {
        overlayX = svgMaxX - padding - scaledWidth;
      }
    }

    // Strictly contain Y with padding
    if (scaledHeight > svgAvailableHeight) {
      overlayY = svgMinY + padding;
    } else {
      if (overlayY < svgMinY + padding) {
        overlayY = svgMinY + padding;
      } else if (overlayY + scaledHeight > svgMaxY - padding) {
        overlayY = svgMaxY - padding - scaledHeight;
      }
    }
  } else {
    // For single panel: prefer staying close to node horizontally
    // Only shift the minimum amount needed to stay in bounds

    // Constrain X - minimal shift, stay close to node
    if (overlayX < svgMinX + padding) {
      overlayX = svgMinX + padding;
    } else if (overlayX + scaledWidth > svgMaxX - padding) {
      overlayX = svgMaxX - padding - scaledWidth;
    }

    // Constrain Y - minimal shift
    if (overlayY < svgMinY + padding) {
      overlayY = svgMinY + padding;
    } else if (overlayY + scaledHeight > svgMaxY - padding) {
      overlayY = svgMaxY - padding - scaledHeight;
    }
  }

  return { x: overlayX, y: overlayY, scaleFactor };
}

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
  hierarchyParentMap: Map<string, string>
): string {
  let fullLabel = node.label;
  if (node.category === 'activator' || node.depth === 7) {
    const ancestorLabel = getActivatorAncestorLabel(node.id, sevenChrDefParentMap, nodeMap, hierarchyParentMap);
    const labelValue = node.label.includes(': ') ? node.label.split(': ').slice(1).join(': ') : node.label;
    if (ancestorLabel) {
      fullLabel = `${ancestorLabel}, ${labelValue}`;
    }
  }
  return fullLabel;
}

/**
 * Get ancestor label for activator nodes.
 * Uses O(1) Map lookup instead of O(n) .find() on edges array.
 */
function getActivatorAncestorLabel(
  nodeId: string,
  sevenChrDefParentMap: Map<string, string>,
  nodeMap: Map<string, GraphNode>,
  hierarchyParentMap: Map<string, string>
): string {
  const parentId = sevenChrDefParentMap.get(nodeId);
  if (!parentId) return '';

  let currentId = parentId;
  while (currentId && currentId !== 'ROOT') {
    const node = nodeMap.get(currentId);
    if (node && node.category !== 'placeholder') {
      return node.label;
    }
    // O(1) lookup instead of O(n) .find()
    const nextParentId = hierarchyParentMap.get(currentId);
    if (nextParentId) {
      currentId = nextParentId;
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
  allowRewind: boolean,
  benchmarkMode: boolean,
  onNodeRewindClick?: (nodeId: string, batchType?: string, feedback?: string) => void,
  maxCharsPerLine: number = MAX_CHARS_PER_LINE
): PanelData[] {
  const panelData: PanelData[] = [];

  nodeDecisions.slice(0, MAX_PANELS).forEach((decision) => {
    const batchName = normalizeBatchName(decision.current_label);
    const labelLines = wrapText(fullLabel, maxCharsPerLine);
    const selectedCandidates = decision.candidates.filter(c => c.selected);
    const showBatch = true;

    const selectedItems: { code: string; labelLines: string[] }[] = [];
    selectedCandidates.forEach((candidate) => {
      const labelMaxChars = maxCharsPerLine - 4;
      selectedItems.push({
        code: candidate.code,
        labelLines: wrapText(candidate.label, labelMaxChars),
      });
    });

    const firstSelected = selectedCandidates[0];
    const reasoningLines = firstSelected?.reasoning
      ? wrapText(firstSelected.reasoning, maxCharsPerLine - 2)
      : [];

    // Calculate content height (2x scale)
    let contentHeight = OVERLAY_PADDING;
    contentHeight += 32; // Code + (Billable) row (2x of 16)
    contentHeight += 16; // Gap after code line (2x of 8)
    contentHeight += labelLines.length * LINE_HEIGHT;

    if (showBatch) {
      contentHeight += 36; // Gap + batch name header (2x of 18)
      contentHeight += 12; // Gap after header (2x of 6)

      if (selectedCandidates.length > 0) {
        selectedItems.forEach((item, idx) => {
          contentHeight += LINE_HEIGHT; // Code line
          contentHeight += item.labelLines.length * LINE_HEIGHT;
          if (idx < selectedItems.length - 1) {
            contentHeight += 12; // 2x of 6
          }
        });
      } else {
        contentHeight += LINE_HEIGHT; // "None Selected"
      }

      if (reasoningLines.length > 0) {
        contentHeight += 32; // Gap before "Reasoning:" header (2x of 16)
        contentHeight += LINE_HEIGHT; // "Reasoning:" header
        contentHeight += reasoningLines.length * LINE_HEIGHT;
      }

      if (allowRewind && !benchmarkMode && node.id !== 'ROOT' && onNodeRewindClick) {
        contentHeight += 24; // Gap before button (2x of 12)
        contentHeight += BUTTON_HEIGHT; // Button height
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
    .attr('rx', 12) // 2x of 6
    .attr('fill', 'rgba(255, 255, 255, 0.98)')
    .attr('stroke', '#e2e8f0')
    .attr('stroke-width', 3) // 2x of 1.5
    .attr('filter', 'drop-shadow(0 4px 12px rgba(0, 0, 0, 0.1))'); // 2x shadow

  let yPos = OVERLAY_PADDING + 28; // 2x of 14

  // Code line
  const codeLineText = panel.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', CODE_FONT_SIZE);

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
  yPos += 20; // 2x of 10
  data.labelLines.forEach((line) => {
    yPos += LINE_HEIGHT;
    panel.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', LABEL_FONT_SIZE)
      .attr('fill', '#334155')
      .text(line);
  });

  if (!data.showBatch) return;

  // Batch name header
  yPos += 40; // 2x of 20
  panel.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', BATCH_HEADER_FONT_SIZE)
    .attr('font-weight', 600)
    .attr('fill', '#7c3aed')
    .text(`${data.batchName}:`);

  yPos += 12; // 2x of 6

  if (data.selectedCandidates.length > 0) {
    data.selectedItems.forEach((item, itemIdx) => {
      yPos += LINE_HEIGHT;
      panel.append('text')
        .attr('x', OVERLAY_PADDING + 16) // 2x of 8
        .attr('y', yPos)
        .attr('font-size', SELECTED_CODE_FONT_SIZE)
        .attr('font-weight', 600)
        .attr('font-family', 'ui-monospace, monospace')
        .attr('fill', '#1e293b')
        .text(item.code);

      item.labelLines.forEach((line) => {
        yPos += LINE_HEIGHT;
        panel.append('text')
          .attr('x', OVERLAY_PADDING + 32) // 2x of 16
          .attr('y', yPos)
          .attr('font-size', SELECTED_LABEL_FONT_SIZE)
          .attr('fill', '#475569')
          .text(line);
      });

      if (itemIdx < data.selectedItems.length - 1) {
        yPos += 8; // 2x of 4
      }
    });
  } else {
    yPos += LINE_HEIGHT;
    panel.append('text')
      .attr('x', OVERLAY_PADDING + 16) // 2x of 8
      .attr('y', yPos)
      .attr('font-size', SELECTED_CODE_FONT_SIZE)
      .attr('font-style', 'italic')
      .attr('fill', '#94a3b8')
      .text('None Selected');
  }

  // Reasoning section
  if (data.reasoningLines.length > 0) {
    yPos += 36; // 2x of 18
    panel.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', SELECTED_CODE_FONT_SIZE)
      .attr('font-weight', 600)
      .attr('fill', '#64748b')
      .text('Reasoning:');

    data.reasoningLines.forEach((line) => {
      yPos += LINE_HEIGHT;
      panel.append('text')
        .attr('x', OVERLAY_PADDING + 16) // 2x of 8
        .attr('y', yPos)
        .attr('font-size', REASONING_FONT_SIZE)
        .attr('font-style', 'italic')
        .attr('fill', '#64748b')
        .text(line);
    });
  }

  // Investigate Batch button (or Regenerate for ROOT)
  if (ctx.allowRewind && !ctx.benchmarkMode && ctx.onNodeRewindClick) {
    const batchType = data.decision.current_label.match(/^(\w+)\s+batch$/)?.[1] || 'children';
    const buttonLabel = 'Investigate Batch';

    yPos += 24; // 2x of 12

    const feedbackBtn = panel.append('g')
      .attr('class', 'feedback-button')
      .attr('transform', `translate(${OVERLAY_PADDING}, ${yPos})`)
      .style('cursor', 'pointer')
      .style('pointer-events', 'auto');

    feedbackBtn.append('rect')
      .attr('width', BUTTON_WIDTH)
      .attr('height', BUTTON_HEIGHT)
      .attr('rx', 8) // 2x of 4
      .attr('fill', '#7c3aed')
      .style('pointer-events', 'auto')
      .style('cursor', 'pointer')
      .on('mousedown', (event: MouseEvent) => event.stopPropagation())
      .on('mouseup', (event: MouseEvent) => event.stopPropagation())
      .on('click', (event: MouseEvent) => {
        event.stopPropagation();
        event.preventDefault();
        ctx.lastInteractionTime.current = Date.now();
        ctx.hideExpandedNode(); // Clear overlays immediately
        ctx.onNodeRewindClick?.(node.id, batchType, '');
      });

    feedbackBtn.append('text')
      .attr('x', BUTTON_WIDTH / 2)
      .attr('y', BUTTON_HEIGHT / 2 + 8) // 2x of 4
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', BUTTON_FONT_SIZE)
      .attr('font-weight', 600)
      .style('pointer-events', 'none')
      .style('user-select', 'none')
      .text(buttonLabel);

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

  // Use wider panel for ROOT (has many children)
  const panelWidth = node.id === 'ROOT' ? ROOT_PANEL_WIDTH : PANEL_WIDTH;
  // Scale chars per line proportionally to panel width
  const maxCharsPerLine = Math.round(MAX_CHARS_PER_LINE * panelWidth / PANEL_WIDTH);

  const fullLabel = getFullLabel(node, ctx.sevenChrDefParentMap, ctx.nodeMap, ctx.hierarchyParentMap);
  const displayCode = getDisplayCode(node, ctx.sevenChrDefParentMap, ctx.nodeMap);

  const panelData = calculatePanelData(
    nodeDecisions,
    fullLabel,
    node,
    ctx.allowRewind,
    ctx.benchmarkMode,
    ctx.onNodeRewindClick,
    maxCharsPerLine
  );

  if (panelData.length === 0) return false;

  // Detect minimal content: single panel with no selected candidates
  const isMinimalContent = panelData.length === 1 &&
    panelData[0].selectedCandidates.length === 0;

  // Calculate dimensions (2x scale for min height)
  const maxPanels = Math.min(panelData.length, MAX_PANELS);
  const panelHeight = Math.max(...panelData.map(p => p.contentHeight), 280); // 2x of 140
  const totalWidth = maxPanels * panelWidth + (maxPanels - 1) * PANEL_GAP;

  // Apply viewport constraints (strict contain for multi-panel)
  const constrained = constrainOverlayPosition(
    pos.x,
    pos.y,
    totalWidth,
    panelHeight,
    ctx.viewportBounds,
    ctx.getTransform,
    true, // strictContain for multi-panel
    isMinimalContent // isSimpleOverlay - use smaller scale for minimal content
  );
  const startX = constrained.x;
  const startY = constrained.y;
  // No scale boost - respect viewport constraints to prevent overflow
  const scaleFactor = constrained.scaleFactor;

  // Render each panel with scale compensation
  panelData.forEach((data, idx) => {
    const panelX = idx * (panelWidth + PANEL_GAP) * scaleFactor;
    const panel = ctx.overlayGroup.append('g')
      .attr('class', 'batch-panel')
      .attr('transform', `translate(${startX + panelX}, ${startY}) scale(${scaleFactor})`)
      .on('click', (event: MouseEvent) => event.stopPropagation())
      .on('mousedown', (event: MouseEvent) => event.stopPropagation())
      .on('mouseenter', () => ctx.cancelHideTimeout())
      .on('mouseleave', () => {
        if (!ctx.pinnedNodeIdRef.current) {
          ctx.hideTimeoutRef.current = setTimeout(() => ctx.hideExpandedNode(), 500);
        }
      });

    renderBatchPanel(panel, data, node, panelWidth, panelHeight, displayCode, ctx);
  });

  // Show indicator if more panels exist
  if (nodeDecisions.length > maxPanels) {
    const indicatorX = startX + (totalWidth + 20) * scaleFactor; // 2x of 10, scaled
    ctx.overlayGroup.append('text')
      .attr('x', indicatorX)
      .attr('y', startY + (panelHeight / 2) * scaleFactor)
      .attr('font-size', 24 * scaleFactor) // 2x of 12, scaled
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

  const fullLabel = getFullLabel(node, ctx.sevenChrDefParentMap, ctx.nodeMap, ctx.hierarchyParentMap);
  const displayCode = getDisplayCode(node, ctx.sevenChrDefParentMap, ctx.nodeMap);
  const labelLines = wrapText(fullLabel, MAX_CHARS_PER_LINE);

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
  const estimatedCodeWidth = displayCode.length * 22 + 200; // 2x of 11 and 100
  const estimatedLabelWidth = longestLabelLine.length * 16; // 2x of 8
  const minWidth = 560; // 2x of 280
  const maxWidth = 840; // 2x of 420
  const overlayWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(estimatedCodeWidth, estimatedLabelWidth) + OVERLAY_PADDING * 2));

  let contentHeight = OVERLAY_PADDING;
  contentHeight += 36; // 2x of 18
  contentHeight += 20; // 2x of 10
  contentHeight += labelLines.length * LINE_HEIGHT;
  if (isExpectedLeaf) {
    contentHeight += 44; // 2x of 22
  }
  contentHeight += OVERLAY_PADDING;
  const overlayHeight = contentHeight;

  // Apply viewport constraints (stay close to node for single panel)
  // This is a simple overlay (just code + label), use smaller scale
  const constrained = constrainOverlayPosition(
    pos.x,
    pos.y,
    overlayWidth,
    overlayHeight,
    ctx.viewportBounds,
    ctx.getTransform,
    false, // prefer staying close to node
    true   // isSimpleOverlay - use smaller scale
  );
  const overlayX = constrained.x;
  const overlayY = constrained.y;
  const scaleFactor = constrained.scaleFactor;

  const bgColor = 'rgba(255, 255, 255, 0.98)';
  const borderColor = '#1e293b';

  const overlay = ctx.overlayGroup.append('g')
    .attr('class', `expanded-node node-${node.category}`)
    .attr('transform', `translate(${overlayX}, ${overlayY}) scale(${scaleFactor})`)
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
    .attr('rx', 12) // 2x of 6
    .attr('ry', 12)
    .attr('fill', bgColor)
    .attr('stroke', borderColor)
    .attr('stroke-width', 4) // 2x of 2
    .attr('filter', 'drop-shadow(0 4px 16px rgba(0, 0, 0, 0.15))'); // 2x shadow

  let yPos = OVERLAY_PADDING + 28; // 2x of 14

  const codeLineText = overlay.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', CODE_FONT_SIZE);

  codeLineText.append('tspan')
    .attr('font-weight', 700)
    .attr('font-family', 'ui-monospace, monospace')
    .attr('fill', '#0f172a')
    .text(displayCode);

  codeLineText.append('tspan')
    .attr('font-weight', 500)
    .attr('fill', node.billable ? '#15803d' : '#64748b')
    .text(` ${billableText}`);

  yPos += 20; // 2x of 10
  labelLines.forEach((line) => {
    yPos += LINE_HEIGHT;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', LABEL_FONT_SIZE)
      .attr('fill', '#334155')
      .text(line);
  });

  if (isExpectedLeaf) {
    yPos += 40; // 2x of 20
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', 24) // 2x of 12
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
      // O(1) lookup instead of O(n) array spread + .find()
      const lateralEdge = ctx.lateralEdgeMap.get(`${node.id}|${childId}`);
      nextCodes.push({ code: childNode.code, rule: lateralEdge?.rule ?? undefined });
    }
  }

  const longestLabelLine = labelLines.reduce((a, b) => a.length > b.length ? a : b, '');
  const estimatedCodeWidth = displayCode.length * 22 + 200; // 2x of 11 and 100
  const estimatedLabelWidth = longestLabelLine.length * 16; // 2x of 8
  const minWidth = 560; // 2x of 280
  const maxWidth = 840; // 2x of 420
  const overlayWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(estimatedCodeWidth, estimatedLabelWidth) + OVERLAY_PADDING * 2));

  // Calculate height including children section (2x scale)
  let contentHeight = OVERLAY_PADDING;
  contentHeight += 36; // Code line (2x of 18)
  contentHeight += 20; // Gap after code (2x of 10)
  contentHeight += labelLines.length * LINE_HEIGHT; // Label lines
  if (nextCodes.length > 0) {
    contentHeight += 32; // Gap + header (2x of 16)
    contentHeight += Math.min(nextCodes.length, 10) * LINE_HEIGHT; // Children items
    if (nextCodes.length > 10) contentHeight += LINE_HEIGHT; // "+N more"
  }
  contentHeight += OVERLAY_PADDING;
  const overlayHeight = contentHeight;

  // Apply viewport constraints (stay close to node for single panel)
  // This is a simple overlay (just code + label + children), use smaller scale
  const constrained = constrainOverlayPosition(
    pos.x,
    pos.y,
    overlayWidth,
    overlayHeight,
    ctx.viewportBounds,
    ctx.getTransform,
    false, // prefer staying close to node
    true   // isSimpleOverlay - use smaller scale
  );
  const overlayX = constrained.x;
  const overlayY = constrained.y;
  const scaleFactor = constrained.scaleFactor;

  const { bgColor, borderColor } = getOverlayColors(node, ctx.finalizedCodesSet);

  const overlay = ctx.overlayGroup.append('g')
    .attr('class', `expanded-node node-${node.category}`)
    .attr('transform', `translate(${overlayX}, ${overlayY}) scale(${scaleFactor})`)
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
    .attr('rx', 12) // 2x of 6
    .attr('ry', 12)
    .attr('fill', bgColor)
    .attr('stroke', borderColor)
    .attr('stroke-width', 4) // 2x of 2
    .attr('filter', 'drop-shadow(0 4px 16px rgba(0, 0, 0, 0.15))'); // 2x shadow

  let yPos = OVERLAY_PADDING + 28; // 2x of 14

  const codeLineText = overlay.append('text')
    .attr('x', OVERLAY_PADDING)
    .attr('y', yPos)
    .attr('font-size', CODE_FONT_SIZE);

  codeLineText.append('tspan')
    .attr('font-weight', 700)
    .attr('font-family', 'ui-monospace, monospace')
    .attr('fill', '#0f172a')
    .text(displayCode);

  codeLineText.append('tspan')
    .attr('font-weight', 500)
    .attr('fill', node.billable ? '#15803d' : '#64748b')
    .text(` ${billableText}`);

  yPos += 20; // 2x of 10
  labelLines.forEach((line) => {
    yPos += LINE_HEIGHT;
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', LABEL_FONT_SIZE)
      .attr('fill', '#334155')
      .text(line);
  });

  // Children codes section
  if (nextCodes.length > 0) {
    yPos += 36; // 2x of 18
    overlay.append('text')
      .attr('x', OVERLAY_PADDING)
      .attr('y', yPos)
      .attr('font-size', SELECTED_CODE_FONT_SIZE)
      .attr('font-weight', 600)
      .attr('fill', '#64748b')
      .text('Children:');

    const displayCount = Math.min(nextCodes.length, 10);
    nextCodes.slice(0, displayCount).forEach((nc) => {
      yPos += LINE_HEIGHT;
      const nextText = overlay.append('text')
        .attr('x', OVERLAY_PADDING + 16) // 2x of 8
        .attr('y', yPos)
        .attr('font-size', SELECTED_CODE_FONT_SIZE)
        .attr('fill', '#475569');

      nextText.append('tspan').text(nc.code);
      if (nc.rule) {
        nextText.append('tspan')
          .attr('fill', '#ea580c')
          .attr('font-weight', 500)
          .text(` (${nc.rule})`);
      }
    });

    if (nextCodes.length > 10) {
      yPos += LINE_HEIGHT;
      overlay.append('text')
        .attr('x', OVERLAY_PADDING + 16) // 2x of 8
        .attr('y', yPos)
        .attr('font-size', SELECTED_CODE_FONT_SIZE)
        .attr('fill', '#94a3b8')
        .attr('font-style', 'italic')
        .text(`+${nextCodes.length - 10} more...`);
    }
  }
}
