import { useEffect, useRef, useMemo, useState, useCallback } from 'react';
import * as d3 from 'd3';
import type { GraphNode, GraphEdge, TraversalStatus, BenchmarkGraphNode, BenchmarkMetrics, OvershootMarker, EdgeMissMarker, DecisionPoint } from '../lib/types';
import { exportSvgToFile, generateSvgFilename } from '../lib/exportSvg';

type SortMode = 'default' | 'asc' | 'desc';

// Debounce helper for resize handling
function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number): T {
  let timeoutId: ReturnType<typeof setTimeout>;
  return ((...args: unknown[]) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), ms);
  }) as T;
}

interface GraphViewerProps {
  nodes: GraphNode[] | BenchmarkGraphNode[];
  edges: GraphEdge[];
  selectedNode: string | null;
  onNodeClick: (nodeId: string) => void;
  finalizedCodes?: string[];
  isTraversing?: boolean;
  currentStep?: string;
  decisionCount?: number;
  status?: TraversalStatus;
  errorMessage?: string | null;
  // Decision history for showing reasoning on hover
  decisions?: DecisionPoint[];
  // Benchmark mode props
  benchmarkMode?: boolean;
  benchmarkMetrics?: BenchmarkMetrics | null;
  overshootMarkers?: OvershootMarker[];
  missedEdgeMarkers?: EdgeMissMarker[];
  expectedLeaves?: Set<string>;
  onRemoveExpectedCode?: (code: string) => void;
  invalidCodes?: Set<string>;
  // Label for the codes bar
  codesBarLabel?: string;
  // Elapsed time (managed by parent for persistence)
  elapsedTime?: number | null;
  // Trigger fit-to-window (increment to trigger)
  triggerFitToWindow?: number;
  // Rewind feature props (TRAVERSE tab only)
  onNodeRewindClick?: (nodeId: string, batchId?: string, feedback?: string) => void;
  allowRewind?: boolean;
  // Controls whether X markers render (default true)
  showXMarkers?: boolean;
}

// Base constants (designed for ~600px container height / 1080p display)
const BASE_NODE_WIDTH = 140;
const BASE_NODE_HEIGHT = 60;
const BASE_LEVEL_HEIGHT = 100;
const REFERENCE_HEIGHT = 600; // Reference container height for scale factor calculation

// Helper to wrap text into lines
function wrapText(text: string, maxWidth: number): string[] {
  if (!text) return [''];
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = '';

  for (const word of words) {
    const testLine = currentLine ? currentLine + ' ' + word : word;
    if (testLine.length <= maxWidth) {
      currentLine = testLine;
    } else {
      if (currentLine) lines.push(currentLine);
      currentLine = word;
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines.length ? lines : [''];
}


export function GraphViewer({
  nodes,
  edges,
  selectedNode,
  onNodeClick,
  finalizedCodes = [],
  isTraversing = false,
  currentStep = '',
  decisionCount = 0,
  status = 'idle',
  errorMessage = null,
  decisions = [],
  benchmarkMode = false,
  benchmarkMetrics = null,
  overshootMarkers = [],
  missedEdgeMarkers = [],
  expectedLeaves = new Set(),
  onRemoveExpectedCode,
  invalidCodes = new Set(),
  codesBarLabel,
  elapsedTime = null,
  triggerFitToWindow,
  onNodeRewindClick,
  allowRewind = false,
  showXMarkers = true,
}: GraphViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [codeSortMode, setCodeSortMode] = useState<SortMode>('default');
  const [pinnedNodeId, setPinnedNodeId] = useState<string | null>(null);
  const pinnedNodeIdRef = useRef<string | null>(null);

  // Keep ref in sync with state for D3 event handlers
  useEffect(() => {
    pinnedNodeIdRef.current = pinnedNodeId;
  }, [pinnedNodeId]);

  // Clear pinned state when the pinned node is no longer in the graph
  useEffect(() => {
    if (pinnedNodeId && !nodes.some(n => n.id === pinnedNodeId)) {
      setPinnedNodeId(null);
    }
  }, [pinnedNodeId, nodes]);
  const hasInitializedZoom = useRef(false);
  const prevIsTraversing = useRef(isTraversing);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const lastInteractionTime = useRef<number>(0);
  const prevNodeCount = useRef<number>(0);
  const autoFitIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Format elapsed time as seconds with 1 decimal
  const formatElapsedTime = (ms: number): string => {
    const seconds = ms / 1000;
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
  };

  const sortedFinalizedCodes = useMemo(() => {
    if (codeSortMode === 'default') return finalizedCodes;
    const sorted = [...finalizedCodes].sort((a, b) => a.localeCompare(b));
    return codeSortMode === 'desc' ? sorted.reverse() : sorted;
  }, [finalizedCodes, codeSortMode]);

  const nodeCount = useMemo(() => {
    return nodes.filter(n => n.id !== 'ROOT').length;
  }, [nodes]);

  // Build a map from node code to reasoning (for tooltip)
  const nodeReasoningMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const decision of decisions) {
      for (const candidate of decision.candidates) {
        if (candidate.selected && candidate.reasoning) {
          map.set(candidate.code, candidate.reasoning);
        }
      }
    }
    return map;
  }, [decisions]);

  // Zoom control handlers
  const handleZoomIn = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(200).call(zoomRef.current.scaleBy, 1.3);
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(200).call(zoomRef.current.scaleBy, 0.7);
  }, []);

  const handleFitToWindow = useCallback(() => {
    if (!svgRef.current || !zoomRef.current || !containerRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Get current graph bounds from the main group
    const g = svg.select('g.main-group');
    if (g.empty()) return;

    const gNode = g.node() as SVGGElement;
    const bbox = gNode.getBBox();

    if (bbox.width === 0 || bbox.height === 0) return;

    // Calculate scale to fit with padding
    const padding = 40;
    const scaleX = (width - padding * 2) / bbox.width;
    const scaleY = (height - padding * 2) / bbox.height;
    const scale = Math.min(scaleX, scaleY, 1.5);

    // Center horizontally, position at top
    const scaledWidth = bbox.width * scale;
    const translateX = (width - scaledWidth) / 2 - bbox.x * scale;
    const topPadding = 20;
    const translateY = topPadding - bbox.y * scale;

    svg.transition().duration(300).call(
      zoomRef.current.transform,
      d3.zoomIdentity.translate(translateX, translateY).scale(scale)
    );
  }, [nodes.length]);

  const handleExportSvg = useCallback(() => {
    if (!svgRef.current) return;
    const prefix = benchmarkMode ? 'graph-benchmark' : isTraversing ? 'graph-traverse' : 'graph-visualize';
    const filename = generateSvgFilename(prefix);
    exportSvgToFile(svgRef.current, filename);
  }, [benchmarkMode, isTraversing]);

  // Trigger fit-to-window when prop changes (with delay for layout to settle)
  useEffect(() => {
    if (triggerFitToWindow && triggerFitToWindow > 0) {
      const timer = setTimeout(() => {
        handleFitToWindow();
      }, 350); // Delay to allow sidebar collapse animation
      return () => clearTimeout(timer);
    }
  }, [triggerFitToWindow, handleFitToWindow]);

  // ResizeObserver: fit-to-window when container resizes (debounced)
  // This handles sidebar collapse, window resize, etc.
  const debouncedFitToWindow = useMemo(
    () => debounce(() => handleFitToWindow(), 150),
    [handleFitToWindow]
  );

  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      debouncedFitToWindow();
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [debouncedFitToWindow]);

  // Periodic fit-to-window: frequent during traversal, sparse after completion
  useEffect(() => {
    // Clear existing interval when traversing state changes
    if (autoFitIntervalRef.current) {
      clearInterval(autoFitIntervalRef.current);
      autoFitIntervalRef.current = null;
    }

    // During traversal: auto-fit every 3 seconds (responsive)
    // After traversal: auto-fit every 20 seconds (sparse, less intrusive)
    const interval = isTraversing ? 3000 : 20000;
    const idleThreshold = isTraversing ? 3000 : 5000;

    autoFitIntervalRef.current = setInterval(() => {
      const timeSinceInteraction = Date.now() - lastInteractionTime.current;
      if (timeSinceInteraction >= idleThreshold) {
        handleFitToWindow();
      }
    }, interval);

    return () => {
      if (autoFitIntervalRef.current) {
        clearInterval(autoFitIntervalRef.current);
      }
    };
  }, [isTraversing, handleFitToWindow]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Calculate scale factor based on container height (clamped to 0.7-1.5x)
    const scaleFactor = Math.max(0.7, Math.min(1.5, height / REFERENCE_HEIGHT));
    const NODE_WIDTH = BASE_NODE_WIDTH * scaleFactor;
    const NODE_HEIGHT = BASE_NODE_HEIGHT * scaleFactor;
    const LEVEL_HEIGHT = BASE_LEVEL_HEIGHT * scaleFactor;

    // Always size SVG to fill container
    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();

    // Only render content when traversing or has nodes
    if (nodes.length === 0 && !isTraversing) return;

    // Create main group for zoom/pan
    const g = svg.append('g').attr('class', 'main-group');

    // Add arrow markers
    const defs = svg.append('defs');

    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#666');

    defs.append('marker')
      .attr('id', 'arrowhead-lateral')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#e67e22');

    // Overshoot arrowhead marker (red)
    defs.append('marker')
      .attr('id', 'arrowhead-overshoot')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#dc2626');

    // Build adjacency for layout
    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    // Build TWO children maps:
    // 1. hierarchyChildren: Only hierarchy edges - used for tree positioning in Phase 2
    // 2. allChildren: All edges - used for subtree width calculation and positioning children of lateral nodes
    const hierarchyChildren = new Map<string, string[]>();
    const allChildren = new Map<string, string[]>();

    edges.forEach(e => {
      const sourceId = String(e.source);
      const targetId = String(e.target);

      // All edges go into allChildren (for width calculation)
      if (!allChildren.has(sourceId)) allChildren.set(sourceId, []);
      allChildren.get(sourceId)!.push(targetId);

      // Hierarchy edges AND sevenChrDef edges go into hierarchyChildren
      // (sevenChrDef is lateral type but represents vertical parent-child relationship)
      const isHierarchy = e.edge_type === 'hierarchy';
      const isSevenChrDef = e.edge_type === 'lateral' && e.rule === 'sevenChrDef';

      if (isHierarchy || isSevenChrDef) {
        if (!hierarchyChildren.has(sourceId)) hierarchyChildren.set(sourceId, []);
        hierarchyChildren.get(sourceId)!.push(targetId);
      }
    });

    // Filter edges by type for rendering
    // - ROOT edges: dark black solid straight lines
    // - Hierarchy edges: solid straight lines (bottom-middle to top-middle)
    // - sevenChrDef edges: dashed straight lines (same path as hierarchy, different style)
    // - Other lateral edges: dashed curved lines
    const rootEdges = edges.filter(e => e.edge_type === 'hierarchy' && String(e.source) === 'ROOT');
    const hierarchyEdges = edges.filter(e => e.edge_type === 'hierarchy' && String(e.source) !== 'ROOT');
    const sevenChrDefEdges = edges.filter(e => e.edge_type === 'lateral' && e.rule === 'sevenChrDef');
    const otherLateralEdges = edges.filter(e => e.edge_type === 'lateral' && e.rule !== 'sevenChrDef');

    // Build map from sevenChrDef target (activator) -> source (parent) for tooltip
    const sevenChrDefParentMap = new Map<string, string>();
    sevenChrDefEdges.forEach(e => {
      sevenChrDefParentMap.set(String(e.target), String(e.source));
    });

    // Helper to get ancestor label for activator nodes
    const getActivatorAncestorLabel = (nodeId: string): string => {
      const parentId = sevenChrDefParentMap.get(nodeId);
      if (!parentId) return '';

      // Traverse up to find non-placeholder ancestor
      let currentId = parentId;
      while (currentId && currentId !== 'ROOT') {
        const node = nodeMap.get(currentId);
        if (node && node.category !== 'placeholder') {
          return node.label;
        }
        // Find parent via hierarchy edges
        const parentEdge = hierarchyEdges.find(e => String(e.target) === currentId);
        if (parentEdge) {
          currentId = String(parentEdge.source);
        } else {
          break;
        }
      }
      return '';
    };

    // Calculate positions using subtree width algorithm (from reference)
    const positions = calculatePositions(nodes, hierarchyChildren, allChildren, width, nodeMap, edges.filter(e => e.edge_type === 'lateral'), NODE_WIDTH, NODE_HEIGHT, LEVEL_HEIGHT);

    // Create lateral edges group FIRST (rendered behind everything else)
    const lateralEdgesGroup = g.append('g').attr('class', 'lateral-edges');

    // Render sevenChrDef edges (straight vertical lines, but styled as lateral)
    lateralEdgesGroup.selectAll('.edge-sevenChrDef')
      .data(sevenChrDefEdges)
      .join('path')
      .attr('class', 'edge edge-lateral edge-sevenChrDef')
      .attr('d', d => createEdgePath(d, positions, NODE_HEIGHT))
      .attr('fill', 'none')
      .attr('stroke', '#e67e22')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,3')
      .attr('marker-end', 'url(#arrowhead-lateral)');

    // Render other lateral edges (curved lines)
    lateralEdgesGroup.selectAll('.edge-other-lateral')
      .data(otherLateralEdges)
      .join('path')
      .attr('class', d => `edge edge-lateral edge-${d.rule || 'lateral'}`)
      .attr('d', d => createCurvedEdgePath(d, positions, NODE_HEIGHT))
      .attr('fill', 'none')
      .attr('stroke', '#e67e22')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,3')
      .attr('marker-end', 'url(#arrowhead-lateral)');

    // Add edge labels for lateral edges - positioned just right of arrowhead
    const allLateralEdges = [...sevenChrDefEdges, ...otherLateralEdges];
    lateralEdgesGroup.selectAll('.edge-label')
      .data(allLateralEdges)
      .join('text')
      .attr('class', 'edge-label')
      .attr('x', d => {
        const tgt = positions.get(String(d.target));
        if (!tgt) return 0;
        // Position just to the right of where arrow meets the node (top center)
        return tgt.x + 12;
      })
      .attr('y', d => {
        const tgt = positions.get(String(d.target));
        if (!tgt) return 0;
        // Position at the arrow entry point (top of node) with small offset
        return tgt.y - NODE_HEIGHT / 2 - 12;
      })
      .attr('text-anchor', 'start')
      .attr('font-size', 9)
      .attr('fill', '#ea580c')
      .attr('font-weight', 500)
      .text(d => d.rule || '');

    // Create hierarchy edges group (rendered after lateral edges, before nodes)
    const edgesGroup = g.append('g').attr('class', 'edges');

    // Render ROOT -> Chapter edges (same style as hierarchy edges)
    edgesGroup.selectAll('.edge-root')
      .data(rootEdges)
      .join('path')
      .attr('class', 'edge edge-root')
      .attr('d', d => createEdgePath(d, positions, NODE_HEIGHT))
      .attr('fill', 'none')
      .attr('stroke', '#64748b')
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrowhead)');

    // Render hierarchy edges (straight vertical lines)
    edgesGroup.selectAll('.edge-hierarchy')
      .data(hierarchyEdges)
      .join('path')
      .attr('class', 'edge edge-hierarchy')
      .attr('d', d => createEdgePath(d, positions, NODE_HEIGHT))
      .attr('fill', 'none')
      .attr('stroke', '#64748b')
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrowhead)');

    // Create nodes group
    const nodesGroup = g.append('g').attr('class', 'nodes');
    const finalizedCodesSet = new Set(finalizedCodes);

    // Only show ROOT node when there are child nodes to connect to
    const hasNonRootNodes = nodes.some(n => n.id !== 'ROOT');

    // Render ROOT node only when there are children
    if (hasNonRootNodes) {
      const rootPos = positions.get('ROOT');
      if (rootPos) {
        const rootGroup = nodesGroup.append('g')
          .attr('class', 'node node-root')
          .attr('transform', `translate(${rootPos.x - NODE_WIDTH / 2}, ${rootPos.y - NODE_HEIGHT / 2})`);

        rootGroup.append('rect')
          .attr('width', NODE_WIDTH)
          .attr('height', NODE_HEIGHT)
          .attr('rx', 4)
          .attr('ry', 4)
          .attr('fill', '#ffffff')
          .attr('stroke', '#334155')
          .attr('stroke-width', 1.5);

        rootGroup.append('text')
          .attr('x', NODE_WIDTH / 2)
          .attr('y', NODE_HEIGHT / 2 + 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', 13)
          .attr('font-weight', 600)
          .attr('fill', '#1e293b')
          .text('ROOT');
      }
    }

    // Create expanded overlay group (rendered on top of everything)
    const expandedOverlayGroup = g.append('g').attr('class', 'expanded-overlay');

    // Helper function to show expanded node overlay
    const showExpandedNode = (d: GraphNode, _nodeGroup: SVGGElement) => {
      // Remove any existing expanded overlays
      expandedOverlayGroup.selectAll('*').remove();

      const pos = positions.get(d.id);
      if (!pos) return;

      // Check if this is a traversed benchmark node with decisions
      const benchmarkNode = d as BenchmarkGraphNode;
      const isTraversedInBenchmark = benchmarkMode &&
        benchmarkNode.benchmarkStatus &&
        benchmarkNode.benchmarkStatus !== 'expected';

      // Find ALL decisions for this node (multiple batches possible)
      // Works for BOTH benchmark mode (traversed nodes) and traverse mode
      const nodeDecisionsRaw = decisions
        ? decisions.filter(dec => dec.current_node === d.id)
        : [];

      // Deduplicate panels with identical batch type + selected codes
      // (Can happen when multiple trajectories converge on the same node due to node deduplication)
      const nodeDecisions = nodeDecisionsRaw.reduce((acc, dec) => {
        // Normalize batch label for deduplication
        const normalizedLabel = normalizeBatchName(dec.current_label);
        // Normalize selected codes (strip to alphanumeric + dots only, sort, join)
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
        // Filter out children batches for nodes that can't have children selections
        .filter(dec => {
          // Use centralized helper for decision filtering
          if (!shouldIncludeDecision(dec, d, finalizedCodesSet)) {
            return false;
          }
          // Keep all batches (including empty selections) to allow Add Feedback
          return true;
        });

      // Determine if we should show batch panels
      // Show for: benchmark traversed nodes with decisions, OR traverse mode with decisions
      const shouldShowBatchPanels = (isTraversedInBenchmark && nodeDecisions.length > 0) ||
        (!benchmarkMode && nodeDecisions.length > 0);

      // If we have decisions, show batch panels (side-by-side for parallel batches)
      if (shouldShowBatchPanels) {
        const panelWidth = 340;
        const panelGap = 16;
        const padding = 14;
        const lineHeight = 15;
        const maxPanels = Math.min(nodeDecisions.length, 3);
        const maxCharsPerLine = 45;

        // Get the full label for the parent node
        let fullLabel = d.label;
        if (d.category === 'activator' || d.depth === 7) {
          const ancestorLabel = getActivatorAncestorLabel(d.id);
          const labelValue = d.label.includes(': ') ? d.label.split(': ').slice(1).join(': ') : d.label;
          if (ancestorLabel) {
            fullLabel = `${ancestorLabel}, ${labelValue}`;
          }
        }

        const billableText = d.billable ? '(Billable)' : '(Non-Billable)';

        // First pass: calculate content heights for all panels to find the max
        const panelData: {
          decision: DecisionPoint;
          batchName: string;
          labelLines: string[];
          selectedCandidates: typeof nodeDecisions[0]['candidates'];
          selectedItems: { code: string; labelLines: string[] }[];
          reasoningLines: string[];
          showBatch: boolean;
          contentHeight: number;
        }[] = [];

        nodeDecisions.slice(0, maxPanels).forEach((decision) => {
          const batchName = normalizeBatchName(decision.current_label);
          const labelLines = wrapText(fullLabel, maxCharsPerLine);
          const selectedCandidates = decision.candidates.filter(c => c.selected);

          // Determine if batch section should be shown (use centralized helper)
          const hasSelections = selectedCandidates.length > 0;
          const showBatch = shouldShowBatch(batchName, d, finalizedCodesSet);

          // For each selected candidate, wrap the label separately
          const selectedItems: { code: string; labelLines: string[] }[] = [];
          selectedCandidates.forEach((candidate) => {
            // Wrap just the label (code shown separately)
            const labelMaxChars = maxCharsPerLine - 4; // Account for indentation
            selectedItems.push({
              code: candidate.code,
              labelLines: wrapText(candidate.label, labelMaxChars),
            });
          });

          // Wrap reasoning
          const firstSelected = selectedCandidates[0];
          const reasoningLines = firstSelected?.reasoning
            ? wrapText(firstSelected.reasoning, maxCharsPerLine - 2)
            : [];

          // Calculate content height for this panel
          let contentHeight = padding; // Top padding
          contentHeight += 16; // Code + (Billable) row
          contentHeight += 8; // Gap after code line
          contentHeight += labelLines.length * lineHeight; // Label lines

          // Batch section (if shown)
          if (showBatch) {
            contentHeight += 18; // Gap + batch name header
            contentHeight += 6; // Gap after header

            if (hasSelections) {
              // Selected codes with labels
              selectedItems.forEach((item, idx) => {
                contentHeight += lineHeight; // Code line
                contentHeight += item.labelLines.length * lineHeight; // Label lines
                if (idx < selectedItems.length - 1) {
                  contentHeight += 6; // Gap between items
                }
              });
            } else {
              // "None Selected" text
              contentHeight += lineHeight;
            }

            // Reasoning section
            if (reasoningLines.length > 0) {
              contentHeight += 16; // Gap before "Reasoning:" header
              contentHeight += lineHeight; // "Reasoning:" header
              contentHeight += reasoningLines.length * lineHeight; // Reasoning lines
            }

            // Add Feedback textarea + button (TRAVERSE tab only)
            if (allowRewind && !benchmarkMode && d.id !== 'ROOT' && onNodeRewindClick) {
              contentHeight += 12; // Gap before textarea
              contentHeight += 50; // Textarea height (2 lines)
              contentHeight += 8; // Gap between textarea and button
              contentHeight += 26; // Button height
            }
          }

          contentHeight += padding; // Bottom padding

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

        // Use the maximum height across all panels for uniform appearance
        const panelHeight = Math.max(...panelData.map(p => p.contentHeight), 140);

        const totalWidth = (maxPanels * panelWidth) + ((maxPanels - 1) * panelGap);
        const startX = pos.x - totalWidth / 2;
        const startY = pos.y - panelHeight / 2;

        // Second pass: render panels with equal heights
        panelData.forEach((data, idx) => {
          const panelX = startX + idx * (panelWidth + panelGap);

          // Create panel group
          const panel = expandedOverlayGroup.append('g')
            .attr('class', 'batch-panel')
            .attr('transform', `translate(${panelX}, ${startY})`)
            .on('click', (event: MouseEvent) => event.stopPropagation());

          // Panel background with uniform height
          panel.append('rect')
            .attr('width', panelWidth)
            .attr('height', panelHeight)
            .attr('rx', 6)
            .attr('fill', 'rgba(255, 255, 255, 0.98)')
            .attr('stroke', '#7c3aed')
            .attr('stroke-width', 2)
            .attr('filter', 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.15))');

          let yPos = padding + 14;

          // Line 1: <Parent Code> (Billable/Non-Billable)
          const codeLineText = panel.append('text')
            .attr('x', padding)
            .attr('y', yPos)
            .attr('font-size', 13);

          codeLineText.append('tspan')
            .attr('font-weight', 700)
            .attr('font-family', 'ui-monospace, monospace')
            .attr('fill', '#0f172a')
            .text(d.code);

          codeLineText.append('tspan')
            .attr('font-weight', 500)
            .attr('fill', d.billable ? '#15803d' : '#64748b')
            .text(` ${billableText}`);

          // Parent label (wrapped, complete)
          yPos += 10;
          data.labelLines.forEach((line) => {
            yPos += lineHeight;
            panel.append('text')
              .attr('x', padding)
              .attr('y', yPos)
              .attr('font-size', 11)
              .attr('fill', '#334155')
              .text(line);
          });

          // Batch section (conditional based on node type)
          if (data.showBatch) {
            // Batch name header (e.g., "children:", "codeFirst:", etc.)
            yPos += 20;
            panel.append('text')
              .attr('x', padding)
              .attr('y', yPos)
              .attr('font-size', 11)
              .attr('font-weight', 600)
              .attr('fill', '#7c3aed')
              .text(`${data.batchName}:`);

            // Selected codes with labels, or "None Selected"
            yPos += 6;
            if (data.selectedItems.length > 0) {
              data.selectedItems.forEach((item, itemIdx) => {
                // Code on its own line (bold, monospace)
                yPos += lineHeight;
                panel.append('text')
                  .attr('x', padding + 8)
                  .attr('y', yPos)
                  .attr('font-size', 11)
                  .attr('font-weight', 600)
                  .attr('font-family', 'ui-monospace, monospace')
                  .attr('fill', '#1e293b')
                  .text(item.code);

                // Label lines (wrapped, indented)
                item.labelLines.forEach((line) => {
                  yPos += lineHeight;
                  panel.append('text')
                    .attr('x', padding + 16)
                    .attr('y', yPos)
                    .attr('font-size', 10)
                    .attr('fill', '#475569')
                    .text(line);
                });

                // Gap between selected items
                if (itemIdx < data.selectedItems.length - 1) {
                  yPos += 4;
                }
              });
            } else {
              // No selections made for this batch
              yPos += lineHeight;
              panel.append('text')
                .attr('x', padding + 8)
                .attr('y', yPos)
                .attr('font-size', 11)
                .attr('font-style', 'italic')
                .attr('fill', '#94a3b8')
                .text('None Selected');
            }

            // Reasoning section with header
            if (data.reasoningLines.length > 0) {
              yPos += 18;
              panel.append('text')
                .attr('x', padding)
                .attr('y', yPos)
                .attr('font-size', 11)
                .attr('font-weight', 600)
                .attr('fill', '#64748b')
                .text('Reasoning:');

              data.reasoningLines.forEach((line) => {
                yPos += lineHeight;
                panel.append('text')
                  .attr('x', padding + 8)
                  .attr('y', yPos)
                  .attr('font-size', 10)
                  .attr('font-style', 'italic')
                  .attr('fill', '#64748b')
                  .text(line);
              });
            }

            // Add Feedback textarea + button for TRAVERSE tab (inside each panel)
            if (allowRewind && !benchmarkMode && d.id !== 'ROOT' && onNodeRewindClick) {
              const textareaWidth = panelWidth - padding * 2;
              const textareaHeight = 50;
              const buttonWidth = 100;
              const buttonHeight = 24;

              // Construct batchId from decision
              const batchType = data.decision.current_label.match(/^(\w+)\s+batch$/)?.[1] || 'children';
              const batchId = `${d.id}|${batchType}`;
              const textareaId = `feedback-textarea-${d.id}-${batchType}`;

              // Gap before textarea
              yPos += 12;

              // Textarea using foreignObject
              const fo = panel.append('foreignObject')
                .attr('x', padding)
                .attr('y', yPos)
                .attr('width', textareaWidth)
                .attr('height', textareaHeight);

              fo.append('xhtml:textarea')
                .attr('id', textareaId)
                .attr('placeholder', 'Enter feedback for this batch...')
                .style('width', '100%')
                .style('height', '100%')
                .style('resize', 'none')
                .style('background-color', '#f8fafc')
                .style('border', '1px solid #d1d5db')
                .style('border-radius', '4px')
                .style('padding', '6px 8px')
                .style('font-size', '11px')
                .style('font-family', 'inherit')
                .style('line-height', '1.4')
                .style('overflow-y', 'auto')
                .style('box-sizing', 'border-box')
                .on('click', (event: MouseEvent) => event.stopPropagation())
                .on('mousedown', (event: MouseEvent) => event.stopPropagation());

              yPos += textareaHeight + 8; // Textarea height + gap before button

              const feedbackBtn = panel.append('g')
                .attr('class', 'feedback-button')
                .attr('transform', `translate(${padding}, ${yPos})`)
                .style('cursor', 'pointer')
                .on('click', (event: MouseEvent) => {
                  event.stopPropagation();
                  // Get textarea value
                  const textareaEl = document.getElementById(textareaId) as HTMLTextAreaElement | null;
                  const feedbackText = textareaEl?.value || '';
                  onNodeRewindClick(d.id, batchId, feedbackText);
                });

              feedbackBtn.append('rect')
                .attr('width', buttonWidth)
                .attr('height', buttonHeight)
                .attr('rx', 4)
                .attr('fill', '#7c3aed');

              feedbackBtn.append('text')
                .attr('x', buttonWidth / 2)
                .attr('y', buttonHeight / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff')
                .attr('font-size', 11)
                .attr('font-weight', 600)
                .text('Add Feedback');

              // Hover effect
              feedbackBtn.on('mouseenter', function () {
                d3.select(this).select('rect').attr('fill', '#6d28d9');
              });
              feedbackBtn.on('mouseleave', function () {
                d3.select(this).select('rect').attr('fill', '#7c3aed');
              });
            }
          }
        });

        // Show indicator if more panels exist
        if (nodeDecisions.length > maxPanels) {
          const indicatorX = startX + totalWidth + 10;
          expandedOverlayGroup.append('text')
            .attr('x', indicatorX)
            .attr('y', startY + panelHeight / 2)
            .attr('font-size', 12)
            .attr('fill', '#7c3aed')
            .attr('font-weight', 600)
            .text(`+${nodeDecisions.length - maxPanels}`);
        }

        return; // Don't show regular overlay
      }

      // === Regular overlay (for non-traversed benchmark nodes or traverse mode) ===

      // Get the full label (for activator nodes, combine with ancestor)
      let fullLabel = d.label;
      if (d.category === 'activator' || d.depth === 7) {
        const ancestorLabel = getActivatorAncestorLabel(d.id);
        const labelValue = d.label.includes(': ') ? d.label.split(': ').slice(1).join(': ') : d.label;
        if (ancestorLabel) {
          fullLabel = `${ancestorLabel}, ${labelValue}`;
        }
      }

      // Find decision for this node (traverse mode - show batch info with selected codes)
      // Use centralized helper to filter out children batches for leaf nodes
      const nodeDecision = !benchmarkMode && decisions
        ? decisions.find(dec => {
          if (dec.current_node !== d.id) return false;
          return shouldIncludeDecision(dec, d, finalizedCodesSet);
        })
        : null;

      // Layout constants
      const padding = 16;
      const lineHeight = 15;
      const maxCharsPerLine = 45;

      // Wrap text for display
      const labelLines = wrapText(fullLabel, maxCharsPerLine);
      const billableText = d.billable ? '(Billable)' : '(Non-Billable)';

      // If we have a decision, show the detailed batch format (same as benchmark)
      if (nodeDecision) {
        const batchName = normalizeBatchName(nodeDecision.current_label);
        const selectedCandidates = nodeDecision.candidates.filter(c => c.selected);

        // Prepare selected items with wrapped labels
        const selectedItems: { code: string; labelLines: string[] }[] = [];
        selectedCandidates.forEach((candidate) => {
          selectedItems.push({
            code: candidate.code,
            labelLines: wrapText(candidate.label, maxCharsPerLine - 4),
          });
        });

        // Wrap reasoning
        const firstSelected = selectedCandidates[0];
        const reasoningLines = firstSelected?.reasoning
          ? wrapText(firstSelected.reasoning, maxCharsPerLine - 2)
          : [];

        // Calculate content height
        let contentHeight = padding; // Top padding
        contentHeight += 16; // Code + (Billable) row
        contentHeight += 8; // Gap after code line
        contentHeight += labelLines.length * lineHeight; // Label lines
        contentHeight += 18; // Gap + batch name header
        contentHeight += 6; // Gap after header

        // Selected codes with labels
        selectedItems.forEach((item, idx) => {
          contentHeight += lineHeight; // Code line
          contentHeight += item.labelLines.length * lineHeight; // Label lines
          if (idx < selectedItems.length - 1) {
            contentHeight += 6; // Gap between items
          }
        });

        // Reasoning section
        if (reasoningLines.length > 0) {
          contentHeight += 16; // Gap before "Reasoning:" header
          contentHeight += lineHeight; // "Reasoning:" header
          contentHeight += reasoningLines.length * lineHeight; // Reasoning lines
        }

        // Add Feedback textarea + button (TRAVERSE tab only)
        if (allowRewind && !benchmarkMode && d.id !== 'ROOT' && onNodeRewindClick) {
          contentHeight += 12; // Gap before textarea
          contentHeight += 50; // Textarea height (2 lines)
          contentHeight += 8; // Gap between textarea and button
          contentHeight += 26; // Button height
        }

        contentHeight += padding; // Bottom padding

        const overlayWidth = 340;
        const overlayHeight = Math.max(contentHeight, 140);
        const overlayX = pos.x - overlayWidth / 2;
        const overlayY = pos.y - overlayHeight / 2;

        // Determine colors using centralized helper (includes finalizedCodesSet check)
        const { bgColor, borderColor } = getOverlayColors(d, finalizedCodesSet);

        // Create overlay group
        const overlay = expandedOverlayGroup.append('g')
          .attr('class', `expanded-node node-${d.category}`)
          .attr('transform', `translate(${overlayX}, ${overlayY})`)
          .on('click', (event: MouseEvent) => event.stopPropagation());

        // Background
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

        let yPos = padding + 14;

        // Line 1: <Parent Code> (Billable/Non-Billable)
        const codeLineText = overlay.append('text')
          .attr('x', padding)
          .attr('y', yPos)
          .attr('font-size', 13);

        codeLineText.append('tspan')
          .attr('font-weight', 700)
          .attr('font-family', 'ui-monospace, monospace')
          .attr('fill', '#0f172a')
          .text(d.code);

        codeLineText.append('tspan')
          .attr('font-weight', 500)
          .attr('fill', d.billable ? '#15803d' : '#64748b')
          .text(` ${billableText}`);

        // Parent label (wrapped, complete)
        yPos += 10;
        labelLines.forEach((line) => {
          yPos += lineHeight;
          overlay.append('text')
            .attr('x', padding)
            .attr('y', yPos)
            .attr('font-size', 11)
            .attr('fill', '#334155')
            .text(line);
        });

        // Batch name header
        yPos += 20;
        overlay.append('text')
          .attr('x', padding)
          .attr('y', yPos)
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .attr('fill', '#7c3aed')
          .text(`${batchName}:`);

        // Selected codes with labels
        yPos += 6;
        selectedItems.forEach((item, itemIdx) => {
          // Code on its own line
          yPos += lineHeight;
          overlay.append('text')
            .attr('x', padding + 8)
            .attr('y', yPos)
            .attr('font-size', 11)
            .attr('font-weight', 600)
            .attr('font-family', 'ui-monospace, monospace')
            .attr('fill', '#1e293b')
            .text(item.code);

          // Label lines (wrapped, indented)
          item.labelLines.forEach((line) => {
            yPos += lineHeight;
            overlay.append('text')
              .attr('x', padding + 16)
              .attr('y', yPos)
              .attr('font-size', 10)
              .attr('fill', '#475569')
              .text(line);
          });

          // Gap between selected items
          if (itemIdx < selectedItems.length - 1) {
            yPos += 4;
          }
        });

        // Reasoning section with header
        if (reasoningLines.length > 0) {
          yPos += 18;
          overlay.append('text')
            .attr('x', padding)
            .attr('y', yPos)
            .attr('font-size', 11)
            .attr('font-weight', 600)
            .attr('fill', '#64748b')
            .text('Reasoning:');

          reasoningLines.forEach((line) => {
            yPos += lineHeight;
            overlay.append('text')
              .attr('x', padding + 8)
              .attr('y', yPos)
              .attr('font-size', 10)
              .attr('font-style', 'italic')
              .attr('fill', '#64748b')
              .text(line);
          });
        }

        // Add Feedback textarea + button for TRAVERSE tab (below reasoning)
        if (allowRewind && !benchmarkMode && d.id !== 'ROOT' && onNodeRewindClick) {
          const textareaWidth = overlayWidth - padding * 2;
          const textareaHeight = 50;
          const buttonWidth = 100;
          const buttonHeight = 24;

          // Construct batchId from decision
          const batchType = nodeDecision.current_label.match(/^(\w+)\s+batch$/)?.[1] || 'children';
          const batchId = `${d.id}|${batchType}`;
          const textareaId = `feedback-textarea-single-${d.id}-${batchType}`;

          // Gap before textarea
          yPos += 12;

          // Textarea using foreignObject
          const fo = overlay.append('foreignObject')
            .attr('x', padding)
            .attr('y', yPos)
            .attr('width', textareaWidth)
            .attr('height', textareaHeight);

          fo.append('xhtml:textarea')
            .attr('id', textareaId)
            .attr('placeholder', 'Enter feedback for this batch...')
            .style('width', '100%')
            .style('height', '100%')
            .style('resize', 'none')
            .style('background-color', '#f8fafc')
            .style('border', '1px solid #d1d5db')
            .style('border-radius', '4px')
            .style('padding', '6px 8px')
            .style('font-size', '11px')
            .style('font-family', 'inherit')
            .style('line-height', '1.4')
            .style('overflow-y', 'auto')
            .style('box-sizing', 'border-box')
            .on('click', (event: MouseEvent) => event.stopPropagation())
            .on('mousedown', (event: MouseEvent) => event.stopPropagation());

          yPos += textareaHeight + 8; // Textarea height + gap before button

          const feedbackBtn = overlay.append('g')
            .attr('class', 'feedback-button')
            .attr('transform', `translate(${padding}, ${yPos})`)
            .style('cursor', 'pointer')
            .on('click', (event: MouseEvent) => {
              event.stopPropagation();
              // Get textarea value
              const textareaEl = document.getElementById(textareaId) as HTMLTextAreaElement | null;
              const feedbackText = textareaEl?.value || '';
              onNodeRewindClick(d.id, batchId, feedbackText);
            });

          feedbackBtn.append('rect')
            .attr('width', buttonWidth)
            .attr('height', buttonHeight)
            .attr('rx', 4)
            .attr('fill', '#7c3aed');

          feedbackBtn.append('text')
            .attr('x', buttonWidth / 2)
            .attr('y', buttonHeight / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', 11)
            .attr('font-weight', 600)
            .text('Add Feedback');

          // Hover effect
          feedbackBtn.on('mouseenter', function () {
            d3.select(this).select('rect').attr('fill', '#6d28d9');
          });
          feedbackBtn.on('mouseleave', function () {
            d3.select(this).select('rect').attr('fill', '#7c3aed');
          });
        }

        return; // Don't continue to basic overlay
      }

      // === Benchmark Expected overlay (expected nodes not yet traversed) ===
      // Shows: code, billable, label, checkered flag if finalized leaf
      // No children data (expected graph doesn't show traversal-discovered children)
      const benchmarkExpected = benchmarkMode &&
        (d as BenchmarkGraphNode).benchmarkStatus === 'expected';

      if (benchmarkExpected) {
        const isExpectedLeaf = expectedLeaves.has(d.id);

        // Calculate content dimensions (no children section)
        const codeText = d.code;
        const longestLabelLine = labelLines.reduce((a, b) => a.length > b.length ? a : b, '');
        const estimatedCodeWidth = codeText.length * 11 + 100;
        const estimatedLabelWidth = longestLabelLine.length * 8;
        const minWidth = 280;
        const maxWidth = 420;
        const overlayWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(estimatedCodeWidth, estimatedLabelWidth) + padding * 2));

        // Calculate height (no children, but may have flag)
        let contentHeight = padding;
        contentHeight += 18; // Code line
        contentHeight += 10; // Gap after code
        contentHeight += labelLines.length * lineHeight; // Label lines
        if (isExpectedLeaf) {
          contentHeight += 22; // Gap + checkered flag row
        }
        contentHeight += padding;
        const overlayHeight = contentHeight;

        const overlayX = pos.x - overlayWidth / 2;
        const overlayY = pos.y - overlayHeight / 2;

        // Black outline, white background (expected/not-yet-traversed style)
        const bgColor = 'rgba(255, 255, 255, 0.98)';
        const borderColor = '#1e293b';

        // Create overlay group
        const overlay = expandedOverlayGroup.append('g')
          .attr('class', `expanded-node node-${d.category}`)
          .attr('transform', `translate(${overlayX}, ${overlayY})`)
          .on('click', (event: MouseEvent) => event.stopPropagation());

        // Background
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

        let yPos = padding + 14;

        // Line 1: <Code> (Billable/Non-Billable)
        const codeLineText = overlay.append('text')
          .attr('x', padding)
          .attr('y', yPos)
          .attr('font-size', 13);

        codeLineText.append('tspan')
          .attr('font-weight', 700)
          .attr('font-family', 'ui-monospace, monospace')
          .attr('fill', '#0f172a')
          .text(codeText);

        codeLineText.append('tspan')
          .attr('font-weight', 500)
          .attr('fill', d.billable ? '#15803d' : '#64748b')
          .text(` ${billableText}`);

        // Label lines
        yPos += 10;
        labelLines.forEach((line) => {
          yPos += lineHeight;
          overlay.append('text')
            .attr('x', padding)
            .attr('y', yPos)
            .attr('font-size', 11)
            .attr('fill', '#334155')
            .text(line);
        });

        // Checkered flag for expected leaves
        if (isExpectedLeaf) {
          yPos += 20;
          overlay.append('text')
            .attr('x', padding)
            .attr('y', yPos)
            .attr('font-size', 14)
            .text('🏁 Expected Finalized');
        }

        return; // Don't continue to basic overlay
      }

      // === Basic overlay (no decision data - show code, label, children) ===

      // Get children codes for "Children:" section
      const childIds = allChildren.get(d.id) || [];
      const nextCodes: { code: string; rule?: string }[] = [];
      for (const childId of childIds) {
        const childNode = nodeMap.get(childId);
        if (childNode) {
          const lateralEdge = [...sevenChrDefEdges, ...otherLateralEdges].find(
            e => String(e.source) === d.id && String(e.target) === childId
          );
          nextCodes.push({ code: childNode.code, rule: lateralEdge?.rule ?? undefined });
        }
      }

      // Calculate content dimensions
      const codeText = d.code;

      const longestLabelLine = labelLines.reduce((a, b) => a.length > b.length ? a : b, '');
      const estimatedCodeWidth = codeText.length * 11 + 100;
      const estimatedLabelWidth = longestLabelLine.length * 8;
      const minWidth = 280;
      const maxWidth = 420;
      const overlayWidth = Math.min(maxWidth, Math.max(minWidth, Math.max(estimatedCodeWidth, estimatedLabelWidth) + padding * 2));

      // Calculate height
      let contentHeight = padding;
      contentHeight += 18; // Code line
      contentHeight += 10; // Gap after code
      contentHeight += labelLines.length * lineHeight; // Label lines
      if (nextCodes.length > 0) {
        contentHeight += 16; // Gap + header
        contentHeight += Math.min(nextCodes.length, 4) * lineHeight; // Children items
        if (nextCodes.length > 4) contentHeight += lineHeight; // "+N more"
      }
      contentHeight += padding;
      const overlayHeight = contentHeight;

      const overlayX = pos.x - overlayWidth / 2;
      const overlayY = pos.y - overlayHeight / 2;

      // Determine colors using centralized helper (includes finalizedCodesSet check)
      const { bgColor, borderColor } = getOverlayColors(d, finalizedCodesSet);

      // Create overlay group
      const overlay = expandedOverlayGroup.append('g')
        .attr('class', `expanded-node node-${d.category}`)
        .attr('transform', `translate(${overlayX}, ${overlayY})`)
        .on('click', (event: MouseEvent) => event.stopPropagation());

      // Background
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

      let yPos = padding + 14;

      // Line 1: <Code> (Billable/Non-Billable)
      const codeLineText = overlay.append('text')
        .attr('x', padding)
        .attr('y', yPos)
        .attr('font-size', 13);

      codeLineText.append('tspan')
        .attr('font-weight', 700)
        .attr('font-family', 'ui-monospace, monospace')
        .attr('fill', '#0f172a')
        .text(codeText);

      codeLineText.append('tspan')
        .attr('font-weight', 500)
        .attr('fill', d.billable ? '#15803d' : '#64748b')
        .text(` ${billableText}`);

      // Label lines (wrapped, complete)
      yPos += 10;
      labelLines.forEach((line) => {
        yPos += lineHeight;
        overlay.append('text')
          .attr('x', padding)
          .attr('y', yPos)
          .attr('font-size', 11)
          .attr('fill', '#334155')
          .text(line);
      });

      // Children codes section
      if (nextCodes.length > 0) {
        yPos += 18;
        overlay.append('text')
          .attr('x', padding)
          .attr('y', yPos)
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .attr('fill', '#64748b')
          .text('Children:');

        const displayCount = Math.min(nextCodes.length, 4);
        nextCodes.slice(0, displayCount).forEach((nc) => {
          yPos += lineHeight;
          const nextText = overlay.append('text')
            .attr('x', padding + 8)
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
          yPos += lineHeight;
          overlay.append('text')
            .attr('x', padding + 8)
            .attr('y', yPos)
            .attr('font-size', 11)
            .attr('fill', '#94a3b8')
            .attr('font-style', 'italic')
            .text(`+${nextCodes.length - 4} more...`);
        }
      }
    };

    // Helper to hide expanded overlay
    const hideExpandedNode = () => {
      expandedOverlayGroup.selectAll('*').remove();
    };

    // Render non-ROOT nodes
    const nodeGroups = nodesGroup.selectAll('.node:not(.node-root)')
      .data(nodes.filter(n => n.id !== 'ROOT'))
      .join('g')
      .attr('class', d => `node node-${d.category}`)
      .attr('transform', d => {
        const pos = positions.get(d.id);
        return pos ? `translate(${pos.x - NODE_WIDTH / 2}, ${pos.y - NODE_HEIGHT / 2})` : '';
      })
      .style('cursor', 'pointer')
      .on('click', function (event: MouseEvent, d) {
        event.stopPropagation(); // Prevent SVG background click handler
        // Pin this node's overlay (clicking another node will pin that one instead)
        setPinnedNodeId(d.id);
        showExpandedNode(d, this as SVGGElement);
        onNodeClick(d.id);
      })
      .on('mouseenter', function (_, d) {
        showExpandedNode(d, this as SVGGElement);
      })
      .on('mouseleave', () => {
        // Only hide if not pinned
        if (!pinnedNodeIdRef.current) {
          hideExpandedNode();
        }
      });

    // Node rectangles - uniform height
    nodeGroups.append('rect')
      .attr('width', NODE_WIDTH)
      .attr('height', NODE_HEIGHT)
      .attr('rx', 4)
      .attr('ry', 4)
      .attr('fill', d => benchmarkMode ? getBenchmarkNodeFill(d as BenchmarkGraphNode) : getNodeFill(d, finalizedCodesSet))
      .attr('stroke', d => {
        if (d.id === selectedNode) return '#0f172a';
        return benchmarkMode ? getBenchmarkNodeStroke(d as BenchmarkGraphNode) : getNodeStroke(d, finalizedCodesSet);
      })
      .attr('stroke-width', d => {
        if (d.id === selectedNode) return 2.5;
        return benchmarkMode ? getBenchmarkNodeStrokeWidth(d as BenchmarkGraphNode) : getNodeStrokeWidth(d, finalizedCodesSet);
      })
      .attr('stroke-dasharray', d => {
        if (benchmarkMode) {
          return getBenchmarkNodeStrokeDasharray(d as BenchmarkGraphNode);
        }
        return d.category === 'placeholder' ? '4,2' : null;
      });

    // Node code text - positioned at top
    nodeGroups.append('text')
      .attr('class', 'node-code')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('font-family', 'ui-monospace, monospace')
      .attr('fill', '#1e293b')
      .text(d => d.code);

    // Billable indicator
    nodeGroups.filter(d => d.billable)
      .append('text')
      .attr('class', 'node-billable')
      .attr('x', NODE_WIDTH - 6)
      .attr('y', 16)
      .attr('text-anchor', 'end')
      .attr('font-size', 12)
      .attr('font-weight', 700)
      .attr('fill', '#16a34a')
      .text('$');

    // Checkered flag indicator for expected leaves (benchmark mode only)
    if (benchmarkMode && expectedLeaves.size > 0) {
      nodeGroups.filter(d => expectedLeaves.has(d.id))
        .append('text')
        .attr('class', 'node-flag')
        .attr('x', 6)
        .attr('y', 16)
        .attr('text-anchor', 'start')
        .attr('font-size', 10)
        .text('🏁');
    }

    // Node label text - word-wrapped, two lines max
    nodeGroups.append('text')
      .attr('class', 'node-label')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', 32)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', '#64748b')
      .text(d => wrapNodeLabel(d.label, 22)[0]);

    nodeGroups.append('text')
      .attr('class', 'node-label')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', 44)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', '#64748b')
      .text(d => wrapNodeLabel(d.label, 22)[1]);

    // Note: Native title tooltip removed - using expanded overlay on hover instead

    // Render overshoot markers (red arrows + optional X) in benchmark mode
    if (benchmarkMode && overshootMarkers.length > 0) {
      // Calculate positions for overshoot X markers
      const overshootPositions = new Map<string, { x: number; y: number }>();
      for (const marker of overshootMarkers) {
        const sourcePos = positions.get(marker.sourceNode);
        if (sourcePos) {
          // Position X marker below the source node at the correct depth
          const y = marker.depth * LEVEL_HEIGHT + 50;
          const x = sourcePos.x; // Same x as source for vertical alignment
          overshootPositions.set(marker.targetCode, { x, y });
        }
      }

      // Render edges pointing to overshoot X positions (always show red arrows)
      const overshootEdgeGroup = g.append('g').attr('class', 'overshoot-edges');
      overshootEdgeGroup.selectAll('.edge-overshoot')
        .data(overshootMarkers)
        .join('path')
        .attr('class', 'edge edge-overshoot')
        .attr('d', d => {
          // Source could be a node or a previous overshoot position
          const srcPos = positions.get(d.sourceNode) || overshootPositions.get(d.sourceNode);
          const tgtPos = overshootPositions.get(d.targetCode);
          if (!srcPos || !tgtPos) return '';
          const x1 = srcPos.x;
          const y1 = srcPos.y + NODE_HEIGHT / 2;
          const x2 = tgtPos.x;
          const y2 = tgtPos.y - 15; // Stop before the X marker
          return `M${x1},${y1} L${x2},${y2}`;
        })
        .attr('fill', 'none')
        .attr('stroke', '#dc2626')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead-overshoot)');

      // Render X markers at overshoot positions (only when showXMarkers is true)
      if (showXMarkers) {
        const overshootGroup = g.append('g').attr('class', 'overshoot-markers');
        overshootGroup.selectAll('.overshoot-x')
          .data([...overshootPositions.entries()])
          .join('g')
          .attr('class', 'overshoot-x')
          .attr('transform', ([, pos]) => `translate(${pos.x}, ${pos.y})`)
          .each(function () {
            const xGroup = d3.select(this);
            // Draw X shape
            xGroup.append('path')
              .attr('d', 'M-10,-10 L10,10 M10,-10 L-10,10')
              .attr('stroke', '#dc2626')
              .attr('stroke-width', 3)
              .attr('fill', 'none')
              .attr('stroke-linecap', 'round');
          });
      }
    }

    // Render missed edge markers (red X near target node's arrowhead) - only when showXMarkers is true
    if (benchmarkMode && missedEdgeMarkers.length > 0 && showXMarkers) {
      const missedGroup = g.append('g').attr('class', 'missed-edge-markers');

      missedEdgeMarkers.forEach(marker => {
        const srcPos = positions.get(marker.edgeSource);
        const tgtPos = positions.get(marker.edgeTarget);
        if (!srcPos || !tgtPos) return;

        // Position X a few pixels before the arrowhead on the edge path
        // For curved edges (when source and target have different x), follow the curve
        const ARROW_PADDING = 8;
        const X_OFFSET_BEFORE_ARROW = 35; // Distance along edge before arrowhead

        // Calculate point on edge path near the arrowhead
        // The arrowhead is at (tgtPos.x, tgtPos.y - NODE_HEIGHT/2 - ARROW_PADDING)
        const arrowX = tgtPos.x;
        const arrowY = tgtPos.y - NODE_HEIGHT / 2 - ARROW_PADDING;

        // For lateral (curved) edges, position X along the approach direction
        // Calculate approach angle from control point to target
        const dx = tgtPos.x - srcPos.x;
        if (Math.abs(dx) > NODE_WIDTH) {
          // This is a curved lateral edge - position X before arrowhead along the curve's approach
          // The curve approaches from above-left, so offset diagonally
          const xPos = arrowX - X_OFFSET_BEFORE_ARROW * 0.7;
          const yPos = arrowY - X_OFFSET_BEFORE_ARROW * 0.5;

          missedGroup.append('path')
            .attr('d', `M${xPos - 8},${yPos - 8} L${xPos + 8},${yPos + 8} M${xPos + 8},${yPos - 8} L${xPos - 8},${yPos + 8}`)
            .attr('stroke', '#dc2626')
            .attr('stroke-width', 3)
            .attr('fill', 'none')
            .attr('stroke-linecap', 'round');
        } else {
          // Vertical/near-vertical edge - position X directly above arrowhead (closer than lateral)
          const xPos = arrowX;
          const yPos = arrowY - 18; // Reduced offset for vertical edges (was X_OFFSET_BEFORE_ARROW = 35)

          missedGroup.append('path')
            .attr('d', `M${xPos - 8},${yPos - 8} L${xPos + 8},${yPos + 8} M${xPos + 8},${yPos - 8} L${xPos - 8},${yPos + 8}`)
            .attr('stroke', '#dc2626')
            .attr('stroke-width', 3)
            .attr('fill', 'none')
            .attr('stroke-linecap', 'round');
        }
      });
    }

    // Raise expanded overlay group to top so it renders above markers
    expandedOverlayGroup.raise();

    // Set up zoom with interaction tracking
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('start', (event) => {
        // Only track ACTUAL user interactions, not programmatic transforms
        // event.sourceEvent is null for programmatic calls, defined for user interactions
        if (event.sourceEvent) {
          lastInteractionTime.current = Date.now();
        }
      })
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        // Only track ACTUAL user interactions
        if (event.sourceEvent) {
          lastInteractionTime.current = Date.now();
        }
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Click on SVG background (any click not stopped by nodes/buttons) releases the pinned overlay
    svg.on('click', () => {
      if (pinnedNodeIdRef.current) {
        setPinnedNodeId(null);
        hideExpandedNode();
      }
    });

    // IMPORTANT: Apply the current transform to the new <g> element
    // This preserves the user's pan/zoom position when the graph re-renders
    const currentTransform = d3.zoomTransform(svg.node()!);
    g.attr('transform', currentTransform.toString());

    // Determine if we should re-center the graph
    // Only re-center on: initial render, traversal completion, or new graph loaded
    const traversalJustCompleted = prevIsTraversing.current && !isTraversing;
    const traversalJustStarted = !prevIsTraversing.current && isTraversing;
    const graphJustLoaded = prevNodeCount.current === 0 && nodes.length > 0;

    // Reset zoom initialization when a new traversal starts or new graph loads
    if (traversalJustStarted || graphJustLoaded) {
      hasInitializedZoom.current = false;
    }

    // Check if user has interacted within the last 3 seconds
    const timeSinceLastInteraction = Date.now() - lastInteractionTime.current;
    const userRecentlyInteracted = lastInteractionTime.current > 0 && timeSinceLastInteraction < 3000;

    const shouldRecenter = ((!hasInitializedZoom.current && nodes.length > 0) || traversalJustCompleted || graphJustLoaded) && !userRecentlyInteracted;

    // Update prev tracking
    prevIsTraversing.current = isTraversing;
    prevNodeCount.current = nodes.length;

    if (shouldRecenter) {
      // Use fit-to-window for proper centering
      handleFitToWindow();
      hasInitializedZoom.current = true;
    }

  }, [nodes, edges, selectedNode, onNodeClick, finalizedCodes, isTraversing, benchmarkMode, overshootMarkers, missedEdgeMarkers, expectedLeaves, nodeReasoningMap, handleFitToWindow, showXMarkers]);

  return (
    <div className="graph-container" ref={containerRef}>
      <div className="view-header-bar">
        <div className="view-status-section">
          <div className="status-line">
            <span className="status-label">Status:</span>
            {status === 'idle' ? (
              <span className="status-value status-idle">IDLE</span>
            ) : status === 'complete' ? (
              <span className="status-value status-complete">COMPLETE</span>
            ) : status === 'error' ? (
              <span className="status-value status-error">ERROR</span>
            ) : (
              <span className="status-value status-processing">PROCESSING</span>
            )}
            {(status === 'complete' || status === 'error') && elapsedTime !== null && (
              <span className="status-elapsed">({formatElapsedTime(elapsedTime)})</span>
            )}
            {status === 'error' && errorMessage && (
              <span className="status-message">{errorMessage}</span>
            )}
            {isTraversing && currentStep && (
              <span className="status-message">{currentStep}</span>
            )}
          </div>
          {benchmarkMode && benchmarkMetrics ? (
            <div className="report-line benchmark-metrics-line">
              <span className="report-label">Benchmark:</span>
              <span className="report-stats">
                <span className="benchmark-stat exact">
                  <strong>{benchmarkMetrics.exactCount}</strong> matched final code
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-stat undershoot">
                  <strong>{benchmarkMetrics.undershootCount}</strong> undershot final code
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-stat overshoot">
                  <strong>{benchmarkMetrics.overshootCount}</strong> overshot final code
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-stat missed">
                  <strong>{benchmarkMetrics.missedCount}</strong> missed decisions
                </span>
                {benchmarkMetrics.otherCount > 0 && (
                  <>
                    <span className="stat-separator">·</span>
                    <span className="benchmark-stat other">
                      <strong>{benchmarkMetrics.otherCount}</strong> other trajectories
                    </span>
                  </>
                )}
                <span className="stat-separator">|</span>
                <span className="benchmark-score">
                  Traversal Recall: <strong>{(benchmarkMetrics.traversalRecall * 100).toFixed(1)}%</strong>
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-score">
                  Final Codes Recall: <strong>{(benchmarkMetrics.finalCodesRecall * 100).toFixed(1)}%</strong>
                </span>
              </span>
            </div>
          ) : benchmarkMode && !benchmarkMetrics && nodeCount > 0 ? (
            // Benchmark mode: before or during traversal
            <div className="report-line">
              <span className="report-label">Report:</span>
              <span className="report-stats">
                {status === 'idle' ? (
                  // Before traversal: show target nodes count
                  <>
                    <strong>{nodeCount}</strong> target nodes
                  </>
                ) : (
                  // During traversal: show intersecting traversed nodes count
                  <>
                    <strong>
                      {(nodes as BenchmarkGraphNode[]).filter(
                        n => n.benchmarkStatus === 'traversed' || n.benchmarkStatus === 'matched'
                      ).length}
                    </strong> nodes traversed
                    <span className="stat-separator">·</span>
                    <strong>{nodeCount}</strong> target nodes
                  </>
                )}
              </span>
            </div>
          ) : (nodeCount > 0 || finalizedCodes.length > 0 || decisionCount > 0) && (
            <div className="report-line">
              <span className="report-label">Report:</span>
              <span className="report-stats">
                {finalizedCodes.length > 0 && status === 'complete' && (
                  <>
                    <strong>{finalizedCodes.length}</strong> codes finalized
                  </>
                )}
                {finalizedCodes.length > 0 && status === 'complete' && nodeCount > 0 && (
                  <span className="stat-separator">·</span>
                )}
                {nodeCount > 0 && (
                  <>
                    <strong>{nodeCount}</strong> nodes explored
                  </>
                )}
                {(nodeCount > 0 || (finalizedCodes.length > 0 && status === 'complete')) && decisionCount > 0 && (
                  <span className="stat-separator">·</span>
                )}
                {decisionCount > 0 && (
                  <>
                    <strong>{decisionCount}</strong> decisions made
                  </>
                )}
              </span>
            </div>
          )}
        </div>
        <button
          className="export-btn"
          onClick={handleExportSvg}
          title="Export as SVG"
          disabled={
            nodes.length === 0 ||
            (benchmarkMode && status !== 'complete' && status !== 'error')
          }
        >
          Export SVG
        </button>
      </div>
      <div className="graph-svg-area">
        <svg ref={svgRef} width="100%" height="100%" />
        <div className="zoom-controls">
          <button className="zoom-btn" onClick={handleZoomIn} title="Zoom in">+</button>
          <button className="zoom-btn" onClick={handleZoomOut} title="Zoom out">−</button>
          <button className="zoom-btn zoom-btn-fit" onClick={handleFitToWindow} title="Fit to window">⤢</button>
        </div>
        {status === 'idle' && nodes.filter(n => n.id !== 'ROOT').length === 0 && (
          <div className="empty-state absolute">
            <span className="empty-icon">📊</span>
            <span className="empty-text">No traversal data yet</span>
            <span className="empty-hint">Enter a clinical note and start traversal</span>
          </div>
        )}
        {isTraversing && nodes.filter(n => n.id !== 'ROOT').length === 0 && (
          <div className="loading-state absolute">
            <div className="spinner" />
            <span>Starting traversal...</span>
          </div>
        )}
      </div>
      <div className="legend">
        {benchmarkMode ? (
          <>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#ffffff', border: '2px dashed #1e293b' }} />
              <span>Expected</span>
            </div>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#ffffff', border: '2px solid #16a34a' }} />
              <span>Traversed</span>
            </div>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#dcfce7', border: '2px solid #16a34a' }} />
              <span>Matched Final Code</span>
            </div>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#fef3c7', border: '2px solid #16a34a' }} />
              <span>Undershot Code</span>
            </div>
            <div className="legend-item">
              <span className="legend-x" style={{ color: '#dc2626', fontWeight: 'bold', fontSize: '16px' }}>✕</span>
              <span>Overshot Code</span>
            </div>
            <div className="legend-item">
              <span className="legend-flag" style={{ fontSize: '14px' }}>🏁</span>
              <span>Target final code</span>
            </div>
          </>
        ) : (
          <>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#f0fdf4', border: '2px solid #22c55e' }} />
              <span>Final Code</span>
            </div>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#ffffff', border: '2px solid #3b82f6' }} />
              <span>7th Char Rule</span>
            </div>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#ffffff', border: '2px dashed #94a3b8' }} />
              <span>Placeholder</span>
            </div>
            <div className="legend-item">
              <span className="legend-box" style={{ background: '#ffffff', border: '2px solid #334155' }} />
              <span>Ancestor</span>
            </div>
            <div className="legend-item">
              <span className="legend-line solid" />
              <span>Child</span>
            </div>
            <div className="legend-item">
              <span className="legend-line dashed" />
              <span>Other Rule</span>
            </div>
          </>
        )}
      </div>
      {finalizedCodes.length > 0 && (
        <div className="finalized-codes-bar">
          {codesBarLabel && <span className="codes-bar-label">{codesBarLabel} ({finalizedCodes.length})</span>}
          <div className="codes-list">
            {sortedFinalizedCodes.map(code => (
              benchmarkMode && onRemoveExpectedCode ? (
                <span
                  key={code}
                  className={`code-badge removable${invalidCodes.has(code) ? ' invalid' : ''}`}
                  onClick={() => onRemoveExpectedCode(code)}
                >
                  {code}
                  <span className="remove-icon">×</span>
                </span>
              ) : (
                <span key={code} className={`code-badge${invalidCodes.has(code) ? ' invalid' : ''}`}>{code}</span>
              )
            ))}
          </div>
          <button
            className="sort-toggle"
            onClick={() => {
              // Cycle: default → asc → desc → default
              if (codeSortMode === 'default') setCodeSortMode('asc');
              else if (codeSortMode === 'asc') setCodeSortMode('desc');
              else setCodeSortMode('default');
            }}
            title={codeSortMode === 'default' ? 'Unsorted' : codeSortMode === 'asc' ? 'Sorted A-Z' : 'Sorted Z-A'}
          >
            <span className="sort-text">Sort</span>
            <span className={`sort-indicator ${codeSortMode !== 'default' ? 'active' : ''}`}>
              {codeSortMode === 'asc' ? '▲' : codeSortMode === 'desc' ? '▼' : '-'}
            </span>
          </button>
        </div>
      )}
    </div>
  );
}

// Wrap node label into two lines with word boundaries (no mid-word breaks)
// Returns [line1, line2] where line2 has ellipsis if there's overflow
function wrapNodeLabel(label: string | undefined, maxCharsPerLine: number): [string, string] {
  if (!label) return ['', ''];

  const words = label.split(' ');
  let line1 = '';
  let line2 = '';
  let wordIndex = 0;

  // Build first line - fit complete words only
  for (; wordIndex < words.length; wordIndex++) {
    const word = words[wordIndex];
    const testLine = line1 ? line1 + ' ' + word : word;
    if (testLine.length <= maxCharsPerLine) {
      line1 = testLine;
    } else {
      break;
    }
  }

  // If no words fit on line1, put the first word (even if too long)
  if (!line1 && words.length > 0) {
    line1 = words[0].length > maxCharsPerLine
      ? words[0].substring(0, maxCharsPerLine - 1) + '…'
      : words[0];
    wordIndex = 1;
  }

  // Build second line from remaining words
  for (; wordIndex < words.length; wordIndex++) {
    const word = words[wordIndex];
    const testLine = line2 ? line2 + ' ' + word : word;
    if (testLine.length <= maxCharsPerLine - 1) { // Reserve space for potential ellipsis
      line2 = testLine;
    } else {
      // More words remain - add ellipsis
      if (line2) {
        line2 = line2 + '…';
      } else {
        // First word of line2 is too long
        line2 = word.substring(0, maxCharsPerLine - 1) + '…';
      }
      return [line1, line2];
    }
  }

  // Check if there were more words that didn't fit
  // (This happens when we break out of the first loop with remaining words)
  if (wordIndex < words.length && line2 && !line2.endsWith('…')) {
    line2 = line2 + '…';
  }

  return [line1, line2];
}

// Check if two node positions collide (with padding)
function hasCollision(
  x1: number, y1: number,
  x2: number, y2: number,
  nodeWidth: number,
  nodeHeight: number,
  padding: number = 20
): boolean {
  return Math.abs(x1 - x2) < nodeWidth + padding &&
    Math.abs(y1 - y2) < nodeHeight + padding;
}

// Find a non-colliding X position for a node
function findNonCollidingX(
  startX: number,
  y: number,
  positions: Map<string, { x: number; y: number }>,
  nodeWidth: number,
  nodeHeight: number,
  excludeId?: string
): number {
  let x = startX;
  let maxAttempts = 30;

  while (maxAttempts-- > 0) {
    let collision = false;

    for (const [id, pos] of positions.entries()) {
      if (id === excludeId) continue;
      if (hasCollision(x, y, pos.x, pos.y, nodeWidth, nodeHeight)) {
        collision = true;
        // Move to the right of the colliding node
        x = pos.x + nodeWidth + 30;
        break;
      }
    }

    if (!collision) return x;
  }

  return x;
}

// Calculate positions using subtree width algorithm (from reference)
function calculatePositions(
  _nodes: GraphNode[],
  hierarchyChildren: Map<string, string[]>,  // Only hierarchy edges - for Phase 2 tree structure
  allChildren: Map<string, string[]>,         // All edges - for width calculation and lateral children
  containerWidth: number,
  nodeMap: Map<string, GraphNode>,
  lateralEdges: GraphEdge[],
  nodeWidth: number,
  nodeHeight: number,
  levelHeight: number
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const hierarchyNodeIds = new Set<string>(); // Track nodes that are part of hierarchy columns (Phase 2)

  // Build set of nodes that have hierarchy parents (connected via hierarchy/sevenChrDef from above)
  // These nodes should stay in their columns even if they're also lateral targets
  const nodesWithHierarchyParent = new Set<string>();
  for (const [, children] of hierarchyChildren.entries()) {
    for (const childId of children) {
      nodesWithHierarchyParent.add(childId);
    }
  }

  const layoutWidth = 160;
  const nodePadding = 15;

  // Helper to get Y position for a node based on depth
  const getYForDepth = (depth: number): number => {
    return depth * levelHeight + 50;
  };

  // Phase 1: Calculate subtree widths for ALL nodes that have children
  // Use allChildren to include lateral subtrees in width calculation
  const subtreeWidth = new Map<string, number>();

  function calcSubtreeWidth(nodeId: string, visited = new Set<string>()): number {
    if (subtreeWidth.has(nodeId)) return subtreeWidth.get(nodeId)!;
    if (visited.has(nodeId)) return layoutWidth + nodePadding; // Cycle detection
    visited.add(nodeId);

    const childIds = allChildren.get(nodeId) || [];
    if (childIds.length === 0) {
      subtreeWidth.set(nodeId, layoutWidth + nodePadding);
      return layoutWidth + nodePadding;
    }

    let totalWidth = 0;
    for (const childId of childIds) {
      totalWidth += calcSubtreeWidth(childId, visited);
    }
    totalWidth = Math.max(totalWidth, layoutWidth + nodePadding);
    subtreeWidth.set(nodeId, totalWidth);
    return totalWidth;
  }

  // Calculate widths starting from ROOT
  calcSubtreeWidth('ROOT');

  // Also calculate widths for any nodes introduced via lateral edges
  for (const edge of lateralEdges) {
    const targetId = String(edge.target);
    if (!subtreeWidth.has(targetId)) {
      calcSubtreeWidth(targetId);
    }
  }

  // Phase 2: Position subtree helper - positions a node and all its hierarchy children
  // parentDepth is used to infer depth for nodes not in nodeMap (edge targets from streaming)
  // isHierarchyPhase: true for Phase 2 (main tree), false for Phase 3 (lateral subtrees)
  //
  // COLUMNAR LAYOUT: Single-child chains are vertically aligned (same X) for straight edges.
  // Multi-child nodes spread children horizontally while keeping the layout balanced.
  function positionSubtree(nodeId: string, x: number, visited: Set<string>, parentDepth: number = -1, isHierarchyPhase: boolean = true) {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);

    const node = nodeMap.get(nodeId);
    // Use node's depth if available, otherwise infer from parent (tree depth)
    const nodeDepth = node?.depth ?? (parentDepth >= 0 ? parentDepth + 1 : 0);

    // Position this node if it hasn't been positioned yet
    // Position even if not in nodeMap - the node might be an edge target that hasn't been
    // added to the nodes array yet (streaming race condition)
    if (!positions.has(nodeId) && nodeId !== 'ROOT') {
      const y = getYForDepth(nodeDepth) + nodeHeight / 2;
      positions.set(nodeId, { x, y });
      // Track hierarchy nodes so they're never moved later
      if (isHierarchyPhase) {
        hierarchyNodeIds.add(nodeId);
      }
    }

    // Use hierarchyChildren for Phase 2, allChildren for lateral subtrees
    const childrenMap = isHierarchyPhase ? hierarchyChildren : allChildren;
    const childIds = childrenMap.get(nodeId) || [];
    if (childIds.length === 0) return;

    const sortedChildren = [...childIds].sort();

    // COLUMNAR LAYOUT: If single child, position directly below parent (same X)
    // This creates straight vertical edges for hierarchy and sevenChrDef relationships
    if (sortedChildren.length === 1) {
      positionSubtree(sortedChildren[0], x, visited, nodeDepth, isHierarchyPhase);
      return;
    }

    // Multiple children: spread horizontally centered under parent
    let totalChildWidth = 0;
    for (const childId of sortedChildren) {
      totalChildWidth += subtreeWidth.get(childId) || (layoutWidth + nodePadding);
    }

    let childX = x - totalChildWidth / 2;

    for (const childId of sortedChildren) {
      const childWidth = subtreeWidth.get(childId) || (layoutWidth + nodePadding);
      const childCenterX = childX + childWidth / 2;
      positionSubtree(childId, childCenterX, visited, nodeDepth, isHierarchyPhase);
      childX += childWidth;
    }
  }

  // Position the full tree starting from ROOT (use hierarchyChildren for vertical column layout)
  positionSubtree('ROOT', containerWidth / 2, new Set<string>(), -1, true);

  // Add ROOT as a virtual positioned node at top center for edge rendering
  positions.set('ROOT', { x: containerWidth / 2, y: 25 });
  hierarchyNodeIds.add('ROOT');

  // Phase 3: Lateral node positioning
  // Position lateral nodes close to their sources by shifting hierarchy columns if needed
  const minGap = 20;
  const step = nodeWidth + minGap;

  // Track lateral targets already processed - each should only be positioned ONCE
  // to prevent accumulated shifts from multiple passes
  const processedLateralTargets = new Set<string>();

  // Track how many lateral nodes have been placed from each source
  // Used to offset multiple lateral targets from the same parent incrementally
  const lateralCountPerSource = new Map<string, number>();

  let maxPasses = 15;
  let madeProgress = true;

  while (madeProgress && maxPasses-- > 0) {
    madeProgress = false;

    for (const edge of lateralEdges) {
      const targetId = String(edge.target);
      const sourceId = String(edge.source);

      // Skip if already processed in Phase 3 - prevents accumulated shifts
      if (processedLateralTargets.has(targetId)) continue;

      // Skip if already positioned AND has hierarchy parent (ancestor linking to it)
      // Nodes with hierarchy parents should stay in their columns
      if (positions.has(targetId) && nodesWithHierarchyParent.has(targetId)) continue;

      // If already positioned but no hierarchy parent, we'll reposition it closer to lateral source
      if (positions.has(targetId)) {
        // This node was positioned but has no hierarchy ancestor - reposition it
        hierarchyNodeIds.delete(targetId);  // Allow Phase 4 to move if needed
      }

      // Need source to be positioned first
      const sourcePos = positions.get(sourceId);
      if (!sourcePos) continue;

      const targetNode = nodeMap.get(targetId);

      // Determine depth: from node if available, otherwise calculate from code structure
      // ICD depth rules: Chapter_* = 1, Ranges (X##-X##) = 2, Codes = length without dots
      const targetDepth = targetNode?.depth ?? calculateDepthFromCode(targetId);
      const targetY = getYForDepth(targetDepth) + nodeHeight / 2;

      // Get how many lateral nodes already placed from this source
      const lateralIndex = lateralCountPerSource.get(sourceId) || 0;

      // Ideal position: to the right of source, offset by how many laterals already placed
      // First lateral at +step, second at +2*step, etc.
      const idealX = sourcePos.x + step * (lateralIndex + 1);

      // Check for ACTUAL collision at the lateral node's position (both X and Y)
      // This determines IF we need to shift - only shift when the lateral node truly collides
      let collidingColumnX: number | null = null;
      for (const [id, pos] of positions.entries()) {
        if (id === sourceId) continue;
        if (hasCollision(idealX, targetY, pos.x, pos.y, nodeWidth, nodeHeight)) {
          collidingColumnX = pos.x;
          break;
        }
      }

      // If there's actual collision, shift the ENTIRE hierarchy column (all Y depths)
      // and everything to its right - columns move as a unit
      if (collidingColumnX !== null) {
        const updates: Array<[string, { x: number; y: number }]> = [];
        for (const [id, pos] of positions.entries()) {
          if (id === sourceId) continue;
          // Shift all nodes at or to the right of the colliding column's X position
          if (pos.x >= collidingColumnX - nodeWidth / 2) {
            updates.push([id, { x: pos.x + step, y: pos.y }]);
          }
        }
        for (const [id, newPos] of updates) {
          positions.set(id, newPos);
        }
      }

      // Place the lateral node at its ideal position (close to source)
      positions.set(targetId, { x: idealX, y: targetY });
      processedLateralTargets.add(targetId);  // Mark as processed to prevent reprocessing
      lateralCountPerSource.set(sourceId, lateralIndex + 1);  // Increment for next lateral from same source
      madeProgress = true;

      // If this lateral node has HIERARCHY children, position them in a column below
      const hierarchyChildIds = hierarchyChildren.get(targetId);
      if (hierarchyChildIds && hierarchyChildIds.length > 0) {
        // Clear positions of all hierarchy descendants so they get repositioned
        // under the lateral target's new position (not their old Phase 2 positions)
        const clearDescendantPositions = (nodeId: string, visited: Set<string>) => {
          if (visited.has(nodeId)) return;
          visited.add(nodeId);
          const children = hierarchyChildren.get(nodeId) || [];
          for (const childId of children) {
            positions.delete(childId);
            hierarchyNodeIds.delete(childId);
            clearDescendantPositions(childId, visited);
          }
        };
        clearDescendantPositions(targetId, new Set<string>());

        positionSubtree(targetId, idealX, new Set<string>(), targetDepth, true);
      }
    }
  }

  // Log any nodes without positions
  for (const node of _nodes) {
    if (!positions.has(node.id)) {
      console.warn(`[POSITION] Node ${node.id} has no position - will not render edges to/from it`);
    }
  }

  // Phase 4: Final collision resolution pass - ONLY move non-hierarchy nodes
  const allPositionedNodes = [...positions.entries()];

  for (let i = 0; i < allPositionedNodes.length; i++) {
    const [idA, posA] = allPositionedNodes[i];

    for (let j = i + 1; j < allPositionedNodes.length; j++) {
      const [idB, posB] = allPositionedNodes[j];

      if (hasCollision(posA.x, posA.y, posB.x, posB.y, nodeWidth, nodeHeight)) {
        // Determine which node to move - NEVER move hierarchy nodes
        const aIsHierarchy = hierarchyNodeIds.has(idA);
        const bIsHierarchy = hierarchyNodeIds.has(idB);

        if (aIsHierarchy && bIsHierarchy) {
          // Both are hierarchy - this shouldn't happen, but don't move either
          console.warn(`[COLLISION] Two hierarchy nodes collide: ${idA} and ${idB}`);
          continue;
        }

        // Move the non-hierarchy node (or B if neither is hierarchy)
        const nodeToMove = aIsHierarchy ? idB : (bIsHierarchy ? idA : idB);
        const posToMove = nodeToMove === idA ? posA : posB;
        const otherPos = nodeToMove === idA ? posB : posA;

        const newX = findNonCollidingX(otherPos.x + nodeWidth + 30, posToMove.y, positions, nodeWidth, nodeHeight, nodeToMove);
        posToMove.x = newX;
        positions.set(nodeToMove, posToMove);

        // Update the array entry
        if (nodeToMove === idA) {
          allPositionedNodes[i] = [idA, posA];
        } else {
          allPositionedNodes[j] = [idB, posB];
        }
      }
    }
  }

  return positions;
}

// Create straight edge path for hierarchy edges
function createEdgePath(
  edge: GraphEdge,
  positions: Map<string, { x: number; y: number }>,
  nodeHeight: number
): string {
  const src = positions.get(String(edge.source));
  const tgt = positions.get(String(edge.target));

  if (!src || !tgt) return '';

  const ARROW_PADDING = 6;

  const x1 = src.x;
  const y1 = src.y + nodeHeight / 2;
  const x2 = tgt.x;
  const y2 = tgt.y - nodeHeight / 2 - ARROW_PADDING;

  return `M${x1},${y1} L${x2},${y2}`;
}

// Create curved edge path for lateral edges (quadratic Bezier)
function createCurvedEdgePath(
  edge: GraphEdge,
  positions: Map<string, { x: number; y: number }>,
  nodeHeight: number
): string {
  const src = positions.get(String(edge.source));
  const tgt = positions.get(String(edge.target));

  if (!src || !tgt) return '';

  const ARROW_PADDING = 8;

  const x1 = src.x;
  const y1 = src.y + nodeHeight / 2;
  const x2 = tgt.x;
  const y2 = tgt.y - nodeHeight / 2 - ARROW_PADDING;

  // Control point for quadratic Bezier curve
  const dx = x2 - x1;
  const cx = x1 + dx / 2;
  const cy = Math.min(y1, y2) - Math.abs(dx) * 0.2 - 20;

  return `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`;
}

// ============================================================================
// Node Type Helpers - Centralized node classification logic
// ============================================================================

// Check if node is a leaf node (billable, finalized category, or in finalized codes list)
function isLeafNode(node: GraphNode, finalizedCodes: Set<string>): boolean {
  return node.billable || node.category === 'finalized' || finalizedCodes.has(node.code);
}

// Check if node should be styled as finalized (same logic as isLeafNode for consistency)
function isFinalizedNode(node: GraphNode, finalizedCodes: Set<string>): boolean {
  return node.category === 'finalized' || node.billable || finalizedCodes.has(node.code);
}

function isActivatorNode(node: GraphNode): boolean {
  return node.category === 'activator';
}

function isPlaceholderNode(node: GraphNode): boolean {
  return node.category === 'placeholder';
}

// ============================================================================
// Batch/Decision Helpers - Centralized batch filtering and normalization
// ============================================================================

// Normalize batch label to lowercase name without " batch" suffix
function normalizeBatchName(label: string | undefined): string {
  return (label || '').replace(' batch', '').toLowerCase();
}

// Check if a decision should be included for overlay display
// Filters out children batches for nodes that can't have children selections
function shouldIncludeDecision(
  dec: DecisionPoint,
  node: GraphNode,
  finalizedCodes: Set<string>
): boolean {
  const batchName = normalizeBatchName(dec.current_label);
  const leaf = isLeafNode(node, finalizedCodes);
  const placeholderOrActivator = isPlaceholderNode(node) || isActivatorNode(node);

  // Filter out children batches for leaf, placeholder, and activator nodes
  if ((leaf || placeholderOrActivator) && batchName === 'children') {
    return false;
  }

  return true;
}

// Check if a batch section should be shown in overlay
// Returns false for children batches on leaf/placeholder nodes
function shouldShowBatch(batchName: string, node: GraphNode, finalizedCodes: Set<string>): boolean {
  const leaf = isLeafNode(node, finalizedCodes);
  const activator = isActivatorNode(node);
  const placeholder = isPlaceholderNode(node);

  // Activator nodes (6th char with sevenChrDef) should show the batch
  if (activator) return true;

  // Leaf nodes with 'children' batch - don't show (no children to select)
  // But DO show lateral batches (useAdditionalCode, codeFirst, codeAlso)
  if (leaf && batchName === 'children') return false;

  // Placeholder nodes with 'children' batch - don't show (implicit)
  if (placeholder && batchName === 'children') return false;

  // All other cases - show the batch section
  return true;
}

// ============================================================================
// Overlay Color Helpers - Centralized color determination for overlays
// ============================================================================

function getOverlayColors(node: GraphNode, finalizedCodes: Set<string>): { bgColor: string; borderColor: string } {
  const finalized = isFinalizedNode(node, finalizedCodes);
  const activator = isActivatorNode(node);
  const placeholder = isPlaceholderNode(node);

  const bgColor = finalized ? 'rgba(240, 253, 244, 0.98)' :
    activator ? 'rgba(239, 246, 255, 0.98)' :
      placeholder ? 'rgba(248, 250, 252, 0.98)' :
        'rgba(255, 255, 255, 0.98)';

  const borderColor = finalized ? '#16a34a' :
    activator ? '#2563eb' :
      placeholder ? '#94a3b8' :
        '#475569';

  return { bgColor, borderColor };
}

// ============================================================================
// Node Styling Helpers - For graph node rendering
// ============================================================================
// Priority: activator (blue) > finalized (green) > placeholder (dashed gray) > ancestor (black)

function getNodeFill(node: GraphNode, finalizedCodes: Set<string>): string {
  // Activator nodes (have sevenChrDef) are NOT finalized - they need 7th char
  if (isActivatorNode(node)) return '#ffffff';
  if (isFinalizedNode(node, finalizedCodes)) return '#f0fdf4';
  if (isPlaceholderNode(node)) return '#ffffff';
  // ROOT styled same as ancestor
  return '#ffffff';
}

function getNodeStroke(node: GraphNode, finalizedCodes: Set<string>): string {
  // Activator nodes get blue border (7th Char Rule indicator)
  if (isActivatorNode(node)) return '#3b82f6';
  if (isFinalizedNode(node, finalizedCodes)) return '#22c55e';
  if (isPlaceholderNode(node)) return '#94a3b8';
  // ROOT and ancestor nodes use same stroke
  return '#334155';
}

function getNodeStrokeWidth(node: GraphNode, finalizedCodes: Set<string>): number {
  if (isActivatorNode(node)) return 2;
  if (isFinalizedNode(node, finalizedCodes)) return 2.5;
  return 1.5;
}

// Calculate ICD depth from code structure
// Chapter_* = 1, Ranges (X##-X##) = 2, Codes = character count without dots
function calculateDepthFromCode(code: string): number {
  if (code.startsWith('Chapter_')) return 1;
  if (code.includes('-')) return 2; // Range like I20-I25
  // For actual codes: count characters excluding dots
  return code.replace(/\./g, '').length;
}

// Benchmark mode node styling
// Styles based on benchmarkStatus: expected, traversed, matched, undershoot
// NOTE: Use white fill (not transparent) to hide lateral edges behind nodes
function getBenchmarkNodeFill(node: BenchmarkGraphNode): string {
  switch (node.benchmarkStatus) {
    case 'expected':
      return '#ffffff';  // White fill - hides edges behind, dashed stroke shows status
    case 'traversed':
      return '#ffffff';  // White fill - hides edges behind, solid green stroke shows status
    case 'matched':
      return '#dcfce7';  // Light green fill - correct finalization
    case 'undershoot':
      return '#fef3c7';  // Amber fill - finalized too early
    default:
      return '#ffffff';  // White fill for any other status
  }
}

function getBenchmarkNodeStroke(node: BenchmarkGraphNode): string {
  switch (node.benchmarkStatus) {
    case 'expected':
      return '#1e293b';  // Black - waiting to be traversed
    case 'traversed':
      return '#16a34a';  // Green - traversed, not finalized
    case 'matched':
      return '#16a34a';  // Green - correct finalization
    case 'undershoot':
      return '#16a34a';  // Green - finalized (but too early)
    default:
      return '#1e293b';
  }
}

function getBenchmarkNodeStrokeWidth(node: BenchmarkGraphNode): number {
  switch (node.benchmarkStatus) {
    case 'matched':
      return 4.5;
    case 'traversed':
    case 'undershoot':
      return 4;
    default:
      return 1.5;
  }
}

// Get stroke dasharray for benchmark nodes (only expected is dashed)
function getBenchmarkNodeStrokeDasharray(node: BenchmarkGraphNode): string | null {
  return node.benchmarkStatus === 'expected' ? '4,2' : null;
}