import { useEffect, useRef, useMemo, useState, useCallback, memo } from 'react';
import * as d3 from 'd3';
import type { GraphNode, GraphEdge, TraversalStatus, BenchmarkGraphNode, BenchmarkMetrics, OvershootMarker, EdgeMissMarker, DecisionPoint } from '../lib/types';
import { exportSvgToFile, generateSvgFilename } from '../lib/exportSvg';
import { wrapNodeLabel, formatElapsedTime } from '../lib/textUtils';
import { calculatePositions } from '../lib/graphPositioning';
import { createEdgePath, createCurvedEdgePath, getPointOnStraightEdge, getPointOnCurvedEdge } from '../lib/edgePaths';
import {
  isFinalizedNode,
  isSevenChrDefFinalizedNode,
  getNodeFill,
  getNodeStroke,
  getNodeStrokeWidth,
  getBenchmarkNodeFill,
  getBenchmarkNodeStroke,
  getBenchmarkNodeStrokeWidth,
  getBenchmarkNodeStrokeDasharray,
} from '../lib/nodeStyles';
import { showNodeOverlay, type OverlayContext } from '../lib/overlayRenderer';

type SortMode = 'default' | 'asc' | 'desc';

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
  // Callback when user interacts with graph (drag, click) - resets idle timer
  onGraphInteraction?: () => void;
  // Rewind feature props (TRAVERSE tab only)
  onNodeRewindClick?: (nodeId: string, batchType?: string, feedback?: string) => void;
  allowRewind?: boolean;
  // Controls whether X markers render (default true)
  showXMarkers?: boolean;
  // Streaming traversed IDs for real-time visual feedback during benchmark traversal
  streamingTraversedIds?: Set<string>;
  // Node ID currently being rewound (shows loading state)
  rewindingNodeId?: string | null;
  // Callback when clicking empty space (not on nodes/overlays)
  onEmptySpaceClick?: () => void;
}

// Base constants (designed for ~600px container height / 1080p display)
const BASE_NODE_WIDTH = 140;
const BASE_NODE_HEIGHT = 60;
const BASE_LEVEL_HEIGHT = 100;
const REFERENCE_HEIGHT = 600; // Reference container height for scale factor calculation

function GraphViewerInner({
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
  onGraphInteraction,
  onNodeRewindClick,
  allowRewind = false,
  showXMarkers = true,
  streamingTraversedIds,
  rewindingNodeId = null,
  onEmptySpaceClick,
}: GraphViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [codeSortMode, setCodeSortMode] = useState<SortMode>('default');
  const [pinnedNodeId, setPinnedNodeId] = useState<string | null>(null);
  const pinnedNodeIdRef = useRef<string | null>(null);
  const showTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeOverlayNodeRef = useRef<string | null>(null);
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Ref for onNodeRewindClick to avoid stale closures in D3 event handlers
  const onNodeRewindClickRef = useRef(onNodeRewindClick);
  // Ref for onEmptySpaceClick to avoid D3 useEffect re-runs from inline lambdas
  const onEmptySpaceClickRef = useRef(onEmptySpaceClick);

  // Keep refs in sync with props/state for D3 event handlers
  useEffect(() => {
    pinnedNodeIdRef.current = pinnedNodeId;
  }, [pinnedNodeId]);

  useEffect(() => {
    onNodeRewindClickRef.current = onNodeRewindClick;
  }, [onNodeRewindClick]);

  useEffect(() => {
    onEmptySpaceClickRef.current = onEmptySpaceClick;
  }, [onEmptySpaceClick]);

  // Clear pinned state when the pinned node is no longer in the graph
  useEffect(() => {
    if (pinnedNodeId && !nodes.some(n => n.id === pinnedNodeId)) {
      setPinnedNodeId(null);
    }
  }, [pinnedNodeId, nodes]);

  // Track rewindingNodeId in ref for D3 access
  const rewindingNodeIdRef = useRef(rewindingNodeId);
  // Track previous fit trigger value to only run fit when trigger actually changes
  const prevFitTriggerRef = useRef(0);
  useEffect(() => {
    rewindingNodeIdRef.current = rewindingNodeId;
  }, [rewindingNodeId]);

  // Close overlay and clear pinned state when rewind starts
  useEffect(() => {
    if (rewindingNodeId) {
      // Clear pinned state - this will trigger overlay close via D3 effect
      setPinnedNodeId(null);
      activeOverlayNodeRef.current = null;
    }
  }, [rewindingNodeId]);

  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const lastInteractionTime = useRef<number>(0);
  // Ref for onGraphInteraction to avoid stale closures in D3 event handlers
  const onGraphInteractionRef = useRef(onGraphInteraction);
  // Ref for handleFitToWindow to avoid effect re-runs when nodes change during traversal
  const handleFitToWindowRef = useRef<() => void>(() => {});

  // Keep onGraphInteraction ref in sync
  useEffect(() => {
    onGraphInteractionRef.current = onGraphInteraction;
  }, [onGraphInteraction]);

  // Build outcome status map for benchmark code-badge coloring
  const outcomeStatusMap = useMemo(() => {
    if (!benchmarkMode || !benchmarkMetrics?.outcomes) return new Map<string, string>();
    const map = new Map<string, string>();
    for (const o of benchmarkMetrics.outcomes) {
      map.set(o.expectedCode, o.status);
    }
    return map;
  }, [benchmarkMode, benchmarkMetrics]);

  // Build code → label map for tooltip on code badges (handles sevenChrDef naming)
  const codeLabelMap = useMemo(() => {
    const map = new Map<string, string>();
    if (nodes.length === 0) return map;

    const nodeMap = new Map<string, GraphNode>();
    for (const n of nodes) {
      if (n.code) nodeMap.set(n.code, n);
    }

    const hierarchyParentMap = new Map<string, string>();
    const sevenChrDefParentMap = new Map<string, string>();
    for (const e of edges) {
      const src = String(e.source);
      const tgt = String(e.target);
      if (e.edge_type === 'hierarchy') {
        hierarchyParentMap.set(tgt, src);
      } else if (e.edge_type === 'lateral' && e.rule === 'sevenChrDef') {
        sevenChrDefParentMap.set(tgt, src);
      }
    }

    for (const code of finalizedCodes) {
      const node = nodeMap.get(code);
      if (!node) continue;

      if (node.depth === 7) {
        const parentId = sevenChrDefParentMap.get(node.id);
        let ancestorLabel = '';
        if (parentId) {
          let currentId = parentId;
          while (currentId && currentId !== 'ROOT') {
            const ancestor = nodeMap.get(currentId);
            if (ancestor && ancestor.category !== 'placeholder') {
              ancestorLabel = ancestor.label;
              break;
            }
            currentId = hierarchyParentMap.get(currentId) ?? '';
          }
        }
        const charLabel = node.label.includes(': ')
          ? node.label.split(': ').slice(1).join(': ')
          : node.label;
        map.set(code, ancestorLabel ? `${ancestorLabel}, ${charLabel}` : charLabel);
      } else {
        map.set(code, node.label);
      }
    }
    return map;
  }, [nodes, edges, finalizedCodes]);

  const sortedFinalizedCodes = useMemo(() => {
    if (codeSortMode === 'default') return finalizedCodes;
    const sorted = [...finalizedCodes].sort((a, b) => a.localeCompare(b));
    return codeSortMode === 'desc' ? sorted.reverse() : sorted;
  }, [finalizedCodes, codeSortMode]);

  const nodeCount = useMemo(() => {
    return nodes.filter(n => n.id !== 'ROOT').length;
  }, [nodes]);

  // Build a set of node IDs that have sevenChrDef children
  // Used to filter out "children batch" for nodes where sevenChrDef IS the children batch
  const nodesWithSevenChrDefChildren = useMemo(() => {
    const set = new Set<string>();
    for (const edge of edges) {
      if (edge.edge_type === 'lateral' && edge.rule === 'sevenChrDef') {
        set.add(String(edge.source));
      }
    }
    return set;
  }, [edges]);

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
    if (!svgRef.current || !zoomRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    // Use graph-svg-area dimensions (SVG's parent) which excludes header bar,
    // legend, and codes bar that are siblings in the flex container
    const svgArea = svgRef.current.parentElement;
    if (!svgArea) return;
    const rect = svgArea.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Get current graph bounds from the main group
    const g = svg.select('g.main-group');
    if (g.empty()) return;

    const gNode = g.node() as SVGGElement;
    const bbox = gNode.getBBox();

    if (bbox.width === 0 || bbox.height === 0) return;

    // Calculate scale to fit both dimensions with padding
    const padding = 40;
    const availableWidth = width - padding * 2;
    const availableHeight = height - padding * 2;

    const scaleX = availableWidth / bbox.width;
    const scaleY = availableHeight / bbox.height;
    // Use the smaller scale to ensure graph fits both horizontally and vertically
    // Cap at 1.0 to avoid zooming in beyond 100%
    const scale = Math.min(scaleX, scaleY, 1.0);

    // Calculate scaled dimensions
    const scaledWidth = bbox.width * scale;
    const scaledHeight = bbox.height * scale;

    // Center horizontally and vertically
    const translateX = (width - scaledWidth) / 2 - bbox.x * scale;
    const translateY = (height - scaledHeight) / 2 - bbox.y * scale;

    svg.transition().duration(300).call(
      zoomRef.current.transform,
      d3.zoomIdentity.translate(translateX, translateY).scale(scale)
    );
  }, [nodes.length]);

  // Keep handleFitToWindow ref in sync so the fit trigger effect always uses the latest version
  useEffect(() => {
    handleFitToWindowRef.current = handleFitToWindow;
  }, [handleFitToWindow]);

  const handleExportSvg = useCallback(() => {
    if (!svgRef.current) return;
    const prefix = benchmarkMode ? 'graph-benchmark' : isTraversing ? 'graph-traverse' : 'graph-visualize';
    const filename = generateSvgFilename(prefix);
    exportSvgToFile(svgRef.current, filename);
  }, [benchmarkMode, isTraversing]);

  // Trigger fit-to-window when prop changes (with delay for layout to settle)
  // Only run when triggerFitToWindow actually increments, not when handleFitToWindow changes
  // Uses handleFitToWindowRef to avoid effect re-runs when nodes change during traversal
  useEffect(() => {
    if (triggerFitToWindow && triggerFitToWindow > prevFitTriggerRef.current) {
      prevFitTriggerRef.current = triggerFitToWindow;
      const timer = setTimeout(() => {
        handleFitToWindowRef.current();
      }, 350); // Delay to allow layout to settle
      return () => clearTimeout(timer);
    }
  }, [triggerFitToWindow]);

  // NOTE: Automatic fit-to-window (ResizeObserver, periodic interval) disabled.
  // The initial render positions content correctly; auto-fit was causing issues.
  // User can manually click the fit-to-window button (⤢) when needed.

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) {
      return;
    }

    const svg = d3.select(svgRef.current);
    // Use graph-svg-area (SVG's parent) for dimensions - this excludes
    // header bar, legend, and codes bar that are siblings in the flex container
    const svgArea = svgRef.current.parentElement;
    if (!svgArea) return;
    const rect = svgArea.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Calculate scale factor based on svg area height (clamped to 0.7-1.5x)
    const scaleFactor = Math.max(0.7, Math.min(1.5, height / REFERENCE_HEIGHT));
    const NODE_WIDTH = BASE_NODE_WIDTH * scaleFactor;
    const NODE_HEIGHT = BASE_NODE_HEIGHT * scaleFactor;
    const LEVEL_HEIGHT = BASE_LEVEL_HEIGHT * scaleFactor;

    // Clear everything and recreate from scratch each render
    // NOTE: SVG uses width="100%" height="100%" from JSX - let CSS handle sizing
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('class', 'main-group');

    // Always attach click handler for empty space (even when graph is empty)
    // This allows sidebar collapse to work on blank/reset graphs
    svg.on('click', (event: MouseEvent) => {
      const target = event.target as Element;
      const isInsideOverlay = target.closest('.expanded-overlay') !== null ||
        target.closest('.expanded-node') !== null ||
        target.closest('.batch-panel') !== null ||
        target.tagName.toLowerCase() === 'textarea' ||
        target.closest('foreignObject') !== null;
      if (isInsideOverlay) return;

      // If overlay is open, just close it (don't collapse sidebar)
      if (pinnedNodeIdRef.current) {
        setPinnedNodeId(null);
        hideExpandedNode();
        return;
      }

      // Only collapse sidebar when no overlay was open
      onEmptySpaceClickRef.current?.();
    });

    // Only render content when traversing or has nodes
    if (nodes.length === 0 && !isTraversing) {
      return;
    }

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

    // Ensure ROOT is in nodeMap for positioning (may be missing during streaming)
    if (!nodeMap.has('ROOT')) {
      nodeMap.set('ROOT', {
        id: 'ROOT',
        code: 'ROOT',
        label: 'ICD-10-CM',
        depth: 0,
        category: 'root' as const,
        billable: false,
      });
    }

    // Build allChildren map for subtree width calculation (includes all edges)
    const allChildren = new Map<string, string[]>();
    edges.forEach(e => {
      const sourceId = String(e.source);
      const targetId = String(e.target);
      if (!allChildren.has(sourceId)) allChildren.set(sourceId, []);
      allChildren.get(sourceId)!.push(targetId);
    });

    // Create finalized codes set for position calculation
    const finalizedCodesSet = new Set(finalizedCodes);

    // =========================================================================
    // ITERATIVE EXPANSION ALGORITHM
    // Correctly classifies nodes as hierarchy-connected vs lateral-only by
    // processing nodes in BFS order from ROOT.
    // =========================================================================
    const rendered = new Set<string>(['ROOT']);
    const hierarchyChildren = new Map<string, string[]>();
    const lateralOnlyNodes = new Set<string>();
    const lateralOnlySources = new Map<string, string>(); // target → lateral source (for positioning)

    // Helper to check if a hierarchy edge is valid (depth ordering)
    const isValidHierarchyEdge = (sourceId: string, targetId: string): boolean => {
      const sourceNode = nodeMap.get(sourceId);
      const targetNode = nodeMap.get(targetId);

      // ROOT → any non-ROOT node is valid
      if (sourceId === 'ROOT' && targetNode) {
        return (targetNode.depth || 0) > 0;
      }

      if (!sourceNode || !targetNode) return false;

      const sourceDepth = sourceNode.depth || 0;
      const targetDepth = targetNode.depth || 0;

      // Valid if target is deeper, or same depth with ID ordering (breaks cycles)
      return targetDepth > sourceDepth || (targetDepth === sourceDepth && sourceId < targetId);
    };

    // Iterative expansion: process nodes in waves until stable
    // IMPORTANT: Handles streaming where edges arrive incrementally in DFS order.
    // A node initially classified as lateral-only may later get a hierarchy path.
    let changed = true;
    while (changed) {
      changed = false;

      // Phase 1: Add hierarchy children of rendered nodes
      // Also handles RECLASSIFICATION for streaming: if a hierarchy edge arrives
      // for a node already classified as lateral-only, demote it and reclassify.
      for (const e of edges) {
        if (e.edge_type !== 'hierarchy') continue;
        const sourceId = String(e.source);
        const targetId = String(e.target);

        if (rendered.has(sourceId) && !rendered.has(targetId)) {
          // Normal case: add new hierarchy child
          if (isValidHierarchyEdge(sourceId, targetId)) {
            if (!hierarchyChildren.has(sourceId)) hierarchyChildren.set(sourceId, []);
            if (!hierarchyChildren.get(sourceId)!.includes(targetId)) {
              hierarchyChildren.get(sourceId)!.push(targetId);
            }
            rendered.add(targetId);
            changed = true;
          }
        } else if (rendered.has(sourceId) && lateralOnlyNodes.has(targetId)) {
          // RECLASSIFICATION: Node was lateral-only but now has a hierarchy path.
          // This happens during streaming when hierarchy edges arrive after lateral edges.
          if (isValidHierarchyEdge(sourceId, targetId)) {
            // Remove from lateral-only classification
            lateralOnlyNodes.delete(targetId);
            lateralOnlySources.delete(targetId);
            // Remove from old parent's children list
            for (const [, children] of hierarchyChildren.entries()) {
              const idx = children.indexOf(targetId);
              if (idx !== -1) {
                children.splice(idx, 1);
                break;
              }
            }
            // Add to new hierarchy parent
            if (!hierarchyChildren.has(sourceId)) hierarchyChildren.set(sourceId, []);
            if (!hierarchyChildren.get(sourceId)!.includes(targetId)) {
              hierarchyChildren.get(sourceId)!.push(targetId);
            }
            changed = true;
          }
        }
      }

      // Phase 2: Add lateral-only targets (source rendered, target not reachable via hierarchy)
      for (const e of edges) {
        if (e.edge_type !== 'lateral' || e.rule === 'sevenChrDef') continue;
        const sourceId = String(e.source);
        const targetId = String(e.target);

        if (rendered.has(sourceId) && !rendered.has(targetId)) {
          // This node can only be reached via lateral edge - mark as lateral-only
          lateralOnlyNodes.add(targetId);
          lateralOnlySources.set(targetId, sourceId);

          // Add to hierarchyChildren for positioning (will be positioned relative to source)
          // The positioning algorithm uses this to place it to the right of the source
          if (!hierarchyChildren.has(sourceId)) hierarchyChildren.set(sourceId, []);
          if (!hierarchyChildren.get(sourceId)!.includes(targetId)) {
            hierarchyChildren.get(sourceId)!.push(targetId);
          }
          rendered.add(targetId);
          changed = true;
        }
      }
    }

    // For backwards compatibility with positioning code, create orphanRescuedNodes alias
    const orphanRescuedNodes = lateralOnlyNodes;

    // Filter edges by type for rendering
    // - ROOT edges: solid straight lines
    // - Hierarchy edges: solid straight lines (but NOT to lateral-only nodes)
    // - sevenChrDef edges: dashed straight lines
    // - Other lateral edges: dashed curved lines
    const rootEdges = edges.filter(e => e.edge_type === 'hierarchy' && String(e.source) === 'ROOT');
    // Exclude edges TO lateral-only nodes - they only have lateral connections
    const hierarchyEdges = edges.filter(e =>
      e.edge_type === 'hierarchy' &&
      String(e.source) !== 'ROOT' &&
      !lateralOnlyNodes.has(String(e.target))
    );
    const sevenChrDefEdges = edges.filter(e => e.edge_type === 'lateral' && e.rule === 'sevenChrDef');
    const otherLateralEdges = edges.filter(e => e.edge_type === 'lateral' && e.rule !== 'sevenChrDef');

    // Build map from sevenChrDef target (activator) -> source (parent) for tooltip
    const sevenChrDefParentMap = new Map<string, string>();
    sevenChrDefEdges.forEach(e => {
      sevenChrDefParentMap.set(String(e.target), String(e.source));
    });

    // Build hierarchy parent map for O(1) ancestor lookup in overlay renderer
    // Maps child node ID -> parent node ID
    const hierarchyParentMap = new Map<string, string>();
    hierarchyEdges.forEach(e => {
      hierarchyParentMap.set(String(e.target), String(e.source));
    });

    // Build lateral edge map for O(1) edge lookup in overlay renderer
    // Maps "source|target" -> edge
    const lateralEdgeMap = new Map<string, GraphEdge>();
    [...sevenChrDefEdges, ...otherLateralEdges].forEach(e => {
      lateralEdgeMap.set(`${String(e.source)}|${String(e.target)}`, e);
    });

    // Helper to get the display code for a node
    // For sevenChrDef targets, combines parent code + 7th character (e.g., "T36.1X5" + "A" = "T36.1X5A")
    const getDisplayCode = (node: GraphNode): string => {
      // If this is a depth-7 finalized node, the code is already complete
      // Backend now sends full code (e.g., "T36.1X5A") in both id and code fields
      if (node.depth === 7) {
        return node.code;
      }

      // For older-style nodes where code might be just the 7th char
      const parentId = sevenChrDefParentMap.get(node.id);
      if (parentId) {
        const parentNode = nodeMap.get(parentId);
        if (parentNode) {
          // Combine parent code with 7th character
          return parentNode.code + node.code;
        }
      }
      return node.code;
    };

    // Calculate positions using subtree width algorithm (from reference)
    const positions = calculatePositions(nodes, hierarchyChildren, allChildren, width, nodeMap, edges.filter(e => e.edge_type === 'lateral'), NODE_WIDTH, NODE_HEIGHT, LEVEL_HEIGHT, finalizedCodesSet, orphanRescuedNodes);

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
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .attr('font-family', 'ui-monospace, monospace')
          .attr('fill', '#1e293b')
          .text('ROOT');

        // Make ROOT node look interactive (event handlers added later after helpers defined)
        rootGroup.style('cursor', 'pointer');
      }
    }

    // Create expanded overlay group (rendered on top of everything)
    const expandedOverlayGroup = g.append('g').attr('class', 'expanded-overlay');

    // Helper to hide expanded overlay
    const hideExpandedNode = () => {
      expandedOverlayGroup.selectAll('*').remove();
      activeOverlayNodeRef.current = null;
    };

    // Helper to cancel any pending show timeout
    const cancelShowTimeout = () => {
      if (showTimeoutRef.current) {
        clearTimeout(showTimeoutRef.current);
        showTimeoutRef.current = null;
      }
    };

    // Helper to cancel any pending hide timeout
    const cancelHideTimeout = () => {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
        hideTimeoutRef.current = null;
      }
    };

    // Create getter for current zoom transform (dynamic, not stale)
    const svgNode = svg.node()!;
    const getTransform = () => d3.zoomTransform(svgNode);

    // Build overlay context for the extracted renderer
    const overlayContext: OverlayContext = {
      overlayGroup: expandedOverlayGroup,
      positions,
      nodeMap,
      decisions,
      finalizedCodesSet,
      nodesWithSevenChrDefChildren,
      sevenChrDefParentMap,
      hierarchyParentMap,
      lateralEdgeMap,
      expectedLeaves,
      benchmarkMode,
      allowRewind,
      pinnedNodeIdRef,
      hideTimeoutRef,
      activeOverlayNodeRef,
      lastInteractionTime,
      onNodeRewindClick: onNodeRewindClickRef.current,
      hideExpandedNode,
      cancelHideTimeout,
      allChildren,
      sevenChrDefEdges,
      otherLateralEdges,
      viewportBounds: { width, height, topMargin: 0, bottomMargin: 0 },
      getTransform,
    };

    // Helper function to show expanded node overlay (uses extracted renderer)
    const showExpandedNode = (d: GraphNode) => {
      showNodeOverlay(d, overlayContext, cancelShowTimeout);
    };

    // Add ROOT node event handlers (now that helper functions are defined)
    const rootNode = nodeMap.get('ROOT');
    if (rootNode) {
      nodesGroup.select<SVGGElement>('.node-root')
        .on('click', function (event: MouseEvent) {
          event.stopPropagation();
          setPinnedNodeId('ROOT');
          pinnedNodeIdRef.current = 'ROOT';
          showExpandedNode(rootNode);
          onNodeClick('ROOT');
        })
        .on('mouseenter', function () {
          if (activeOverlayNodeRef.current === 'ROOT') {
            cancelHideTimeout();
            return;
          }
          cancelShowTimeout();
          cancelHideTimeout();
          if (pinnedNodeIdRef.current === 'ROOT') {
            showExpandedNode(rootNode);
            return;
          }
          showTimeoutRef.current = setTimeout(() => {
            showExpandedNode(rootNode);
          }, 1000);
        })
        .on('mouseleave', () => {
          cancelShowTimeout();
          if (!pinnedNodeIdRef.current) {
            hideTimeoutRef.current = setTimeout(() => {
              hideExpandedNode();
            }, 500);
          }
        });
    }

    // Render non-ROOT nodes with simple join pattern
    const nonRootNodes = nodes.filter(n => n.id !== 'ROOT');

    const nodeGroups = nodesGroup.selectAll<SVGGElement, GraphNode>('.node:not(.node-root)')
      .data(nonRootNodes, d => d.id)
      .join('g')
      .attr('class', d => `node node-${d.category}`)
      .attr('transform', d => {
        const pos = positions.get(d.id);
        return pos ? `translate(${pos.x - NODE_WIDTH / 2}, ${pos.y - NODE_HEIGHT / 2})` : '';
      })
      .style('cursor', 'pointer');

    // Add node rectangle
    nodeGroups.append('rect')
      .attr('class', d => d.id === rewindingNodeId ? 'node-rect rewinding' : 'node-rect')
      .attr('width', NODE_WIDTH)
      .attr('height', NODE_HEIGHT)
      .attr('rx', 4)
      .attr('ry', 4)
      .attr('fill', d => benchmarkMode ? getBenchmarkNodeFill(d as BenchmarkGraphNode, streamingTraversedIds) : getNodeFill(d, finalizedCodesSet))
      .attr('stroke', d => {
        if (d.id === rewindingNodeId) return '#7c3aed';
        if (d.id === selectedNode) return '#0f172a';
        return benchmarkMode ? getBenchmarkNodeStroke(d as BenchmarkGraphNode, streamingTraversedIds) : getNodeStroke(d, finalizedCodesSet);
      })
      .attr('stroke-width', d => {
        if (d.id === rewindingNodeId) return 3;
        if (d.id === selectedNode) return 2.5;
        return benchmarkMode ? getBenchmarkNodeStrokeWidth(d as BenchmarkGraphNode, streamingTraversedIds) : getNodeStrokeWidth(d, finalizedCodesSet);
      })
      .attr('stroke-dasharray', d => {
        if (benchmarkMode) {
          return getBenchmarkNodeStrokeDasharray(d as BenchmarkGraphNode, streamingTraversedIds);
        }
        if (isSevenChrDefFinalizedNode(d, finalizedCodesSet)) return null;
        if (isFinalizedNode(d, finalizedCodesSet)) return null;
        return d.category === 'placeholder' ? '4,2' : null;
      });

    // Add node code text (y scales with node height: 18/60 = 0.3)
    nodeGroups.append('text')
      .attr('class', 'node-code')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', NODE_HEIGHT * 0.30)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('font-family', 'ui-monospace, monospace')
      .attr('fill', '#1e293b')
      .text(d => getDisplayCode(d));

    // Add billable indicator (y scales with node height: 16/60 = 0.267)
    nodeGroups.append('text')
      .attr('class', 'node-billable')
      .attr('x', NODE_WIDTH - 6)
      .attr('y', NODE_HEIGHT * 0.267)
      .attr('text-anchor', 'end')
      .attr('font-size', 12)
      .attr('font-weight', 700)
      .attr('fill', '#16a34a')
      .text(d => d.billable ? '$' : '');

    // Add label line 1 (y scales with node height: 32/60 = 0.533)
    nodeGroups.append('text')
      .attr('class', 'node-label node-label-1')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', NODE_HEIGHT * 0.533)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', '#64748b')
      .text(d => wrapNodeLabel(d.label, 22)[0]);

    // Add label line 2 (y scales with node height: 44/60 = 0.733)
    nodeGroups.append('text')
      .attr('class', 'node-label node-label-2')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', NODE_HEIGHT * 0.733)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', '#64748b')
      .text(d => wrapNodeLabel(d.label, 22)[1]);

    // Add event handlers
    nodeGroups
      .on('click', function (event: MouseEvent, d) {
        event.stopPropagation();
        setPinnedNodeId(d.id);
        pinnedNodeIdRef.current = d.id;
        showExpandedNode(d);
        onNodeClick(d.id);
      })
      .on('mouseenter', function (_, d) {
        if (activeOverlayNodeRef.current === d.id) {
          cancelHideTimeout();
          return;
        }
        cancelShowTimeout();
        cancelHideTimeout();
        if (pinnedNodeIdRef.current === d.id) {
          showExpandedNode(d);
          return;
        }
        showTimeoutRef.current = setTimeout(() => {
          showExpandedNode(d);
        }, 1000);
      })
      .on('mouseleave', () => {
        cancelShowTimeout();
        if (!pinnedNodeIdRef.current) {
          hideTimeoutRef.current = setTimeout(() => {
            hideExpandedNode();
          }, 500);
        }
      });

    // Add expected leaf flags in benchmark mode
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

    // Add simple throbber below rewinding node
    if (rewindingNodeId) {
      const throbberGroup = nodeGroups.filter(d => d.id === rewindingNodeId)
        .append('g')
        .attr('class', 'rewind-throbber')
        .attr('transform', `translate(${NODE_WIDTH / 2}, ${NODE_HEIGHT + 12})`);

      // Three dots throbber with staggered animation classes
      [-12, 0, 12].forEach((x, i) => {
        throbberGroup.append('circle')
          .attr('cx', x)
          .attr('cy', 0)
          .attr('r', 3)
          .attr('fill', '#7c3aed')
          .attr('class', `throbber-dot throbber-dot-${i + 1}`);
      });
    }

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

        // Find the actual edge to determine if it uses curved path
        // Lateral edges (except sevenChrDef) use curved paths
        const edge = edges.find(e =>
          String(e.source) === marker.edgeSource &&
          String(e.target) === marker.edgeTarget
        );
        const isCurvedEdge = edge?.edge_type === 'lateral' && edge?.rule !== 'sevenChrDef';

        // Get point along the actual edge path (near arrowhead)
        // Use different t values: curved edges can be closer, straight edges need more clearance
        const markerPos = isCurvedEdge
          ? getPointOnCurvedEdge(srcPos, tgtPos, NODE_HEIGHT, 0.92)
          : getPointOnStraightEdge(srcPos, tgtPos, NODE_HEIGHT, 0.85);

        missedGroup.append('path')
          .attr('d', `M${markerPos.x - 6},${markerPos.y - 6} L${markerPos.x + 6},${markerPos.y + 6} M${markerPos.x + 6},${markerPos.y - 6} L${markerPos.x - 6},${markerPos.y + 6}`)
          .attr('stroke', '#dc2626')
          .attr('stroke-width', 2.5)
          .attr('fill', 'none')
          .attr('stroke-linecap', 'round');
      });
    }

    // Raise expanded overlay group to top so it renders above markers
    expandedOverlayGroup.raise();

    // Set up zoom with interaction tracking
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .filter((event) => {
        // Block zoom for events originating from expanded overlay elements
        // This allows buttons and textareas in overlays to receive click events
        const target = event.target as Element;
        if (target && target.closest('.expanded-overlay')) {
          return false; // Don't zoom - let the overlay handle the event
        }
        return true; // Allow zoom for all other events
      })
      .on('start', (event) => {
        // Only track ACTUAL user interactions, not programmatic transforms
        // event.sourceEvent is null for programmatic calls, defined for user interactions
        if (event.sourceEvent) {
          lastInteractionTime.current = Date.now();
          onGraphInteractionRef.current?.();
        }
      })
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        // Only track ACTUAL user interactions
        if (event.sourceEvent) {
          lastInteractionTime.current = Date.now();
          onGraphInteractionRef.current?.();
        }
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Restore overlay for pinned node if it exists (handles redraws)
    if (pinnedNodeIdRef.current) {
      const pinnedNode = nodes.find(n => n.id === pinnedNodeIdRef.current);
      if (pinnedNode) {
        // Pass null for nodeGroup as it's not actually used for positioning (positions map is used)
        showExpandedNode(pinnedNode);
      }
    }

    // IMPORTANT: Apply the current transform to the new <g> element
    // This preserves the user's pan/zoom position when the graph re-renders
    const savedTransform = d3.zoomTransform(svg.node()!);
    g.attr('transform', savedTransform.toString());

    // NOTE: Automatic fit-to-window removed - D3 render positions content correctly.
    // User can manually click the fit-to-window button (⤢) when needed.

    // Cleanup: cancel any pending show timeout
    return () => {
      if (showTimeoutRef.current) {
        clearTimeout(showTimeoutRef.current);
      }
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    };

  }, [nodes, edges, selectedNode, onNodeClick, finalizedCodes, isTraversing, benchmarkMode, overshootMarkers, missedEdgeMarkers, expectedLeaves, showXMarkers, rewindingNodeId]);

  // Separate useEffect for streaming traversal style updates (avoids full re-render)
  // This only runs when streamingTraversedIds changes, updating node styles efficiently
  // Uses requestAnimationFrame to batch DOM updates and sync with browser paint cycle
  useEffect(() => {
    if (!svgRef.current || !benchmarkMode || !streamingTraversedIds) return;

    const svg = d3.select(svgRef.current);

    // Helper to get node data from a rect's parent group
    const getNodeData = (rect: SVGRectElement): BenchmarkGraphNode | null => {
      const nodeGroup = rect.parentElement as unknown as SVGGElement | null;
      if (!nodeGroup) return null;
      return d3.select<SVGGElement, BenchmarkGraphNode>(nodeGroup).datum();
    };

    // Wrap in RAF to batch DOM updates and sync with browser paint cycle
    // This prevents layout thrashing when updates arrive rapidly
    const rafId = requestAnimationFrame(() => {
      // Update only the node rectangles' visual properties without recreating them
      svg.selectAll<SVGRectElement, BenchmarkGraphNode>('.node-rect')
        .attr('fill', function() {
          const nodeData = getNodeData(this);
          if (!nodeData) return '#ffffff';
          return getBenchmarkNodeFill(nodeData, streamingTraversedIds);
        })
        .attr('stroke', function() {
          const nodeData = getNodeData(this);
          if (!nodeData) return '#1e293b';
          return getBenchmarkNodeStroke(nodeData, streamingTraversedIds);
        })
        .attr('stroke-width', function() {
          const nodeData = getNodeData(this);
          if (!nodeData) return 1.5;
          return getBenchmarkNodeStrokeWidth(nodeData, streamingTraversedIds);
        })
        .attr('stroke-dasharray', function() {
          const nodeData = getNodeData(this);
          if (!nodeData) return null;
          return getBenchmarkNodeStrokeDasharray(nodeData, streamingTraversedIds);
        });
    });

    // Cleanup: cancel RAF if component unmounts or deps change before it executes
    return () => cancelAnimationFrame(rafId);
  }, [streamingTraversedIds, benchmarkMode]);

  return (
    <div className="graph-container" ref={containerRef}>
      <div className="view-header-bar">
        <div className="view-header-info">
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
            {status !== 'idle' && elapsedTime !== null && (
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
          <>
            <div className="report-line">
              <span className="report-label">Benchmark:</span>
              <span className="benchmark-scores">
                <span className="benchmark-score">
                  Traversal Recall: <strong>{(benchmarkMetrics.traversalRecall * 100).toFixed(1)}%</strong>
                  {' '}({benchmarkMetrics.expectedNodesTraversed}/{benchmarkMetrics.expectedNodesCount})
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-score">
                  Final Codes Recall: <strong>{(benchmarkMetrics.finalCodesRecall * 100).toFixed(1)}%</strong>
                  {' '}({benchmarkMetrics.exactCount}/{benchmarkMetrics.expectedCount})
                </span>
              </span>
            </div>
            <div className="report-line">
              <span className="report-label">Alignment:</span>
              <span className="benchmark-outcome-counts">
                <span className="benchmark-outcome-prefix">Final Code(s):</span>
                <span className="benchmark-stat exact">
                  {benchmarkMetrics.exactCount} <span className="outcome-label">matched</span>
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-stat missed">
                  {benchmarkMetrics.missedCount} <span className="outcome-label">missed</span>
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-stat undershoot">
                  {benchmarkMetrics.undershootCount} <span className="outcome-label">undershot</span>
                </span>
                <span className="stat-separator">·</span>
                <span className="benchmark-stat overshoot">
                  {benchmarkMetrics.overshootCount} <span className="outcome-label">overshot</span>
                </span>
              </span>
            </div>
          </>
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
              ) : (() => {
                const traversedCount = status === 'traversing' && streamingTraversedIds
                  ? streamingTraversedIds.size
                  : (nodes as BenchmarkGraphNode[]).filter(
                      n => n.benchmarkStatus === 'traversed' || n.benchmarkStatus === 'matched'
                    ).length;
                const expectedNodeIds = new Set(nodes.filter(n => n.id !== 'ROOT').map(n => n.id));
                const expectedNodesTraversed = status === 'traversing' && streamingTraversedIds
                  ? [...expectedNodeIds].filter(id => streamingTraversedIds.has(id)).length
                  : (nodes as BenchmarkGraphNode[]).filter(
                      n => n.benchmarkStatus === 'traversed' || n.benchmarkStatus === 'matched'
                    ).length;
                const extraCount = traversedCount - expectedNodesTraversed;
                return (
                  <>
                    <strong>{traversedCount}</strong> <strong style={{ color: 'var(--text)' }}>explored</strong> nodes
                    {' '}(<strong style={{ color: '#16a34a' }}>aligned</strong>{' '}
                    <strong>{expectedNodesTraversed}</strong>/<strong>{expectedNodeIds.size}</strong>{' '}
                    to target nodes,
                    {' '}<strong>{extraCount}</strong>{' '}
                    <strong style={{ color: '#6b7280' }}>extra</strong> nodes)
                  </>
                );
              })()}
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
        <div className="view-header-actions">
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
            <span className="empty-text">{benchmarkMode ? 'No benchmark data yet' : 'No traversal data yet'}</span>
            <span className="empty-hint">{benchmarkMode ? 'Add expected codes and run a benchmark' : 'Enter a clinical note and start traversal'}</span>
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
              <span className="legend-box" style={{ background: '#fecaca', border: '2px solid #dc2626' }} />
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
            {sortedFinalizedCodes.map(code => {
              const outcomeClass = outcomeStatusMap.get(code) ?? '';
              const tooltip = codeLabelMap.get(code) ?? '';
              const outcomeLabel = outcomeClass === 'exact' ? '\t\t\u00A0\u00A0\u00A0\u00A0\u00A0Alignment:\u00A0MATCHED'
                : outcomeClass === 'undershoot' ? '\t\t\u00A0\u00A0\u00A0\u00A0\u00A0Alignment:\u00A0UNDERSHOT'
                : outcomeClass === 'overshoot' ? '\t\t\u00A0\u00A0\u00A0\u00A0\u00A0Alignment:\u00A0OVERSHOT'
                : outcomeClass === 'missed' ? '\t\t\u00A0\u00A0\u00A0\u00A0\u00A0Alignment:\u00A0MISSED'
                : '';
              const paddedCode = code.padEnd(8, '\u00A0');
              const codeLine = tooltip ? `${paddedCode}\t${tooltip}` : code;
              const fullTooltip = outcomeLabel
                ? `${codeLine}\n${outcomeLabel}`
                : codeLine;
              return benchmarkMode && onRemoveExpectedCode ? (
                <span
                  key={code}
                  className={`code-badge removable${invalidCodes.has(code) ? ' invalid' : ''}${outcomeClass ? ` outcome-${outcomeClass}` : ''}`}
                  title={fullTooltip}
                  onClick={() => onRemoveExpectedCode(code)}
                >
                  {code}
                  <span className="remove-icon">×</span>
                </span>
              ) : (
                <span key={code} className={`code-badge${invalidCodes.has(code) ? ' invalid' : ''}${outcomeClass ? ` outcome-${outcomeClass}` : ''}`} title={fullTooltip}>{code}</span>
              );
            })}
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

// Custom comparison function for memo - prevents unnecessary re-renders from parent
// The key optimization: streamingTraversedIds changes are ALLOWED through because:
// 1. They're already throttled to 10/sec in App.tsx
// 2. The main D3 useEffect doesn't depend on streamingTraversedIds (won't rebuild graph)
// 3. Only the lightweight style-update useEffect runs (just updates fill/stroke)
function arePropsEqual(prev: GraphViewerProps, next: GraphViewerProps): boolean {
  // Core data changes - always re-render
  if (prev.nodes !== next.nodes) return false;
  if (prev.edges !== next.edges) return false;
  if (prev.status !== next.status) return false;

  // streamingTraversedIds - ALLOW through (throttled in App.tsx, lightweight update)
  if (prev.streamingTraversedIds !== next.streamingTraversedIds) return false;

  // Other props that affect rendering
  if (prev.selectedNode !== next.selectedNode) return false;
  if (prev.rewindingNodeId !== next.rewindingNodeId) return false;
  if (prev.triggerFitToWindow !== next.triggerFitToWindow) return false;
  if (prev.finalizedCodes !== next.finalizedCodes) return false;
  if (prev.isTraversing !== next.isTraversing) return false;
  if (prev.benchmarkMode !== next.benchmarkMode) return false;
  if (prev.benchmarkMetrics !== next.benchmarkMetrics) return false;
  if (prev.overshootMarkers !== next.overshootMarkers) return false;
  if (prev.missedEdgeMarkers !== next.missedEdgeMarkers) return false;
  if (prev.showXMarkers !== next.showXMarkers) return false;
  if (prev.allowRewind !== next.allowRewind) return false;
  if (prev.decisionCount !== next.decisionCount) return false;
  if (prev.elapsedTime !== next.elapsedTime) return false;

  return true;
}

export const GraphViewer = memo(GraphViewerInner, arePropsEqual);
