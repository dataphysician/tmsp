import { useEffect, useRef, useMemo, useState, useCallback, memo } from 'react';
import * as d3 from 'd3';
import type { GraphNode, GraphEdge, TraversalStatus, BenchmarkGraphNode, BenchmarkMetrics, OvershootMarker, EdgeMissMarker, DecisionPoint } from '../lib/types';
import { exportSvgToFile, generateSvgFilename } from '../lib/exportSvg';
import { wrapNodeLabel, formatElapsedTime } from '../lib/textUtils';
import { calculatePositions } from '../lib/graphPositioning';
import { createEdgePath, createCurvedEdgePath } from '../lib/edgePaths';
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

  // Keep refs in sync with props/state for D3 event handlers
  useEffect(() => {
    pinnedNodeIdRef.current = pinnedNodeId;
  }, [pinnedNodeId]);

  useEffect(() => {
    onNodeRewindClickRef.current = onNodeRewindClick;
  }, [onNodeRewindClick]);

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

    // Build TWO children maps:
    // 1. hierarchyChildren: Only hierarchy edges - used for tree positioning in Phase 2
    // 2. allChildren: All edges - used for subtree width calculation and positioning children of lateral nodes
    const hierarchyChildren = new Map<string, string[]>();
    const allChildren = new Map<string, string[]>();

    // Create finalized codes set for position calculation
    const finalizedCodesSet = new Set(finalizedCodes);

    // Track which nodes have a hierarchy parent (for orphan detection)
    const hasHierarchyParent = new Set<string>();

    // First pass: build maps from hierarchy edges
    // IMPORTANT: For ROOT edges, we relax the sourceNode check since ROOT may not be
    // in the nodes array during streaming (it's added via STATE_SNAPSHOT but may not
    // have propagated yet when edges arrive via STATE_DELTA)
    edges.forEach(e => {
      const sourceId = String(e.source);
      const targetId = String(e.target);

      // All edges go into allChildren (for width calculation)
      if (!allChildren.has(sourceId)) allChildren.set(sourceId, []);
      allChildren.get(sourceId)!.push(targetId);

      // Only hierarchy edges go into hierarchyChildren
      // We enforce a strict depth check (source.depth < target.depth) to break cycles needed for the tree layout.
      // This ensures that "back-edges" or "cross-edges" don't confuse the layout algorithm.
      const isHierarchy = e.edge_type === 'hierarchy';
      const sourceNode = nodeMap.get(sourceId);
      const targetNode = nodeMap.get(targetId);

      // Special handling for ROOT edges: ROOT has depth 0, so any chapter (depth 1+) is deeper
      // We don't require sourceNode to exist for ROOT since it may not be in nodeMap during streaming
      if (isHierarchy && sourceId === 'ROOT' && targetNode) {
        const targetDepth = targetNode.depth || 0;
        if (targetDepth > 0) {  // Any non-ROOT node is a valid child of ROOT
          if (!hierarchyChildren.has('ROOT')) hierarchyChildren.set('ROOT', []);
          hierarchyChildren.get('ROOT')!.push(targetId);
          hasHierarchyParent.add(targetId);
        }
      } else if (isHierarchy && sourceNode && targetNode) {
        // Only treat as structural hierarchy if moving "down" the tree (deeper)
        // If depth is missing/zero (e.g. ROOT), treat it as lower depth
        const sourceDepth = sourceNode.depth || 0;
        const targetDepth = targetNode.depth || 0;

        // Break cycles:
        // 1. Strictly favor deeper targets (parent -> child)
        // 2. If same depth (sibling/circular dependency), use ID tie-breaker to pick ONE direction
        //    This ensures we don't remove BOTH edges in a tight cycle (A<->B), which would orphan one node.
        const isDeeper = targetDepth > sourceDepth;
        const isSameDepthAndOrdered = targetDepth === sourceDepth && sourceId < targetId;

        if (isDeeper || isSameDepthAndOrdered) {
          if (!hierarchyChildren.has(sourceId)) hierarchyChildren.set(sourceId, []);
          hierarchyChildren.get(sourceId)!.push(targetId);
          hasHierarchyParent.add(targetId);
        }
      }
    });

    // Track nodes that are "orphan rescued" via lateral edges
    // These should be positioned as hierarchy children, not as lateral targets
    const orphanRescuedNodes = new Set<string>();

    // Second pass: include lateral edges that connect orphaned subtrees
    // These are lateral edges (codeFirst, codeAlso, useAdditionalCode) where the target
    // has no hierarchy parent and would otherwise be disconnected from the tree
    edges.forEach(e => {
      const sourceId = String(e.source);
      const targetId = String(e.target);

      // Only process lateral edges (not sevenChrDef which is handled separately)
      const isLateral = e.edge_type === 'lateral' && e.rule !== 'sevenChrDef';
      if (!isLateral) return;

      // If target already has a hierarchy parent, skip (it's already positioned)
      if (hasHierarchyParent.has(targetId)) return;

      // For orphaned subtrees, always add the lateral edge regardless of depth ordering
      // This rescues subtrees that are connected via lateral links (codeFirst, codeAlso, etc.)
      // We don't use alphabetical ordering here because the whole point is to rescue orphans
      if (!hierarchyChildren.has(sourceId)) hierarchyChildren.set(sourceId, []);
      if (!hierarchyChildren.get(sourceId)!.includes(targetId)) {
        hierarchyChildren.get(sourceId)!.push(targetId);
        hasHierarchyParent.add(targetId);
        orphanRescuedNodes.add(targetId);  // Track for hierarchyParent mapping
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
          .attr('font-size', 13)
          .attr('font-weight', 600)
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
    };

    // Helper function to show expanded node overlay (uses extracted renderer)
    const showExpandedNode = (d: GraphNode, _nodeGroup: SVGGElement) => {
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
          showExpandedNode(rootNode, this as SVGGElement);
          onNodeClick('ROOT');
        })
        .on('mouseenter', function () {
          if (activeOverlayNodeRef.current === 'ROOT') {
            cancelHideTimeout();
            return;
          }
          cancelShowTimeout();
          cancelHideTimeout();
          const nodeGroup = this as SVGGElement;
          if (pinnedNodeIdRef.current === 'ROOT') {
            showExpandedNode(rootNode, nodeGroup);
            return;
          }
          showTimeoutRef.current = setTimeout(() => {
            showExpandedNode(rootNode, nodeGroup);
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

    // Add node code text
    nodeGroups.append('text')
      .attr('class', 'node-code')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('font-family', 'ui-monospace, monospace')
      .attr('fill', '#1e293b')
      .text(d => getDisplayCode(d));

    // Add billable indicator
    nodeGroups.append('text')
      .attr('class', 'node-billable')
      .attr('x', NODE_WIDTH - 6)
      .attr('y', 16)
      .attr('text-anchor', 'end')
      .attr('font-size', 12)
      .attr('font-weight', 700)
      .attr('fill', '#16a34a')
      .text(d => d.billable ? '$' : '');

    // Add label line 1
    nodeGroups.append('text')
      .attr('class', 'node-label node-label-1')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', 32)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', '#64748b')
      .text(d => wrapNodeLabel(d.label, 22)[0]);

    // Add label line 2
    nodeGroups.append('text')
      .attr('class', 'node-label node-label-2')
      .attr('x', NODE_WIDTH / 2)
      .attr('y', 44)
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
        showExpandedNode(d, this as SVGGElement);
        onNodeClick(d.id);
      })
      .on('mouseenter', function (_, d) {
        if (activeOverlayNodeRef.current === d.id) {
          cancelHideTimeout();
          return;
        }
        cancelShowTimeout();
        cancelHideTimeout();
        const nodeGroup = this as SVGGElement;
        if (pinnedNodeIdRef.current === d.id) {
          showExpandedNode(d, nodeGroup);
          return;
        }
        showTimeoutRef.current = setTimeout(() => {
          showExpandedNode(d, nodeGroup);
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
        showExpandedNode(pinnedNode, null as unknown as SVGGElement);
      }
    }

    // Click on SVG background (any click not stopped by nodes/buttons) releases the pinned overlay
    svg.on('click', (event: MouseEvent) => {
      if (pinnedNodeIdRef.current) {
        // Don't unpin if clicking inside the overlay (foreignObject clicks may not stop propagation correctly)
        const target = event.target as Element;
        const isInsideOverlay = target.closest('.expanded-overlay') !== null ||
          target.closest('.expanded-node') !== null ||
          target.closest('.batch-panel') !== null ||
          target.tagName.toLowerCase() === 'textarea' ||
          target.closest('foreignObject') !== null;
        if (isInsideOverlay) return;

        setPinnedNodeId(null);
        hideExpandedNode();
      }
    });

    // IMPORTANT: Apply the current transform to the new <g> element
    // This preserves the user's pan/zoom position when the graph re-renders
    const currentTransform = d3.zoomTransform(svg.node()!);
    g.attr('transform', currentTransform.toString());

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
