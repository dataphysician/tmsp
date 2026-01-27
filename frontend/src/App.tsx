import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { applyPatch, type Operation } from 'fast-json-patch';
import { GraphViewer } from './components/GraphViewer';
import { TrajectoryViewer } from './components/TrajectoryViewer';
import { BenchmarkReportViewer } from './components/BenchmarkReportViewer';
import { VisualizeReportViewer } from './components/VisualizeReportViewer';
import { LLMSettingsPanel, NodeRewindModal } from './components/shared';
import { streamTraversal, streamRewind, buildGraph, type AGUIEvent } from './lib/api';
import { extractBatchType, getBatchSpecificDescendants } from './lib/graphUtils';
import type {
  TraversalState,
  DecisionPoint,
  CandidateDecision,
  GraphNode,
  GraphEdge,
  BenchmarkGraphNode,
  BenchmarkMetrics,
  TraversalStatus,
  OvershootMarker,
  EdgeMissMarker,
  LLMConfig,
} from './lib/types';
import {
  buildAncestorMap,
  compareFinalizedCodes,
  computeBenchmarkMetrics,
  computeFinalizedComparison,
  initializeExpectedNodes,
  resetNodesToIdle,
} from './lib/benchmark';
import {
  INITIAL_TRAVERSAL_STATE,
  LLM_SYSTEM_PROMPT,
  LLM_SYSTEM_PROMPT_NON_SCAFFOLDED,
  type ViewTab,
  type SidebarTab,
} from './lib/constants';
import './App.css';

type FeatureTab = 'visualize' | 'traverse' | 'benchmark';

// Merge helpers for two-stage rendering (Stream + Reconcile)
function mergeById(existing: GraphNode[], incoming: GraphNode[]): GraphNode[] {
  const map = new Map(existing.map(n => [n.id, n]));
  for (const node of incoming) {
    // Simply overwrite - STEP_FINISHED has authoritative depth from selected_details
    map.set(node.id, node);
  }
  return [...map.values()];
}

function mergeByKey(existing: GraphEdge[], incoming: GraphEdge[]): GraphEdge[] {
  const key = (e: GraphEdge) => `${e.source}|${e.target}`;
  const map = new Map(existing.map(e => [key(e), e]));
  for (const edge of incoming) {
    map.set(key(edge), edge);
  }
  return [...map.values()];
}

// Calculate ICD depth from code structure
// Chapter_* = 1, Ranges (X##-X##) = 2, Codes = character count without dots
function calculateDepthFromCode(code: string): number {
  if (code.startsWith('Chapter_')) return 1;
  if (code.includes('-')) return 2; // Range like I20-I25
  // For actual codes: count characters excluding dots
  return code.replace(/\./g, '').length;
}


function TraversalUI() {
  const [clinicalNote, setClinicalNote] = useState('');
  const [state, setState] = useState<TraversalState>(INITIAL_TRAVERSAL_STATE);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [batchCount, setBatchCount] = useState(0);
  const controllerRef = useRef<AbortController | null>(null);
  // Refs for traverse tab state (updated during streaming, applied at RUN_FINISHED)
  const traverseNodesRef = useRef<GraphNode[]>([]);
  const traverseEdgesRef = useRef<GraphEdge[]>([]);
  const traverseDecisionsRef = useRef<DecisionPoint[]>([]);
  const traverseBatchCountRef = useRef<number>(0);
  // Per-feature view tab states (Graph vs Report)
  const [visualizeViewTab, setVisualizeViewTab] = useState<ViewTab>('graph');
  const [traverseViewTab, setTraverseViewTab] = useState<ViewTab>('graph');
  const [benchmarkViewTab, setBenchmarkViewTab] = useState<ViewTab>('graph');
  const [activeFeatureTab, setActiveFeatureTab] = useState<FeatureTab>('traverse');

  // LLM Configuration
  const [llmConfig, setLlmConfig] = useState<LLMConfig>({
    provider: 'vertexai',
    apiKey: '',
    model: 'gemini-2.5-flash',
    maxTokens: 64000,
    temperature: 0.0,
    extra: { auth_type: 'api_key', location: '', project_id: '' },
    systemPrompt: LLM_SYSTEM_PROMPT,
    scaffolded: true,
  });
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('clinical-note');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // VISUALIZE tab state
  const [inputCodes, setInputCodes] = useState<Set<string>>(new Set());
  const [codeInput, setCodeInput] = useState('');
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [visualizeFitTrigger, setVisualizeFitTrigger] = useState(0);

  // Zero-shot visualization graph (for Traverse tab when visualizePrediction is ON)
  const [zeroShotVisualization, setZeroShotVisualization] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [isLoadingZeroShotViz, setIsLoadingZeroShotViz] = useState(false);
  const [traverseFitTrigger, setTraverseFitTrigger] = useState(0);
  const traverseLastInteractionRef = useRef<number>(0);

  // BENCHMARK tab state
  const [benchmarkExpectedCodes, setBenchmarkExpectedCodes] = useState<Set<string>>(new Set());
  const [benchmarkCodeInput, setBenchmarkCodeInput] = useState('');
  const [benchmarkExpectedGraph, setBenchmarkExpectedGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [benchmarkTraversedNodes, setBenchmarkTraversedNodes] = useState<GraphNode[]>([]);
  const [_benchmarkTraversedEdges, setBenchmarkTraversedEdges] = useState<GraphEdge[]>([]);
  // Refs to track latest values for use in async callbacks (avoid stale closures)
  const benchmarkTraversedNodesRef = useRef<GraphNode[]>([]);
  const benchmarkTraversedEdgesRef = useRef<GraphEdge[]>([]);
  const benchmarkDecisionsRef = useRef<DecisionPoint[]>([]);
  // Map refs for O(1) updates during streaming (avoid O(n) mergeById/mergeByKey on every event)
  const benchmarkNodesMapRef = useRef<Map<string, GraphNode>>(new Map());
  const benchmarkEdgesMapRef = useRef<Map<string, GraphEdge>>(new Map());
  const benchmarkCombinedNodesRef = useRef<BenchmarkGraphNode[]>([]);
  // Phase 1: Track all selected_ids during streaming for Phase 2 marker computation
  const benchmarkStreamedIdsRef = useRef<Set<string>>(new Set());
  // Cache expectedNodeIds to avoid creating a new Set on every STEP_FINISHED event
  const benchmarkExpectedNodeIdsRef = useRef<Set<string>>(new Set());
  // Batch count ref for streaming (state only updated at end)
  const benchmarkBatchCountRef = useRef<number>(0);
  // Track pending RAF for node reset (to cancel on cached replay completion)
  const benchmarkResetRafRef = useRef<number | null>(null);
  // Track if current run is a cached replay (for UI display)
  const benchmarkIsCachedReplayRef = useRef<boolean>(false);
  const [benchmarkCombinedNodes, setBenchmarkCombinedNodes] = useState<BenchmarkGraphNode[]>([]);
  const [benchmarkCombinedEdges, setBenchmarkCombinedEdges] = useState<GraphEdge[]>([]);
  const [benchmarkOvershootMarkers, setBenchmarkOvershootMarkers] = useState<OvershootMarker[]>([]);
  const [benchmarkMissedEdgeMarkers, setBenchmarkMissedEdgeMarkers] = useState<EdgeMissMarker[]>([]);
  const [benchmarkMetrics, setBenchmarkMetrics] = useState<BenchmarkMetrics | null>(null);
  const [benchmarkDecisions, setBenchmarkDecisions] = useState<DecisionPoint[]>([]);
  const [benchmarkStatus, setBenchmarkStatus] = useState<TraversalStatus>('idle');
  const [benchmarkCurrentStep, setBenchmarkCurrentStep] = useState('');
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [benchmarkClinicalNote, setBenchmarkClinicalNote] = useState('');
  const [benchmarkBatchCount, setBenchmarkBatchCount] = useState(0);
  const [benchmarkFitTrigger, setBenchmarkFitTrigger] = useState(0);
  const benchmarkLastInteractionRef = useRef<number>(0);
  // Streaming traversed IDs - updated during STEP_FINISHED for real-time visual feedback
  const [streamingTraversedIds, setStreamingTraversedIds] = useState<Set<string>>(new Set());
  // Ref to track streaming IDs without triggering re-renders
  const streamingTraversedIdsRef = useRef<Set<string>>(new Set());
  const lastVisualUpdateTimeRef = useRef<number>(0);
  // Throttle visual updates to max 10/sec during streaming (prevents re-render storm)
  const VISUAL_UPDATE_THROTTLE_MS = 100; // Max 10 visual updates per second during streaming
  const benchmarkControllerRef = useRef<AbortController | null>(null);
  const [benchmarkSidebarTab, setBenchmarkSidebarTab] = useState<SidebarTab>('clinical-note');
  const [benchmarkInvalidCodes, setBenchmarkInvalidCodes] = useState<Set<string>>(new Set());

  // Benchmark: Infer Precursor Nodes toggle for zero-shot mode
  const [benchmarkInferPrecursors, setBenchmarkInferPrecursors] = useState(false);

  // Cached finalized-only view (no X markers - default view)
  const [benchmarkFinalizedView, setBenchmarkFinalizedView] = useState<{
    nodes: BenchmarkGraphNode[];
    edges: GraphEdge[];
    metrics: BenchmarkMetrics;
    overshootMarkers: OvershootMarker[];
  } | null>(null);

  // Cached inferred view (merged graph with X markers)
  const [benchmarkInferredView, setBenchmarkInferredView] = useState<{
    nodes: BenchmarkGraphNode[];
    edges: GraphEdge[];
    graphNodes: BenchmarkGraphNode[];  // Inferred graph nodes with benchmark status for Graph view
    metrics: BenchmarkMetrics;
    overshootMarkers: OvershootMarker[];
    missedEdgeMarkers: EdgeMissMarker[];
    inferredNodes: GraphNode[];  // Raw inferred graph nodes for Report interim computation
  } | null>(null);

  // NOTE: Cross-run caching for zero-shot is now handled by the backend (Burr + SQLite)
  // The backend caches LLM responses based on (clinical_note, provider, model, temperature, system_prompt)
  // Frontend processes results into comparison views in handleBenchmarkEvent RUN_FINISHED handler

  // Memoize benchmark finalized codes array to prevent unnecessary GraphViewer re-renders
  // During streaming, benchmarkExpectedCodes doesn't change, so this array stays stable
  const benchmarkFinalizedCodesArray = useMemo(() => {
    return [...benchmarkExpectedCodes];
  }, [benchmarkExpectedCodes]);

  // Rewind state (for spot rewind feature in TRAVERSE tab)
  const [rewindTargetNode, setRewindTargetNode] = useState<GraphNode | null>(null);
  const [rewindTargetBatchId, setRewindTargetBatchId] = useState<string | null>(null);
  const [rewindFeedbackText, setRewindFeedbackText] = useState<string>('');
  const [isRewindModalOpen, setIsRewindModalOpen] = useState(false);
  const [isRewinding, setIsRewinding] = useState(false);
  const [rewindingNodeId, setRewindingNodeId] = useState<string | null>(null);
  const [rewindError, setRewindError] = useState<string | null>(null);
  const rewindControllerRef = useRef<AbortController | null>(null);

  // Elapsed time tracking (lifted up from child components for persistence across tab switches)
  const [traverseElapsedTime, setTraverseElapsedTime] = useState<number | null>(null);
  const traverseStartTimeRef = useRef<number | null>(null);
  const prevTraverseStatusRef = useRef<TraversalStatus>('idle');

  const [benchmarkElapsedTime, setBenchmarkElapsedTime] = useState<number | null>(null);
  const benchmarkStartTimeRef = useRef<number | null>(null);
  const prevBenchmarkStatusRef = useRef<TraversalStatus>('idle');

  // Track traverse elapsed time
  useEffect(() => {
    const currentStatus = state.status;
    // Started: any status -> traversing (handles re-runs from complete/error)
    if (prevTraverseStatusRef.current !== 'traversing' && currentStatus === 'traversing') {
      traverseStartTimeRef.current = Date.now();
      setTraverseElapsedTime(null);
    }
    // Finished: traversing -> complete or error
    if (prevTraverseStatusRef.current === 'traversing' && (currentStatus === 'complete' || currentStatus === 'error')) {
      if (traverseStartTimeRef.current) {
        setTraverseElapsedTime(Date.now() - traverseStartTimeRef.current);
      }
    }
    // Reset when going back to idle
    if (currentStatus === 'idle') {
      traverseStartTimeRef.current = null;
      setTraverseElapsedTime(null);
    }
    prevTraverseStatusRef.current = currentStatus;
  }, [state.status]);

  // Track benchmark elapsed time
  useEffect(() => {
    // Started: any status -> traversing (handles re-runs from complete/error)
    if (prevBenchmarkStatusRef.current !== 'traversing' && benchmarkStatus === 'traversing') {
      benchmarkStartTimeRef.current = Date.now();
      setBenchmarkElapsedTime(null);
    }
    // Finished: traversing -> complete or error
    if (prevBenchmarkStatusRef.current === 'traversing' && (benchmarkStatus === 'complete' || benchmarkStatus === 'error')) {
      if (benchmarkStartTimeRef.current) {
        setBenchmarkElapsedTime(Date.now() - benchmarkStartTimeRef.current);
      }
    }
    // Reset when going back to idle
    if (benchmarkStatus === 'idle') {
      benchmarkStartTimeRef.current = null;
      setBenchmarkElapsedTime(null);
    }
    prevBenchmarkStatusRef.current = benchmarkStatus;
  }, [benchmarkStatus]);

  // Callbacks to reset idle timer on graph interaction
  const handleTraverseGraphInteraction = useCallback(() => {
    traverseLastInteractionRef.current = Date.now();
  }, []);

  const handleBenchmarkGraphInteraction = useCallback(() => {
    benchmarkLastInteractionRef.current = Date.now();
  }, []);

  // Fit-to-window when nodes first appear during Traverse tab traversal
  const traverseHadNodesRef = useRef(false);
  useEffect(() => {
    if (state.status === 'traversing' && !isRewinding) {
      if (state.nodes.length > 0 && !traverseHadNodesRef.current) {
        // Nodes just appeared - trigger fit after short delay
        traverseHadNodesRef.current = true;
        const timer = setTimeout(() => {
          setTraverseFitTrigger(prev => prev + 1);
        }, 350);
        return () => clearTimeout(timer);
      }
    } else if (state.status === 'idle') {
      // Reset when going back to idle
      traverseHadNodesRef.current = false;
    }
  }, [state.status, state.nodes.length, isRewinding]);

  // Periodic fit-to-window during Traverse tab traversal (after 5 seconds idle, not during rewind)
  useEffect(() => {
    if (state.status === 'traversing' && !isRewinding) {
      // Reset interaction time when traversal starts
      traverseLastInteractionRef.current = Date.now();
      const interval = setInterval(() => {
        const idleTime = Date.now() - traverseLastInteractionRef.current;
        if (idleTime >= 5000) {
          setTraverseFitTrigger(prev => prev + 1);
          traverseLastInteractionRef.current = Date.now(); // Reset after fit
        }
      }, 1000); // Check every second
      return () => clearInterval(interval);
    }
  }, [state.status, isRewinding]);

  // Final fit-to-window 5 seconds after Traverse tab traversal completes (resets on interaction)
  useEffect(() => {
    if (state.status === 'complete' && !isRewinding) {
      traverseLastInteractionRef.current = Date.now();
      const interval = setInterval(() => {
        const idleTime = Date.now() - traverseLastInteractionRef.current;
        if (idleTime >= 5000) {
          setTraverseFitTrigger(prev => prev + 1);
          clearInterval(interval); // Only trigger once after completion
        }
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [state.status, isRewinding]);

  // Fit-to-window when nodes first appear during Benchmark tab traversal
  const benchmarkHadNodesRef = useRef(false);
  useEffect(() => {
    if (benchmarkStatus === 'traversing') {
      if (benchmarkCombinedNodes.length > 0 && !benchmarkHadNodesRef.current) {
        // Nodes just appeared - trigger fit after short delay
        benchmarkHadNodesRef.current = true;
        const timer = setTimeout(() => {
          setBenchmarkFitTrigger(prev => prev + 1);
        }, 350);
        return () => clearTimeout(timer);
      }
    } else if (benchmarkStatus === 'idle') {
      // Reset when going back to idle
      benchmarkHadNodesRef.current = false;
    }
  }, [benchmarkStatus, benchmarkCombinedNodes.length]);

  // Periodic fit-to-window during Benchmark tab traversal (after 5 seconds idle)
  useEffect(() => {
    if (benchmarkStatus === 'traversing') {
      // Reset interaction time when traversal starts
      benchmarkLastInteractionRef.current = Date.now();
      const interval = setInterval(() => {
        const idleTime = Date.now() - benchmarkLastInteractionRef.current;
        if (idleTime >= 5000) {
          setBenchmarkFitTrigger(prev => prev + 1);
          benchmarkLastInteractionRef.current = Date.now(); // Reset after fit
        }
      }, 1000); // Check every second
      return () => clearInterval(interval);
    }
  }, [benchmarkStatus]);

  // Final fit-to-window 5 seconds after Benchmark tab traversal completes (resets on interaction)
  useEffect(() => {
    if (benchmarkStatus === 'complete') {
      benchmarkLastInteractionRef.current = Date.now();
      const interval = setInterval(() => {
        const idleTime = Date.now() - benchmarkLastInteractionRef.current;
        if (idleTime >= 5000) {
          setBenchmarkFitTrigger(prev => prev + 1);
          clearInterval(interval); // Only trigger once after completion
        }
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [benchmarkStatus]);

  // Handle AG-UI events
  // For cached replays, server sends STATE_SNAPSHOT (complete graph) instead of individual events
  const handleAGUIEvent = useCallback((event: AGUIEvent) => {
    switch (event.type) {
      case 'RUN_STARTED':
        if (event.metadata?.cached) {
          console.log('[Traverse] Cached replay - expecting STATE_SNAPSHOT');
        }
        setState(prev => ({
          ...prev,
          status: 'traversing',
          current_step: 'Starting traversal',
        }));
        break;

      case 'STATE_SNAPSHOT':
        // AG-UI Protocol: Complete state replacement for cached replays
        // Server sends a single snapshot instead of 500+ individual events
        if (event.state) {
          const nodes = event.state.nodes as GraphNode[];
          const edges = event.state.edges as GraphEdge[];
          // Update refs
          traverseNodesRef.current = nodes;
          traverseEdgesRef.current = edges;
          // For cached replays, this is the complete graph - update state immediately
          // For live traversals, this may be an initial snapshot
          setState(prev => ({
            ...prev,
            nodes,
            edges,
          }));
          console.log(`[STATE_SNAPSHOT] Complete graph: ${nodes.length} nodes, ${edges.length} edges`);
        }
        break;

      case 'STATE_DELTA':
        // Live traversal: Apply JSON Patch for incremental updates
        // Note: Cached replays use STATE_SNAPSHOT instead, so this only runs for live traversals
        if (event.delta) {
          setState(prev => {
            const graphState = {
              nodes: [...prev.nodes],
              edges: [...prev.edges],
              finalized: [...prev.finalized_codes],
            };

            try {
              const result = applyPatch(graphState, event.delta as Operation[], true, false);
              // Update refs as well
              traverseNodesRef.current = result.newDocument.nodes;
              traverseEdgesRef.current = result.newDocument.edges;
              return {
                ...prev,
                nodes: result.newDocument.nodes,
                edges: result.newDocument.edges,
                finalized_codes: result.newDocument.finalized || prev.finalized_codes,
              };
            } catch (error) {
              console.error('Failed to apply JSON Patch:', error);
              return prev;
            }
          });
        }
        break;

      case 'STEP_STARTED':
        // Live traversal: Update current step
        // Note: Cached replays use STATE_SNAPSHOT instead, so this only runs for live traversals
        setState(prev => ({
          ...prev,
          current_step: event.step_id || '',
        }));
        break;

      case 'STEP_FINISHED':
        traverseBatchCountRef.current += 1;

        // Convert to decision point for history
        if (event.metadata) {
          const metadata = event.metadata;
          const candidates = (metadata.candidates || {}) as Record<string, string>;
          const selectedIds = (metadata.selected_ids || []) as string[];
          const reasoning = (metadata.reasoning || '') as string;
          const nodeId = (metadata.node_id || event.step_id) as string;
          const batchType = (metadata.batch_type || 'children') as string;
          const selectedDetails = (metadata.selected_details || {}) as Record<string, {
            depth: number;
            category: string;
            billable: boolean;
          }>;

          // Stage 2: Reconcile - Derive graph data from batch metadata
          // This ensures graph is complete even if STATE_DELTA had race conditions
          // Use calculateDepthFromCode for deterministic depth based on ICD structure
          // Backend handles all sevenChrDef X-padding and validation - frontend just uses codes as-is
          const edgeSource = nodeId && nodeId !== 'ROOT' ? nodeId : 'ROOT';
          const isFromRoot = edgeSource === 'ROOT';

          const newNodes: GraphNode[] = [];
          const newEdges: GraphEdge[] = [];

          // All batches handled uniformly - backend sends full validated codes
          for (const code of selectedIds) {
            newNodes.push({
              id: code,
              code: code,
              label: candidates[code] || code,
              depth: calculateDepthFromCode(code),
              category: (selectedDetails[code]?.category ?? 'ancestor') as GraphNode['category'],
              billable: selectedDetails[code]?.billable ?? false,
            });

            newEdges.push({
              source: edgeSource,
              target: code,
              edge_type: (isFromRoot || batchType === 'children') ? 'hierarchy' as const : 'lateral' as const,
              rule: (isFromRoot || batchType === 'children') ? null : batchType,
            });
          }

          // Trajectory: Build decision point for history
          const candidateDecisions: CandidateDecision[] = Object.entries(candidates).map(
            ([code, label]) => ({
              code,
              label,
              selected: selectedIds.includes(code),
              confidence: selectedIds.includes(code) ? 1.0 : 0.0,
              evidence: null,
              reasoning: selectedIds.includes(code) ? reasoning : '',
              billable: selectedDetails[code]?.billable ?? false,
            })
          );

          const decision: DecisionPoint = {
            current_node: nodeId,
            current_label: `${batchType} batch`,
            depth: (event.step_id?.split('|').length || 1),
            candidates: candidateDecisions,
            selected_codes: selectedIds,
          };

          // Live traversal: merge and update state for real-time feedback
          // Note: Cached replays use STATE_SNAPSHOT + decisions in RUN_FINISHED metadata
          traverseNodesRef.current = mergeById(traverseNodesRef.current, newNodes);
          traverseEdgesRef.current = mergeByKey(traverseEdgesRef.current, newEdges);
          traverseDecisionsRef.current = [...traverseDecisionsRef.current, decision];
          setBatchCount(traverseBatchCountRef.current);
          setState(prev => ({
            ...prev,
            nodes: traverseNodesRef.current,
            edges: traverseEdgesRef.current,
            decision_history: traverseDecisionsRef.current,
          }));
        }
        break;

      case 'RUN_FINISHED':
        if (event.metadata?.error) {
          setState(prev => ({
            ...prev,
            status: 'error',
            error: event.metadata!.error as string,
            current_step: 'Error',
          }));
        } else {
          // Deduplicate final nodes (in case parallel batches reached same node)
          const finalNodesRaw = (event.metadata?.final_nodes || []) as string[];
          const finalNodes = [...new Set(finalNodesRaw)];

          // AG-UI Protocol: For snapshot mode, decisions come in RUN_FINISHED metadata
          // This eliminates the need for 500+ STEP_FINISHED events
          const snapshotDecisions = event.metadata?.decisions as Array<{
            batch_id: string;
            node_id: string;
            batch_type: string;
            candidates: Record<string, string>;
            selected_ids: string[];
            reasoning: string;
            selected_details?: Record<string, { depth: number; category: string; billable: boolean }>;
          }> | undefined;

          if (snapshotDecisions && snapshotDecisions.length > 0) {
            // Convert server decisions to frontend DecisionPoint format
            const decisions: DecisionPoint[] = snapshotDecisions.map(d => ({
              current_node: d.node_id,
              current_label: `${d.batch_type} batch`,
              depth: (d.batch_id?.split('|').length || 1),
              candidates: Object.entries(d.candidates).map(([code, label]) => ({
                code,
                label,
                selected: d.selected_ids.includes(code),
                confidence: d.selected_ids.includes(code) ? 1.0 : 0.0,
                evidence: null,
                reasoning: d.selected_ids.includes(code) ? d.reasoning : '',
                billable: d.selected_details?.[code]?.billable ?? false,
              })),
              selected_codes: d.selected_ids,
            }));
            traverseDecisionsRef.current = decisions;
            traverseBatchCountRef.current = decisions.length;
            console.log(`[RUN_FINISHED] Snapshot mode: ${decisions.length} decisions from metadata`);
          }

          // Apply data from refs
          setBatchCount(traverseBatchCountRef.current);
          setState(prev => ({
            ...prev,
            status: 'complete',
            nodes: traverseNodesRef.current,
            edges: traverseEdgesRef.current,
            decision_history: traverseDecisionsRef.current,
            finalized_codes: finalNodes,
            current_step: `Complete - ${finalNodes.length} codes found`,
          }));
          // Trigger fit-to-window after traversal completes
          setTraverseFitTrigger(prev => prev + 1);
        }
        setIsLoading(false);
        break;
    }
  }, []);

  const handleError = useCallback((error: Error) => {
    setState(prev => ({
      ...prev,
      status: 'error',
      error: error.message,
    }));
    setIsLoading(false);
  }, []);

  const handleTraverse = useCallback((): boolean => {
    if (!clinicalNote.trim()) return false;
    if (!llmConfig.apiKey) {
      alert('Please configure your API key in LLM Settings');
      setSidebarTab('llm-settings');
      return false;
    }

    // Cancel any existing traverse stream
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }

    // Also cancel any pending benchmark stream to prevent cross-tab interference
    if (benchmarkControllerRef.current) {
      benchmarkControllerRef.current.abort();
      benchmarkControllerRef.current = null;
      setBenchmarkStatus('idle');
    }

    setIsLoading(true);
    setBatchCount(0);
    // Reset refs
    traverseNodesRef.current = [];
    traverseEdgesRef.current = [];
    traverseDecisionsRef.current = [];
    traverseBatchCountRef.current = 0;
    setState({
      ...INITIAL_TRAVERSAL_STATE,
      status: 'traversing',
      current_step: 'Starting traversal',
    });

    controllerRef.current = streamTraversal(
      {
        clinical_note: clinicalNote,
        provider: llmConfig.provider,
        api_key: llmConfig.apiKey,
        model: llmConfig.model || undefined,
        selector: 'llm',
        max_tokens: llmConfig.maxTokens,
        temperature: llmConfig.temperature,
        extra: llmConfig.extra,
        system_prompt: llmConfig.systemPrompt || undefined,
        scaffolded: llmConfig.scaffolded ?? true,
      },
      handleAGUIEvent,
      handleError
    );
    return true;
  }, [clinicalNote, llmConfig, handleAGUIEvent, handleError]);

  const handleCancel = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    setIsLoading(false);
    setState(prev => ({
      ...prev,
      status: 'idle',
      current_step: 'Cancelled',
    }));
  }, []);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(prev => (prev === nodeId ? null : nodeId));
  }, []);

  // Rewind handlers (for spot rewind feature in TRAVERSE tab)
  const handleRewindSubmit = useCallback(async (nodeId: string, feedback: string, providedBatchId?: string) => {
    setIsRewinding(true);
    setRewindingNodeId(nodeId);  // Track which node is being rewound
    setRewindError(null);
    setSelectedNode(null);  // Clear selection to reset SVG state

    try {
      // Use provided batchId, then stored batchId, otherwise construct from first matching decision
      let batchId = providedBatchId || rewindTargetBatchId;
      if (!batchId) {
        const decision = state.decision_history.find(d => d.current_node === nodeId);
        const batchType = decision ? extractBatchType(decision.current_label) : 'children';
        batchId = `${nodeId}|${batchType}`;
      }

      // Cancel any existing rewind stream
      if (rewindControllerRef.current) {
        rewindControllerRef.current.abort();
      }

      // Extract batchType from batchId for batch-aware pruning
      const batchType = batchId.split('|')[1] || 'children';

      // Use batch-specific pruning inside setState to ensure we use latest state
      // This prevents stale closure issues when rapid events are being processed
      setState(prev => {
        // Calculate descendants to prune using CURRENT state (prev), not stale closure
        const descendantIds = getBatchSpecificDescendants(
          nodeId,
          batchType,
          prev.decision_history,
          prev.edges
        );

        // Filter functions for reuse
        const filterNodes = (nodes: typeof prev.nodes) => nodes.filter(n => !descendantIds.has(n.id));
        const filterEdges = (edges: typeof prev.edges) => edges.filter(e =>
          !descendantIds.has(e.source as string) &&
          !descendantIds.has(e.target as string)
        );
        const filterDecisions = (decisions: typeof prev.decision_history) => decisions.filter(d => {
          // Remove the specific batch decision being rewound
          if (d.current_node === nodeId && extractBatchType(d.current_label) === batchType) {
            return false;
          }
          // Remove decisions for pruned descendants
          if (descendantIds.has(d.current_node)) {
            return false;
          }
          return true;
        });

        // CRITICAL: Also update refs so STEP_FINISHED events don't append to stale data
        // Without this, the old batch decision remains in the ref and we get duplicates
        traverseNodesRef.current = filterNodes(traverseNodesRef.current);
        traverseEdgesRef.current = filterEdges(traverseEdgesRef.current);
        traverseDecisionsRef.current = filterDecisions(traverseDecisionsRef.current);

        return {
          ...prev,
          nodes: filterNodes(prev.nodes),
          edges: filterEdges(prev.edges),
          // Remove only descendants from finalized_codes (not the rewound node)
          // The backend will compute correct final_nodes after rewind:
          // - If children selected → rewound node is NOT a leaf → NOT in final_nodes
          // - If no children selected → rewound node IS a leaf → IN final_nodes
          // RUN_FINISHED will replace finalized_codes with the correct final_nodes
          finalized_codes: prev.finalized_codes.filter(c => !descendantIds.has(c)),
          // Filter decision_history: remove specific batch + pruned descendants
          decision_history: filterDecisions(prev.decision_history),
          status: 'traversing',
          current_step: `Rewinding from ${nodeId} (${batchType} batch)...`,
        };
      });

      // Close modal
      setIsRewindModalOpen(false);
      setRewindTargetNode(null);
      setRewindTargetBatchId(null);

      // Stream the rewind traversal
      // Pass existing node IDs so backend can link lateral targets to existing ancestors
      const existingNodeIds = state.nodes.map(n => n.id);
      rewindControllerRef.current = streamRewind(
        {
          batch_id: batchId,
          feedback,
          clinical_note: clinicalNote,
          existing_nodes: existingNodeIds,
          provider: llmConfig.provider,
          api_key: llmConfig.apiKey,
          model: llmConfig.model || undefined,
          selector: 'llm',
          max_tokens: llmConfig.maxTokens,
          temperature: llmConfig.temperature,
          extra: llmConfig.extra,
          system_prompt: llmConfig.systemPrompt || undefined,
          scaffolded: llmConfig.scaffolded ?? true,
        },
        handleAGUIEvent,
        (error) => {
          setRewindError(error.message);
          setIsRewinding(false);
          setState(prev => ({
            ...prev,
            status: 'error',
            error: error.message,
          }));
        }
      );
    } catch (error) {
      setRewindError(error instanceof Error ? error.message : 'Rewind failed');
      setIsRewinding(false);
    }
  }, [state.decision_history, state.nodes, clinicalNote, llmConfig, handleAGUIEvent, rewindTargetBatchId]);

  const handleRewindClose = useCallback(() => {
    if (!isRewinding) {
      setIsRewindModalOpen(false);
      setRewindTargetNode(null);
      setRewindTargetBatchId(null);
      setRewindFeedbackText('');
      setRewindError(null);
    }
  }, [isRewinding]);

  const handleNodeRewindClick = useCallback((nodeId: string, batchType?: string, feedback?: string) => {
    // Find the node in current state
    const node = state.nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Only allow rewind on non-ROOT nodes during/after traversal
    if (nodeId === 'ROOT' || state.status === 'idle') return;

    // Construct the composite batch_id for backend
    const batchId = batchType ? `${nodeId}|${batchType}` : null;

    // If feedback is provided directly, skip modal and rewind immediately
    if (feedback && feedback.trim().length > 0 && batchId) {
      handleRewindSubmit(nodeId, feedback, batchId);
      return;
    }

    // Otherwise, open modal for user to enter feedback
    setRewindTargetNode(node);
    setRewindTargetBatchId(batchId);
    setRewindFeedbackText(feedback || '');
    setIsRewindModalOpen(true);
    setRewindError(null);
  }, [state.nodes, state.status, handleRewindSubmit]);

  // Effect to reset isRewinding when traversal completes
  useEffect(() => {
    if (isRewinding && (state.status === 'complete' || state.status === 'error')) {
      setIsRewinding(false);
      setRewindingNodeId(null);
    }
  }, [isRewinding, state.status]);

  // Effect to visualize zero-shot predictions when enabled
  useEffect(() => {
    const isComplete = state.status === 'complete';
    const isZeroShot = !(llmConfig.scaffolded ?? true);
    const vizEnabled = llmConfig.visualizePrediction ?? false;
    const hasCodes = state.finalized_codes.length > 0;
    const shouldVisualize = isComplete && isZeroShot && vizEnabled && hasCodes;

    console.log('[ZeroShotViz] Effect triggered:', {
      isComplete,
      isZeroShot,
      vizEnabled,
      hasCodes,
      shouldVisualize,
      finalized_codes: state.finalized_codes,
    });

    if (shouldVisualize) {
      // Build graph from finalized codes for Traverse tab visualization
      (async () => {
        setIsLoadingZeroShotViz(true);
        try {
          console.log('[ZeroShotViz] Building graph from codes:', state.finalized_codes);
          const result = await buildGraph(state.finalized_codes);
          console.log('[ZeroShotViz] Graph built successfully:', result.nodes.length, 'nodes');
          setZeroShotVisualization({ nodes: result.nodes, edges: result.edges });
        } catch (err) {
          console.error('[ZeroShotViz] Failed to build visualization:', err);
          setZeroShotVisualization(null);
        } finally {
          setIsLoadingZeroShotViz(false);
        }
      })();
    } else if (state.status === 'idle' || (llmConfig.scaffolded ?? true)) {
      // Clear visualization when returning to idle or switching to scaffolded mode
      console.log('[ZeroShotViz] Clearing visualization');
      setZeroShotVisualization(null);
    }
  }, [state.status, state.finalized_codes, llmConfig.scaffolded, llmConfig.visualizePrediction]);

  // Get the decision for the rewind target node
  // If batchId is set, find the matching decision by batch type
  const rewindTargetDecision = rewindTargetNode
    ? (() => {
        if (rewindTargetBatchId) {
          // batchId format: "nodeId|batchType"
          const batchType = rewindTargetBatchId.split('|')[1];
          return state.decision_history.find(d =>
            d.current_node === rewindTargetNode.id &&
            extractBatchType(d.current_label) === batchType
          ) || null;
        }
        // Fallback to first matching decision
        return state.decision_history.find(d => d.current_node === rewindTargetNode.id) || null;
      })()
    : null;

  // VISUALIZE tab handlers
  const handleAddCode = useCallback(async () => {
    // Split by comma, newline, tab, or space and filter empty strings
    const codes = codeInput
      .split(/[,\n\t\s]+/)
      .map(c => c.trim().toUpperCase())
      .filter(c => c.length > 0);

    if (codes.length > 0) {
      const newSet = new Set(inputCodes);
      codes.forEach(code => newSet.add(code));
      setInputCodes(newSet);
      setCodeInput('');

      // Auto-build graph with all codes
      setIsLoadingGraph(true);
      setGraphError(null);
      try {
        const result = await buildGraph([...newSet]);
        setGraphData({ nodes: result.nodes, edges: result.edges });
        // Trigger fit-to-window after graph is built
        setVisualizeFitTrigger(prev => prev + 1);
      } catch (err) {
        setGraphError(err instanceof Error ? err.message : 'Failed to build graph');
      } finally {
        setIsLoadingGraph(false);
      }
    }
  }, [codeInput, inputCodes]);

  const handleRemoveCode = useCallback(async (code: string) => {
    const newCodes = new Set(inputCodes);
    newCodes.delete(code);
    setInputCodes(newCodes);

    // Auto-refresh graph with remaining codes
    if (newCodes.size > 0) {
      setIsLoadingGraph(true);
      try {
        const result = await buildGraph([...newCodes]);
        setGraphData({ nodes: result.nodes, edges: result.edges });
      } catch (err) {
        setGraphError(err instanceof Error ? err.message : 'Failed to build graph');
      } finally {
        setIsLoadingGraph(false);
      }
    } else {
      setGraphData(null);
    }
  }, [inputCodes]);

  const handleClearCodes = useCallback(() => {
    setInputCodes(new Set());
    setGraphData(null);
    setGraphError(null);
  }, []);

  // BENCHMARK tab handlers
  // Build expected graph from codes
  const buildExpectedGraph = useCallback(async (codes: Set<string>) => {
    if (codes.size === 0) return;

    setBenchmarkStatus('traversing');
    setBenchmarkCurrentStep('Building expected graph...');

    try {
      const result = await buildGraph([...codes]);
      setBenchmarkExpectedGraph(result);
      // Cache expectedNodeIds for efficient lookups during STEP_FINISHED events
      benchmarkExpectedNodeIdsRef.current = new Set(result.nodes.map(n => n.id));

      // Initialize combined graph with expected nodes (all marked as 'expected')
      const combinedNodes = initializeExpectedNodes(result.nodes);
      benchmarkCombinedNodesRef.current = combinedNodes;
      setBenchmarkCombinedNodes(combinedNodes);
      setBenchmarkCombinedEdges(result.edges);

      // Clear invalid codes on success
      setBenchmarkInvalidCodes(new Set());
      setBenchmarkStatus('idle');
      setBenchmarkCurrentStep('');
      // Trigger fit-to-window after graph is built
      setBenchmarkFitTrigger(prev => prev + 1);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to build expected graph';

      // Parse invalid codes from error message (format: "Invalid ICD-10-CM codes: X, Y, Z")
      const match = errorMsg.match(/Invalid ICD-10-CM codes?: (.+)/i);
      if (match) {
        const invalidCodes = match[1].split(',').map(c => c.trim());
        setBenchmarkInvalidCodes(new Set(invalidCodes));

        // Try to build graph with valid codes only
        const validCodes = new Set([...codes].filter(c => !invalidCodes.includes(c)));
        if (validCodes.size > 0) {
          try {
            const result = await buildGraph([...validCodes]);
            setBenchmarkExpectedGraph(result);
            benchmarkExpectedNodeIdsRef.current = new Set(result.nodes.map(n => n.id));
            const combinedNodes = initializeExpectedNodes(result.nodes);
            benchmarkCombinedNodesRef.current = combinedNodes;
            setBenchmarkCombinedNodes(combinedNodes);
            setBenchmarkCombinedEdges(result.edges);
            setBenchmarkStatus('idle');
            setBenchmarkCurrentStep('');
            // Trigger fit-to-window after graph is built
            setBenchmarkFitTrigger(prev => prev + 1);
            return;
          } catch {
            // Fall through to error state if valid codes also fail
          }
        }
      }

      setBenchmarkError(errorMsg);
      setBenchmarkStatus('error');
    }
  }, []);

  // Add codes and automatically build/rebuild the expected graph
  const handleBenchmarkAddCode = useCallback(async () => {
    const codes = benchmarkCodeInput
      .split(/[,\n\t\s]+/)
      .map(c => c.trim().toUpperCase())
      .filter(c => c.length > 0);

    if (codes.length > 0) {
      // If benchmark is complete, clear traversal state first (new benchmark setup)
      if (benchmarkStatus === 'complete') {
        benchmarkCombinedNodesRef.current = [];
        setBenchmarkCombinedNodes([]);
        setBenchmarkCombinedEdges([]);
        setBenchmarkOvershootMarkers([]);
        setBenchmarkMissedEdgeMarkers([]);
        setBenchmarkMetrics(null);
        setBenchmarkStatus('idle');
      }

      const newSet = new Set(benchmarkExpectedCodes);
      codes.forEach(code => newSet.add(code));
      setBenchmarkExpectedCodes(newSet);
      setBenchmarkCodeInput('');

      // Auto-build the expected graph
      await buildExpectedGraph(newSet);
    }
  }, [benchmarkCodeInput, benchmarkExpectedCodes, buildExpectedGraph, benchmarkStatus]);

  // Remove an expected code and rebuild the graph
  const handleBenchmarkRemoveCode = useCallback(async (code: string) => {
    // If benchmark is complete, clear traversal state first (new benchmark setup)
    if (benchmarkStatus === 'complete') {
      benchmarkCombinedNodesRef.current = [];
      setBenchmarkCombinedNodes([]);
      setBenchmarkCombinedEdges([]);
      setBenchmarkOvershootMarkers([]);
      setBenchmarkMissedEdgeMarkers([]);
      setBenchmarkMetrics(null);
      setBenchmarkStatus('idle');
    }

    const newSet = new Set(benchmarkExpectedCodes);
    newSet.delete(code);
    setBenchmarkExpectedCodes(newSet);

    // Also remove from invalid codes if present
    setBenchmarkInvalidCodes(prev => {
      const updated = new Set(prev);
      updated.delete(code);
      return updated;
    });

    if (newSet.size > 0) {
      // Rebuild graph with remaining codes
      await buildExpectedGraph(newSet);
    } else {
      // No codes left - clear the graph
      setBenchmarkExpectedGraph(null);
      benchmarkExpectedNodeIdsRef.current = new Set();
      benchmarkCombinedNodesRef.current = [];
      setBenchmarkCombinedNodes([]);
      setBenchmarkCombinedEdges([]);
      setBenchmarkOvershootMarkers([]);
      setBenchmarkMissedEdgeMarkers([]);
      setBenchmarkInvalidCodes(new Set());
    }
  }, [benchmarkExpectedCodes, buildExpectedGraph, benchmarkStatus]);

  const handleBenchmarkEvent = useCallback((event: AGUIEvent) => {
    switch (event.type) {
      case 'RUN_STARTED':
        // Track cached status - single source of truth from server
        benchmarkIsCachedReplayRef.current = event.metadata?.cached === true;

        // Cancel any pending RAF from previous run
        if (benchmarkResetRafRef.current !== null) {
          cancelAnimationFrame(benchmarkResetRafRef.current);
          benchmarkResetRafRef.current = null;
        }

        // Synchronously reset combined nodes to idle status BEFORE any data arrives
        // This ensures users see the "freshly added" state before results stream in
        // Critical for cached replays where STATE_SNAPSHOT arrives immediately
        if (benchmarkCombinedNodesRef.current.length > 0) {
          const idleNodes = resetNodesToIdle(benchmarkCombinedNodesRef.current);
          benchmarkCombinedNodesRef.current = idleNodes;
          setBenchmarkCombinedNodes(idleNodes);
        }

        setBenchmarkStatus('traversing');
        setBenchmarkCurrentStep(
          benchmarkIsCachedReplayRef.current
            ? 'Loading cached results...'
            : 'Starting benchmark traversal'
        );
        break;

      case 'STATE_SNAPSHOT':
        // AG-UI Protocol: Complete state replacement for cached replays
        // Server sends a single snapshot instead of 500+ individual events
        if (event.state) {
          const nodes = event.state.nodes as GraphNode[];
          const edges = event.state.edges as GraphEdge[];
          // Update refs only - state will be applied in RUN_FINISHED
          // This prevents double state updates which can cause hangs
          benchmarkTraversedNodesRef.current = nodes;
          benchmarkTraversedEdgesRef.current = edges;

          // Track all node IDs for streaming progress (marks them as traversed)
          const traversedIds = new Set(nodes.map(n => n.id));
          benchmarkStreamedIdsRef.current = traversedIds;

          console.log(`[STATE_SNAPSHOT] Benchmark complete graph: ${nodes.length} nodes, ${edges.length} edges`);
        }
        break;

      case 'STATE_DELTA':
        // SKIP for benchmark mode - we accumulate via STEP_FINISHED using O(1) Maps
        // STATE_DELTA with array spreading is O(n) per event = O(n²) total
        // Only used for non-benchmark traverse mode
        break;

      case 'STEP_STARTED':
        // Skip state update during streaming - tracked in RUN_FINISHED
        break;

      case 'STEP_FINISHED':
        benchmarkBatchCountRef.current += 1;

        if (event.metadata) {
          const metadata = event.metadata;
          const candidates = (metadata.candidates || {}) as Record<string, string>;
          const selectedIds = (metadata.selected_ids || []) as string[];
          const reasoning = (metadata.reasoning || '') as string;
          const nodeId = (metadata.node_id || event.step_id) as string;
          const batchType = (metadata.batch_type || 'children') as string;
          const selectedDetails = (metadata.selected_details || {}) as Record<string, {
            depth: number;
            category: string;
            billable: boolean;
          }>;

          // Build new traversed nodes and edges
          const edgeSource = nodeId && nodeId !== 'ROOT' ? nodeId : 'ROOT';
          const isFromRoot = edgeSource === 'ROOT';

          const newNodes: GraphNode[] = [];
          const newEdges: GraphEdge[] = [];

          for (const code of selectedIds) {
            newNodes.push({
              id: code,
              code: code,
              label: candidates[code] || code,
              depth: calculateDepthFromCode(code),
              category: (selectedDetails[code]?.category ?? 'ancestor') as GraphNode['category'],
              billable: selectedDetails[code]?.billable ?? false,
            });

            newEdges.push({
              source: edgeSource,
              target: code,
              edge_type: (isFromRoot || batchType === 'children') ? 'hierarchy' as const : 'lateral' as const,
              rule: (isFromRoot || batchType === 'children') ? null : batchType,
            });
          }

          // Build decision point - use Set for O(1) lookups instead of Array.includes O(n)
          const selectedSet = new Set(selectedIds);
          const candidateDecisions: CandidateDecision[] = Object.entries(candidates).map(
            ([code, label]) => ({
              code,
              label,
              selected: selectedSet.has(code),
              confidence: selectedSet.has(code) ? 1.0 : 0.0,
              evidence: null,
              reasoning: selectedSet.has(code) ? reasoning : '',
              billable: selectedDetails[code]?.billable ?? false,
            })
          );

          const decision: DecisionPoint = {
            current_node: nodeId,
            current_label: `${batchType} batch`,
            depth: (event.step_id?.split('|').length || 1),
            candidates: candidateDecisions,
            selected_codes: selectedIds,
          };

          // Accumulate in Map refs for O(1) updates (NO state updates during streaming)
          // Maps are converted to arrays at RUN_FINISHED
          for (const node of newNodes) {
            benchmarkNodesMapRef.current.set(node.id, node);
          }
          for (const edge of newEdges) {
            const edgeKey = `${edge.source}|${edge.target}`;
            benchmarkEdgesMapRef.current.set(edgeKey, edge);
          }
          benchmarkDecisionsRef.current.push(decision);

          // Track streamed IDs for Phase 2 marker computation AND visual feedback
          for (const id of selectedIds) {
            benchmarkStreamedIdsRef.current.add(id);
            streamingTraversedIdsRef.current.add(id);
          }

          // Throttled visual updates during streaming (max 10/sec to avoid UI overload)
          // This triggers GraphViewer's lightweight style-update useEffect
          const now = Date.now();
          if (now - lastVisualUpdateTimeRef.current >= VISUAL_UPDATE_THROTTLE_MS) {
            lastVisualUpdateTimeRef.current = now;
            setStreamingTraversedIds(new Set(streamingTraversedIdsRef.current));
          }
        }
        break;

      case 'RUN_FINISHED':
        if (event.metadata?.error) {
          setBenchmarkStatus('error');
          setBenchmarkError(event.metadata.error as string);
          setBenchmarkCurrentStep('Error');
        } else {
          const finalNodesRaw = (event.metadata?.final_nodes || []) as string[];
          const finalNodes = new Set(finalNodesRaw);
          const isZeroShot = !(llmConfig.scaffolded ?? true);

          // Safety net: Cancel any pending RAF (primary reset happens in RUN_STARTED,
          // but this catches edge cases where RAF might still be pending)
          if (benchmarkResetRafRef.current !== null) {
            cancelAnimationFrame(benchmarkResetRafRef.current);
            benchmarkResetRafRef.current = null;
          }

          // AG-UI Protocol: For snapshot mode, decisions come in RUN_FINISHED metadata
          // This eliminates the need for 500+ STEP_FINISHED events
          const snapshotDecisions = event.metadata?.decisions as Array<{
            batch_id: string;
            node_id: string;
            batch_type: string;
            candidates: Record<string, string>;
            selected_ids: string[];
            reasoning: string;
            selected_details?: Record<string, { depth: number; category: string; billable: boolean }>;
          }> | undefined;

          if (snapshotDecisions && snapshotDecisions.length > 0) {
            // Convert server decisions to frontend DecisionPoint format
            // Use Set for O(1) lookups instead of Array.includes O(n)
            const decisions: DecisionPoint[] = snapshotDecisions.map(d => {
              const selectedSet = new Set(d.selected_ids);
              return {
                current_node: d.node_id,
                current_label: `${d.batch_type} batch`,
                depth: (d.batch_id?.split('|').length || 1),
                candidates: Object.entries(d.candidates).map(([code, label]) => ({
                  code,
                  label,
                  selected: selectedSet.has(code),
                  confidence: selectedSet.has(code) ? 1.0 : 0.0,
                  evidence: null,
                  reasoning: selectedSet.has(code) ? d.reasoning : '',
                  billable: d.selected_details?.[code]?.billable ?? false,
                })),
                selected_codes: d.selected_ids,
              };
            });
            benchmarkDecisionsRef.current = decisions;
            benchmarkBatchCountRef.current = decisions.length;
            console.log(`[RUN_FINISHED] Snapshot mode: ${decisions.length} decisions from metadata`);
          }

          // Convert Maps to arrays (only done once at RUN_FINISHED)
          // During streaming, we used Map refs for O(1) updates
          if (benchmarkNodesMapRef.current.size > 0) {
            benchmarkTraversedNodesRef.current = [...benchmarkNodesMapRef.current.values()];
            benchmarkTraversedEdgesRef.current = [...benchmarkEdgesMapRef.current.values()];
          }

          // Apply state from refs
          setBenchmarkTraversedNodes(benchmarkTraversedNodesRef.current);
          setBenchmarkTraversedEdges(benchmarkTraversedEdgesRef.current);
          setBenchmarkDecisions(benchmarkDecisionsRef.current);
          setBenchmarkBatchCount(benchmarkBatchCountRef.current);

          // Log cache status
          const wasCached = event.metadata?.cached ?? false;
          if (wasCached) {
            console.log('[BACKEND CACHE HIT] Using cached results:', finalNodesRaw.length, 'codes');
          }

          // Phase 2: Compute final statuses and markers OUTSIDE of setState callbacks
          // This prevents nested setState calls and React batching issues
          if (benchmarkExpectedGraph) {
            const latestTraversedEdges = benchmarkTraversedEdgesRef.current;
            const streamedIds = benchmarkStreamedIdsRef.current;
            const currentCombinedNodes = benchmarkCombinedNodesRef.current;

            // For zero-shot finalized-only view: use empty set for streamedIds
            const finalizedViewStreamedIds = isZeroShot ? new Set<string>() : streamedIds;

            // Compute final comparison with markers
            const {
              nodes: finalCombinedNodes,
              missedEdgeMarkers,
              overshootMarkers,
              traversedSet,
            } = computeFinalizedComparison(
              currentCombinedNodes,
              benchmarkExpectedGraph.edges,
              finalizedViewStreamedIds,
              finalNodes,
              benchmarkExpectedCodes,
              latestTraversedEdges,
              { finalizedOnlyMode: isZeroShot }
            );

            // Compute metrics
            const expectedAncestorMap = buildAncestorMap(benchmarkExpectedGraph.edges);
            const traversedAncestorMap = buildAncestorMap(latestTraversedEdges);

            const { outcomes, otherCodes } = compareFinalizedCodes(
              benchmarkExpectedCodes,
              expectedAncestorMap,
              finalNodes,
              traversedAncestorMap,
              benchmarkExpectedGraph.edges
            );

            const expectedNodeIds = new Set(
              benchmarkExpectedGraph.nodes.map(n => n.id).filter(id => id !== 'ROOT')
            );
            const traversedNodeIdsExcludingRoot = new Set(
              [...traversedSet].filter(id => id !== 'ROOT')
            );

            const metrics = computeBenchmarkMetrics(
              benchmarkExpectedCodes,
              finalNodes,
              expectedNodeIds,
              traversedNodeIdsExcludingRoot,
              outcomes,
              otherCodes
            );

            // Prepare finalized view cache
            const finalizedView = {
              nodes: finalCombinedNodes,
              edges: benchmarkExpectedGraph.edges,
              metrics: metrics,
              overshootMarkers: overshootMarkers,
            };

            // Update ref before state
            benchmarkCombinedNodesRef.current = finalCombinedNodes;

            // Apply ALL state updates in a single batch (no nested callbacks)
            setBenchmarkCombinedNodes(finalCombinedNodes);
            setBenchmarkMissedEdgeMarkers(missedEdgeMarkers);
            setBenchmarkOvershootMarkers(overshootMarkers);
            setBenchmarkMetrics(metrics);
            setBenchmarkFinalizedView(finalizedView);

            // For zero-shot mode: build inferred view AFTER main state updates
            console.log('[BenchmarkInferredView] Checking conditions:', { isZeroShot, finalNodesCount: finalNodesRaw.length });
            if (isZeroShot && finalNodesRaw.length > 0) {
              // Use setTimeout to ensure main render completes first
              console.log('[BenchmarkInferredView] Building inferred view for codes:', finalNodesRaw);
              setTimeout(async () => {
                try {
                  const inferredGraph = await buildGraph(finalNodesRaw);

                  // Log any invalid codes that were filtered out by the server
                  if (inferredGraph.invalid_codes && inferredGraph.invalid_codes.length > 0) {
                    console.log('[BenchmarkInferredView] Filtered out invalid codes:', inferredGraph.invalid_codes);
                  }

                  // If no valid nodes were returned, skip building the inferred view
                  if (inferredGraph.nodes.length === 0) {
                    console.log('[BenchmarkInferredView] No valid codes - skipping inferred view');
                    return;
                  }

                  const inferredTraversedIds = new Set(inferredGraph.nodes.map(n => n.id));

                  const initializedExpectedNodes: BenchmarkGraphNode[] = benchmarkExpectedGraph.nodes.map(node => ({
                    ...node,
                    benchmarkStatus: 'expected' as const,
                  }));

                  const {
                    nodes: inferredViewNodes,
                    missedEdgeMarkers: inferredMissedMarkers,
                    overshootMarkers: inferredOvershootMarkers,
                    traversedSet: inferredTraversedSet,
                  } = computeFinalizedComparison(
                    initializedExpectedNodes,
                    benchmarkExpectedGraph.edges,
                    inferredTraversedIds,
                    finalNodes,
                    benchmarkExpectedCodes,
                    inferredGraph.edges
                  );

                  const inferredAncestorMap = buildAncestorMap(inferredGraph.edges);
                  const { outcomes: inferredOutcomes, otherCodes: inferredOtherCodes } = compareFinalizedCodes(
                    benchmarkExpectedCodes,
                    expectedAncestorMap,
                    finalNodes,
                    inferredAncestorMap,
                    benchmarkExpectedGraph.edges
                  );

                  const expectedNodeIdsForMetrics = new Set(
                    benchmarkExpectedGraph.nodes.map(n => n.id).filter(id => id !== 'ROOT')
                  );
                  const inferredTraversedNodeIds = new Set(
                    [...inferredTraversedSet].filter(id => id !== 'ROOT')
                  );

                  const inferredMetrics = {
                    ...computeBenchmarkMetrics(
                      benchmarkExpectedCodes,
                      finalNodes,
                      expectedNodeIdsForMetrics,
                      inferredTraversedNodeIds,
                      inferredOutcomes,
                      inferredOtherCodes
                    ),
                    expectedCount: expectedNodeIdsForMetrics.size,
                    traversedCount: inferredTraversedNodeIds.size,
                  };

                  // Create BenchmarkGraphNode[] from inferred graph with proper status
                  // All nodes in inferred graph are "traversed" by definition
                  // Nodes matching expected codes are "matched"
                  const inferredGraphNodesWithStatus: BenchmarkGraphNode[] = inferredGraph.nodes.map(node => {
                    const nodeCode = node.code || node.id;
                    const isFinalized = finalNodes.has(nodeCode);
                    const isExpected = benchmarkExpectedCodes.has(nodeCode);

                    let benchmarkStatus: 'expected' | 'traversed' | 'matched';
                    if (isFinalized && isExpected) {
                      benchmarkStatus = 'matched';
                    } else {
                      // All nodes in inferred graph are part of the traversal path
                      benchmarkStatus = 'traversed';
                    }

                    return {
                      ...node,
                      benchmarkStatus,
                    };
                  });

                  console.log('[BenchmarkInferredView] Built successfully:', {
                    inferredViewNodesCount: inferredViewNodes.length,
                    inferredGraphNodesCount: inferredGraph.nodes.length,
                  });
                  setBenchmarkInferredView({
                    nodes: inferredViewNodes,
                    edges: inferredGraph.edges,  // Use inferred edges for Graph view
                    graphNodes: inferredGraphNodesWithStatus,  // Inferred nodes with benchmark status
                    metrics: inferredMetrics,
                    overshootMarkers: inferredOvershootMarkers,
                    missedEdgeMarkers: inferredMissedMarkers,
                    inferredNodes: inferredGraph.nodes,
                  });
                } catch (err) {
                  console.error('[BenchmarkInferredView] Failed to build:', err);
                }
              }, 0);
            }
          }

          // Final visual sync - ensure all streaming IDs are reflected
          setStreamingTraversedIds(new Set(benchmarkStreamedIdsRef.current));

          setBenchmarkStatus('complete');
          setBenchmarkCurrentStep(`Complete - ${finalNodes.size} codes finalized`);
          // Trigger fit-to-window after traversal completes
          setBenchmarkFitTrigger(prev => prev + 1);
        }
        break;
    }
  // Note: benchmarkTraversedNodes/Edges are accessed via refs (benchmarkTraversedNodesRef/EdgesRef)
  // to avoid stale closures, so they're not needed in the dependency array
  }, [benchmarkExpectedGraph, benchmarkExpectedCodes, llmConfig]);

  const handleBenchmarkError = useCallback((error: Error) => {
    setBenchmarkStatus('error');
    setBenchmarkError(error.message);
  }, []);

  const handleBenchmarkRun = useCallback((): boolean => {
    if (!benchmarkClinicalNote.trim()) return false;
    if (!llmConfig.apiKey) {
      alert('Please configure your API key in LLM Settings');
      setBenchmarkSidebarTab('llm-settings');
      return false;
    }

    // Cancel any existing benchmark stream
    if (benchmarkControllerRef.current) {
      benchmarkControllerRef.current.abort();
      benchmarkControllerRef.current = null;
    }

    // Also cancel any pending traverse stream to prevent cross-tab interference
    // This handles the case where user switched from Traverse tab with pending request
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
      setIsLoading(false);
    }

    // NOTE: Cross-run caching is now handled by the backend (Burr + SQLite)
    // The frontend always sends the request; backend returns cached results if available

    // Reset traversal state but keep expected graph
    setBenchmarkTraversedNodes([]);
    setBenchmarkTraversedEdges([]);
    setBenchmarkDecisions([]);
    setBenchmarkBatchCount(0);
    // Reset all refs used during streaming
    benchmarkTraversedNodesRef.current = [];
    benchmarkTraversedEdgesRef.current = [];
    benchmarkDecisionsRef.current = [];
    benchmarkBatchCountRef.current = 0;
    benchmarkStreamedIdsRef.current = new Set();
    streamingTraversedIdsRef.current = new Set();
    lastVisualUpdateTimeRef.current = 0;
    // Reset Map refs used for O(1) streaming updates
    benchmarkNodesMapRef.current = new Map();
    benchmarkEdgesMapRef.current = new Map();
    setStreamingTraversedIds(new Set());
    setBenchmarkOvershootMarkers([]);
    setBenchmarkMissedEdgeMarkers([]);
    setBenchmarkMetrics(null);
    setBenchmarkStatus('traversing');
    setBenchmarkCurrentStep('Starting benchmark traversal');
    setBenchmarkError(null);
    // Reset cached views for new benchmark (toggle state persists - user preference)
    setBenchmarkFinalizedView(null);
    setBenchmarkInferredView(null);
    // Reset cached replay flag
    benchmarkIsCachedReplayRef.current = false;

    // Reset graph to idle state first (plain black nodes)
    // This ensures a clean visual slate before traversal colors nodes
    if (benchmarkCombinedNodes.length > 0) {
      const idleNodes = resetNodesToIdle(benchmarkCombinedNodes);
      benchmarkCombinedNodesRef.current = idleNodes;
      setBenchmarkCombinedNodes(idleNodes);
    }

    benchmarkControllerRef.current = streamTraversal(
      {
        clinical_note: benchmarkClinicalNote,
        provider: llmConfig.provider,
        api_key: llmConfig.apiKey,
        model: llmConfig.model || undefined,
        selector: 'llm',
        max_tokens: llmConfig.maxTokens,
        temperature: llmConfig.temperature,
        extra: llmConfig.extra,
        system_prompt: llmConfig.systemPrompt || undefined,
        scaffolded: llmConfig.scaffolded ?? true,
      },
      handleBenchmarkEvent,
      handleBenchmarkError
    );
    return true;
  }, [benchmarkClinicalNote, llmConfig, benchmarkCombinedNodes, handleBenchmarkEvent, handleBenchmarkError]);

  const handleBenchmarkCancel = useCallback(() => {
    if (benchmarkControllerRef.current) {
      benchmarkControllerRef.current.abort();
      benchmarkControllerRef.current = null;
    }
    setBenchmarkStatus('idle');
    setBenchmarkCurrentStep('Cancelled');
  }, []);

  const handleBenchmarkReset = useCallback(() => {
    // Cancel any running traversal
    if (benchmarkControllerRef.current) {
      benchmarkControllerRef.current.abort();
      benchmarkControllerRef.current = null;
    }

    // Reset all benchmark state
    setBenchmarkExpectedCodes(new Set());
    setBenchmarkCodeInput('');
    setBenchmarkExpectedGraph(null);
    benchmarkExpectedNodeIdsRef.current = new Set(); // Reset cached expectedNodeIds
    setBenchmarkTraversedNodes([]);
    setBenchmarkTraversedEdges([]);
    benchmarkTraversedNodesRef.current = [];
    benchmarkTraversedEdgesRef.current = [];
    benchmarkDecisionsRef.current = [];
    benchmarkBatchCountRef.current = 0;
    benchmarkStreamedIdsRef.current = new Set();
    streamingTraversedIdsRef.current = new Set();
    lastVisualUpdateTimeRef.current = 0;
    // Reset Map refs used for O(1) streaming updates
    benchmarkNodesMapRef.current = new Map();
    benchmarkEdgesMapRef.current = new Map();
    setStreamingTraversedIds(new Set());
    benchmarkCombinedNodesRef.current = [];
    setBenchmarkCombinedNodes([]);
    setBenchmarkCombinedEdges([]);
    setBenchmarkOvershootMarkers([]);
    setBenchmarkMissedEdgeMarkers([]);
    setBenchmarkMetrics(null);
    setBenchmarkDecisions([]);
    setBenchmarkStatus('idle');
    setBenchmarkCurrentStep('');
    setBenchmarkError(null);
    setBenchmarkClinicalNote('');
    setBenchmarkBatchCount(0);
    setBenchmarkInvalidCodes(new Set());
    // Reset cached views, toggle, and tracking refs
    setBenchmarkFinalizedView(null);
    setBenchmarkInferredView(null);
    setBenchmarkInferPrecursors(false);
    benchmarkIsCachedReplayRef.current = false;
    // Cancel any pending RAF
    if (benchmarkResetRafRef.current !== null) {
      cancelAnimationFrame(benchmarkResetRafRef.current);
      benchmarkResetRafRef.current = null;
    }
  }, []);

  return (
    <div className="app">
      <aside
        className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}
        onClick={sidebarCollapsed ? () => setSidebarCollapsed(false) : undefined}
        style={sidebarCollapsed ? { cursor: 'pointer' } : undefined}
      >
        <div
          className="sidebar-header"
          onClick={(e) => {
            e.stopPropagation();
            setSidebarCollapsed(!sidebarCollapsed);
          }}
          style={{ cursor: 'pointer' }}
        >
          <h1>Test for Medical Stepwise Predictions</h1>
          <span
            className="sidebar-toggle-btn"
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <span className="chevron-icon">{sidebarCollapsed ? '\u00BB' : '\u00AB'}</span>
          </span>
        </div>

        {/* VISUALIZE Sidebar */}
        {activeFeatureTab === 'visualize' && !sidebarCollapsed && (
          <>
            <div className="sidebar-tab-content">
              {/* Code Input */}
              <div className="input-group">
                <label className="input-label">Add ICD-10-CM Codes</label>
                <div className="input-row">
                  <textarea
                    value={codeInput}
                    onChange={(e) => setCodeInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleAddCode();
                      }
                    }}
                    placeholder="e.g., I25.10, E11.9"
                    className="code-input"
                    rows={1}
                  />
                  <button
                    onClick={handleAddCode}
                    disabled={!codeInput.trim() || isLoadingGraph}
                    className="add-btn"
                  >
                    {isLoadingGraph ? '...' : 'Add'}
                  </button>
                </div>
              </div>

              {/* Input Codes List */}
              <div className="input-group flex-grow">
                <div className="input-label-row">
                  <label className="input-label">Input Codes{inputCodes.size > 0 ? ` (${inputCodes.size})` : ''}</label>
                  {inputCodes.size > 0 && (
                    <button onClick={handleClearCodes} className="clear-btn">
                      Clear
                    </button>
                  )}
                </div>
                <div
                  className="input-codes-table"
                  onClick={(e) => {
                    // Focus textarea when clicking on table background (not on code rows)
                    if (e.target === e.currentTarget || (e.target as HTMLElement).classList.contains('empty-hint')) {
                      const textarea = document.querySelector<HTMLTextAreaElement>('textarea.code-input');
                      textarea?.focus();
                    }
                  }}
                >
                  {inputCodes.size === 0 ? (
                    <span className="empty-hint">No codes added yet</span>
                  ) : (
                    <table>
                      <tbody>
                        {[...inputCodes].sort().map(code => {
                          // For 7th char codes (7 chars after removing dot), combine ancestor label with sevenChrDef value
                          const codeLen = code.replace(/\./g, '').length;
                          let displayLabel = '';
                          if (graphData) {
                            if (codeLen === 7) {
                              // Get the 7th char node's label (e.g., "D: subsequent encounter")
                              const node = graphData.nodes.find(n => n.code === code);
                              const nodeLabel = node?.label || '';
                              // Extract value part from "Key: Value" format
                              const labelValue = nodeLabel.includes(': ') ? nodeLabel.split(': ').slice(1).join(': ') : nodeLabel;

                              // Traverse up hierarchy until finding non-placeholder node
                              let ancestorLabel = '';
                              let ancestorCode = code.slice(0, -1); // Start with parent
                              while (ancestorCode.length > 0) {
                                const ancestorNode = graphData.nodes.find(n => n.code === ancestorCode);
                                if (ancestorNode && ancestorNode.category !== 'placeholder') {
                                  ancestorLabel = ancestorNode.label;
                                  break;
                                }
                                // Go up one more level (handle dot position)
                                ancestorCode = ancestorCode.slice(0, -1);
                                if (ancestorCode.endsWith('.')) {
                                  ancestorCode = ancestorCode.slice(0, -1);
                                }
                              }

                              // Combine ancestor label with sevenChrDef value (matching GraphViewer hover behavior)
                              if (ancestorLabel && labelValue) {
                                displayLabel = `${ancestorLabel}, ${labelValue}`;
                              } else if (ancestorLabel) {
                                displayLabel = ancestorLabel;
                              } else {
                                displayLabel = nodeLabel;
                              }
                            } else {
                              const node = graphData.nodes.find(n => n.code === code);
                              displayLabel = node?.label || '';
                            }
                          }
                          return (
                            <tr key={code} className="code-row" onClick={() => handleRemoveCode(code)}>
                              <td className="code-cell">
                                <span className="code-badge">{code}</span>
                              </td>
                              <td className="label-cell">{displayLabel}</td>
                              <td className="remove-cell">×</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>

          </>
        )}

        {/* TRAVERSE Sidebar */}
        {activeFeatureTab === 'traverse' && !sidebarCollapsed && (
          <>
            {/* Sidebar Tabs */}
            <div className="sidebar-tabs">
              <button
                className={`sidebar-tab-btn ${sidebarTab === 'clinical-note' ? 'active' : ''}`}
                onClick={() => setSidebarTab('clinical-note')}
              >
                Clinical Note
              </button>
              <button
                className={`sidebar-tab-btn ${sidebarTab === 'llm-settings' ? 'active' : ''}`}
                onClick={() => setSidebarTab('llm-settings')}
              >
                LLM Settings
              </button>
            </div>

            {/* Clinical Note Tab */}
            {sidebarTab === 'clinical-note' && (
              <div className="sidebar-tab-content">
                <textarea
                  key="traverse-clinical-note"
                  value={clinicalNote}
                  onChange={(e) => {
                    const newValue = e.target.value;
                    // Defensive check: reject if value matches system prompts
                    if (newValue === LLM_SYSTEM_PROMPT || newValue === LLM_SYSTEM_PROMPT_NON_SCAFFOLDED) {
                      console.error('[BUG DETECTED] Clinical note onChange received system prompt value!');
                      console.error('[BUG] Event type:', e.type, 'isTrusted:', e.isTrusted);
                      console.error('[BUG] Target id:', e.target.id, 'className:', e.target.className);
                      console.error('[BUG] Stack trace:', new Error().stack);
                      return; // Do NOT update state with system prompt
                    }
                    setClinicalNote(newValue);
                  }}
                  onPaste={(e) => {
                    const pastedText = e.clipboardData.getData('text');
                    console.log('[DEBUG paste] length:', pastedText.length, 'preview:', pastedText.substring(0, 50));
                    if (pastedText === LLM_SYSTEM_PROMPT || pastedText === LLM_SYSTEM_PROMPT_NON_SCAFFOLDED) {
                      console.error('[BUG DETECTED] Clipboard contains system prompt!');
                    }
                  }}
                  placeholder="Paste or type a clinical note..."
                  disabled={isLoading}
                />
              </div>
            )}

            {/* LLM Settings Tab */}
            {sidebarTab === 'llm-settings' && (
              <div className="sidebar-tab-content">
                <LLMSettingsPanel
                  config={llmConfig}
                  onChange={setLlmConfig}
                  disabled={isLoading}
                />
              </div>
            )}

            {/* Action Buttons */}
            <div className="sidebar-actions">
              <button
                onClick={() => {
                  if (handleTraverse()) {
                    setSidebarCollapsed(true);
                    setTraverseFitTrigger(prev => prev + 1);
                  }
                }}
                disabled={isLoading || !clinicalNote.trim()}
                className="primary-btn"
              >
                {isLoading ? `Traversing... (${batchCount})` : 'Start Traversal'}
              </button>
              {isLoading && (
                <button onClick={handleCancel} className="cancel-btn">
                  Cancel
                </button>
              )}
            </div>
          </>
        )}

        {/* BENCHMARK Sidebar */}
        {activeFeatureTab === 'benchmark' && !sidebarCollapsed && (
          <div className="benchmark-sidebar">
            {/* Expected Codes Input - compact at top */}
            <div className="input-group benchmark-codes-input">
              <label className="input-label">Add Expected Codes</label>
              <div className="code-input-row">
                <textarea
                  value={benchmarkCodeInput}
                  onChange={(e) => setBenchmarkCodeInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleBenchmarkAddCode();
                    }
                  }}
                  placeholder="e.g., I25.10, E11.9"
                  className="code-input"
                  rows={1}
                  disabled={benchmarkStatus === 'traversing'}
                />
                <button
                  onClick={handleBenchmarkAddCode}
                  disabled={!benchmarkCodeInput.trim() || benchmarkStatus === 'traversing'}
                  className={`benchmark-action-btn${benchmarkCodeInput.trim() ? ' has-input' : ''}`}
                >
                  Add
                </button>
              </div>
            </div>

            {/* Sidebar Tabs - always visible */}
            <div className="sidebar-tabs">
              <button
                className={`sidebar-tab-btn ${benchmarkSidebarTab === 'clinical-note' ? 'active' : ''}`}
                onClick={() => setBenchmarkSidebarTab('clinical-note')}
              >
                Clinical Note
              </button>
              <button
                className={`sidebar-tab-btn ${benchmarkSidebarTab === 'llm-settings' ? 'active' : ''}`}
                onClick={() => setBenchmarkSidebarTab('llm-settings')}
              >
                LLM Settings
              </button>
            </div>

            {/* Content area - flex-grow to fill available space */}
            <div className="sidebar-tab-content benchmark-content">
              {/* Clinical Note Tab */}
              {benchmarkSidebarTab === 'clinical-note' && (
                <div className="input-group flex-grow">
                  <textarea
                    key="benchmark-clinical-note"
                    value={benchmarkClinicalNote}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      // Defensive check: reject if value matches system prompts
                      if (newValue === LLM_SYSTEM_PROMPT || newValue === LLM_SYSTEM_PROMPT_NON_SCAFFOLDED) {
                        console.error('[BUG DETECTED] Benchmark clinical note onChange received system prompt value!');
                        return;
                      }
                      setBenchmarkClinicalNote(newValue);
                    }}
                    placeholder="Paste or type a clinical note to benchmark against expected codes..."
                    disabled={benchmarkStatus === 'traversing'}
                    className="clinical-note-input"
                  />
                </div>
              )}

              {/* LLM Settings Tab */}
              {benchmarkSidebarTab === 'llm-settings' && (
                <LLMSettingsPanel
                  config={llmConfig}
                  onChange={setLlmConfig}
                  disabled={benchmarkStatus === 'traversing'}
                  benchmarkMode={true}
                  benchmarkInferPrecursors={benchmarkInferPrecursors}
                  onBenchmarkInferPrecursorsChange={setBenchmarkInferPrecursors}
                  benchmarkComplete={benchmarkStatus === 'complete'}
                />
              )}
            </div>

            {/* Action Buttons - always at bottom */}
            <div className="benchmark-bottom-actions">
              <div className="benchmark-action-row">
                <button
                  onClick={() => {
                    if (handleBenchmarkRun()) {
                      setSidebarCollapsed(true);
                      setBenchmarkFitTrigger(prev => prev + 1);
                    }
                  }}
                  disabled={benchmarkStatus === 'traversing' || !benchmarkClinicalNote.trim() || benchmarkExpectedCodes.size === 0}
                  className="primary-btn"
                >
                  {benchmarkStatus === 'traversing' ? `Running... (${benchmarkBatchCount})` : 'Run Benchmark'}
                </button>
                {benchmarkStatus === 'traversing' ? (
                  <button onClick={handleBenchmarkCancel} className="benchmark-action-btn">
                    Cancel
                  </button>
                ) : (
                  <button onClick={handleBenchmarkReset} className="benchmark-action-btn">
                    Reset
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </aside>

      <main className="main-content">
        <div className="main-header">
          <div className="feature-tabs">
            <button
              className={`feature-tab ${activeFeatureTab === 'visualize' ? 'active' : ''}`}
              onClick={() => {
                if (activeFeatureTab === 'visualize') {
                  // If in report view, switch to graph; otherwise toggle sidebar
                  if (visualizeViewTab === 'trajectory') {
                    setVisualizeViewTab('graph');
                  } else {
                    setSidebarCollapsed(prev => !prev);
                  }
                } else {
                  setActiveFeatureTab('visualize');
                  setSidebarCollapsed(false);
                }
              }}
            >
              VISUALIZE
            </button>
            <button
              className={`feature-tab ${activeFeatureTab === 'traverse' ? 'active' : ''}`}
              onClick={() => {
                if (activeFeatureTab === 'traverse') {
                  // If in report view, switch to graph; otherwise toggle sidebar
                  if (traverseViewTab === 'trajectory') {
                    setTraverseViewTab('graph');
                  } else {
                    setSidebarCollapsed(prev => !prev);
                  }
                } else {
                  setActiveFeatureTab('traverse');
                  setSidebarCollapsed(false);
                  setSidebarTab('clinical-note');
                }
              }}
            >
              TRAVERSE
            </button>
            <button
              className={`feature-tab ${activeFeatureTab === 'benchmark' ? 'active' : ''}`}
              onClick={() => {
                if (activeFeatureTab === 'benchmark') {
                  // If in report view, switch to graph; otherwise toggle sidebar
                  if (benchmarkViewTab === 'trajectory') {
                    setBenchmarkViewTab('graph');
                  } else {
                    setSidebarCollapsed(prev => !prev);
                  }
                } else {
                  setActiveFeatureTab('benchmark');
                  setSidebarCollapsed(false);
                  setBenchmarkSidebarTab('clinical-note');
                }
              }}
            >
              BENCHMARK
            </button>
          </div>
          <div className="view-switcher">
            <span className="view-label">View:</span>
            <button
              className={`view-btn ${(activeFeatureTab === 'visualize' ? visualizeViewTab :
                activeFeatureTab === 'traverse' ? traverseViewTab : benchmarkViewTab) === 'graph' ? 'active' : ''
                }`}
              onClick={() => {
                if (activeFeatureTab === 'visualize') setVisualizeViewTab('graph');
                else if (activeFeatureTab === 'traverse') setTraverseViewTab('graph');
                else setBenchmarkViewTab('graph');
              }}
            >
              Graph
            </button>
            <button
              className={`view-btn ${(activeFeatureTab === 'visualize' ? visualizeViewTab :
                activeFeatureTab === 'traverse' ? traverseViewTab : benchmarkViewTab) === 'trajectory' ? 'active' : ''
                }`}
              onClick={() => {
                if (activeFeatureTab === 'visualize') setVisualizeViewTab('trajectory');
                else if (activeFeatureTab === 'traverse') setTraverseViewTab('trajectory');
                else setBenchmarkViewTab('trajectory');
              }}
            >
              Report
            </button>
          </div>
        </div>

        <div className="tab-content">
          {activeFeatureTab === 'visualize' && (
            visualizeViewTab === 'graph' ? (
              <GraphViewer
                nodes={graphData?.nodes ?? []}
                edges={graphData?.edges ?? []}
                selectedNode={selectedNode}
                onNodeClick={handleNodeClick}
                finalizedCodes={[...inputCodes]}
                isTraversing={isLoadingGraph}
                status={isLoadingGraph ? 'traversing' : graphData ? 'complete' : 'idle'}
                errorMessage={graphError}
                codesBarLabel="Submitted Codes"
                triggerFitToWindow={visualizeFitTrigger}
              />
            ) : (
              <VisualizeReportViewer
                nodes={graphData?.nodes ?? []}
                edges={graphData?.edges ?? []}
                inputCodes={inputCodes}
              />
            )
          )}
          {activeFeatureTab === 'traverse' && (
            traverseViewTab === 'graph' ? (
              // Use visualization graph when zero-shot + visualizePrediction is ON
              (() => {
                const isZeroShot = !(llmConfig.scaffolded ?? true);
                const vizEnabled = llmConfig.visualizePrediction ?? false;
                const hasVizData = zeroShotVisualization !== null;
                const useVisualization = isZeroShot && vizEnabled && hasVizData;

                console.log('[TraverseGraphViewer] Render:', {
                  isZeroShot,
                  vizEnabled,
                  hasVizData,
                  useVisualization,
                  vizNodes: zeroShotVisualization?.nodes.length ?? 0,
                  stateNodes: state.nodes.length,
                });

                return (
                  <GraphViewer
                    nodes={useVisualization ? zeroShotVisualization.nodes : state.nodes}
                    edges={useVisualization ? zeroShotVisualization.edges : state.edges}
                    selectedNode={selectedNode}
                    onNodeClick={handleNodeClick}
                    finalizedCodes={state.finalized_codes}
                    isTraversing={state.status === 'traversing' || isLoadingZeroShotViz}
                    currentStep={isLoadingZeroShotViz ? 'Building visualization...' : state.current_step}
                    decisionCount={state.decision_history.length}
                    status={isLoadingZeroShotViz ? 'traversing' : state.status}
                    errorMessage={state.error}
                    decisions={state.decision_history}
                    codesBarLabel="Extracted Codes"
                    elapsedTime={traverseElapsedTime}
                    onNodeRewindClick={handleNodeRewindClick}
                    allowRewind={state.status !== 'idle' && state.nodes.length > 1}
                    rewindingNodeId={rewindingNodeId}
                    triggerFitToWindow={traverseFitTrigger}
                    onGraphInteraction={handleTraverseGraphInteraction}
                  />
                );
              })()
            ) : (
              <TrajectoryViewer
                decisions={state.decision_history}
                finalizedCodes={state.finalized_codes}
                status={state.status}
                currentStep={state.current_step}
                errorMessage={state.error}
              />
            )
          )}
          {activeFeatureTab === 'benchmark' && (
            benchmarkViewTab === 'graph' ? (
              // Select view based on toggle: inferred (with X markers) or finalized (no X markers)
              (() => {
                const isZeroShot = !(llmConfig.scaffolded ?? true);
                const isComplete = benchmarkStatus === 'complete';
                const useInferredView = isZeroShot && isComplete && benchmarkInferPrecursors && benchmarkInferredView !== null;
                const useFinalizedView = isZeroShot && isComplete && !benchmarkInferPrecursors && benchmarkFinalizedView !== null;

                console.log('[BenchmarkGraphViewer] Render:', {
                  isZeroShot,
                  isComplete,
                  benchmarkInferPrecursors,
                  hasInferredView: benchmarkInferredView !== null,
                  hasFinalizedView: benchmarkFinalizedView !== null,
                  useInferredView,
                  useFinalizedView,
                });

                // Get active view data
                const activeView = useInferredView
                  ? benchmarkInferredView
                  : useFinalizedView
                    ? benchmarkFinalizedView
                    : null;

                // When inferred view is active, use the comparison-based nodes (expected graph with
                // benchmark status based on inferred traversal coverage) and expected graph edges.
                // This shows which expected nodes were/weren't covered by the inferred traversal.
                // For finalized view, use expected graph structure with benchmark status highlighting.
                const activeNodes = useInferredView
                  ? (benchmarkInferredView?.nodes ?? benchmarkCombinedNodes)
                  : (activeView?.nodes ?? benchmarkCombinedNodes);
                const activeEdges = useInferredView
                  ? (benchmarkExpectedGraph?.edges ?? benchmarkCombinedEdges)
                  : (activeView?.edges ?? benchmarkCombinedEdges);
                // Ensure metrics are available: use inferred view metrics, then finalized view metrics, then state
                const activeMetrics = useInferredView
                  ? (benchmarkInferredView?.metrics ?? benchmarkFinalizedView?.metrics ?? benchmarkMetrics)
                  : (activeView?.metrics ?? benchmarkMetrics);
                const activeOvershootMarkers = activeView?.overshootMarkers ?? benchmarkOvershootMarkers;
                const activeMissedEdgeMarkers = useInferredView && benchmarkInferredView
                  ? benchmarkInferredView.missedEdgeMarkers
                  : benchmarkMissedEdgeMarkers;

                // Show X markers when:
                // - Zero-shot mode with infer precursors enabled, OR
                // - Scaffolded mode (always has full traversal path)
                const showXMarkers = !isZeroShot || benchmarkInferPrecursors;

                return (
                  <GraphViewer
                    nodes={activeNodes}
                    edges={activeEdges}
                    selectedNode={selectedNode}
                    onNodeClick={handleNodeClick}
                    finalizedCodes={
                      isComplete && activeMetrics
                        ? activeMetrics.outcomes
                          .filter(o => o.status === 'exact')
                          .map(o => o.expectedCode)
                        : benchmarkFinalizedCodesArray
                    }
                    isTraversing={benchmarkStatus === 'traversing'}
                    currentStep={benchmarkCurrentStep}
                    decisionCount={benchmarkDecisions.length}
                    status={benchmarkStatus}
                    errorMessage={benchmarkError}
                    decisions={benchmarkDecisions}
                    benchmarkMode={true}
                    benchmarkMetrics={activeMetrics}
                    overshootMarkers={activeOvershootMarkers}
                    missedEdgeMarkers={activeMissedEdgeMarkers}
                    expectedLeaves={benchmarkExpectedCodes}
                    onRemoveExpectedCode={isComplete ? undefined : handleBenchmarkRemoveCode}
                    invalidCodes={benchmarkInvalidCodes}
                    codesBarLabel={isComplete ? 'Matched Final Codes' : 'Target Final Codes'}
                    elapsedTime={benchmarkElapsedTime}
                    triggerFitToWindow={benchmarkFitTrigger}
                    onGraphInteraction={handleBenchmarkGraphInteraction}
                    showXMarkers={showXMarkers}
                    streamingTraversedIds={benchmarkStatus === 'traversing' ? streamingTraversedIds : undefined}
                  />
                );
              })()
            ) : (
              // Report view also uses active metrics based on toggle
              (() => {
                const isZeroShot = !(llmConfig.scaffolded ?? true);
                const isComplete = benchmarkStatus === 'complete';
                const useInferredView = isZeroShot && isComplete && benchmarkInferPrecursors && benchmarkInferredView !== null;
                const useFinalizedView = isZeroShot && isComplete && !benchmarkInferPrecursors && benchmarkFinalizedView !== null;

                // Ensure metrics are available with fallback chain
                const activeMetrics = useInferredView
                  ? (benchmarkInferredView?.metrics ?? benchmarkFinalizedView?.metrics ?? benchmarkMetrics)
                  : useFinalizedView
                    ? (benchmarkFinalizedView?.metrics ?? benchmarkMetrics)
                    : benchmarkMetrics;

                // For zero-shot mode, use cached view nodes for correct interim computation
                // Finalized-only: empty arrays = no interim nodes shown at all
                // Inferred: use cached view nodes for interim display
                const activeCombinedNodes = useInferredView
                  ? benchmarkInferredView?.nodes ?? []
                  : useFinalizedView
                    ? []  // No interim nodes in finalized-only view
                    : benchmarkCombinedNodes;

                const activeTraversedNodes = useInferredView
                  ? benchmarkInferredView?.inferredNodes ?? []
                  : useFinalizedView
                    ? []  // No interim nodes in finalized-only view
                    : benchmarkTraversedNodes;

                return (
                  <BenchmarkReportViewer
                    metrics={activeMetrics}
                    decisions={benchmarkDecisions}
                    status={benchmarkStatus}
                    currentStep={benchmarkCurrentStep}
                    errorMessage={benchmarkError}
                    onCodeClick={handleNodeClick}
                    expectedGraph={benchmarkExpectedGraph}
                    expectedCodes={benchmarkExpectedCodes}
                    combinedNodes={activeCombinedNodes}
                    traversedNodes={activeTraversedNodes}
                    elapsedTime={benchmarkElapsedTime}
                    hideOvershootUndershoot={useFinalizedView}
                  />
                );
              })()
            )
          )}
        </div>
      </main>

      {/* Rewind Modal */}
      <NodeRewindModal
        node={rewindTargetNode}
        decision={rewindTargetDecision}
        isOpen={isRewindModalOpen}
        onClose={handleRewindClose}
        onSubmit={handleRewindSubmit}
        isSubmitting={isRewinding}
        error={rewindError}
        initialFeedback={rewindFeedbackText}
      />
    </div>
  );
}

function App() {
  return <TraversalUI />;
}

export default App;