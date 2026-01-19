import { useState, useCallback, useRef, useEffect } from 'react';
import { applyPatch, type Operation } from 'fast-json-patch';
import { GraphViewer } from './components/GraphViewer';
import { TrajectoryViewer } from './components/TrajectoryViewer';
import { BenchmarkReportViewer } from './components/BenchmarkReportViewer';
import { VisualizeReportViewer } from './components/VisualizeReportViewer';
import { LLMSettingsPanel, NodeRewindModal } from './components/shared';
import { streamTraversal, streamRewind, buildGraph, type AGUIEvent } from './lib/api';
import { getDescendantNodeIds, extractBatchType } from './lib/graphUtils';
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
  updateTraversedNodes,
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

  // Zero-shot visualization graph (for Traverse tab when visualizePrediction is ON)
  const [zeroShotVisualization, setZeroShotVisualization] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [isLoadingZeroShotViz, setIsLoadingZeroShotViz] = useState(false);

  // BENCHMARK tab state
  const [benchmarkExpectedCodes, setBenchmarkExpectedCodes] = useState<Set<string>>(new Set());
  const [benchmarkCodeInput, setBenchmarkCodeInput] = useState('');
  const [benchmarkExpectedGraph, setBenchmarkExpectedGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [benchmarkTraversedNodes, setBenchmarkTraversedNodes] = useState<GraphNode[]>([]);
  const [benchmarkTraversedEdges, setBenchmarkTraversedEdges] = useState<GraphEdge[]>([]);
  // Refs to track latest values for use in async callbacks (avoid stale closures)
  const benchmarkTraversedNodesRef = useRef<GraphNode[]>([]);
  const benchmarkTraversedEdgesRef = useRef<GraphEdge[]>([]);
  // Phase 1: Track all selected_ids during streaming for Phase 2 marker computation
  const benchmarkStreamedIdsRef = useRef<Set<string>>(new Set());
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
    metrics: BenchmarkMetrics;
    overshootMarkers: OvershootMarker[];
    missedEdgeMarkers: EdgeMissMarker[];
    inferredNodes: GraphNode[];  // Raw inferred graph nodes for Report interim computation
  } | null>(null);

  // NOTE: Cross-run caching for zero-shot is now handled by the backend (Burr + SQLite)
  // The backend caches LLM responses based on (clinical_note, provider, model, temperature, system_prompt)
  // Frontend processes results into comparison views in handleBenchmarkEvent RUN_FINISHED handler

  // Rewind state (for spot rewind feature in TRAVERSE tab)
  const [rewindTargetNode, setRewindTargetNode] = useState<GraphNode | null>(null);
  const [rewindTargetBatchId, setRewindTargetBatchId] = useState<string | null>(null);
  const [rewindFeedbackText, setRewindFeedbackText] = useState<string>('');
  const [isRewindModalOpen, setIsRewindModalOpen] = useState(false);
  const [isRewinding, setIsRewinding] = useState(false);
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

  // Handle AG-UI events
  const handleAGUIEvent = useCallback((event: AGUIEvent) => {
    switch (event.type) {
      case 'RUN_STARTED':
        setState(prev => ({
          ...prev,
          status: 'traversing',
          current_step: 'Starting traversal',
        }));
        break;

      case 'STATE_SNAPSHOT':
        if (event.state) {
          setState(prev => ({
            ...prev,
            nodes: event.state!.nodes as GraphNode[],
            edges: event.state!.edges as GraphEdge[],
          }));
        }
        break;

      case 'STATE_DELTA':
        if (event.delta) {
          // Log all additions for debugging
          event.delta.forEach((op) => {
            if (op.op === 'add') {
              if (op.path === '/edges/-') {
                console.log('[EDGE ADDED]', op.value);
              } else if (op.path === '/nodes/-') {
                console.log('[NODE ADDED]', op.value);
              }
            }
          });

          setState(prev => {
            // Create mutable copy of graph state for JSON Patch
            const graphState = {
              nodes: [...prev.nodes],
              edges: [...prev.edges],
              finalized: [...prev.finalized_codes],
            };

            try {
              // Apply JSON Patch operations
              const result = applyPatch(graphState, event.delta as Operation[], true, false);
              console.log(`[STATE] After delta: ${result.newDocument.nodes.length} nodes, ${result.newDocument.edges.length} edges`);
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
        setState(prev => ({
          ...prev,
          current_step: event.step_id || '',
        }));
        break;

      case 'STEP_FINISHED':
        setBatchCount(prev => prev + 1);

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

          // Merge graph data and add decision to history
          setState(prev => ({
            ...prev,
            nodes: mergeById(prev.nodes, newNodes),
            edges: mergeByKey(prev.edges, newEdges),
            decision_history: [...prev.decision_history, decision],
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
          setState(prev => ({
            ...prev,
            status: 'complete',
            finalized_codes: finalNodes,
            current_step: `Complete - ${finalNodes.length} codes found`,
          }));
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

    // Cancel any existing stream
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    setIsLoading(true);
    setBatchCount(0);
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
  const handleNodeRewindClick = useCallback((nodeId: string, batchId?: string, feedback?: string) => {
    // Find the node in current state
    const node = state.nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Only allow rewind on non-ROOT nodes during/after traversal
    if (nodeId === 'ROOT' || state.status === 'idle') return;

    setRewindTargetNode(node);
    setRewindTargetBatchId(batchId || null);
    setRewindFeedbackText(feedback || '');
    setIsRewindModalOpen(true);
    setRewindError(null);
  }, [state.nodes, state.status]);

  const handleRewindClose = useCallback(() => {
    if (!isRewinding) {
      setIsRewindModalOpen(false);
      setRewindTargetNode(null);
      setRewindTargetBatchId(null);
      setRewindFeedbackText('');
      setRewindError(null);
    }
  }, [isRewinding]);

  const handleRewindSubmit = useCallback(async (nodeId: string, feedback: string) => {
    setIsRewinding(true);
    setRewindError(null);

    try {
      // Use stored batchId if available, otherwise construct from first matching decision
      let batchId = rewindTargetBatchId;
      if (!batchId) {
        const decision = state.decision_history.find(d => d.current_node === nodeId);
        const batchType = decision ? extractBatchType(decision.current_label) : 'children';
        batchId = `${nodeId}|${batchType}`;
      }

      // Cancel any existing rewind stream
      if (rewindControllerRef.current) {
        rewindControllerRef.current.abort();
      }

      // Remove descendant nodes immediately (optimistic update)
      const descendantIds = getDescendantNodeIds(nodeId, state.edges);

      setState(prev => ({
        ...prev,
        nodes: prev.nodes.filter(n => !descendantIds.has(n.id)),
        edges: prev.edges.filter(e =>
          !descendantIds.has(e.source as string) &&
          !descendantIds.has(e.target as string)
        ),
        finalized_codes: prev.finalized_codes.filter(c => !descendantIds.has(c)),
        decision_history: prev.decision_history.filter(d => !descendantIds.has(d.current_node)),
        status: 'traversing',
        current_step: `Rewinding from ${nodeId}...`,
      }));

      // Close modal
      setIsRewindModalOpen(false);
      setRewindTargetNode(null);
      setRewindTargetBatchId(null);

      // Stream the rewind traversal
      rewindControllerRef.current = streamRewind(
        {
          batch_id: batchId,
          feedback,
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
  }, [state.decision_history, state.edges, llmConfig, handleAGUIEvent]);

  // Effect to reset isRewinding when traversal completes
  useEffect(() => {
    if (isRewinding && (state.status === 'complete' || state.status === 'error')) {
      setIsRewinding(false);
    }
  }, [isRewinding, state.status]);

  // Effect to visualize zero-shot predictions when enabled
  useEffect(() => {
    const shouldVisualize =
      state.status === 'complete' &&
      !(llmConfig.scaffolded ?? true) &&
      (llmConfig.visualizePrediction ?? false) &&
      state.finalized_codes.length > 0;

    if (shouldVisualize) {
      // Build graph from finalized codes for Traverse tab visualization
      (async () => {
        setIsLoadingZeroShotViz(true);
        try {
          const result = await buildGraph(state.finalized_codes);
          setZeroShotVisualization({ nodes: result.nodes, edges: result.edges });
        } catch (err) {
          console.error('Failed to build zero-shot visualization:', err);
          setZeroShotVisualization(null);
        } finally {
          setIsLoadingZeroShotViz(false);
        }
      })();
    } else if (state.status === 'idle' || (llmConfig.scaffolded ?? true)) {
      // Clear visualization when returning to idle or switching to scaffolded mode
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

      // Initialize combined graph with expected nodes (all marked as 'expected')
      const combinedNodes = initializeExpectedNodes(result.nodes);
      setBenchmarkCombinedNodes(combinedNodes);
      setBenchmarkCombinedEdges(result.edges);

      // Clear invalid codes on success
      setBenchmarkInvalidCodes(new Set());
      setBenchmarkStatus('idle');
      setBenchmarkCurrentStep('');
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
            const combinedNodes = initializeExpectedNodes(result.nodes);
            setBenchmarkCombinedNodes(combinedNodes);
            setBenchmarkCombinedEdges(result.edges);
            setBenchmarkStatus('idle');
            setBenchmarkCurrentStep('');
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
        setBenchmarkStatus('traversing');
        setBenchmarkCurrentStep('Starting benchmark traversal');
        break;

      case 'STATE_SNAPSHOT':
        if (event.state) {
          setBenchmarkTraversedNodes(event.state.nodes as GraphNode[]);
          setBenchmarkTraversedEdges(event.state.edges as GraphEdge[]);
        }
        break;

      case 'STATE_DELTA':
        if (event.delta) {
          setBenchmarkTraversedNodes(prev => {
            const graphState = { nodes: [...prev], edges: [] };
            try {
              const result = applyPatch(graphState, event.delta as Operation[], true, false);
              const updated = result.newDocument.nodes;
              // Update ref so benchmark comparison has latest data
              benchmarkTraversedNodesRef.current = updated;
              return updated;
            } catch {
              return prev;
            }
          });
          setBenchmarkTraversedEdges(prev => {
            const graphState = { nodes: [], edges: [...prev] };
            try {
              const result = applyPatch(graphState, event.delta as Operation[], true, false);
              const updated = result.newDocument.edges;
              // Update ref so benchmark comparison has latest data
              benchmarkTraversedEdgesRef.current = updated;
              return updated;
            } catch {
              return prev;
            }
          });
        }
        break;

      case 'STEP_STARTED':
        setBenchmarkCurrentStep(event.step_id || '');
        break;

      case 'STEP_FINISHED':
        setBenchmarkBatchCount(prev => prev + 1);

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

          // Build decision point
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

          // Update traversed state and refs (refs avoid stale closures in RUN_FINISHED)
          setBenchmarkTraversedNodes(prev => {
            const updated = mergeById(prev, newNodes);
            benchmarkTraversedNodesRef.current = updated;
            return updated;
          });
          setBenchmarkTraversedEdges(prev => {
            const updated = mergeByKey(prev, newEdges);
            benchmarkTraversedEdgesRef.current = updated;
            return updated;
          });
          setBenchmarkDecisions(prev => [...prev, decision]);

          // Phase 1a: Accumulate streamed IDs for Phase 2 marker computation
          for (const id of selectedIds) {
            benchmarkStreamedIdsRef.current.add(id);
          }

          // Phase 1b: Update traversed status in real-time (visual feedback)
          // Updates expected nodes that were traversed (nodeId + selected_ids) to 'traversed' status
          // nodeId is included because it represents the node being processed in the current batch
          setBenchmarkCombinedNodes(prevCombined => {
            if (!benchmarkExpectedGraph || prevCombined.length === 0) return prevCombined;

            const expectedNodeIds = new Set(benchmarkExpectedGraph.nodes.map(n => n.id));
            return updateTraversedNodes(prevCombined, selectedIds, expectedNodeIds, nodeId);
          });
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

          // NOTE: Cross-run caching is now handled by the backend (Burr + SQLite)
          // The backend reports cache status in event.metadata.cached
          const wasCached = event.metadata?.cached ?? false;
          if (wasCached) {
            console.log('[BACKEND CACHE HIT] Using cached zero-shot results:', finalNodesRaw.length, 'codes');
          }

          // Phase 2: Compute final statuses and markers
          // Use refs to get latest traversed data (avoid stale closure)
          if (benchmarkExpectedGraph) {
            const latestTraversedEdges = benchmarkTraversedEdgesRef.current;
            const streamedIds = benchmarkStreamedIdsRef.current;

            // For zero-shot finalized-only view: use empty set for streamedIds
            // Zero-shot doesn't actually traverse - intermediate predictions shouldn't affect status
            // Only the actual final codes matter for comparison
            const finalizedViewStreamedIds = isZeroShot ? new Set<string>() : streamedIds;

            // Phase 2: Compute final comparison with markers
            // Uses accumulated streamedIds from Phase 1 + finalizedCodes (scaffolded only)
            setBenchmarkCombinedNodes(prevCombined => {
              const {
                nodes: finalCombinedNodes,
                missedEdgeMarkers,
                overshootMarkers,
                traversedSet,
              } = computeFinalizedComparison(
                prevCombined,
                benchmarkExpectedGraph.edges,
                finalizedViewStreamedIds,
                finalNodes,
                benchmarkExpectedCodes,
                latestTraversedEdges,
                // Zero-shot finalized-only mode: no "traversed" status, only endpoint comparison
                { finalizedOnlyMode: isZeroShot }
              );

              // Set markers (side effect, but needed here for traversedSet)
              setBenchmarkMissedEdgeMarkers(missedEdgeMarkers);
              setBenchmarkOvershootMarkers(overshootMarkers);

              // Compute metrics (separate concern - used for report)
              const expectedAncestorMap = buildAncestorMap(benchmarkExpectedGraph.edges);
              const traversedAncestorMap = buildAncestorMap(latestTraversedEdges);

              const { outcomes, otherCodes } = compareFinalizedCodes(
                benchmarkExpectedCodes,
                expectedAncestorMap,
                finalNodes,
                traversedAncestorMap
              );

              // Exclude ROOT from node counts - it's a virtual node, not a real ICD-10 code
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

              setBenchmarkMetrics(metrics);

              // Cache finalized view (no X markers - overshoot arrows only)
              setBenchmarkFinalizedView({
                nodes: finalCombinedNodes,
                edges: benchmarkExpectedGraph.edges,
                metrics: metrics,
                overshootMarkers: overshootMarkers,
              });

              // For zero-shot mode: build inferred view with X markers
              // The inferred graph represents the conceptual traversal path from final codes
              // We compare this against the UNCHANGED expected graph, same as scaffolded mode
              if (isZeroShot && finalNodesRaw.length > 0) {
                (async () => {
                  try {
                    // Build inferred graph from final codes (represents what LLM "traversed")
                    const inferredGraph = await buildGraph(finalNodesRaw);

                    // Use inferred node IDs as the "traversed" set (simulating scaffolded traversal)
                    const inferredTraversedIds = new Set(inferredGraph.nodes.map(n => n.id));

                    // Initialize expected nodes with 'expected' status
                    const initializedExpectedNodes: BenchmarkGraphNode[] = benchmarkExpectedGraph.nodes.map(node => ({
                      ...node,
                      benchmarkStatus: 'expected' as const,
                    }));

                    // Run the same comparison as scaffolded mode:
                    // Compare inferred traversal against expected graph
                    const {
                      nodes: inferredViewNodes,
                      missedEdgeMarkers: inferredMissedMarkers,
                      overshootMarkers: inferredOvershootMarkers,
                      traversedSet: inferredTraversedSet,
                    } = computeFinalizedComparison(
                      initializedExpectedNodes,
                      benchmarkExpectedGraph.edges,  // Use expected edges (not merged)
                      inferredTraversedIds,          // Inferred nodes as "streamed" traversal
                      finalNodes,                    // Finalized codes
                      benchmarkExpectedCodes,        // Expected leaf codes
                      inferredGraph.edges            // Inferred edges for ancestor expansion
                    );

                    // Compute metrics using expected graph structure
                    const expectedAncestorMap = buildAncestorMap(benchmarkExpectedGraph.edges);
                    const inferredAncestorMap = buildAncestorMap(inferredGraph.edges);

                    const { outcomes: inferredOutcomes, otherCodes: inferredOtherCodes } = compareFinalizedCodes(
                      benchmarkExpectedCodes,
                      expectedAncestorMap,
                      finalNodes,
                      inferredAncestorMap
                    );

                    // Exclude ROOT from node counts
                    const expectedNodeIdsForMetrics = new Set(
                      benchmarkExpectedGraph.nodes.map(n => n.id).filter(id => id !== 'ROOT')
                    );
                    const inferredTraversedNodeIds = new Set(
                      [...inferredTraversedSet].filter(id => id !== 'ROOT')
                    );

                    const baseInferredMetrics = computeBenchmarkMetrics(
                      benchmarkExpectedCodes,
                      finalNodes,
                      expectedNodeIdsForMetrics,
                      inferredTraversedNodeIds,
                      inferredOutcomes,
                      inferredOtherCodes
                    );

                    // Override display counts to show graph node counts (not just leaf codes)
                    // This makes "Expected" show 19 (all expected nodes) instead of 3 (leaf codes)
                    // and "Traversed" show 65 (all inferred nodes) instead of 22 (finalized codes)
                    const inferredMetrics = {
                      ...baseInferredMetrics,
                      expectedCount: expectedNodeIdsForMetrics.size,
                      traversedCount: inferredTraversedNodeIds.size,
                    };

                    // Cache inferred view - uses expected graph structure, not merged
                    setBenchmarkInferredView({
                      nodes: inferredViewNodes,
                      edges: benchmarkExpectedGraph.edges,  // Expected edges unchanged
                      metrics: inferredMetrics,
                      overshootMarkers: inferredOvershootMarkers,
                      missedEdgeMarkers: inferredMissedMarkers,
                      inferredNodes: inferredGraph.nodes,  // Raw inferred nodes for Report interim
                    });
                  } catch (err) {
                    console.error('Failed to build inferred view:', err);
                  }
                })();
              }

              return finalCombinedNodes;
            });
          }

          setBenchmarkStatus('complete');
          setBenchmarkCurrentStep(`Complete - ${finalNodes.size} codes finalized`);
        }
        break;
    }
  }, [benchmarkExpectedGraph, benchmarkExpectedCodes, benchmarkTraversedNodes, benchmarkTraversedEdges, llmConfig]);

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

    // Cancel any existing stream
    if (benchmarkControllerRef.current) {
      benchmarkControllerRef.current.abort();
    }

    // NOTE: Cross-run caching is now handled by the backend (Burr + SQLite)
    // The frontend always sends the request; backend returns cached results if available

    // Reset traversal state but keep expected graph
    setBenchmarkTraversedNodes([]);
    setBenchmarkTraversedEdges([]);
    benchmarkTraversedNodesRef.current = [];
    benchmarkTraversedEdgesRef.current = [];
    benchmarkStreamedIdsRef.current = new Set(); // Reset Phase 1 streamed IDs
    setBenchmarkOvershootMarkers([]);
    setBenchmarkMissedEdgeMarkers([]);
    setBenchmarkDecisions([]);
    setBenchmarkMetrics(null);
    setBenchmarkBatchCount(0);
    setBenchmarkStatus('traversing');
    setBenchmarkCurrentStep('Starting benchmark traversal');
    setBenchmarkError(null);
    // Reset cached views for new benchmark (toggle state persists - user preference)
    setBenchmarkFinalizedView(null);
    setBenchmarkInferredView(null);

    // Reset graph to idle state first (plain black nodes)
    // This ensures a clean visual slate before traversal colors nodes
    if (benchmarkCombinedNodes.length > 0) {
      const idleNodes = resetNodesToIdle(benchmarkCombinedNodes);
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
    setBenchmarkTraversedNodes([]);
    setBenchmarkTraversedEdges([]);
    benchmarkTraversedNodesRef.current = [];
    benchmarkTraversedEdgesRef.current = [];
    benchmarkStreamedIdsRef.current = new Set(); // Reset Phase 1 streamed IDs
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
    // Reset cached views and toggle
    setBenchmarkFinalizedView(null);
    setBenchmarkInferredView(null);
    setBenchmarkInferPrecursors(false);
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
                <div className="input-codes-table">
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
                const useVisualization =
                  !(llmConfig.scaffolded ?? true) &&
                  (llmConfig.visualizePrediction ?? false) &&
                  zeroShotVisualization !== null;

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

                // Get active view data
                const activeView = useInferredView
                  ? benchmarkInferredView
                  : useFinalizedView
                    ? benchmarkFinalizedView
                    : null;

                const activeNodes = activeView?.nodes ?? benchmarkCombinedNodes;
                const activeEdges = activeView?.edges ?? benchmarkCombinedEdges;
                const activeMetrics = activeView?.metrics ?? benchmarkMetrics;
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
                        : [...benchmarkExpectedCodes]
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
                    showXMarkers={showXMarkers}
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

                const activeMetrics = useInferredView
                  ? benchmarkInferredView?.metrics
                  : useFinalizedView
                    ? benchmarkFinalizedView?.metrics
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