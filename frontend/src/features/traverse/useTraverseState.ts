import { useState, useCallback, useRef, useEffect } from 'react';
import { useElapsedTime } from '../../hooks/useElapsedTime';
import { applyPatch, type Operation } from 'fast-json-patch';
import {
  streamTraversal,
  streamRewind,
  type AGUIEvent,
  type StepFinishedMetadata,
  type RunFinishedMetadata,
} from '../../lib/api';
import {
  mergeById,
  mergeByKey,
  calculateDepthFromCode,
  extractBatchType,
} from '../../lib/graphUtils';
import type {
  TraversalState,
  GraphNode,
  GraphEdge,
  DecisionPoint,
  CandidateDecision,
  LLMConfig,
} from '../../lib/types';
import { INITIAL_TRAVERSAL_STATE, type SidebarTab } from '../../lib/constants';
import { cancelTraversal } from '../../lib/api';

interface UseTraverseStateProps {
  llmConfig: LLMConfig;
  setSidebarTab: (tab: SidebarTab) => void;
}

export function useTraverseState({ llmConfig, setSidebarTab }: UseTraverseStateProps) {
  // State
  const [clinicalNote, setClinicalNote] = useState('');
  const [state, setState] = useState<TraversalState>(INITIAL_TRAVERSAL_STATE);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [batchCount, setBatchCount] = useState(0);

  const [traverseFitTrigger, setTraverseFitTrigger] = useState(0);
  const [traverseTimerRunning, setTraverseTimerRunning] = useState(false);
  const [traverseElapsedTime, resetTraverseTimer] = useElapsedTime(traverseTimerRunning);

  // Rewind state
  const [rewindTargetNode, setRewindTargetNode] = useState<GraphNode | null>(null);
  const [rewindTargetBatchId, setRewindTargetBatchId] = useState<string | null>(null);
  const [rewindFeedbackText, setRewindFeedbackText] = useState<string>('');
  const [isRewindModalOpen, setIsRewindModalOpen] = useState(false);
  const [isRewinding, setIsRewinding] = useState(false);
  const [rewindingNodeId, setRewindingNodeId] = useState<string | null>(null);
  const [rewindError, setRewindError] = useState<string | null>(null);

  // Refs
  const controllerRef = useRef<AbortController | null>(null);
  const rewindControllerRef = useRef<AbortController | null>(null);
  const traverseNodesRef = useRef<GraphNode[]>([]);
  const traverseEdgesRef = useRef<GraphEdge[]>([]);
  const traverseDecisionsRef = useRef<DecisionPoint[]>([]);
  const traverseBatchCountRef = useRef<number>(0);
  const traverseLastInteractionRef = useRef<number>(0);
  const traverseHadNodesRef = useRef(false);
  const wasRewindRef = useRef(false);
  const wasZeroShotRef = useRef(false);

  // Fit-to-window logic (skip during rewind)
  useEffect(() => {
    if (state.status === 'traversing' && !wasRewindRef.current) {
      if (state.nodes.length > 0 && !traverseHadNodesRef.current) {
        traverseHadNodesRef.current = true;
        const timer = setTimeout(() => {
          setTraverseFitTrigger(prev => prev + 1);
        }, 350);
        return () => clearTimeout(timer);
      }
    } else if (state.status === 'idle') {
      traverseHadNodesRef.current = false;
    }
  }, [state.status, state.nodes.length]);

  // Periodic fit during traversal, and instant fit at completion (skip during rewind)
  useEffect(() => {
    if (state.status === 'traversing' && !wasRewindRef.current) {
      traverseLastInteractionRef.current = Date.now();
      const interval = setInterval(() => {
        const idleTime = Date.now() - traverseLastInteractionRef.current;
        if (idleTime >= 1000) {
          setTraverseFitTrigger(prev => prev + 1);
          traverseLastInteractionRef.current = Date.now();
        }
      }, 1000);
      return () => clearInterval(interval);
    } else if (state.status === 'complete' && !wasRewindRef.current) {
      setTraverseFitTrigger(prev => prev + 1);
    }
  }, [state.status]);

  const handleTraverseGraphInteraction = useCallback(() => {
    traverseLastInteractionRef.current = Date.now();
  }, []);

  const handleTraverseEvent = useCallback((event: AGUIEvent) => {
    // Note: In strict mode, React invokes this twice.
    // Refs help avoid double-processing issues where possible.

    switch (event.type) {
      case 'RUN_STARTED':
        setTraverseTimerRunning(true);
        setState(prev => ({
          ...prev,
          status: 'traversing',
          current_step: wasZeroShotRef.current ? 'Zero-Shot' : 'Starting traversal',
        }));

        // Reset refs
        traverseNodesRef.current = [];
        traverseEdgesRef.current = [];
        traverseDecisionsRef.current = [];
        traverseBatchCountRef.current = 0;
        break;

      case 'STATE_SNAPSHOT':
        if (event.snapshot) {
          // GraphStateSnapshot already types nodes/edges correctly
          const { nodes, edges } = event.snapshot;

          traverseNodesRef.current = nodes;
          traverseEdgesRef.current = edges;

          // During rewind, prune decision_history and finalized_codes to match snapshot
          if (wasRewindRef.current) {
            const snapshotNodeIds = new Set(nodes.map(n => n.id));
            const prunedDecisions = traverseDecisionsRef.current.filter(
              d => snapshotNodeIds.has(d.current_node)
            );
            const prunedFinalized = nodes
              .filter(n => n.category === 'finalized')
              .map(n => n.id);

            traverseDecisionsRef.current = prunedDecisions;

            setState(prev => ({
              ...prev,
              nodes,
              edges,
              decision_history: prunedDecisions,
              finalized_codes: prunedFinalized,
            }));
          } else {
            setState(prev => ({
              ...prev,
              nodes,
              edges,
            }));

            // Trigger fit-to-window for cached replays (all nodes arrive at once)
            if (nodes && nodes.length > 0) {
              setTimeout(() => {
                setTraverseFitTrigger(prev => prev + 1);
              }, 350);
            }
          }
        }
        break;

      case 'STATE_DELTA':
        if (event.delta) {
          // For delta updates, we need the LATEST state which might be in refs or state
          // But useState is async, so refs are safer for "current world state" if we maintained them fully
          // Here we use functional update to get latest 'prev' from React
          // Include finalized array since backend may send patches for /finalized/-
          setState(prev => {
            try {
              const doc = {
                nodes: prev.nodes,
                edges: prev.edges,
                finalized: prev.finalized_codes,
              };
              const result = applyPatch(doc, event.delta as Operation[], true, false);
              const nextNodes = result.newDocument.nodes;
              const nextEdges = result.newDocument.edges;
              const nextFinalized = result.newDocument.finalized || prev.finalized_codes;

              traverseNodesRef.current = nextNodes;
              traverseEdgesRef.current = nextEdges;

              return {
                ...prev,
                nodes: nextNodes,
                edges: nextEdges,
                finalized_codes: nextFinalized,
              };
            } catch (e) {
              // During rewind, patches may reference indices that no longer exist
              // after node filtering - safely ignore and keep previous state
              console.warn('[STATE_DELTA] Patch failed (likely stale during rewind):', e);
              return prev;
            }
          });
        }
        break;

      case 'STEP_STARTED':
        setState(prev => ({
          ...prev,
          current_step: event.stepName || '',
        }));
        break;

      case 'STEP_FINISHED':
        traverseBatchCountRef.current += 1;
        setBatchCount(traverseBatchCountRef.current); // Sync for UI

        if (event.metadata) {
          const metadata: StepFinishedMetadata = event.metadata;
          const candidates: Record<string, string> = metadata.candidates ?? {};
          const selectedIds = metadata.selected_ids ?? [];
          const reasoning = metadata.reasoning ?? '';
          const nodeId = metadata.node_id ?? event.stepName;
          const batchType = metadata.batch_type ?? 'children';
          const selectedDetails = metadata.selected_details ?? {};

          // Reconstruct nodes/edges logic
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
            depth: (event.stepName?.split('|').length || 1),
            candidates: candidateDecisions,
            selected_codes: selectedIds,
          };

          traverseDecisionsRef.current.push(decision);

          setState(prev => ({
            ...prev,
            nodes: mergeById(prev.nodes, newNodes),
            edges: mergeByKey(prev.edges, newEdges),
            decision_history: [...prev.decision_history, decision],
          }));
        }
        break;

      case 'RUN_ERROR':
        // Handle dedicated error events (AG-UI protocol)
        setTraverseTimerRunning(false);
        setState(prev => ({
          ...prev,
          status: 'error',
          error: event.error,
          current_step: 'Error',
        }));
        setIsLoading(false);
        break;

      case 'RUN_FINISHED': {
        const finishedMeta: RunFinishedMetadata | undefined = event.metadata;
        const finalNodes = finishedMeta?.final_nodes ?? [];

        // Handle snapshot decisions if present (cached replays and rewinds)
        const snapshotDecisions = finishedMeta?.decisions;
        if (snapshotDecisions && snapshotDecisions.length > 0) {
          const decisions: DecisionPoint[] = snapshotDecisions.map(d => {
            const selectedSet = new Set(d.selected_ids);
            return {
              current_node: d.node_id,
              current_label: `${d.batch_type} batch`,
              depth: (d.batch_id?.split('|').length || 1),
              candidates: Object.entries(d.candidates ?? {}).map(([code, label]) => ({
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
          traverseDecisionsRef.current = decisions;
          setState(prev => ({
            ...prev,
            decision_history: decisions,
          }));
        }

        setTraverseTimerRunning(false);
        setState(prev => ({
          ...prev,
          status: 'complete',
          finalized_codes: finalNodes,
          current_step: `Complete - ${finalNodes.length} codes found`,
          wasZeroShot: wasZeroShotRef.current,
        }));
        setIsLoading(false);
        break;
      }
    }
  }, []);

  const handleTraverseError = useCallback((error: Error) => {
    setTraverseTimerRunning(false);
    setState(prev => ({
      ...prev,
      status: 'error',
      error: error.message,
      current_step: 'Error',
    }));
    setIsLoading(false);
  }, []);

  const handleTraverse = useCallback((options?: { bypassCache?: boolean }): boolean => {
    if (!clinicalNote.trim()) return false;
    if (!llmConfig.apiKey) {
      alert('Please configure your API key in LLM Settings');
      setSidebarTab('llm-settings');
      return false;
    }

    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    setIsLoading(true);
    setBatchCount(0);
    setState({
      ...INITIAL_TRAVERSAL_STATE,
      status: 'traversing',
      current_step: options?.bypassCache ? 'Regenerating (bypassing cache)' : 'Starting traversal',
    });

    // Reset refs
    traverseNodesRef.current = [];
    traverseEdgesRef.current = [];
    traverseDecisionsRef.current = [];
    traverseBatchCountRef.current = 0;
    wasRewindRef.current = false;
    wasZeroShotRef.current = !(llmConfig.scaffolded ?? true);

    // When bypassCache is true, set persist_cache to false to skip cache lookup
    const useCache = options?.bypassCache ? false : (llmConfig.persistCache ?? true);

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
        persist_cache: useCache,
      },
      handleTraverseEvent,
      handleTraverseError
    );

    return true;
  }, [clinicalNote, llmConfig, handleTraverseEvent, handleTraverseError, setSidebarTab]);

  const handleCancel = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    // Also cancel any rewind
    if (rewindControllerRef.current) {
      rewindControllerRef.current.abort();
      rewindControllerRef.current = null;
    }

    // Signal server to cancel (also cleans up backend state)
    cancelTraversal().catch(console.error);

    setIsLoading(false);
    setIsRewinding(false);
    setTraverseTimerRunning(false);
    resetTraverseTimer();

    // Clear refs
    traverseNodesRef.current = [];
    traverseEdgesRef.current = [];
    traverseDecisionsRef.current = [];
    traverseBatchCountRef.current = 0;
    traverseHadNodesRef.current = false;

    setState({
      ...INITIAL_TRAVERSAL_STATE,
      status: 'idle',
      current_step: 'Cancelled',
    });
  }, []);

  // Clear all state back to initial (like a freshly started app)
  const handleClear = useCallback(() => {
    // Abort any running operations
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    if (rewindControllerRef.current) {
      rewindControllerRef.current.abort();
      rewindControllerRef.current = null;
    }

    // Reset clinical note
    setClinicalNote('');

    // Reset all state to initial
    setIsLoading(false);
    setBatchCount(0);
    setSelectedNode(null);
    setTraverseFitTrigger(0);
    setTraverseTimerRunning(false);
    resetTraverseTimer();
    setState(INITIAL_TRAVERSAL_STATE);

    // Reset rewind state
    setRewindTargetNode(null);
    setRewindTargetBatchId(null);
    setRewindFeedbackText('');
    setIsRewindModalOpen(false);
    setIsRewinding(false);
    setRewindingNodeId(null);
    setRewindError(null);

    // Reset refs
    traverseNodesRef.current = [];
    traverseEdgesRef.current = [];
    traverseDecisionsRef.current = [];
    traverseBatchCountRef.current = 0;
    traverseHadNodesRef.current = false;
    wasRewindRef.current = false;
  }, []);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(prev => (prev === nodeId ? null : nodeId));
  }, []);

  // Rewind Handlers
  // Backend is the authoritative source for pre-rewind state - it loads cached state,
  // prunes to rewind point, and emits STATE_SNAPSHOT. Frontend just renders what it receives.
  const handleRewindSubmit = useCallback(async (nodeId: string, feedback: string, providedBatchId?: string) => {
    setIsRewinding(true);
    wasRewindRef.current = true;
    setRewindingNodeId(nodeId);
    setRewindError(null);
    setSelectedNode(null);

    try {
      let batchId = providedBatchId || rewindTargetBatchId;
      if (!batchId) {
        const decision = state.decision_history.find(d => d.current_node === nodeId);
        const batchType = decision ? extractBatchType(decision.current_label) : 'children';
        batchId = `${nodeId}|${batchType}`;
      }

      if (rewindControllerRef.current) {
        rewindControllerRef.current.abort();
      }

      const batchType = batchId!.split('|')[1] || 'children';

      // Set status to traversing while we wait for backend STATE_SNAPSHOT
      // Clear any previous error state
      setState(prev => ({
        ...prev,
        status: 'traversing',
        error: null,
        current_step: `Rewinding from ${nodeId} (${batchType} batch)...`,
      }));

      setIsRewindModalOpen(false);
      setRewindTargetNode(null);
      setRewindTargetBatchId(null);

      // Backend will emit STATE_SNAPSHOT with pre-rewind graph, then STEP_* events for new traversal
      rewindControllerRef.current = streamRewind(
        {
          batch_id: batchId!,
          feedback,
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
          persist_cache: llmConfig.persistCache ?? true,
        },
        handleTraverseEvent,
        (error) => {
          setRewindError(error.message);
          setIsRewinding(false);
          setTraverseTimerRunning(false);
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
  }, [state.decision_history, clinicalNote, llmConfig, handleTraverseEvent, rewindTargetBatchId]);

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
    const node = state.nodes.find(n => n.id === nodeId);
    if (!node) return;
    if (state.status === 'idle') return;

    // Every node with "Investigate Batch" owns its own batch
    // batch_id is simply nodeId|batchType - no traversal needed
    // For ROOT, batchType will be 'children' and batch_id will be 'ROOT|children'
    const batchId = batchType ? `${nodeId}|${batchType}` : null;

    if (feedback && feedback.trim().length > 0 && batchId) {
      handleRewindSubmit(nodeId, feedback, batchId);
      return;
    }

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
      wasRewindRef.current = false;
    }
  }, [isRewinding, state.status]);

  return {
    state: {
      clinicalNote,
      state,
      selectedNode,
      isLoading,
      batchCount,
      traverseFitTrigger,
      traverseElapsedTime,
      // Rewind state
      rewindTargetNode,
      rewindTargetBatchId,
      rewindFeedbackText,
      isRewindModalOpen,
      isRewinding,
      rewindingNodeId,
      rewindError,
    },
    setters: {
      setClinicalNote,
      setSelectedNode,
      setState,
      setTraverseFitTrigger,
      // Rewind setters
      setRewindFeedbackText,
      setIsRewindModalOpen,
    },
    actions: {
      handleTraverse,
      handleCancel,
      handleClear,
      handleNodeClick,
      handleTraverseGraphInteraction,
      // Rewind actions
      handleRewindSubmit,
      handleRewindClose,
      handleNodeRewindClick,
    },
    controllerRef,
  };
}