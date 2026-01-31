import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
    streamTraversal,
    buildGraph,
    type AGUIEvent,
    type StepFinishedMetadata,
    type RunFinishedMetadata,
} from '../../lib/api';
import { calculateDepthFromCode } from '../../lib/graphUtils';
import type {
    GraphNode,
    GraphEdge,
    BenchmarkGraphNode,
    BenchmarkMetrics,
    DecisionPoint,
    CandidateDecision,
    TraversalStatus,
    OvershootMarker,
    EdgeMissMarker,
    LLMConfig,
} from '../../lib/types';
import { type SidebarTab } from '../../lib/constants';
import {
    buildAncestorMap,
    compareFinalizedCodes,
    computeBenchmarkMetrics,
    computeFinalizedComparison,
    resetNodesToIdle,
} from '../../lib/benchmark';

interface UseBenchmarkStateProps {
    llmConfig: LLMConfig;
    setSidebarTab: (tab: SidebarTab) => void;
}

export function useBenchmarkState({ llmConfig, setSidebarTab }: UseBenchmarkStateProps) {
    // State
    const [benchmarkExpectedCodes, setBenchmarkExpectedCodes] = useState<Set<string>>(new Set());
    const [benchmarkCodeInput, setBenchmarkCodeInput] = useState('');
    const [benchmarkExpectedGraph, setBenchmarkExpectedGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
    const [benchmarkTraversedNodes, setBenchmarkTraversedNodes] = useState<GraphNode[]>([]);
    const [_benchmarkTraversedEdges, setBenchmarkTraversedEdges] = useState<GraphEdge[]>([]);
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
    const [streamingTraversedIds, setStreamingTraversedIds] = useState<Set<string>>(new Set());
    const [benchmarkInvalidCodes, setBenchmarkInvalidCodes] = useState<Set<string>>(new Set());
    const [benchmarkInferPrecursors, setBenchmarkInferPrecursors] = useState(false);
    const [benchmarkWasZeroShot, setBenchmarkWasZeroShot] = useState(false);
    const [benchmarkIsStarting, setBenchmarkIsStarting] = useState(false);
    // Inferred view for zero-shot mode when "Infer Precursor Nodes" is toggled ON
    // - expectedNodesColored: Expected graph with nodes colored by intersection with inferred graph (for Benchmark)
    // - graphNodes: The actual inferred graph (for Traverse tab's zero-shot visualization)
    const [benchmarkInferredView, setBenchmarkInferredView] = useState<{
        expectedNodesColored: BenchmarkGraphNode[];  // Expected graph colored by inferred intersection
        metrics: BenchmarkMetrics;
        overshootMarkers: OvershootMarker[];
        missedEdgeMarkers: EdgeMissMarker[];
    } | null>(null);

    // Refs
    const benchmarkControllerRef = useRef<AbortController | null>(null);
    const benchmarkTraversedNodesRef = useRef<GraphNode[]>([]);
    const benchmarkTraversedEdgesRef = useRef<GraphEdge[]>([]);
    const benchmarkDecisionsRef = useRef<DecisionPoint[]>([]);
    const benchmarkNodesMapRef = useRef<Map<string, GraphNode>>(new Map());
    const benchmarkEdgesMapRef = useRef<Map<string, GraphEdge>>(new Map());
    const benchmarkCombinedNodesRef = useRef<BenchmarkGraphNode[]>([]);
    const benchmarkStreamedIdsRef = useRef<Set<string>>(new Set());
    const streamingTraversedIdsRef = useRef<Set<string>>(new Set());
    const benchmarkBatchCountRef = useRef<number>(0);
    const benchmarkResetRafRef = useRef<number | null>(null);
    const benchmarkIsCachedReplayRef = useRef<boolean>(false);
    const benchmarkWasZeroShotRef = useRef<boolean>(false);
    const lastVisualUpdateTimeRef = useRef<number>(0);
    const benchmarkStartTimeRef = useRef<number | null>(null);
    const prevBenchmarkStatusRef = useRef<TraversalStatus>('idle');
    const benchmarkLastInteractionRef = useRef<number>(0);

    const VISUAL_UPDATE_THROTTLE_MS = 100;

    // Track elapsed time
    const [benchmarkElapsedTime, setBenchmarkElapsedTime] = useState<number | null>(null);

    // Track exact matched codes (for "Matched Final Codes" display after benchmark completes)
    const [benchmarkExactMatchedCodes, setBenchmarkExactMatchedCodes] = useState<Set<string>>(new Set());

    useEffect(() => {
        if (prevBenchmarkStatusRef.current !== 'traversing' && benchmarkStatus === 'traversing') {
            benchmarkStartTimeRef.current = Date.now();
            setBenchmarkElapsedTime(null);
        }
        if (prevBenchmarkStatusRef.current === 'traversing' && (benchmarkStatus === 'complete' || benchmarkStatus === 'error')) {
            if (benchmarkStartTimeRef.current) {
                setBenchmarkElapsedTime(Date.now() - benchmarkStartTimeRef.current);
            }
        }
        if (benchmarkStatus === 'idle') {
            benchmarkStartTimeRef.current = null;
            setBenchmarkElapsedTime(null);
        }
        prevBenchmarkStatusRef.current = benchmarkStatus;
    }, [benchmarkStatus]);

    // Fit-to-window logic
    const benchmarkHadNodesRef = useRef(false);
    useEffect(() => {
        if (benchmarkStatus === 'traversing') {
            if (benchmarkCombinedNodes.length > 0 && !benchmarkHadNodesRef.current) {
                benchmarkHadNodesRef.current = true;
                const timer = setTimeout(() => {
                    setBenchmarkFitTrigger(prev => prev + 1);
                }, 350);
                return () => clearTimeout(timer);
            }
        } else if (benchmarkStatus === 'idle') {
            benchmarkHadNodesRef.current = false;
        }
    }, [benchmarkStatus, benchmarkCombinedNodes.length]);

    useEffect(() => {
        if (benchmarkStatus === 'traversing') {
            benchmarkLastInteractionRef.current = Date.now();
            const interval = setInterval(() => {
                const idleTime = Date.now() - benchmarkLastInteractionRef.current;
                if (idleTime >= 5000) {
                    setBenchmarkFitTrigger(prev => prev + 1);
                    benchmarkLastInteractionRef.current = Date.now();
                }
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [benchmarkStatus]);

    useEffect(() => {
        if (benchmarkStatus === 'complete') {
            benchmarkLastInteractionRef.current = Date.now();
            const interval = setInterval(() => {
                const idleTime = Date.now() - benchmarkLastInteractionRef.current;
                if (idleTime >= 5000) {
                    setBenchmarkFitTrigger(prev => prev + 1);
                    clearInterval(interval);
                }
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [benchmarkStatus]);

    const handleBenchmarkGraphInteraction = useCallback(() => {
        benchmarkLastInteractionRef.current = Date.now();
    }, []);

    const handleBenchmarkEvent = useCallback((event: AGUIEvent) => {
        switch (event.type) {
            case 'RUN_STARTED':
                if (event.metadata?.cached) {
                    console.log('[Benchmark] Cached replay - expecting STATE_SNAPSHOT');
                    benchmarkIsCachedReplayRef.current = true;
                } else {
                    benchmarkIsCachedReplayRef.current = false;
                }

                benchmarkWasZeroShotRef.current = !(llmConfig.scaffolded ?? true);
                setBenchmarkWasZeroShot(benchmarkWasZeroShotRef.current);

                if (benchmarkResetRafRef.current !== null) {
                    cancelAnimationFrame(benchmarkResetRafRef.current);
                    benchmarkResetRafRef.current = null;
                }

                // Clear previous run's inferred view (critical for zero-shot reset)
                setBenchmarkInferredView(null);

                if (benchmarkCombinedNodesRef.current.length > 0) {
                    const idleNodes = resetNodesToIdle(benchmarkCombinedNodesRef.current);
                    benchmarkCombinedNodesRef.current = idleNodes;
                    setBenchmarkCombinedNodes(idleNodes);
                }

                setBenchmarkStatus('traversing');
                if (benchmarkIsCachedReplayRef.current) {
                    setBenchmarkCurrentStep('Loading cached results...');
                } else if (benchmarkWasZeroShotRef.current) {
                    setBenchmarkCurrentStep('Zero-Shot');
                } else {
                    setBenchmarkCurrentStep('Starting benchmark traversal');
                }
                setBenchmarkIsStarting(false);
                break;

            case 'STATE_SNAPSHOT':
                if (event.snapshot) {
                    const nodes = event.snapshot.nodes as GraphNode[];
                    const edges = event.snapshot.edges as GraphEdge[];
                    benchmarkTraversedNodesRef.current = nodes;
                    benchmarkTraversedEdgesRef.current = edges;

                    const traversedIds = new Set(nodes.map(n => n.id));
                    benchmarkStreamedIdsRef.current = traversedIds;

                    console.log(`[STATE_SNAPSHOT] Benchmark complete graph: ${nodes.length} nodes, ${edges.length} edges`);
                }
                break;

            case 'STATE_DELTA':
                break;

            case 'STEP_STARTED':
                setBenchmarkCurrentStep(event.stepName || 'Processing...');
                break;

            case 'STEP_FINISHED':
                benchmarkBatchCountRef.current += 1;

                if (event.metadata) {
                    const metadata: StepFinishedMetadata = event.metadata;
                    const candidates: Record<string, string> = metadata.candidates ?? {};
                    const selectedIds = metadata.selected_ids ?? [];
                    const reasoning = metadata.reasoning ?? '';
                    const nodeId = metadata.node_id ?? event.stepName;
                    const batchType = metadata.batch_type ?? 'children';
                    const selectedDetails = metadata.selected_details ?? {};

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

                    for (const node of newNodes) {
                        benchmarkNodesMapRef.current.set(node.id, node);
                    }
                    for (const edge of newEdges) {
                        const edgeKey = `${edge.source}|${edge.target}`;
                        benchmarkEdgesMapRef.current.set(edgeKey, edge);
                    }
                    benchmarkDecisionsRef.current.push(decision);

                    for (const id of selectedIds) {
                        benchmarkStreamedIdsRef.current.add(id);
                        streamingTraversedIdsRef.current.add(id);
                    }

                    const now = Date.now();
                    if (now - lastVisualUpdateTimeRef.current >= VISUAL_UPDATE_THROTTLE_MS) {
                        lastVisualUpdateTimeRef.current = now;
                        setStreamingTraversedIds(new Set(streamingTraversedIdsRef.current));
                    }
                }
                break;

            case 'RUN_ERROR':
                // Handle dedicated error events (AG-UI protocol)
                setBenchmarkStatus('error');
                setBenchmarkError(event.error);
                setBenchmarkCurrentStep('Error');
                break;

            case 'RUN_FINISHED': {
                const finishedMeta: RunFinishedMetadata | undefined = event.metadata;
                const finalNodesRaw = finishedMeta?.final_nodes ?? [];
                const finalNodes = new Set(finalNodesRaw);
                const wasZeroShot = benchmarkWasZeroShotRef.current;

                if (benchmarkResetRafRef.current !== null) {
                    cancelAnimationFrame(benchmarkResetRafRef.current);
                    benchmarkResetRafRef.current = null;
                }

                const snapshotDecisions = finishedMeta?.decisions;

                    if (snapshotDecisions && snapshotDecisions.length > 0) {
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
                    }

                    if (benchmarkNodesMapRef.current.size > 0) {
                        benchmarkTraversedNodesRef.current = [...benchmarkNodesMapRef.current.values()];
                        benchmarkTraversedEdgesRef.current = [...benchmarkEdgesMapRef.current.values()];
                    }

                    setBenchmarkTraversedNodes(benchmarkTraversedNodesRef.current);
                    setBenchmarkTraversedEdges(benchmarkTraversedEdgesRef.current);
                    setBenchmarkDecisions(benchmarkDecisionsRef.current);
                    setBenchmarkBatchCount(benchmarkBatchCountRef.current);

                    const wasCached = finishedMeta?.cached ?? false;
                    if (wasCached) {
                        console.log('[BACKEND CACHE HIT] Using cached results:', finalNodesRaw.length, 'codes');
                    }

                    if (benchmarkExpectedGraph) {
                        const latestTraversedEdges = benchmarkTraversedEdgesRef.current;
                        const streamedIds = benchmarkStreamedIdsRef.current;
                        const currentCombinedNodes = benchmarkCombinedNodesRef.current;

                        const finalizedViewStreamedIds = wasZeroShot ? new Set<string>() : streamedIds;

                        // 1. Compute outcomes first to get exact matches (including lateral matches)
                        const expectedAncestorMap = buildAncestorMap(benchmarkExpectedGraph.edges);
                        const traversedAncestorMap = buildAncestorMap(latestTraversedEdges);

                        const { outcomes, otherCodes } = compareFinalizedCodes(
                            benchmarkExpectedCodes,
                            expectedAncestorMap,
                            finalNodes,
                            traversedAncestorMap,
                            benchmarkExpectedGraph.edges
                        );

                        // 2. Build exact match set for visual green fill (includes lateral matches)
                        const exactMatchedCodes = new Set(
                            outcomes.filter(o => o.status === 'exact').map(o => o.expectedCode)
                        );

                        // Store exact matched codes for display in codes bar
                        setBenchmarkExactMatchedCodes(exactMatchedCodes);

                        // 3. Compute visual node statuses with exact match info
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
                            { finalizedOnlyMode: wasZeroShot, exactMatchedCodes }
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

                        benchmarkCombinedNodesRef.current = finalCombinedNodes;

                        setBenchmarkCombinedNodes(finalCombinedNodes);
                        setBenchmarkMissedEdgeMarkers(missedEdgeMarkers);
                        setBenchmarkOvershootMarkers(overshootMarkers);
                        setBenchmarkMetrics(metrics);

                        if (wasZeroShot && finalNodesRaw.length > 0) {
                            setTimeout(async () => {
                                try {
                                    const inferredGraph = await buildGraph(finalNodesRaw);

                                    if (inferredGraph.invalid_codes && inferredGraph.invalid_codes.length > 0) {
                                        console.log('[BenchmarkInferredView] Filtered out invalid codes:', inferredGraph.invalid_codes);
                                    }

                                    if (inferredGraph.nodes.length === 0) {
                                        return;
                                    }

                                    const inferredTraversedIds = new Set(inferredGraph.nodes.map(n => n.id));

                                    const initializedExpectedNodes: BenchmarkGraphNode[] = benchmarkExpectedGraph.nodes.map(node => ({
                                        ...node,
                                        benchmarkStatus: 'expected' as const,
                                    }));

                                    // 1. Compute outcomes first to get exact matches (including lateral matches)
                                    const inferredAncestorMap = buildAncestorMap(inferredGraph.edges);
                                    const { outcomes: inferredOutcomes, otherCodes: inferredOtherCodes } = compareFinalizedCodes(
                                        benchmarkExpectedCodes,
                                        expectedAncestorMap,
                                        finalNodes,
                                        inferredAncestorMap,
                                        benchmarkExpectedGraph.edges
                                    );

                                    // 2. Build exact match set for visual green fill
                                    const inferredExactMatchedCodes = new Set(
                                        inferredOutcomes.filter(o => o.status === 'exact').map(o => o.expectedCode)
                                    );

                                    // 3. Compute visual node statuses with exact match info
                                    const {
                                        nodes: expectedNodesColored,
                                        missedEdgeMarkers: inferredMissedMarkers,
                                        overshootMarkers: inferredOvershootMarkers,
                                        traversedSet: inferredTraversedSet,
                                    } = computeFinalizedComparison(
                                        initializedExpectedNodes,
                                        benchmarkExpectedGraph.edges,
                                        inferredTraversedIds,
                                        finalNodes,
                                        benchmarkExpectedCodes,
                                        inferredGraph.edges,
                                        { exactMatchedCodes: inferredExactMatchedCodes }
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

                                    setBenchmarkInferredView({
                                        expectedNodesColored,
                                        metrics: inferredMetrics,
                                        overshootMarkers: inferredOvershootMarkers,
                                        missedEdgeMarkers: inferredMissedMarkers,
                                    });
                                } catch (err) {
                                    console.error('[BenchmarkInferredView] Failed to build:', err);
                                }
                            }, 0);
                        }
                    }

                setStreamingTraversedIds(new Set(benchmarkStreamedIdsRef.current));
                setBenchmarkStatus('complete');
                setBenchmarkCurrentStep(`Complete - ${finalNodes.size} codes finalized`);
                setBenchmarkFitTrigger(prev => prev + 1);
                break;
            }
        }
    }, [benchmarkExpectedGraph, benchmarkExpectedCodes, llmConfig]);

    const handleBenchmarkError = useCallback((error: Error) => {
        setBenchmarkStatus('error');
        setBenchmarkError(error.message);
    }, []);

    const handleBenchmarkRun = useCallback((): boolean => {
        if (!benchmarkClinicalNote.trim()) return false;
        if (!llmConfig.apiKey) {
            alert('Please configure your API key in LLM Settings');
            setSidebarTab('llm-settings');
            return false;
        }

        if (benchmarkControllerRef.current) {
            benchmarkControllerRef.current.abort();
            benchmarkControllerRef.current = null;
        }

        setBenchmarkTraversedNodes([]);
        setBenchmarkTraversedEdges([]);
        setBenchmarkDecisions([]);
        setBenchmarkBatchCount(0);
        benchmarkTraversedNodesRef.current = [];
        benchmarkTraversedEdgesRef.current = [];
        benchmarkDecisionsRef.current = [];
        benchmarkBatchCountRef.current = 0;
        benchmarkStreamedIdsRef.current = new Set();
        streamingTraversedIdsRef.current = new Set();
        lastVisualUpdateTimeRef.current = 0;
        benchmarkNodesMapRef.current = new Map();
        benchmarkEdgesMapRef.current = new Map();
        setStreamingTraversedIds(new Set());
        setBenchmarkOvershootMarkers([]);
        setBenchmarkMissedEdgeMarkers([]);
        setBenchmarkMetrics(null);
        setBenchmarkStatus('traversing');
        setBenchmarkCurrentStep('Starting benchmark traversal');
        setBenchmarkIsStarting(true);
        setBenchmarkError(null);
        setBenchmarkInferredView(null);
        benchmarkIsCachedReplayRef.current = false;

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
                persist_cache: llmConfig.persistCache ?? true,
            },
            handleBenchmarkEvent,
            handleBenchmarkError
        );

        return true;
    }, [benchmarkClinicalNote, llmConfig, handleBenchmarkEvent, handleBenchmarkError, benchmarkCombinedNodes, setSidebarTab]);

    const handleBenchmarkCancel = useCallback(() => {
        if (benchmarkControllerRef.current) {
            benchmarkControllerRef.current.abort();
            benchmarkControllerRef.current = null;
        }
        setBenchmarkStatus('idle');
        setBenchmarkCurrentStep('Cancelled');
    }, []);

    const handleBenchmarkReset = useCallback(() => {
        // Abort any running operation
        if (benchmarkControllerRef.current) {
            benchmarkControllerRef.current.abort();
            benchmarkControllerRef.current = null;
        }

        // Reset clinical note and expected codes (full reset like a fresh app)
        setBenchmarkClinicalNote('');
        setBenchmarkExpectedCodes(new Set());
        setBenchmarkCodeInput('');
        setBenchmarkExpectedGraph(null);
        setBenchmarkInvalidCodes(new Set());

        // Reset traversal state
        setBenchmarkStatus('idle');
        setBenchmarkCurrentStep('');
        setBenchmarkError(null);
        setBenchmarkMetrics(null);
        setBenchmarkBatchCount(0);
        setStreamingTraversedIds(new Set());
        setBenchmarkWasZeroShot(false);
        setBenchmarkInferredView(null);
        setBenchmarkElapsedTime(null);
        setBenchmarkExactMatchedCodes(new Set());
        setBenchmarkTraversedNodes([]);
        setBenchmarkTraversedEdges([]);
        setBenchmarkDecisions([]);
        setBenchmarkOvershootMarkers([]);
        setBenchmarkMissedEdgeMarkers([]);

        // Clear combined graph (clears SVG area)
        benchmarkCombinedNodesRef.current = [];
        setBenchmarkCombinedNodes([]);
        setBenchmarkCombinedEdges([]);

        // Reset refs
        benchmarkTraversedNodesRef.current = [];
        benchmarkTraversedEdgesRef.current = [];
        benchmarkDecisionsRef.current = [];
        benchmarkNodesMapRef.current = new Map();
        benchmarkEdgesMapRef.current = new Map();
        benchmarkStreamedIdsRef.current = new Set();
        streamingTraversedIdsRef.current = new Set();
        benchmarkBatchCountRef.current = 0;
        benchmarkIsCachedReplayRef.current = false;
        benchmarkWasZeroShotRef.current = false;
        benchmarkHadNodesRef.current = false;
        benchmarkStartTimeRef.current = null;
    }, []);

    // Build expected graph from codes
    const buildExpectedGraph = useCallback(async (codes: Set<string>) => {
        if (codes.size === 0) return;

        setBenchmarkStatus('traversing');
        setBenchmarkCurrentStep('Building expected graph...');

        try {
            const result = await buildGraph([...codes]);
            setBenchmarkExpectedGraph(result);

            // Initialize combined graph
            const combinedNodes = result.nodes.map(node => ({
                ...node,
                benchmarkStatus: 'expected' as const,
                category: node.category || 'ancestor', // Ensure category exists
                billable: node.billable || false,       // Ensure billable exists
            }));

            benchmarkCombinedNodesRef.current = combinedNodes;
            setBenchmarkCombinedNodes(combinedNodes);
            setBenchmarkCombinedEdges(result.edges);

            setBenchmarkInvalidCodes(new Set());
            setBenchmarkStatus('idle');
            setBenchmarkCurrentStep('');
            setBenchmarkFitTrigger(prev => prev + 1);
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Failed to build expected graph';
            const match = errorMsg.match(/Invalid ICD-10-CM codes?: (.+)/i);

            if (match) {
                const invalidCodes = match[1].split(',').map(c => c.trim());
                setBenchmarkInvalidCodes(new Set(invalidCodes));

                const validCodes = new Set([...codes].filter(c => !invalidCodes.includes(c)));
                if (validCodes.size > 0) {
                    try {
                        const result = await buildGraph([...validCodes]);
                        setBenchmarkExpectedGraph(result);
                        const combinedNodes = result.nodes.map(node => ({
                            ...node,
                            benchmarkStatus: 'expected' as const,
                            category: node.category || 'ancestor',
                            billable: node.billable || false,
                        }));
                        benchmarkCombinedNodesRef.current = combinedNodes;
                        setBenchmarkCombinedNodes(combinedNodes);
                        setBenchmarkCombinedEdges(result.edges);
                        setBenchmarkStatus('idle');
                        setBenchmarkCurrentStep('');
                        setBenchmarkFitTrigger(prev => prev + 1);
                        return;
                    } catch {
                        // Fall through
                    }
                }
            }
            setBenchmarkError(errorMsg);
            setBenchmarkStatus('error');
        }
    }, []);

    const handleBenchmarkAddCode = useCallback(async () => {
        const codes = benchmarkCodeInput
            .split(/[,\n\t\s]+/)
            .map(c => c.trim().toUpperCase())
            .filter(c => c.length > 0);

        if (codes.length > 0) {
            if (benchmarkStatus === 'complete') {
                benchmarkCombinedNodesRef.current = [];
                setBenchmarkCombinedNodes([]);
                setBenchmarkCombinedEdges([]);
                setBenchmarkOvershootMarkers([]);
                setBenchmarkMissedEdgeMarkers([]);
                setBenchmarkMetrics(null);
                setBenchmarkInferredView(null);
                setBenchmarkStatus('idle');
            }

            const newSet = new Set(benchmarkExpectedCodes);
            codes.forEach(code => newSet.add(code));
            setBenchmarkExpectedCodes(newSet);
            setBenchmarkCodeInput('');

            await buildExpectedGraph(newSet);
        }
    }, [benchmarkCodeInput, benchmarkExpectedCodes, buildExpectedGraph, benchmarkStatus]);

    const handleBenchmarkRemoveCode = useCallback(async (code: string) => {
        if (benchmarkStatus === 'complete') {
            benchmarkCombinedNodesRef.current = [];
            setBenchmarkCombinedNodes([]);
            setBenchmarkCombinedEdges([]);
            setBenchmarkOvershootMarkers([]);
            setBenchmarkMissedEdgeMarkers([]);
            setBenchmarkMetrics(null);
            setBenchmarkInferredView(null);
            setBenchmarkStatus('idle');
        }

        const newSet = new Set(benchmarkExpectedCodes);
        newSet.delete(code);
        setBenchmarkExpectedCodes(newSet);

        setBenchmarkInvalidCodes(prev => {
            const updated = new Set(prev);
            updated.delete(code);
            return updated;
        });

        if (newSet.size > 0) {
            await buildExpectedGraph(newSet);
        } else {
            setBenchmarkExpectedGraph(null);
            benchmarkCombinedNodesRef.current = [];
            setBenchmarkCombinedNodes([]);
            setBenchmarkCombinedEdges([]);
            setBenchmarkOvershootMarkers([]);
            setBenchmarkMissedEdgeMarkers([]);
            setBenchmarkInvalidCodes(new Set());
        }
    }, [benchmarkExpectedCodes, buildExpectedGraph, benchmarkStatus]);

    const memoizedFinalizedCodesArray = useMemo(() => {
        return [...benchmarkExpectedCodes];
    }, [benchmarkExpectedCodes]);

    return {
        state: {
            benchmarkExpectedCodes,
            benchmarkCodeInput,
            benchmarkExpectedGraph,
            benchmarkTraversedNodes,
            benchmarkCombinedNodes,
            benchmarkCombinedEdges,
            benchmarkOvershootMarkers,
            benchmarkMissedEdgeMarkers,
            benchmarkMetrics,
            benchmarkDecisions,
            benchmarkStatus,
            benchmarkCurrentStep,
            benchmarkError,
            benchmarkClinicalNote,
            benchmarkBatchCount,
            benchmarkFitTrigger,
            streamingTraversedIds,
            benchmarkInvalidCodes,
            benchmarkInferPrecursors,
            benchmarkWasZeroShot,
            benchmarkIsStarting,
            benchmarkInferredView,
            benchmarkElapsedTime,
            benchmarkExactMatchedCodes,
            benchmarkFinalizedCodesArray: memoizedFinalizedCodesArray,
        },
        setters: {
            setBenchmarkExpectedCodes,
            setBenchmarkCodeInput,
            setBenchmarkExpectedGraph,
            setBenchmarkClinicalNote,
            setBenchmarkInferPrecursors,
            setBenchmarkInvalidCodes,
        },
        actions: {
            handleBenchmarkRun,
            handleBenchmarkCancel,
            handleBenchmarkReset,
            handleBenchmarkGraphInteraction,
            handleBenchmarkAddCode,
            handleBenchmarkRemoveCode,
        },
        controllerRef: benchmarkControllerRef,
    };
}
