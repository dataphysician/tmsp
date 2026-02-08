import { useState, useCallback, useEffect } from 'react';
import { GraphViewer } from './components/GraphViewer';
import { TrajectoryViewer } from './components/TrajectoryViewer';
import { BenchmarkReportViewer } from './components/BenchmarkReportViewer';
import { VisualizeReportViewer } from './components/VisualizeReportViewer';
import { VisualizeSidebar } from './features/visualize/VisualizeSidebar';
import { TraverseSidebar } from './features/traverse/TraverseSidebar';
import { BenchmarkSidebar } from './features/benchmark/BenchmarkSidebar'; // New
import { NodeRewindModal } from './components/shared';
import { buildGraph } from './lib/api';
import { parseCodeInput } from './lib/graphUtils';
import type {
  GraphNode,
  GraphEdge,
  LLMConfig,
} from './lib/types';
import {
  LLM_SYSTEM_PROMPT,
  type ViewTab,
  type SidebarTab,
  SIDEBAR_TABS,
} from './lib/constants';
import './App.css';
import { useBenchmarkState } from './features/benchmark/useBenchmarkState';
import { useTraverseState } from './features/traverse/useTraverseState';
import { DEFAULT_LLM_CONFIG } from './hooks/useLLMConfig';

type FeatureTab = 'visualize' | 'traverse' | 'benchmark';

function TraversalUI() {
  // Shared State
  const [activeFeatureTab, setActiveFeatureTab] = useState<FeatureTab>('traverse');
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
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>(SIDEBAR_TABS.CLINICAL_NOTE);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [, setSidebarRefElement] = useState<HTMLElement | null>(null);

  // Per-feature view tab states (Graph vs Report)
  const [visualizeViewTab, setVisualizeViewTab] = useState<ViewTab>('graph');
  const [traverseViewTab, setTraverseViewTab] = useState<ViewTab>('graph');
  const [benchmarkViewTab, setBenchmarkViewTab] = useState<ViewTab>('graph');

  // Custom Hooks
  const traverse = useTraverseState({ llmConfig, setSidebarTab });
  const benchmark = useBenchmarkState({ llmConfig, setSidebarTab });

  // VISUALIZE tab state (Keeping inline for now as per plan)
  const [inputCodes, setInputCodes] = useState<Set<string>>(new Set());
  const [codeInput, setCodeInput] = useState('');
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);
  const [, setGraphError] = useState<string | null>(null);
  const [visualizeFitTrigger, setVisualizeFitTrigger] = useState(0);

  // Zero-shot visualization graph (for Traverse tab when visualizePrediction is ON)
  const [zeroShotVisualization, setZeroShotVisualization] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [, setIsLoadingZeroShotViz] = useState(false);

  // Effect to visualize zero-shot predictions when enabled
  useEffect(() => {
    const isComplete = traverse.state.state.status === 'complete';
    const wasZeroShot = traverse.state.state.wasZeroShot ?? false;
    const vizEnabled = llmConfig.visualizePrediction ?? false;
    const hasCodes = traverse.state.state.finalized_codes.length > 0;
    const shouldVisualize = isComplete && wasZeroShot && vizEnabled && hasCodes;

    if (shouldVisualize) {
      (async () => {
        setIsLoadingZeroShotViz(true);
        try {
          console.log('[ZeroShotViz] Building graph from codes:', traverse.state.state.finalized_codes);
          const result = await buildGraph(traverse.state.state.finalized_codes);
          console.log('[ZeroShotViz] Graph built successfully:', result.nodes.length, 'nodes');
          setZeroShotVisualization({ nodes: result.nodes, edges: result.edges });
        } catch (err) {
          console.error('[ZeroShotViz] Failed to build visualization:', err);
          setZeroShotVisualization(null);
        } finally {
          setIsLoadingZeroShotViz(false);
        }
      })();
    } else if (traverse.state.state.status === 'idle' || !wasZeroShot) {
      setZeroShotVisualization(null);
    }
  }, [
    traverse.state.state.status,
    traverse.state.state.finalized_codes,
    traverse.state.state.wasZeroShot,
    llmConfig.visualizePrediction
  ]);

  // VISUALIZE tab handlers
  const handleAddCode = useCallback(async () => {
    const codes = parseCodeInput(codeInput);
    if (codes.length > 0) {
      const newSet = new Set(inputCodes);
      codes.forEach(code => newSet.add(code));
      setInputCodes(newSet);
      setCodeInput('');

      setIsLoadingGraph(true);
      setGraphError(null);
      try {
        const result = await buildGraph([...newSet]);
        setGraphData({ nodes: result.nodes, edges: result.edges });
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

  // Clear all Traverse state back to defaults (clinical note, LLM config, graph, status)
  const handleClearTraverse = useCallback(() => {
    traverse.actions.handleClear();
    setLlmConfig(DEFAULT_LLM_CONFIG);
    setZeroShotVisualization(null);
  }, [traverse.actions]);

  // Reset all Benchmark state back to defaults (clinical note, expected codes, LLM config, graph, status)
  const handleResetBenchmark = useCallback(() => {
    benchmark.actions.handleBenchmarkReset();
    setLlmConfig(DEFAULT_LLM_CONFIG);
  }, [benchmark.actions]);

  return (
    <div className="app">
      {/* Sidebar */}
      <aside
        ref={setSidebarRefElement}
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
          <span className="sidebar-toggle-btn">
            <span className="chevron-icon">{sidebarCollapsed ? '\u00BB' : '\u00AB'}</span>
          </span>
        </div>

        {activeFeatureTab === 'visualize' && !sidebarCollapsed && (
          <VisualizeSidebar
              inputCodes={inputCodes}
              codeInput={codeInput}
              graphData={graphData}
              isLoading={isLoadingGraph}
              onCodeInputChange={setCodeInput}
              onAddCode={handleAddCode}
              onRemoveCode={handleRemoveCode}
              onClearCodes={handleClearCodes}
            />
          )}

        {activeFeatureTab === 'traverse' && !sidebarCollapsed && (
          <TraverseSidebar
              sidebarTab={sidebarTab}
              onSidebarTabChange={setSidebarTab}
              clinicalNote={traverse.state.clinicalNote}
              onClinicalNoteChange={traverse.setters.setClinicalNote}
              llmConfig={llmConfig}
              onLLMConfigChange={setLlmConfig}
              isLoading={traverse.state.state.status === 'traversing'}
              onTraverse={traverse.actions.handleTraverse}
              onCancel={traverse.actions.handleCancel}
              onClear={handleClearTraverse}
              batchCount={traverse.state.batchCount}
              onCollapseSidebar={() => setSidebarCollapsed(true)}
            />
          )}

        {activeFeatureTab === 'benchmark' && !sidebarCollapsed && (
          <BenchmarkSidebar
              clinicalNote={benchmark.state.benchmarkClinicalNote}
              onClinicalNoteChange={benchmark.setters.setBenchmarkClinicalNote}
              llmConfig={llmConfig}
              onLLMConfigChange={setLlmConfig}
              sidebarTab={sidebarTab}
              onSidebarTabChange={setSidebarTab}
              isLoading={benchmark.state.benchmarkStatus === 'traversing'}
              onStartBenchmark={benchmark.actions.handleBenchmarkRun}
              onCancelBenchmark={benchmark.actions.handleBenchmarkCancel}
              onResetBenchmark={handleResetBenchmark}
              expectedCodes={benchmark.state.benchmarkExpectedCodes}
              expectedCodesInput={benchmark.state.benchmarkCodeInput}
              onExpectedCodesInputChange={benchmark.setters.setBenchmarkCodeInput}
              onAddExpectedCode={benchmark.actions.handleBenchmarkAddCode}
              benchmarkInferPrecursors={benchmark.state.benchmarkInferPrecursors}
              onBenchmarkInferPrecursorsChange={benchmark.setters.setBenchmarkInferPrecursors}
              benchmarkComplete={benchmark.state.benchmarkStatus === 'complete'}
          />
        )}
      </aside>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Header */}
        <header className="main-header">
          <div className="feature-tabs">
            {(['visualize', 'traverse', 'benchmark'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => {
                  if (tab === activeFeatureTab) {
                    // Clicking active tab: toggle sidebar or switch from report to graph
                    const viewTab = tab === 'visualize' ? visualizeViewTab
                                  : tab === 'traverse' ? traverseViewTab
                                  : benchmarkViewTab;
                    if (viewTab === 'report') {
                      // Switch from report to graph view
                      if (tab === 'visualize') setVisualizeViewTab('graph');
                      else if (tab === 'traverse') setTraverseViewTab('graph');
                      else setBenchmarkViewTab('graph');
                    } else {
                      // Toggle sidebar
                      setSidebarCollapsed(prev => !prev);
                    }
                  } else {
                    // Clicking inactive tab: activate and expand
                    setActiveFeatureTab(tab);
                    setSidebarCollapsed(false);
                    if (tab !== 'visualize') {
                      setSidebarTab(SIDEBAR_TABS.CLINICAL_NOTE);
                    }
                  }
                }}
                className={`feature-tab ${activeFeatureTab === tab ? 'active' : ''}`}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="view-switcher">
            <span className="view-label">View:</span>
            {activeFeatureTab === 'visualize' && (
              <>
                <button onClick={() => setVisualizeViewTab('graph')} className={`view-btn ${visualizeViewTab === 'graph' ? 'active' : ''}`}>Graph</button>
                <button onClick={() => setVisualizeViewTab('report')} className={`view-btn ${visualizeViewTab === 'report' ? 'active' : ''}`}>Report</button>
              </>
            )}
            {activeFeatureTab === 'traverse' && (
              <>
                <button onClick={() => setTraverseViewTab('graph')} className={`view-btn ${traverseViewTab === 'graph' ? 'active' : ''}`}>Graph</button>
                <button onClick={() => setTraverseViewTab('report')} className={`view-btn ${traverseViewTab === 'report' ? 'active' : ''}`}>Report</button>
              </>
            )}
            {activeFeatureTab === 'benchmark' && (
              <>
                <button onClick={() => setBenchmarkViewTab('graph')} className={`view-btn ${benchmarkViewTab === 'graph' ? 'active' : ''}`}>Graph</button>
                <button onClick={() => setBenchmarkViewTab('report')} className={`view-btn ${benchmarkViewTab === 'report' ? 'active' : ''}`}>Report</button>
              </>
            )}
          </div>
        </header>

        {/* Tab Content */}
        <div className="tab-content">
          {activeFeatureTab === 'visualize' && (
            <>
              <div className="graph-container">
                {visualizeViewTab === 'graph' ? (
                  <GraphViewer
                    nodes={graphData?.nodes || []}
                    edges={graphData?.edges || []}
                    finalizedCodes={[...inputCodes]}
                    triggerFitToWindow={visualizeFitTrigger}
                    selectedNode={null}
                    onNodeClick={() => { }}
                    codesBarLabel="Submitted Codes"
                    onEmptySpaceClick={() => setSidebarCollapsed(true)}
                  />
                ) : (
                  <VisualizeReportViewer
                    nodes={graphData?.nodes || []}
                    edges={graphData?.edges || []}
                    inputCodes={inputCodes}
                  />
                )}
                {isLoadingGraph && (
                  <div className="loading-state absolute">
                    <div className="spinner"></div>
                    <span>Building Graph...</span>
                  </div>
                )}
              </div>
            </>
          )}

          {activeFeatureTab === 'traverse' && (() => {
            // Only use zero-shot visualization when toggle is ON and data exists
            const useVisualization =
              !(llmConfig.scaffolded ?? true) &&
              (llmConfig.visualizePrediction ?? false) &&
              zeroShotVisualization !== null;

            return (
              <>
                <div className="graph-container">
                  {traverseViewTab === 'graph' ? (
                    <GraphViewer
                      nodes={useVisualization ? zeroShotVisualization.nodes : traverse.state.state.nodes}
                      edges={useVisualization ? zeroShotVisualization.edges : traverse.state.state.edges}
                      selectedNode={traverse.state.selectedNode}
                      onNodeClick={traverse.actions.handleNodeClick}
                      finalizedCodes={traverse.state.state.finalized_codes}
                      isTraversing={traverse.state.state.status === 'traversing'}
                      status={traverse.state.state.status}
                      currentStep={traverse.state.state.current_step}
                      decisions={traverse.state.state.decision_history}
                      triggerFitToWindow={traverse.state.traverseFitTrigger}
                      onGraphInteraction={traverse.actions.handleTraverseGraphInteraction}
                      allowRewind={true}
                      rewindingNodeId={traverse.state.rewindingNodeId}
                      onNodeRewindClick={traverse.actions.handleNodeRewindClick}
                      elapsedTime={traverse.state.traverseElapsedTime}
                      codesBarLabel="Extracted Codes"
                      onEmptySpaceClick={() => setSidebarCollapsed(true)}
                    />
                  ) : (
                    <TrajectoryViewer
                      decisions={traverse.state.state.decision_history}
                      finalizedCodes={traverse.state.state.finalized_codes}
                    />
                  )}
                </div>
              </>
            );
          })()}

          {activeFeatureTab === 'benchmark' && (() => {
            // Simplified view logic: use inferred graph only when zero-shot + complete + toggle ON
            const isZeroShot = !(llmConfig.scaffolded ?? true);
            const isComplete = benchmark.state.benchmarkStatus === 'complete';
            const inferredView = benchmark.state.benchmarkInferredView;
            const showInferredGraph = isZeroShot && isComplete && benchmark.state.benchmarkInferPrecursors && inferredView !== null;

            // When toggle is ON and inferred view exists, use expected graph colored by inferred intersection
            const activeNodes = showInferredGraph ? inferredView.expectedNodesColored : benchmark.state.benchmarkCombinedNodes;
            const activeMetrics = showInferredGraph ? inferredView.metrics : benchmark.state.benchmarkMetrics;
            const activeOvershootMarkers = showInferredGraph ? inferredView.overshootMarkers : benchmark.state.benchmarkOvershootMarkers;
            const activeMissedEdgeMarkers = showInferredGraph ? inferredView.missedEdgeMarkers : benchmark.state.benchmarkMissedEdgeMarkers;
            // Hide overshoot/undershoot in report when zero-shot without infer precursors (no traversal path to compare)
            const hideOvershootUndershoot = isZeroShot && isComplete && !benchmark.state.benchmarkInferPrecursors;
            // Show X markers when scaffolded (always has full traversal path) OR when zero-shot with infer precursors enabled
            const showXMarkers = !isZeroShot || benchmark.state.benchmarkInferPrecursors;

            return (
            <>
              <div className="graph-container">
                {benchmarkViewTab === 'graph' ? (
                  <GraphViewer
                    nodes={activeNodes}
                    edges={benchmark.state.benchmarkCombinedEdges}
                    triggerFitToWindow={benchmark.state.benchmarkFitTrigger}
                    overshootMarkers={activeOvershootMarkers}
                    missedEdgeMarkers={activeMissedEdgeMarkers}
                    onGraphInteraction={benchmark.actions.handleBenchmarkGraphInteraction}
                    selectedNode={null}
                    onNodeClick={() => { }}
                    isTraversing={benchmark.state.benchmarkStatus === 'traversing'}
                    status={benchmark.state.benchmarkStatus}
                    currentStep={benchmark.state.benchmarkCurrentStep}
                    benchmarkMode={true}
                    benchmarkMetrics={activeMetrics}
                    streamingTraversedIds={benchmark.state.streamingTraversedIds}
                    showXMarkers={showXMarkers}
                    expectedLeaves={benchmark.state.benchmarkExpectedCodes}
                    finalizedCodes={[...benchmark.state.benchmarkExpectedCodes]}
                    codesBarLabel={
                      benchmark.state.benchmarkStatus === 'complete'
                        ? "Benchmarked Final Codes"
                        : "Target Codes"
                    }
                    onRemoveExpectedCode={
                      benchmark.state.benchmarkStatus === 'idle'
                        ? benchmark.actions.handleBenchmarkRemoveCode
                        : undefined
                    }
                    elapsedTime={benchmark.state.benchmarkElapsedTime}
                    onEmptySpaceClick={() => setSidebarCollapsed(true)}
                  />
                ) : (
                  <BenchmarkReportViewer
                    metrics={activeMetrics}
                    decisions={benchmark.state.benchmarkDecisions}
                    expectedCodes={benchmark.state.benchmarkExpectedCodes}
                    expectedGraph={benchmark.state.benchmarkExpectedGraph}
                    combinedNodes={activeNodes}
                    traversedNodes={benchmark.state.benchmarkTraversedNodes}
                    status={benchmark.state.benchmarkStatus}
                    currentStep={benchmark.state.benchmarkCurrentStep}
                    errorMessage={benchmark.state.benchmarkError}
                    elapsedTime={benchmark.state.benchmarkElapsedTime}
                    hideOvershootUndershoot={hideOvershootUndershoot}
                  />
                )}
                {benchmark.state.benchmarkIsStarting && (
                  <div className="loading-state absolute">
                    <div className="spinner"></div>
                    <span>{benchmark.state.benchmarkCurrentStep || 'Running Benchmark...'}</span>
                  </div>
                )}
              </div>
            </>
          );
          })()}
        </div>
      </div>

      <NodeRewindModal
        node={traverse.state.rewindTargetNode}
        decision={traverse.state.state.decision_history.find(d => d.current_node === traverse.state.rewindTargetNode?.id) || null}
        isOpen={traverse.state.isRewindModalOpen}
        onClose={traverse.actions.handleRewindClose}
        onSubmit={async (nodeId, feedback) => {
          await traverse.actions.handleRewindSubmit(
            nodeId,
            feedback,
            traverse.state.rewindTargetBatchId || undefined
          );
        }}
        isSubmitting={traverse.state.isRewinding}
        error={traverse.state.rewindError}
        initialFeedback={traverse.state.rewindFeedbackText}
      />
    </div>
  );
}

export default TraversalUI;