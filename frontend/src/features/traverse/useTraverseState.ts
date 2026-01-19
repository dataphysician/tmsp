import { useState, useCallback, useRef } from 'react';
import type { TraversalState, LLMConfig } from '../../lib/types';
import { streamTraversal } from '../../lib/api';
import { INITIAL_TRAVERSAL_STATE, type SidebarTab } from '../../lib/constants';
import { useAGUIStream, mergeById, mergeByKey } from '../../hooks/useAGUIStream';
import { useElapsedTime } from '../../hooks/useElapsedTime';

export function useTraverseState(llmConfig: LLMConfig) {
  const [clinicalNote, setClinicalNote] = useState('');
  const [state, setState] = useState<TraversalState>(INITIAL_TRAVERSAL_STATE);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [batchCount, setBatchCount] = useState(0);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('clinical-note');
  const controllerRef = useRef<AbortController | null>(null);

  const elapsedTime = useElapsedTime(state.status);

  // Get current state for delta operations
  const getCurrentState = useCallback(() => ({
    nodes: state.nodes,
    edges: state.edges,
  }), [state.nodes, state.edges]);

  const { handleEvent } = useAGUIStream({
    onStart: () => {
      setState(prev => ({
        ...prev,
        status: 'traversing',
        current_step: 'Starting traversal',
      }));
    },
    onSnapshot: (nodes, edges) => {
      setState(prev => ({
        ...prev,
        nodes,
        edges,
      }));
    },
    onDelta: (nodes, edges) => {
      setState(prev => ({
        ...prev,
        nodes,
        edges,
      }));
    },
    onStepStart: (stepId) => {
      setState(prev => ({
        ...prev,
        current_step: stepId,
      }));
    },
    onStepFinish: (result) => {
      setBatchCount(prev => prev + 1);
      setState(prev => ({
        ...prev,
        nodes: mergeById(prev.nodes, result.nodes),
        edges: mergeByKey(prev.edges, result.edges),
        decision_history: [...prev.decision_history, result.decision],
      }));
    },
    onComplete: (finalNodes) => {
      setState(prev => ({
        ...prev,
        status: 'complete',
        finalized_codes: finalNodes,
        current_step: `Complete - ${finalNodes.length} codes found`,
      }));
      setIsLoading(false);
    },
    onError: (error) => {
      setState(prev => ({
        ...prev,
        status: 'error',
        error,
        current_step: 'Error',
      }));
      setIsLoading(false);
    },
  }, getCurrentState);

  const handleTraverse = useCallback((): { success: boolean; error?: string } => {
    if (!clinicalNote.trim()) {
      return { success: false, error: 'Please enter a clinical note' };
    }
    if (!llmConfig.apiKey) {
      return { success: false, error: 'Please configure your API key in LLM Settings' };
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
        extra: llmConfig.extra,  // Provider-specific config (e.g., Vertex AI location/projectId)
        system_prompt: llmConfig.systemPrompt || undefined,
        scaffolded: llmConfig.scaffolded ?? true,
      },
      handleEvent,
      (error: Error) => {
        setState(prev => ({
          ...prev,
          status: 'error',
          error: error.message,
        }));
        setIsLoading(false);
      }
    );

    return { success: true };
  }, [clinicalNote, llmConfig, handleEvent]);

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

  return {
    // State
    clinicalNote,
    state,
    selectedNode,
    isLoading,
    batchCount,
    sidebarTab,
    elapsedTime,

    // Setters
    setClinicalNote,
    setSidebarTab,

    // Handlers
    handleTraverse,
    handleCancel,
    handleNodeClick,
  };
}