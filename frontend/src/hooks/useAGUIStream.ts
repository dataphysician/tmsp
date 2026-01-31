import { useCallback } from 'react';
import { applyPatch, type Operation } from 'fast-json-patch';
import type { AGUIEvent } from '../lib/api';
import type {
  GraphNode,
  GraphEdge,
  DecisionPoint,
  CandidateDecision,
} from '../lib/types';
import { calculateDepthFromCode, mergeById, mergeByKey } from '../lib/graphUtils';

/**
 * Metadata structure from STEP_FINISHED events.
 */
interface StepMetadata {
  candidates: Record<string, string>;
  selected_ids: string[];
  reasoning: string;
  node_id: string;
  batch_type: string;
  selected_details: Record<string, {
    depth: number;
    category: string;
    billable: boolean;
  }>;
}

/**
 * Result from processing a STEP_FINISHED event.
 */
export interface StepResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  decision: DecisionPoint;
}

/**
 * Callbacks for handling AG-UI stream events.
 */
export interface AGUIStreamCallbacks {
  /** Called on RUN_STARTED */
  onStart?: () => void;

  /** Called on STATE_SNAPSHOT with the full state */
  onSnapshot?: (nodes: GraphNode[], edges: GraphEdge[]) => void;

  /** Called on STATE_DELTA with the patched state */
  onDelta?: (nodes: GraphNode[], edges: GraphEdge[]) => void;

  /** Called on STEP_STARTED with the step ID */
  onStepStart?: (stepId: string) => void;

  /** Called on STEP_FINISHED with processed nodes, edges, and decision */
  onStepFinish?: (result: StepResult) => void;

  /** Called on RUN_FINISHED success with final node codes */
  onComplete?: (finalNodes: string[]) => void;

  /** Called on RUN_FINISHED error */
  onError?: (error: string) => void;
}

/**
 * Process STATE_DELTA by applying JSON Patch operations.
 */
function applyStateDelta(
  currentNodes: GraphNode[],
  currentEdges: GraphEdge[],
  delta: Operation[]
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const graphState = {
    nodes: [...currentNodes],
    edges: [...currentEdges],
  };

  try {
    const result = applyPatch(graphState, delta, true, false);
    return {
      nodes: result.newDocument.nodes,
      edges: result.newDocument.edges,
    };
  } catch {
    // Return unchanged on patch failure
    return { nodes: currentNodes, edges: currentEdges };
  }
}

/**
 * Process STEP_FINISHED metadata to extract nodes, edges, and decision.
 */
function processStepFinished(
  metadata: StepMetadata,
  stepId: string | undefined
): StepResult {
  const {
    candidates,
    selected_ids: selectedIds,
    reasoning,
    node_id: nodeId,
    batch_type: batchType,
    selected_details: selectedDetails,
  } = metadata;

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
      edge_type: (isFromRoot || batchType === 'children') ? 'hierarchy' : 'lateral',
      rule: (isFromRoot || batchType === 'children') ? null : batchType,
    });
  }

  // Build decision point for history
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
    depth: stepId?.split('|').length || 1,
    candidates: candidateDecisions,
    selected_codes: selectedIds,
  };

  return { nodes: newNodes, edges: newEdges, decision };
}

/**
 * Hook to create an AG-UI event handler with customizable callbacks.
 *
 * @param callbacks - Object containing callback functions for each event type
 * @param getCurrentState - Function to get current nodes/edges for delta operations
 * @returns Event handler function
 */
export function useAGUIStream(
  callbacks: AGUIStreamCallbacks,
  getCurrentState?: () => { nodes: GraphNode[]; edges: GraphEdge[] }
) {
  const handleEvent = useCallback((event: AGUIEvent) => {
    switch (event.type) {
      case 'RUN_STARTED':
        callbacks.onStart?.();
        break;

      case 'STATE_SNAPSHOT':
        if (event.snapshot) {
          callbacks.onSnapshot?.(
            event.snapshot.nodes as GraphNode[],
            event.snapshot.edges as GraphEdge[]
          );
        }
        break;

      case 'STATE_DELTA':
        if (event.delta && getCurrentState) {
          const { nodes, edges } = getCurrentState();
          const patched = applyStateDelta(nodes, edges, event.delta as Operation[]);
          callbacks.onDelta?.(patched.nodes, patched.edges);
        }
        break;

      case 'STEP_STARTED':
        callbacks.onStepStart?.(event.stepName || '');
        break;

      case 'STEP_FINISHED':
        if (event.metadata) {
          const result = processStepFinished(
            event.metadata as unknown as StepMetadata,
            event.stepName
          );
          callbacks.onStepFinish?.(result);
        }
        break;

      case 'RUN_ERROR':
        callbacks.onError?.(event.error);
        break;

      case 'RUN_FINISHED': {
        const finalNodesRaw = (event.metadata?.final_nodes ?? []) as string[];
        const finalNodes = [...new Set(finalNodesRaw)];
        callbacks.onComplete?.(finalNodes);
        break;
      }
    }
  }, [callbacks, getCurrentState]);

  return { handleEvent };
}

// Re-export utilities for use in feature modules
export { mergeById, mergeByKey, calculateDepthFromCode };