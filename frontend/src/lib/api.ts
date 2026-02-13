/**
 * AG-UI streaming API client for ICD-10-CM traversal.
 *
 * Uses the official @ag-ui/client HttpAgent for SSE streaming with
 * AG-UI protocol compliance. Non-streaming REST endpoints use fetch directly.
 */

// Re-export types for consumers
export {
  EventType,
  type BaseEvent,
  type RunStartedEvent,
  type RunFinishedEvent,
  type RunErrorEvent,
  type StepStartedEvent,
  type StepFinishedEvent,
  type StateSnapshotEvent,
  type StateDeltaEvent,
  type ReasoningMessageContentEvent,
  type CustomEvent,
  type SSEDecisionData,
  type StepMetadata,
  type RunMetadata,
  type RunResult,
  type TraversalConfig,
  type RewindConfig,
  type GraphResponse,
  type DeleteCacheConfig,
  type DeleteCacheResponse,
  type ClearAllCacheResponse,
  type InvalidateCacheConfig,
  type InvalidateCacheResponse,
} from './api.types';

import type { BaseEvent, TraversalConfig, RewindConfig, GraphResponse, DeleteCacheResponse, DeleteCacheConfig, ClearAllCacheResponse, InvalidateCacheConfig, InvalidateCacheResponse } from './api.types';
import { TMSPAgent, type StreamHandle } from './agent';

/**
 * Create the request body for traversal/rewind API calls.
 * Handles null coalescing for optional fields.
 */
function buildTraversalRequestBody(config: {
  clinical_note: string;
  provider?: string;
  api_key?: string;
  model?: string | null;
  selector?: string;
  max_tokens?: number | null;
  temperature?: number | null;
  extra?: Record<string, string> | null;
  system_prompt?: string | null;
  scaffolded?: boolean;
  persist_cache?: boolean;
  // Additional fields for rewind
  batch_id?: string;
  feedback?: string;
  existing_nodes?: string[];
}): Record<string, unknown> {
  const body: Record<string, unknown> = {
    clinical_note: config.clinical_note,
    provider: config.provider ?? 'openai',
    api_key: config.api_key ?? '',
    model: config.model ?? null,
    selector: config.selector ?? 'llm',
    max_tokens: config.max_tokens ?? null,
    temperature: config.temperature ?? null,
    extra: config.extra ?? null,
    system_prompt: config.system_prompt ?? null,
    scaffolded: config.scaffolded ?? true,
    persist_cache: config.persist_cache ?? true,
  };

  // Add rewind-specific fields if present
  if (config.batch_id !== undefined) {
    body.batch_id = config.batch_id;
  }
  if (config.feedback !== undefined) {
    body.feedback = config.feedback;
  }
  if (config.existing_nodes !== undefined) {
    body.existing_nodes = config.existing_nodes;
  }

  return body;
}

/**
 * Stream ICD-10-CM traversal results using AG-UI protocol over SSE.
 *
 * Uses the official @ag-ui/client HttpAgent for SSE parsing and event streaming.
 *
 * @param config - Traversal configuration
 * @param onEvent - Called for each AG-UI event
 * @param onError - Called on error
 * @returns StreamHandle with abort() to cancel the stream
 */
export function streamTraversal(
  config: TraversalConfig,
  onEvent: (event: BaseEvent) => void,
  onError: (error: Error) => void
): StreamHandle {
  const agent = new TMSPAgent('/api/traverse/stream');
  const body = buildTraversalRequestBody(config);
  const sub = agent.stream(body).subscribe({
    next: onEvent,
    error: onError,
  });
  return {
    abort: () => {
      agent.abortRun();
      sub.unsubscribe();
    },
  };
}

/**
 * Non-streaming traversal API (for simpler use cases).
 *
 * @param config - Traversal configuration
 * @returns Promise with final nodes and batch data
 */
export async function runTraversal(
  config: TraversalConfig
): Promise<{ final_nodes: string[]; batch_data: Record<string, unknown> }> {
  const response = await fetch('/api/traverse', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(buildTraversalRequestBody(config)),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  return response.json();
}

/**
 * Stream a rewind traversal from a specific batch with feedback.
 *
 * Uses Burr's fork pattern to branch from checkpoint and inject feedback.
 * Returns AG-UI SSE stream same as streamTraversal().
 *
 * @param config - Rewind configuration (batch_id, feedback, LLM settings)
 * @param onEvent - Called for each AG-UI event
 * @param onError - Called on error
 * @returns StreamHandle with abort() to cancel the stream
 */
export function streamRewind(
  config: RewindConfig,
  onEvent: (event: BaseEvent) => void,
  onError: (error: Error) => void
): StreamHandle {
  const agent = new TMSPAgent('/api/traverse/rewind');
  const body = buildTraversalRequestBody({
    ...config,
    batch_id: config.batch_id,
    feedback: config.feedback,
  });
  const sub = agent.stream(body).subscribe({
    next: onEvent,
    error: onError,
  });
  return {
    abort: () => {
      agent.abortRun();
      sub.unsubscribe();
    },
  };
}

// Re-export StreamHandle type for consumers
export type { StreamHandle } from './agent';

// ---------- Non-streaming REST API functions ----------

/**
 * Build a graph from ICD-10-CM codes.
 *
 * @param codes - Array of ICD-10-CM codes to visualize
 * @param fullPaths - If true, use full paths to ROOT (no nearest-anchor optimization). Useful for Benchmark mode.
 * @returns Promise with nodes, edges, and stats
 */
export async function buildGraph(codes: string[], fullPaths = false): Promise<GraphResponse> {
  const response = await fetch('/api/graph', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ codes, full_paths: fullPaths }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to build graph');
  }

  return response.json();
}

/**
 * Delete a cached traversal state.
 */
export async function deleteCacheEntry(config: DeleteCacheConfig): Promise<DeleteCacheResponse> {
  const response = await fetch('/api/cache/delete', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      clinical_note: config.clinical_note,
      provider: config.provider,
      model: config.model,
      temperature: config.temperature ?? 0.0,
      scaffolded: config.scaffolded ?? true,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete cache entry');
  }

  return response.json();
}

/**
 * Cancel the currently running traversal.
 */
export async function cancelTraversal(): Promise<{ cancelled: boolean }> {
  const response = await fetch('/api/traverse/cancel', {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to cancel traversal');
  }

  return response.json();
}

/**
 * Clear only the most recent traversal run.
 */
export async function clearLastCache(): Promise<ClearAllCacheResponse> {
  const response = await fetch('/api/cache/clear-last', {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to clear last run cache');
  }

  return response.json();
}

/**
 * Clear cached state from this server session only.
 */
export async function clearSessionCache(): Promise<ClearAllCacheResponse> {
  const response = await fetch('/api/cache/clear-session', {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to clear session cache');
  }

  return response.json();
}

/**
 * Clear all cached state (database + in-memory cache).
 */
export async function clearAllCache(): Promise<ClearAllCacheResponse> {
  const response = await fetch('/api/cache/clear-all', {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to clear cache');
  }

  return response.json();
}

/**
 * Invalidate cache for a specific configuration.
 */
export async function invalidateCache(
  config: InvalidateCacheConfig
): Promise<InvalidateCacheResponse> {
  const response = await fetch('/api/cache/invalidate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      clinical_note: config.clinical_note,
      provider: config.provider,
      model: config.model,
      temperature: config.temperature ?? 0.0,
      system_prompt: config.system_prompt ?? null,
      scaffolded: config.scaffolded ?? true,
    }),
  });

  if (!response.ok) {
    let errorMessage = `Failed to invalidate cache: ${response.status} ${response.statusText}`;
    try {
      const errorBody = await response.json() as { detail?: string };
      if (errorBody.detail) {
        errorMessage = errorBody.detail;
      }
    } catch {
      // Response body wasn't valid JSON, use default message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}
