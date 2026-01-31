/**
 * AG-UI streaming API client for ICD-10-CM traversal.
 *
 * Implements AG-UI protocol over SSE with JSON Patch (RFC 6902)
 * for incremental graph updates.
 */

// Re-export types for backwards compatibility
export type {
  AGUIEventType,
  JsonPatchOp,
  GraphStateSnapshot,
  AGUIEvent,
  SSEDecisionData,
  StepFinishedMetadata,
  RunFinishedMetadata,
  TraversalConfig,
  RewindConfig,
  GraphResponse,
  DeleteCacheConfig,
  DeleteCacheResponse,
  ClearAllCacheResponse,
  InvalidateCacheConfig,
  InvalidateCacheResponse,
} from './api.types';

import type { AGUIEvent, TraversalConfig, RewindConfig, GraphResponse, DeleteCacheResponse, DeleteCacheConfig, ClearAllCacheResponse, InvalidateCacheConfig, InvalidateCacheResponse } from './api.types';
import { processSSEStream, buildTraversalRequestBody } from './sse';

/**
 * Stream ICD-10-CM traversal results using AG-UI protocol over SSE.
 *
 * @param config - Traversal configuration
 * @param onEvent - Called for each AG-UI event
 * @param onError - Called on error
 * @returns AbortController to cancel the stream
 *
 * @example
 * ```tsx
 * const controller = streamTraversal(
 *   { clinical_note: "Patient with diabetes...", provider: "openai", api_key: "sk-..." },
 *   (event) => {
 *     switch (event.type) {
 *       case 'STATE_SNAPSHOT':
 *         setNodes(event.snapshot.nodes);
 *         break;
 *       case 'STATE_DELTA':
 *         applyPatch(graphState, event.delta);
 *         break;
 *       case 'RUN_ERROR':
 *         console.error("Run error:", event.error);
 *         break;
 *     }
 *   },
 *   (error) => console.error("Error:", error)
 * );
 *
 * // To cancel:
 * controller.abort();
 * ```
 */
export function streamTraversal(
  config: TraversalConfig,
  onEvent: (event: AGUIEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch('/api/traverse/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(buildTraversalRequestBody(config)),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      await processSSEStream(response, {
        onEvent,
        onError,
        signal: controller.signal,
        logPrefix: 'API',
      });
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError(error);
      }
    }
  })();

  return controller;
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
 * @returns AbortController to cancel the stream
 *
 * @example
 * ```tsx
 * const controller = streamRewind(
 *   {
 *     batch_id: "E08.3|children",
 *     feedback: "Select E08.32 instead - patient has diabetic retinopathy",
 *     provider: "openai",
 *     api_key: "sk-..."
 *   },
 *   (event) => {
 *     switch (event.type) {
 *       case 'STATE_DELTA':
 *         applyPatch(graphState, event.delta);
 *         break;
 *     }
 *   },
 *   (error) => console.error("Rewind error:", error)
 * );
 * ```
 */
export function streamRewind(
  config: RewindConfig,
  onEvent: (event: AGUIEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch('/api/traverse/rewind', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(buildTraversalRequestBody({
          ...config,
          batch_id: config.batch_id,
          feedback: config.feedback,
        })),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      await processSSEStream(response, {
        onEvent,
        onError,
        signal: controller.signal,
        logPrefix: 'API Rewind',
      });
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError(error);
      }
    }
  })();

  return controller;
}

/**
 * Build a graph from ICD-10-CM codes.
 *
 * @param codes - Array of ICD-10-CM codes to visualize
 * @returns Promise with nodes, edges, and stats
 */
export async function buildGraph(codes: string[]): Promise<GraphResponse> {
  const response = await fetch('/api/graph', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ codes }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to build graph');
  }

  return response.json();
}

/**
 * Delete a cached traversal state.
 *
 * Use this when a traversal is cancelled or reset to ensure a fresh run.
 *
 * @param config - The same config used for the traversal (to generate matching partition key)
 * @returns Promise with deletion result
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
 *
 * Sets cancelled=True in Burr state and deletes partial cache.
 * Call this when user clicks Cancel to ensure clean termination.
 *
 * @returns Promise with cancellation result
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
 *
 * @returns Promise with clear result
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
 *
 * Preserves older cached runs from previous sessions.
 *
 * @returns Promise with clear result
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
 *
 * Clears everything including older cached runs from previous sessions.
 *
 * @returns Promise with clear result
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
 *
 * Soft invalidation: increments the cache version on the server,
 * causing subsequent runs to miss the old cache and create new entries.
 * Old entries are orphaned but not deleted.
 *
 * @param config - Configuration to invalidate cache for
 * @returns Promise with invalidation result
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
    const error = await response.json();
    throw new Error(error.detail || 'Failed to invalidate cache');
  }

  return response.json();
}
