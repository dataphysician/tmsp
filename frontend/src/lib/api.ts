/**
 * AG-UI streaming API client for ICD-10-CM traversal.
 *
 * Implements AG-UI protocol over SSE with JSON Patch (RFC 6902)
 * for incremental graph updates.
 */

import type { GraphNode, GraphEdge } from './types';

// AG-UI Event Types
export type AGUIEventType =
  | 'RUN_STARTED'
  | 'RUN_FINISHED'
  | 'STEP_STARTED'
  | 'STEP_FINISHED'
  | 'STATE_SNAPSHOT'
  | 'STATE_DELTA';

// JSON Patch operation (RFC 6902)
export interface JsonPatchOp {
  op: 'add' | 'remove' | 'replace';
  path: string;
  value?: unknown;
}

// Graph state for STATE_SNAPSHOT
export interface GraphStateSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// AG-UI Event
export interface AGUIEvent {
  type: AGUIEventType;
  step_id?: string;
  state?: GraphStateSnapshot;
  delta?: JsonPatchOp[];
  metadata?: Record<string, unknown>;
}

export interface TraversalConfig {
  clinical_note: string;
  provider?: 'openai' | 'cerebras' | 'sambanova' | 'anthropic' | 'vertexai' | 'other';
  api_key?: string;
  model?: string;
  selector?: 'llm' | 'manual';
  max_tokens?: number;
  temperature?: number;
  extra?: Record<string, string>;  // Provider-specific config (e.g., Vertex AI location/project_id)
  system_prompt?: string;  // Custom system prompt (uses default if empty)
  scaffolded?: boolean;    // true = tree traversal (default), false = single-shot
}

/**
 * Stream ICD-10-CM traversal results using AG-UI protocol over SSE.
 *
 * @param config - Traversal configuration
 * @param onEvent - Called for each AG-UI event
 * @param onError - Called on error
 *
 * @returns AbortController to cancel the stream
 *
 * @example
 * ```tsx
 * const controller = streamTraversal(
 *   { clinical_note: "Patient with diabetes...", provider: "openai", api_key: "sk-..." },
 *   (event) => {
 *     switch (event.type) {
 *       case 'STATE_SNAPSHOT':
 *         setNodes(event.state.nodes);
 *         break;
 *       case 'STATE_DELTA':
 *         applyPatch(graphState, event.delta);
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
        body: JSON.stringify({
          clinical_note: config.clinical_note,
          provider: config.provider ?? 'openai',
          api_key: config.api_key ?? '',
          model: config.model ?? null,
          selector: config.selector ?? 'llm',
          max_tokens: config.max_tokens ?? null,
          temperature: config.temperature ?? null,
          extra: config.extra ?? null,  // Provider-specific config (e.g., Vertex AI location/project_id)
          system_prompt: config.system_prompt ?? null,
          scaffolded: config.scaffolded ?? true,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)) as AGUIEvent;
              onEvent(data);
            } catch (parseError) {
              console.error('Failed to parse SSE event:', parseError);
            }
          }
        }
      }
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
    body: JSON.stringify({
      clinical_note: config.clinical_note,
      provider: config.provider ?? 'openai',
      api_key: config.api_key ?? '',
      model: config.model ?? null,
      selector: config.selector ?? 'llm',
      max_tokens: config.max_tokens ?? null,
      temperature: config.temperature ?? null,
      extra: config.extra ?? null,  // Provider-specific config (e.g., Vertex AI location/project_id)
      system_prompt: config.system_prompt ?? null,
      scaffolded: config.scaffolded ?? true,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  return response.json();
}

// Rewind configuration for spot rewind feature
export interface RewindConfig {
  batch_id: string;
  feedback: string;
  provider?: 'openai' | 'cerebras' | 'sambanova' | 'anthropic' | 'vertexai' | 'other';
  api_key?: string;
  model?: string;
  selector?: 'llm' | 'manual';
  max_tokens?: number;
  temperature?: number;
  extra?: Record<string, string>;
  system_prompt?: string;  // Custom system prompt (uses default if empty)
  scaffolded?: boolean;    // true = tree traversal (default), false = single-shot
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
 *
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
        body: JSON.stringify({
          batch_id: config.batch_id,
          feedback: config.feedback,
          provider: config.provider ?? 'openai',
          api_key: config.api_key ?? '',
          model: config.model ?? null,
          selector: config.selector ?? 'llm',
          max_tokens: config.max_tokens ?? null,
          temperature: config.temperature ?? null,
          extra: config.extra ?? null,
          system_prompt: config.system_prompt ?? null,
          scaffolded: config.scaffolded ?? true,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)) as AGUIEvent;
              onEvent(data);
            } catch (parseError) {
              console.error('Failed to parse SSE event:', parseError);
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError(error);
      }
    }
  })();

  return controller;
}

// Graph API response for VISUALIZE tab
export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: { input_count: number; node_count: number };
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