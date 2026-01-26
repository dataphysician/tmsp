/**
 * Server-Sent Events (SSE) streaming utilities for AG-UI protocol.
 *
 * Simplified after AG-UI Protocol alignment: server now emits STATE_SNAPSHOT
 * for cached replays instead of 500+ individual events. The frontend event
 * handlers (handleAGUIEvent, handleBenchmarkEvent) process snapshots directly.
 */

import type { AGUIEvent } from './api.types';

export interface SSEStreamOptions {
  /** Called for each parsed AG-UI event */
  onEvent: (event: AGUIEvent) => void;
  /** Called when an error occurs */
  onError: (error: Error) => void;
  /** Abort signal to cancel the stream */
  signal: AbortSignal;
  /** Optional label for logging (e.g., 'Traversal', 'Rewind') */
  logPrefix?: string;
}

/**
 * Process an SSE response stream.
 *
 * Events are processed immediately as they arrive. For cached replays,
 * the server emits STATE_SNAPSHOT with complete graph instead of individual
 * STEP_STARTED/STATE_DELTA/STEP_FINISHED events, so no accumulation is needed.
 */
export async function processSSEStream(
  response: Response,
  options: SSEStreamOptions
): Promise<void> {
  const { onEvent, onError, signal: _signal, logPrefix = 'API' } = options;

  if (!response.body) {
    throw new Error('No response body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
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
            const event = JSON.parse(line.slice(6)) as AGUIEvent;
            // AG-UI Protocol: Process all events immediately
            // For cached replays, server sends STATE_SNAPSHOT + RUN_FINISHED (3 events)
            // instead of 500+ individual events
            onEvent(event);
          } catch (parseError) {
            console.error(`[${logPrefix}] Failed to parse SSE event:`, parseError);
          }
        }
      }
    }
  } catch (error) {
    if (error instanceof Error && error.name !== 'AbortError') {
      onError(error);
    }
  }
}

/**
 * Create the request body for traversal/rewind API calls.
 * Handles null coalescing for optional fields.
 */
export function buildTraversalRequestBody(config: {
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
