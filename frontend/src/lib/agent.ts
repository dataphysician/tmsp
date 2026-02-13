/**
 * TMSPAgent — AG-UI HttpAgent subclass for TMSP streaming.
 *
 * Uses the standard RunAgentInput format with domain config in the `state` field.
 * The default HttpAgent.requestInit() handles serialization and SSE headers.
 */

import { HttpAgent, randomUUID } from '@ag-ui/client';
import type { BaseEvent } from '@ag-ui/core';
import type { Observable } from 'rxjs';

/**
 * Handle returned from streaming functions. Provides abort() to cancel.
 */
export interface StreamHandle {
  abort: () => void;
}

/**
 * AG-UI HttpAgent configured for TMSP endpoints.
 *
 * Usage:
 *   const agent = new TMSPAgent('/api/traverse/stream');
 *   const sub = agent.stream(body).subscribe({ next: onEvent, error: onError });
 *   // To cancel: agent.abortRun(); sub.unsubscribe();
 */
export class TMSPAgent extends HttpAgent {
  constructor(url: string) {
    super({ url });
  }

  /**
   * Start streaming with domain config passed via RunAgentInput.state.
   * Returns an Observable<BaseEvent> from the official AG-UI SSE parser.
   */
  stream(config: Record<string, unknown>): Observable<BaseEvent> {
    return this.run({
      threadId: randomUUID(),
      runId: randomUUID(),
      messages: [],
      tools: [],
      context: [],
      state: config,
      forwardedProps: {},
    });
  }
}
