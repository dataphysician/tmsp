/**
 * Type definitions for AG-UI streaming API.
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

// Provider types supported by the API
export type LLMProvider = 'openai' | 'cerebras' | 'sambanova' | 'anthropic' | 'vertexai' | 'other';

// Base configuration for all traversal operations
export interface BaseTraversalConfig {
  clinical_note: string;
  provider?: LLMProvider;
  api_key?: string;
  model?: string;
  selector?: 'llm' | 'manual';
  max_tokens?: number;
  temperature?: number;
  extra?: Record<string, string>;  // Provider-specific config (e.g., Vertex AI location/project_id)
  system_prompt?: string;  // Custom system prompt (uses default if empty)
  scaffolded?: boolean;    // true = tree traversal (default), false = single-shot
}

// Configuration for streamTraversal
export interface TraversalConfig extends BaseTraversalConfig {}

// Configuration for streamRewind
export interface RewindConfig extends BaseTraversalConfig {
  batch_id: string;
  feedback: string;
  existing_nodes?: string[]; // Node IDs already in the graph (for lateral target parent lookup)
}

// Graph API response for VISUALIZE tab
export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: { input_count: number; node_count: number };
  invalid_codes?: string[];  // Codes that were filtered out (not in flat index)
}

// Cache deletion configuration
export interface DeleteCacheConfig {
  clinical_note: string;
  provider: string;
  model: string;
  temperature?: number;
  scaffolded?: boolean;  // true for tree traversal, false for zero-shot
}

// Cache deletion response
export interface DeleteCacheResponse {
  deleted: boolean;
  partition_key: string;
  message: string;
}
