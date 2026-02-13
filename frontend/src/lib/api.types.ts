/**
 * Type definitions for AG-UI streaming API.
 *
 * AG-UI protocol types (EventType, BaseEvent, and all typed events) are
 * imported from the official @ag-ui/core package. Domain-specific types
 * (step metadata, run result, config types) are defined here.
 */

// Re-export AG-UI protocol types from official package
export { EventType } from '@ag-ui/core';
export type {
  BaseEvent,
  RunStartedEvent,
  RunFinishedEvent,
  RunErrorEvent,
  StepStartedEvent,
  StepFinishedEvent,
  StateSnapshotEvent,
  StateDeltaEvent,
  ReasoningMessageContentEvent,
  CustomEvent,
} from '@ag-ui/core';

// ----- Domain-specific types (NOT part of AG-UI protocol) -----

/** Decision data from cached STATE_SNAPSHOT (server-side format) */
export interface SSEDecisionData {
  batch_id: string;
  node_id: string;
  batch_type: string;
  candidates: Record<string, string>;  // code -> label
  selected_ids: string[];
  reasoning: string;
  selected_details?: Record<string, { depth?: number; category?: string; billable?: boolean }>;
}

/**
 * Domain metadata carried in CustomEvent(name="step_metadata").
 * Reasoning is now sent via REASONING events (AG-UI standard).
 */
export interface StepMetadata {
  node_id: string;
  batch_type: 'children' | 'lateral' | 'zero-shot' | string;
  selected_ids: string[];
  candidates: Record<string, string>;  // code -> label
  error?: boolean;
  selected_details?: Record<string, { depth?: number; category?: string; billable?: boolean }>;
}

/**
 * Domain metadata carried in CustomEvent(name="run_metadata").
 * Previously was on RUN_STARTED.metadata — now a separate CUSTOM event.
 */
export interface RunMetadata {
  clinical_note?: string;
  cached?: boolean;
  rewind_from?: string;  // For rewind runs
  feedback?: string;     // For rewind runs
}

/**
 * Domain result carried in RunFinishedEvent.result.
 * Previously was on RUN_FINISHED.metadata — now in the result field.
 */
export interface RunResult {
  final_nodes?: string[];
  batch_count?: number;
  mode?: 'scaffolded' | 'zero-shot';
  cached?: boolean;
  decisions?: SSEDecisionData[];
  rewind_from?: string;  // For rewind runs
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
  persist_cache?: boolean; // true = cache traversals in SQLite (default), false = no caching
}

// Configuration for streamTraversal
export interface TraversalConfig extends BaseTraversalConfig {}

// Configuration for streamRewind
// Note: Backend is the authoritative source for pre-rewind state - it loads
// cached state, prunes to rewind point, and emits STATE_SNAPSHOT
export interface RewindConfig extends BaseTraversalConfig {
  batch_id: string;
  feedback: string;
}

// Graph API response for VISUALIZE tab
export interface GraphResponse {
  nodes: import('./types').GraphNode[];
  edges: import('./types').GraphEdge[];
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

// Clear all cache response
export interface ClearAllCacheResponse {
  success: boolean;
  message: string;
  entries_deleted: number;
}

// Cache invalidation configuration (soft invalidation via versioning)
export interface InvalidateCacheConfig {
  clinical_note: string;
  provider: LLMProvider;
  model: string;
  temperature?: number;
  system_prompt?: string;
  scaffolded?: boolean;
}

// Cache invalidation response
export interface InvalidateCacheResponse {
  success: boolean;
  new_version: number;
  message: string;
}
