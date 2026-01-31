/**
 * Type definitions for AG-UI streaming API.
 */

import type { GraphNode, GraphEdge } from './types';

// AG-UI Event Types (AG-UI protocol v0.1)
export type AGUIEventType =
  | 'RUN_STARTED'
  | 'RUN_FINISHED'
  | 'RUN_ERROR'
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

// AG-UI Event base fields
interface AGUIEventBase {
  threadId?: string;
  runId?: string;
  parentRunId?: string;  // For rewind runs that fork from a previous run
}

// Discriminated union types for AG-UI events (AG-UI protocol v0.1)
export interface AGUIRunStartedEvent extends AGUIEventBase {
  type: 'RUN_STARTED';
  metadata?: RunStartedMetadata;
}

export interface AGUIRunFinishedEvent extends AGUIEventBase {
  type: 'RUN_FINISHED';
  metadata?: RunFinishedMetadata;
}

export interface AGUIRunErrorEvent extends AGUIEventBase {
  type: 'RUN_ERROR';
  error: string;
  metadata?: Record<string, unknown>;
}

export interface AGUIStepStartedEvent {
  type: 'STEP_STARTED';
  stepName: string;
}

export interface AGUIStepFinishedEvent {
  type: 'STEP_FINISHED';
  stepName: string;
  metadata?: StepFinishedMetadata;
}

export interface AGUIStateSnapshotEvent {
  type: 'STATE_SNAPSHOT';
  snapshot: GraphStateSnapshot;
}

export interface AGUIStateDeltaEvent {
  type: 'STATE_DELTA';
  delta: JsonPatchOp[];
}

// Union type for all AG-UI events (provides type narrowing via switch/case on type)
export type AGUIEvent =
  | AGUIRunStartedEvent
  | AGUIRunFinishedEvent
  | AGUIRunErrorEvent
  | AGUIStepStartedEvent
  | AGUIStepFinishedEvent
  | AGUIStateSnapshotEvent
  | AGUIStateDeltaEvent;

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

/** Typed metadata for RUN_STARTED events */
export interface RunStartedMetadata {
  clinical_note?: string;
  cached?: boolean;
  rewind_from?: string;  // For rewind runs
  feedback?: string;     // For rewind runs
}

/** Typed metadata for STEP_FINISHED events */
export interface StepFinishedMetadata {
  node_id: string;
  batch_type: 'children' | 'lateral' | 'zero-shot' | string;
  selected_ids: string[];
  reasoning: string;
  candidates: Record<string, string>;  // code -> label
  error?: boolean;
  selected_details?: Record<string, { depth?: number; category?: string; billable?: boolean }>;
}

/** Typed metadata for RUN_FINISHED events */
export interface RunFinishedMetadata {
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
