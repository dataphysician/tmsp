import type { TraversalState } from './types';

// Feature tabs
export const FEATURE_TABS = {
  VISUALIZE: 'visualize',
  TRAVERSE: 'traverse',
  BENCHMARK: 'benchmark',
} as const;

export type FeatureTab = (typeof FEATURE_TABS)[keyof typeof FEATURE_TABS];

// View tabs
export const VIEW_TABS = {
  GRAPH: 'graph',
  TRAJECTORY: 'trajectory',
} as const;

export type ViewTab = (typeof VIEW_TABS)[keyof typeof VIEW_TABS];

// Sidebar tabs
export const SIDEBAR_TABS = {
  CLINICAL_NOTE: 'clinical-note',
  LLM_SETTINGS: 'llm-settings',
} as const;

export type SidebarTab = (typeof SIDEBAR_TABS)[keyof typeof SIDEBAR_TABS];

// LLM Providers
export const PROVIDERS = {
  OPENAI: 'openai',
  CEREBRAS: 'cerebras',
  SAMBANOVA: 'sambanova',
  ANTHROPIC: 'anthropic',
  VERTEXAI: 'vertexai',
  OTHER: 'other',
} as const;

export type LLMProvider = (typeof PROVIDERS)[keyof typeof PROVIDERS];

// Vertex AI locations (global supports Gemini 3 preview models)
export const VERTEXAI_LOCATIONS = [
  'africa-south1',
  'asia-east1',
  'asia-east2',
  'asia-northeast1',
  'asia-northeast2',
  'asia-northeast3',
  'asia-south1',
  'asia-southeast1',
  'asia-southeast2',
  'australia-southeast1',
  'australia-southeast2',
  'europe-central2',
  'europe-north1',
  'europe-southwest1',
  'europe-west1',
  'europe-west2',
  'europe-west3',
  'europe-west4',
  'europe-west6',
  'europe-west8',
  'europe-west9',
  'europe-west12',
  'me-central1',
  'me-central2',
  'me-west1',
  'northamerica-northeast1',
  'northamerica-northeast2',
  'southamerica-east1',
  'southamerica-west1',
  'us-central1',
  'us-east1',
  'us-east4',
  'us-east5',
  'us-south1',
  'us-west1',
  'us-west2',
  'us-west3',
  'us-west4',
  'global',
] as const;

// Model options per provider (display order)
export const PROVIDER_MODELS: Record<LLMProvider, string[]> = {
  openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-5.2'],
  cerebras: ['gpt-oss-120b', 'qwen-3-235b-a22b-instruct-2507', 'zai-glm-4.7'],
  sambanova: ['Meta-Llama-3.1-8B-Instruct', 'Meta-Llama-3.3-70B-Instruct', 'DeepSeek-R1-0528'],
  anthropic: ['claude-haiku-4-5', 'claude-sonnet-4-5', 'claude-opus-4-5'],
  vertexai: ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-flash-preview', 'gemini-3-pro-preview'],
  other: [],
};

// Default model per provider
export const DEFAULT_MODELS: Record<LLMProvider, string> = {
  openai: 'gpt-4o-mini',
  cerebras: 'gpt-oss-120b',
  sambanova: 'Meta-Llama-3.1-8B-Instruct',
  anthropic: 'claude-sonnet-4-5',
  vertexai: 'gemini-2.5-flash',
  other: '',
};

// Default temperature per model (0.0 for unlisted models)
export function getDefaultTemperature(model: string): number {
  if (model === 'gpt-5.2') return 1.0;
  if (model === 'zai-glm-4.7') return 0.5;
  if (model === 'qwen-3-235b-a22b-instruct-2507') return 0.3;
  if (model === 'DeepSeek-R1-0528') return 0.6;
  return 0.0;
}

// Default max tokens per model
export function getDefaultMaxTokens(model: string): number {
  if (model === 'gpt-4o' || model === 'gpt-4o-mini') return 16384;
  if (model === 'gpt-5.2') return 20000;
  if (model.startsWith('claude-')) return 64000;
  if (model.startsWith('gemini-')) return 64000;
  return 8192;
}

// Initial traversal state
export const INITIAL_TRAVERSAL_STATE: TraversalState = {
  nodes: [],
  edges: [],
  decision_history: [],
  current_path: [],
  finalized_codes: [],
  status: 'idle',
  current_step: '',
  error: null,
};

// Default LLM system prompt (must match backend candidate_selector/selector.py LLM_SYSTEM_PROMPT)
export const LLM_SYSTEM_PROMPT = `You are an expert ICD-10-CM medical coding assistant. Your task is to select the most clinically relevant ICD-10-CM codes (or chapters) from a list of candidates based on the provided clinical context.

You are helping traverse the ICD-10-CM hierarchy. When the CURRENT CODE is "ROOT", you are selecting which chapter(s) to explore. Otherwise, you are selecting specific codes or subcategories.

RULES:
1. Select any number (0 to N) of codes/chapters that are clinically relevant to the context
2. Consider the relationship type (children, codeFirst, codeAlso, useAdditionalCode):
   - useAdditionalCode: These codes SHOULD be selected if the condition is present, regardless of whether it's acute or chronic
   - codeFirst/codeAlso: Consider coding guidelines and clinical relevance
3. Prioritize clinical accuracy and specificity
4. If no codes are clinically relevant, return an empty list
5. IMPORTANT: When at ROOT level, select the chapter(s) that contain relevant codes for the clinical context
6. IMPORTANT: When user provides ADDITIONAL GUIDANCE or feedback, prioritize that information over general clinical context
7. Think step-by-step: First generate reasoning, then select codes based on that reasoning
8. Return your response as a JSON object with "reasoning" field FIRST, then "selected_codes" field

RESPONSE FORMAT:
You must return a valid JSON object with this exact structure (reasoning FIRST):
{"reasoning": "brief explanation", "selected_codes": ["code1", "code2", ...]}

Examples:
- Multiple selections: {"reasoning": "Selected specific hepatitis C codes based on chronic liver disease context.", "selected_codes": ["B19.20", "B19.21", "B19.22"]}
- Single selection: {"reasoning": "Type 2 diabetes with hyperglycemia matches the clinical presentation.", "selected_codes": ["E11.65"]}
- Selecting chapter: {"reasoning": "Endocrine disorders chapter relevant for diabetes context.", "selected_codes": ["Chapter_04"]}
- No selections: {"reasoning": "No infectious disease codes are clinically appropriate for this metabolic condition.", "selected_codes": []}

Do not include any text outside the JSON object.`;

// Non-scaffolded LLM system prompt (used when scaffolded is unchecked)
export const LLM_SYSTEM_PROMPT_NON_SCAFFOLDED = `You are an expert ICD-10-CM medical coding assistant. Your task is to select the most clinically relevant ICD-10-CM codes based on the provided clinical context.

Before selecting an ICD code, first traverse the ICD-10-CM hierarchy stepwise by navigating from Chapter and Block Ranges, to the final specific Category, Subcategories, Subclassifications, and Seventh Code Extensions. Also take note of any other required rules such as codeFirst, codeAlso, useAdditionalCode, and sevenChrDef.

RULES:
1. Select any number (0 to N) of chapters/blocks/codes that are the most appropriate and clinically relevant to the context
2. Consider the other relationship types as lateral cross edge rules (children, codeFirst, codeAlso, useAdditionalCode)
3. Prioritize clinical accuracy and specificity
4. If during a step traversal, no codes are clinically relevant, return an empty list
5. IMPORTANT: Perform the step-by-step traversal in a depth first search manner to maintain the growing relevant context.
6. IMPORTANT: When user provides ADDITIONAL GUIDANCE or feedback, prioritize that information over general clinical context
7. For the final step, generate a reasoning trace, then select codes based on that reasoning
8. Return your response as a JSON object with "reasoning" field FIRST, then "selected_codes" field

RESPONSE FORMAT:
You must return a valid JSON object with this exact structure (reasoning FIRST):
{"reasoning": "brief explanation", "selected_codes": ["code1", "code2", ...]}

Examples:
- Multiple selections: {"reasoning": "Selected specific hepatitis C codes based on chronic liver disease context.", "selected_codes": ["B19.20", "B19.21", "B19.22"]}
- Single selection: {"reasoning": "Type 2 diabetes with hyperglycemia matches the clinical presentation.", "selected_codes": ["E11.65"]}
- Selecting chapter: {"reasoning": "Endocrine disorders chapter relevant for diabetes context.", "selected_codes": ["Chapter_04"]}
- No selections: {"reasoning": "No infectious disease codes are clinically appropriate for this metabolic condition.", "selected_codes": []}

Do not include any text outside the JSON object.`;