import { useState, useCallback } from 'react';
import type { LLMConfig, LLMProvider } from '../lib/types';
import {
  DEFAULT_MODELS,
  getDefaultTemperature,
  getDefaultMaxTokens,
} from '../lib/constants';

export const DEFAULT_LLM_CONFIG: LLMConfig = {
  provider: 'vertexai',
  apiKey: '',
  model: 'gemini-2.5-flash',
  maxTokens: 64000,
  temperature: 0.0,
  extra: { auth_type: 'api_key', location: 'global', project_id: '' },
  systemPrompt: '',
  scaffolded: true,
};

/**
 * Hook to manage LLM configuration state with smart defaults.
 * Automatically updates model defaults when provider or model changes.
 */
export function useLLMConfig(initialConfig?: Partial<LLMConfig>) {
  const [config, setConfig] = useState<LLMConfig>({
    ...DEFAULT_LLM_CONFIG,
    ...initialConfig,
  });

  const updateProvider = useCallback((provider: LLMProvider) => {
    const model = DEFAULT_MODELS[provider];
    setConfig(prev => ({
      ...prev,
      provider,
      model,
      temperature: getDefaultTemperature(model),
      maxTokens: getDefaultMaxTokens(model),
      // Initialize extra for Vertex AI, clear for other providers
      extra: provider === 'vertexai'
        ? { auth_type: 'api_key', location: prev.extra?.location || 'global', project_id: prev.extra?.project_id ?? '' }
        : undefined,
    }));
  }, []);

  const updateModel = useCallback((model: string) => {
    setConfig(prev => ({
      ...prev,
      model,
      temperature: getDefaultTemperature(model),
      maxTokens: getDefaultMaxTokens(model),
    }));
  }, []);

  const updateApiKey = useCallback((apiKey: string) => {
    setConfig(prev => ({ ...prev, apiKey }));
  }, []);

  const updateMaxTokens = useCallback((maxTokens: number) => {
    setConfig(prev => ({ ...prev, maxTokens }));
  }, []);

  const updateTemperature = useCallback((temperature: number) => {
    setConfig(prev => ({ ...prev, temperature }));
  }, []);

  return {
    config,
    setConfig,
    updateProvider,
    updateModel,
    updateApiKey,
    updateMaxTokens,
    updateTemperature,
  };
}