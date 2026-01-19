import { createContext, useContext, type ReactNode } from 'react';
import { useLLMConfig } from '../hooks/useLLMConfig';
import type { LLMConfig, LLMProvider } from '../lib/types';

interface LLMConfigContextValue {
  config: LLMConfig;
  setConfig: (config: LLMConfig) => void;
  updateProvider: (provider: LLMProvider) => void;
  updateModel: (model: string) => void;
  updateApiKey: (apiKey: string) => void;
  updateMaxTokens: (maxTokens: number) => void;
  updateTemperature: (temperature: number) => void;
}

const LLMConfigContext = createContext<LLMConfigContextValue | null>(null);

interface LLMConfigProviderProps {
  children: ReactNode;
}

export function LLMConfigProvider({ children }: LLMConfigProviderProps) {
  const llmConfig = useLLMConfig();

  return (
    <LLMConfigContext.Provider value={llmConfig}>
      {children}
    </LLMConfigContext.Provider>
  );
}

export function useLLMConfigContext(): LLMConfigContextValue {
  const context = useContext(LLMConfigContext);
  if (!context) {
    throw new Error('useLLMConfigContext must be used within a LLMConfigProvider');
  }
  return context;
}