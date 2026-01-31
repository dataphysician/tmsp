import type { LLMConfig, LLMProvider } from '../../lib/types';
import {
  PROVIDER_MODELS,
  DEFAULT_MODELS,
  VERTEXAI_LOCATIONS,
  LLM_SYSTEM_PROMPT,
  LLM_SYSTEM_PROMPT_NON_SCAFFOLDED,
  getDefaultTemperature,
  getDefaultMaxTokens,
} from '../../lib/constants';
import { SystemPromptAccordion } from './SystemPromptAccordion';

interface LLMSettingsPanelProps {
  config: LLMConfig;
  onChange: (config: LLMConfig) => void;
  disabled?: boolean;
  // Cache invalidation callback (called when user clicks "Clear Cache")
  onInvalidateCache?: () => void | Promise<void>;
  // Benchmark mode props for independent Infer Precursor Nodes toggle
  benchmarkMode?: boolean;
  benchmarkInferPrecursors?: boolean;
  onBenchmarkInferPrecursorsChange?: (value: boolean) => void;
  benchmarkComplete?: boolean;  // Only enable toggle after benchmark completes
}

export function LLMSettingsPanel({
  config,
  onChange,
  disabled,
  onInvalidateCache,
  benchmarkMode,
  benchmarkInferPrecursors,
  onBenchmarkInferPrecursorsChange,
  benchmarkComplete,
}: LLMSettingsPanelProps) {
  const handleProviderChange = (provider: LLMProvider) => {
    const model = DEFAULT_MODELS[provider];
    onChange({
      ...config,
      provider,
      model,
      temperature: getDefaultTemperature(model),
      maxTokens: getDefaultMaxTokens(model),
      // Initialize extra for Vertex AI, clear for other providers
      extra: provider === 'vertexai'
        ? { auth_type: 'api_key', location: config.extra?.location || 'global', project_id: config.extra?.project_id ?? '' }
        : undefined,
    });
  };

  const vertexAuthType = config.extra?.auth_type ?? 'api_key';

  const handleExtraChange = (key: string, value: string) => {
    // When switching to ADC mode, ensure location is a valid region (not 'global' or empty)
    let newExtra = { ...config.extra, [key]: value };
    if (key === 'auth_type' && value === 'adc') {
      const currentLoc = config.extra?.location;
      if (!currentLoc || currentLoc === 'global' || currentLoc === '') {
        newExtra.location = 'us-central1';  // Default to us-central1 for ADC
      }
    }
    onChange({
      ...config,
      extra: newExtra,
    });
  };

  const handleModelChange = (model: string) => {
    onChange({
      ...config,
      model,
      temperature: getDefaultTemperature(model),
      maxTokens: getDefaultMaxTokens(model),
    });
  };

  const isCustomModel = config.provider !== 'other' &&
    PROVIDER_MODELS[config.provider].length > 0 &&
    !PROVIDER_MODELS[config.provider].includes(config.model);

  const currentLocation = config.extra?.location || 'global';
  const isCustomLocation = !VERTEXAI_LOCATIONS.includes(currentLocation as typeof VERTEXAI_LOCATIONS[number]);

  return (
    <div className="llm-settings-wrapper">
      <div className="llm-settings-content">
        <div className="setting-row">
          <label>Provider</label>
          <select
            value={config.provider}
            onChange={(e) => handleProviderChange(e.target.value as LLMProvider)}
            disabled={disabled}
          >
            <option value="vertexai">Vertex AI</option>
            <option value="anthropic">Anthropic</option>
            <option value="openai">OpenAI</option>
            <option value="cerebras">Cerebras</option>
            <option value="sambanova">SambaNova</option>
            <option value="other">Other</option>
          </select>
        </div>

        {config.provider === 'vertexai' && (
          <>
            <div className="setting-row">
              <label>Authentication</label>
              <div className="auth-toggle">
                <button
                  type="button"
                  className={`auth-toggle-btn ${vertexAuthType === 'api_key' ? 'active' : ''}`}
                  onClick={() => handleExtraChange('auth_type', 'api_key')}
                  disabled={disabled}
                >
                  API Key
                </button>
                <button
                  type="button"
                  className={`auth-toggle-btn ${vertexAuthType === 'adc' ? 'active' : ''}`}
                  onClick={() => handleExtraChange('auth_type', 'adc')}
                  disabled={disabled}
                >
                  ADC
                </button>
              </div>
            </div>
            <div className="setting-row">
              <label>Location</label>
              <select
                value={isCustomLocation ? '__other__' : currentLocation}
                onChange={(e) => {
                  const value = e.target.value;
                  if (value === '__other__') {
                    handleExtraChange('location', '');
                  } else {
                    handleExtraChange('location', value);
                  }
                }}
                disabled={disabled}
              >
                {VERTEXAI_LOCATIONS.map(loc => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
                <option value="__other__">Other</option>
              </select>
            </div>
            {isCustomLocation && (
              <div className="setting-row">
                <label>Custom Location</label>
                <input
                  type="text"
                  value={currentLocation}
                  onChange={(e) => handleExtraChange('location', e.target.value)}
                  placeholder="e.g., us-west5"
                  disabled={disabled}
                />
              </div>
            )}
            {vertexAuthType === 'adc' && (
              <div className="setting-row">
                <label>Project ID</label>
                <input
                  type="text"
                  value={config.extra?.project_id ?? ''}
                  onChange={(e) => handleExtraChange('project_id', e.target.value)}
                  placeholder="e.g., my-gcp-project-123"
                  disabled={disabled}
                />
              </div>
            )}
          </>
        )}

        {config.provider === 'other' && (
          <div className="setting-row">
            <label>Base URL</label>
            <input
              type="text"
              placeholder="https://api.example.com/v1"
              disabled
            />
          </div>
        )}

        <div className="setting-row">
          <label>{config.provider === 'vertexai' && vertexAuthType === 'adc' ? 'Access Token' : 'API Key'}</label>
          <input
            type="password"
            value={config.apiKey}
            onChange={(e) => onChange({ ...config, apiKey: e.target.value })}
            placeholder={config.provider === 'vertexai' && vertexAuthType === 'adc' ? 'gcloud auth print-access-token' : 'Enter API key...'}
            disabled={disabled}
          />
        </div>

        {config.provider !== 'other' && PROVIDER_MODELS[config.provider].length > 0 ? (
          <>
            <div className="setting-row">
              <label>Model</label>
              <select
                value={isCustomModel ? '__custom__' : config.model}
                onChange={(e) => {
                  const value = e.target.value;
                  if (value === '__custom__') {
                    onChange({ ...config, model: '', temperature: 0.0, maxTokens: 8192 });
                  } else {
                    handleModelChange(value);
                  }
                }}
                disabled={disabled}
              >
                {PROVIDER_MODELS[config.provider].map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
                <option value="__custom__">Other</option>
              </select>
            </div>
            {isCustomModel && (
              <div className="setting-row">
                <label>Custom Model</label>
                <input
                  type="text"
                  value={config.model}
                  onChange={(e) => handleModelChange(e.target.value)}
                  placeholder="Enter model name..."
                  disabled={disabled}
                />
              </div>
            )}
          </>
        ) : (
          <div className="setting-row">
            <label>Model</label>
            <input
              type="text"
              value={config.model}
              onChange={(e) => handleModelChange(e.target.value)}
              placeholder="Enter model name..."
              disabled={disabled}
            />
          </div>
        )}

        <div className="setting-row">
          <label>Max Completion Tokens</label>
          <input
            type="number"
            value={config.maxTokens}
            onChange={(e) => onChange({ ...config, maxTokens: parseInt(e.target.value) || 8192 })}
            min={1}
            max={200000}
            step={1}
            disabled={disabled}
          />
        </div>

        <div className="setting-row">
          <label>Temperature: <span className="temperature-value">{config.temperature.toFixed(1)}</span></label>
          <input
            type="range"
            value={config.temperature}
            onChange={(e) => onChange({ ...config, temperature: parseFloat(e.target.value) })}
            min={0}
            max={2}
            step={0.1}
            disabled={disabled}
          />
        </div>

        <div className="setting-row trajectory-section">
          <label>
            Trajectory Type
            <span
              className="info-tooltip"
              title="Scaffolded: Traverses the ICD-10-CM hierarchy step-by-step, exploring chapters, blocks, categories, subcategories, subclassifications, and extension codes. Also explores cross-references (codeFirst, codeAlso, useAdditionalCode, sevenChrDef).&#10;&#10;Zero-shot: Directly predicts final ICD-10-CM codes without hierarchical traversal."
            >ⓘ</span>
          </label>
          {!(config.scaffolded ?? true) && (
            <label className="infer-precursor-toggle">
              <span>Infer Precursor Nodes</span>
              <input
                type="checkbox"
                checked={benchmarkMode ? (benchmarkInferPrecursors ?? false) : (config.visualizePrediction ?? false)}
                onChange={(e) => {
                  if (benchmarkMode && onBenchmarkInferPrecursorsChange) {
                    onBenchmarkInferPrecursorsChange(e.target.checked);
                  } else {
                    onChange({ ...config, visualizePrediction: e.target.checked });
                  }
                }}
                disabled={disabled || (benchmarkMode && !benchmarkComplete)}
              />
            </label>
          )}
          <div className="trajectory-toggle">
            <button
              type="button"
              className={`trajectory-toggle-btn ${config.scaffolded ?? true ? 'active' : ''}`}
              onClick={() => {
                if (!(config.scaffolded ?? true)) {
                  onChange({ ...config, scaffolded: true, systemPrompt: LLM_SYSTEM_PROMPT });
                }
              }}
              disabled={disabled}
            >
              Scaffolded
            </button>
            <button
              type="button"
              className={`trajectory-toggle-btn ${!(config.scaffolded ?? true) ? 'active' : ''}`}
              onClick={() => {
                if (config.scaffolded ?? true) {
                  onChange({ ...config, scaffolded: false, systemPrompt: LLM_SYSTEM_PROMPT_NON_SCAFFOLDED });
                }
              }}
              disabled={disabled}
            >
              Zero-shot
            </button>
          </div>
        </div>

        {/* Cache */}
        <div className="setting-row cache-row">
          <label className="cache-header-label">
            <span>Cached</span>
            <input
              type="checkbox"
              checked={config.persistCache ?? true}
              onChange={(e) => onChange({ ...config, persistCache: e.target.checked })}
              disabled={disabled}
            />
          </label>
          <button
            type="button"
            className="clear-cache-btn"
            onClick={() => onInvalidateCache?.()}
            disabled={disabled || !(config.persistCache ?? true) || !onInvalidateCache}
            title={config.persistCache ?? true
              ? 'Invalidate cache for current settings'
              : 'Enable caching first'}
          >
            Clear Cache
          </button>
        </div>
      </div>

      <SystemPromptAccordion
        systemPrompt={config.systemPrompt ?? ''}
        scaffolded={config.scaffolded ?? true}
        defaultPromptScaffolded={LLM_SYSTEM_PROMPT}
        defaultPromptNonScaffolded={LLM_SYSTEM_PROMPT_NON_SCAFFOLDED}
        onSystemPromptChange={(value) => onChange({ ...config, systemPrompt: value })}
        disabled={disabled}
      />
    </div>
  );
}