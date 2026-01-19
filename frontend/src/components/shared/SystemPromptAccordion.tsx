import { useState, useRef, useEffect } from 'react';

interface SystemPromptAccordionProps {
  systemPrompt: string;
  scaffolded: boolean;
  defaultPromptScaffolded: string;
  defaultPromptNonScaffolded: string;
  onSystemPromptChange: (value: string) => void;
  disabled?: boolean;
}

export function SystemPromptAccordion({
  systemPrompt,
  scaffolded,
  defaultPromptScaffolded,
  defaultPromptNonScaffolded,
  onSystemPromptChange,
  disabled,
}: SystemPromptAccordionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [localPrompt, setLocalPrompt] = useState(systemPrompt);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync local prompt with prop when accordion opens
  useEffect(() => {
    if (isExpanded) {
      setLocalPrompt(systemPrompt);
    }
  }, [isExpanded, systemPrompt]);

  // Focus textarea when expanded
  useEffect(() => {
    if (isExpanded && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isExpanded]);

  // Check if prompt is empty (after trimming)
  const isPromptEmpty = !localPrompt.trim();

  // Check if save is allowed (prompt must not be empty)
  const canSave = !isPromptEmpty;

  // Check if collapse is allowed (prompt must not be empty)
  const canCollapse = !isPromptEmpty;

  const handleSave = () => {
    if (!canSave) return;
    onSystemPromptChange(localPrompt);
    setIsExpanded(false);
  };

  const handleDefault = () => {
    const defaultPrompt = scaffolded ? defaultPromptScaffolded : defaultPromptNonScaffolded;
    setLocalPrompt(defaultPrompt);
    onSystemPromptChange(defaultPrompt);
  };

  const handleCollapse = () => {
    if (!canCollapse) return;
    setLocalPrompt(systemPrompt);
    setIsExpanded(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Escape to close without saving (only if prompt is not empty)
    if (e.key === 'Escape') {
      if (canCollapse) {
        setLocalPrompt(systemPrompt);
        setIsExpanded(false);
      }
    }
    // Ctrl+Enter to save (only if prompt is not empty)
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      if (canSave) {
        handleSave();
      }
    }
  };

  return (
    <div className={`system-prompt-accordion ${isExpanded ? 'expanded' : ''}`}>
      {/* Toggle - always visible, at TOP when expanded */}
      <button
        type="button"
        className={`system-prompt-toggle ${isExpanded ? 'expanded' : ''}`}
        onClick={() => isExpanded ? handleCollapse() : setIsExpanded(true)}
        disabled={disabled || (isExpanded && !canCollapse)}
        title={isExpanded && !canCollapse ? 'Cannot close with empty prompt' : undefined}
      >
        <span className="toggle-chevron">{isExpanded ? '▼' : '▲'}</span>
        <span className="toggle-label">
          {isExpanded ? 'Hide System Prompt' : 'Edit System Prompt'}
        </span>
      </button>

      {/* Expanded content - appears BELOW the toggle */}
      {isExpanded && (
        <div className="system-prompt-content">
          {/* Textarea */}
          <textarea
            key="system-prompt-editor"
            ref={textareaRef}
            className="system-prompt-textarea"
            value={localPrompt}
            onChange={(e) => setLocalPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter system prompt..."
            disabled={disabled}
          />

          {/* Footer with action buttons */}
          <div className="system-prompt-footer">
            <div className="system-prompt-actions">
              <button
                type="button"
                className="default-btn"
                onClick={handleDefault}
                disabled={disabled}
              >
                Default
              </button>
              <button
                type="button"
                className={`save-btn ${!canSave ? 'disabled' : ''}`}
                onClick={handleSave}
                disabled={disabled || !canSave}
                title={!canSave ? 'Cannot save empty prompt' : 'Save'}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
