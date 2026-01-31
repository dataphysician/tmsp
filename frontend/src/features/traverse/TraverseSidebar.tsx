import { useCallback, useEffect, useState } from 'react';
import type { LLMConfig } from '../../lib/types';
import { LLMSettingsPanel, ConfirmModal } from '../../components/shared';
import { invalidateCache } from '../../lib/api';
import {
  SIDEBAR_TABS,
  type SidebarTab,
  LLM_SYSTEM_PROMPT,
  LLM_SYSTEM_PROMPT_NON_SCAFFOLDED,
} from '../../lib/constants';

interface TraverseSidebarProps {
  clinicalNote: string;
  onClinicalNoteChange: (value: string) => void;
  llmConfig: LLMConfig;
  onLLMConfigChange: (config: LLMConfig) => void;
  sidebarTab: SidebarTab;
  onSidebarTabChange: (tab: SidebarTab) => void;
  isLoading: boolean;
  batchCount: number;
  onTraverse: () => boolean; // Updated to return only boolean success
  onCancel: () => void;
  onClear: () => void; // Reset all state to initial values
  onCollapseSidebar: () => void;
}

export function TraverseSidebar({
  clinicalNote,
  onClinicalNoteChange,
  llmConfig,
  onLLMConfigChange,
  sidebarTab,
  onSidebarTabChange,
  isLoading,
  batchCount,
  onTraverse,
  onCancel,
  onClear,
  onCollapseSidebar,
}: TraverseSidebarProps) {
  // State for clear cache confirmation modal
  const [showClearCacheModal, setShowClearCacheModal] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  const handleTraverseClick = useCallback(() => {
    // onTraverse handles validation and sidebar switching internally
    // It returns true if traversal started successfully
    if (onTraverse()) {
      onCollapseSidebar();
    }
  }, [onTraverse, onCollapseSidebar]);

  // Ctrl+Enter keyboard shortcut to start traversal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'Enter') {
        const canTraverse = !isLoading && clinicalNote.trim();
        if (canTraverse) {
          e.preventDefault();
          handleTraverseClick();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isLoading, clinicalNote, handleTraverseClick]);

  const handleNoteChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    // Defensive check: reject if value matches system prompts
    if (newValue === LLM_SYSTEM_PROMPT || newValue === LLM_SYSTEM_PROMPT_NON_SCAFFOLDED) {
      console.error('[BUG DETECTED] Clinical note onChange received system prompt value!');
      return; // Do NOT update state with system prompt
    }
    onClinicalNoteChange(newValue);
  };

  const handleNotePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const pastedText = e.clipboardData.getData('text');
    if (pastedText === LLM_SYSTEM_PROMPT || pastedText === LLM_SYSTEM_PROMPT_NON_SCAFFOLDED) {
      console.error('[BUG DETECTED] Clipboard contains system prompt!');
    }
  };

  // Show confirmation modal when Clear Cache is clicked
  const handleClearCacheClick = useCallback(() => {
    if (!clinicalNote.trim()) return;
    setShowClearCacheModal(true);
  }, [clinicalNote]);

  // Actually perform cache invalidation after confirmation
  const handleConfirmClearCache = useCallback(async () => {
    setIsClearing(true);
    try {
      await invalidateCache({
        clinical_note: clinicalNote,
        provider: llmConfig.provider,
        model: llmConfig.model || '',
        temperature: llmConfig.temperature,
        system_prompt: llmConfig.systemPrompt,
        scaffolded: llmConfig.scaffolded ?? true,
      });
      setShowClearCacheModal(false);
    } catch (error) {
      console.error('Failed to invalidate cache:', error);
    } finally {
      setIsClearing(false);
    }
  }, [clinicalNote, llmConfig]);

  return (
    <div className="sidebar-tab-content">
      {/* Sidebar Tabs */}
      <div className="sidebar-tabs">
        <button
          className={`sidebar-tab-btn ${sidebarTab === SIDEBAR_TABS.CLINICAL_NOTE ? 'active' : ''}`}
          onClick={() => onSidebarTabChange(SIDEBAR_TABS.CLINICAL_NOTE)}
        >
          Clinical Note
        </button>
        <button
          className={`sidebar-tab-btn ${sidebarTab === SIDEBAR_TABS.LLM_SETTINGS ? 'active' : ''}`}
          onClick={() => onSidebarTabChange(SIDEBAR_TABS.LLM_SETTINGS)}
        >
          LLM Settings
        </button>
      </div>

      {/* Clinical Note Tab */}
      {sidebarTab === SIDEBAR_TABS.CLINICAL_NOTE && (
        <textarea
          key="traverse-clinical-note"
          value={clinicalNote}
          onChange={handleNoteChange}
          onPaste={handleNotePaste}
          placeholder="Paste or type a clinical note..."
          disabled={isLoading}
          className="clinical-note-textarea"
        />
      )}

      {/* LLM Settings Tab */}
      {sidebarTab === SIDEBAR_TABS.LLM_SETTINGS && (
        <div className="llm-settings-scroll-container">
          <LLMSettingsPanel
            config={llmConfig}
            onChange={onLLMConfigChange}
            disabled={isLoading}
            onInvalidateCache={handleClearCacheClick}
          />
        </div>
      )}

      {/* Clear Cache Confirmation Modal */}
      <ConfirmModal
        isOpen={showClearCacheModal}
        title="Clear Cache"
        message="Confirm clear cache?"
        confirmText="Ok"
        cancelText="Cancel"
        onConfirm={handleConfirmClearCache}
        onCancel={() => setShowClearCacheModal(false)}
        isConfirming={isClearing}
      />

      {/* Action Buttons */}
      <div className="sidebar-actions">
        <button
          onClick={handleTraverseClick}
          disabled={isLoading || !clinicalNote.trim()}
          className="primary-btn"
        >
          {isLoading ? `Traversing... (${batchCount})` : 'Start Traversal'}
        </button>
        {isLoading ? (
          <button onClick={onCancel} className="cancel-btn">
            Cancel
          </button>
        ) : (
          <button onClick={onClear} className="secondary-btn">
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
