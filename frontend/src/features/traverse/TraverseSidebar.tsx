import type { LLMConfig } from '../../lib/types';
import { LLMSettingsPanel } from '../../components/shared/LLMSettingsPanel';
import { SIDEBAR_TABS, type SidebarTab } from '../../lib/constants';

interface TraverseSidebarProps {
  clinicalNote: string;
  onClinicalNoteChange: (value: string) => void;
  llmConfig: LLMConfig;
  onLLMConfigChange: (config: LLMConfig) => void;
  sidebarTab: SidebarTab;
  onSidebarTabChange: (tab: SidebarTab) => void;
  isLoading: boolean;
  batchCount: number;
  onTraverse: () => { success: boolean; error?: string };
  onCancel: () => void;
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
  onCollapseSidebar,
}: TraverseSidebarProps) {
  const handleTraverseClick = () => {
    const result = onTraverse();
    if (result.success) {
      onCollapseSidebar();
    } else if (result.error?.includes('API key')) {
      onSidebarTabChange(SIDEBAR_TABS.LLM_SETTINGS);
    }
  };

  return (
    <>
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
        <div className="sidebar-tab-content">
          <textarea
            value={clinicalNote}
            onChange={(e) => onClinicalNoteChange(e.target.value)}
            placeholder="Paste or type a clinical note..."
            disabled={isLoading}
          />
        </div>
      )}

      {/* LLM Settings Tab */}
      {sidebarTab === SIDEBAR_TABS.LLM_SETTINGS && (
        <div className="sidebar-tab-content">
          <LLMSettingsPanel
            config={llmConfig}
            onChange={onLLMConfigChange}
            disabled={isLoading}
          />
        </div>
      )}

      {/* Action Buttons */}
      <div className="sidebar-actions">
        <button
          onClick={handleTraverseClick}
          disabled={isLoading || !clinicalNote.trim()}
          className="primary-btn"
        >
          {isLoading ? `Traversing... (${batchCount})` : 'Start Traversal'}
        </button>
        {isLoading && (
          <button onClick={onCancel} className="cancel-btn">
            Cancel
          </button>
        )}
      </div>
    </>
  );
}