import { useCallback, useState } from 'react';
import type { LLMConfig } from '../../lib/types';
import { LLMSettingsPanel, ConfirmModal } from '../../components/shared';
import { invalidateCache } from '../../lib/api';
import {
    SIDEBAR_TABS,
    type SidebarTab,
} from '../../lib/constants';

interface BenchmarkSidebarProps {
    clinicalNote: string;
    onClinicalNoteChange: (value: string) => void;
    llmConfig: LLMConfig;
    onLLMConfigChange: (config: LLMConfig) => void;
    sidebarTab: SidebarTab;
    onSidebarTabChange: (tab: SidebarTab) => void;
    isLoading: boolean;
    onStartBenchmark: () => void;
    onCancelBenchmark: () => void;
    onResetBenchmark: () => void;

    // Benchmark specific props
    expectedCodes: Set<string>;
    expectedCodesInput: string;
    onExpectedCodesInputChange: (value: string) => void;
    onAddExpectedCode: () => void;

    // Infer precursor nodes props (for zero-shot benchmark)
    benchmarkInferPrecursors: boolean;
    onBenchmarkInferPrecursorsChange: (value: boolean) => void;
    benchmarkComplete: boolean;
}

export function BenchmarkSidebar({
    clinicalNote,
    onClinicalNoteChange,
    llmConfig,
    onLLMConfigChange,
    sidebarTab,
    onSidebarTabChange,
    isLoading,
    onStartBenchmark,
    onCancelBenchmark,
    onResetBenchmark,
    expectedCodes,
    expectedCodesInput,
    onExpectedCodesInputChange,
    onAddExpectedCode,
    benchmarkInferPrecursors,
    onBenchmarkInferPrecursorsChange,
    benchmarkComplete,
}: BenchmarkSidebarProps) {
    // State for clear cache confirmation modal
    const [showClearCacheModal, setShowClearCacheModal] = useState(false);
    const [isClearing, setIsClearing] = useState(false);

    const handleStartClick = useCallback(() => {
        onStartBenchmark();
        // Don't auto-collapse for benchmark, users might want to watch progress or adjust settings
    }, [onStartBenchmark]);

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

    // Handle expected codes input keydown (Enter to add)
    const handleCodesKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onAddExpectedCode();
        }
    };

    return (
        <div className="sidebar-tab-content">
            {/* Add Expected Codes - Always visible at top */}
            <div className="input-group">
                <label className="input-label">Add Expected Codes</label>
                <div className="input-row">
                    <input
                        type="text"
                        value={expectedCodesInput}
                        onChange={(e) => onExpectedCodesInputChange(e.target.value)}
                        onKeyDown={handleCodesKeyDown}
                        placeholder="e.g., I25.10, E11.9"
                        className="code-input"
                        disabled={isLoading}
                    />
                    <button
                        onClick={onAddExpectedCode}
                        disabled={isLoading || !expectedCodesInput.trim()}
                        className="add-btn"
                    >
                        Add
                    </button>
                </div>
            </div>

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
                    key="benchmark-clinical-note"
                    value={clinicalNote}
                    onChange={(e) => onClinicalNoteChange(e.target.value)}
                    placeholder="Paste or type a clinical note for benchmarking..."
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
                        benchmarkMode={true}
                        benchmarkInferPrecursors={benchmarkInferPrecursors}
                        onBenchmarkInferPrecursorsChange={onBenchmarkInferPrecursorsChange}
                        benchmarkComplete={benchmarkComplete}
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
                    onClick={handleStartClick}
                    disabled={isLoading || !clinicalNote.trim() || expectedCodes.size === 0}
                    className="primary-btn"
                >
                    {isLoading ? 'Running...' : 'Run Benchmark'}
                </button>
                {isLoading ? (
                    <button onClick={onCancelBenchmark} className="secondary-btn">
                        Cancel
                    </button>
                ) : (
                    <button onClick={onResetBenchmark} className="secondary-btn">
                        Reset
                    </button>
                )}
            </div>
        </div>
    );
}
