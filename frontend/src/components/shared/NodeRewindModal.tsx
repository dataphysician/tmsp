/**
 * Modal for rewinding traversal from a specific node with feedback.
 *
 * Shows node details (similar to hover overlay) and provides a textarea
 * for corrective feedback that will guide the LLM to make different selections.
 */

import { useState, useEffect } from 'react';
import type { GraphNode, DecisionPoint } from '../../lib/types';

export interface NodeRewindModalProps {
  /** The node being rewound from */
  node: GraphNode | null;

  /** Decision data for this node (candidates, previous selections, reasoning) */
  decision: DecisionPoint | null;

  /** Whether the modal is open */
  isOpen: boolean;

  /** Close handler */
  onClose: () => void;

  /** Submit handler - triggers rewind */
  onSubmit: (nodeId: string, feedback: string) => Promise<void>;

  /** Loading state during rewind */
  isSubmitting: boolean;

  /** Error message if rewind failed */
  error: string | null;

  /** Initial feedback text (from hover overlay textarea) */
  initialFeedback?: string;
}

export function NodeRewindModal({
  node,
  decision,
  isOpen,
  onClose,
  onSubmit,
  isSubmitting,
  error,
  initialFeedback = '',
}: NodeRewindModalProps) {
  const [feedback, setFeedback] = useState('');

  // Set feedback when modal opens (use initial feedback from hover overlay if provided)
  useEffect(() => {
    if (isOpen) {
      setFeedback(initialFeedback);
    }
  }, [isOpen, node?.id, initialFeedback]);

  const handleSubmit = async () => {
    if (!node || !feedback.trim()) return;
    await onSubmit(node.id, feedback);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Submit on Ctrl/Cmd + Enter
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && feedback.trim() && !isSubmitting) {
      e.preventDefault();
      handleSubmit();
    }
    // Close on Escape
    if (e.key === 'Escape' && !isSubmitting) {
      onClose();
    }
  };

  if (!isOpen || !node) return null;

  // Find the selected candidate's reasoning
  const selectedReasoning = decision?.candidates.find(c => c.selected)?.reasoning || '';

  return (
    <div className="rewind-modal-overlay" onClick={onClose}>
      <div
        className="rewind-modal-content"
        onClick={e => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="rewind-modal-header">
          <h2>Rewind from {node.code}</h2>
          <button
            className="rewind-modal-close-btn"
            onClick={onClose}
            disabled={isSubmitting}
            aria-label="Close modal"
          >
            &times;
          </button>
        </div>

        {/* Node Info Section */}
        <div className="rewind-modal-node-info">
          <div className="rewind-modal-code-line">
            <span className="rewind-modal-code">{node.code}</span>
            <span className={`rewind-modal-billable ${node.billable ? 'is-billable' : ''}`}>
              {node.billable ? '$ Billable' : 'Non-Billable'}
            </span>
          </div>
          <div className="rewind-modal-label">{node.label}</div>
        </div>

        {/* Previous Decision Section */}
        {decision && decision.candidates.length > 0 && (
          <div className="rewind-modal-decision-section">
            <h3>Previous Selection ({decision.current_label})</h3>
            <div className="rewind-modal-candidates-list">
              {decision.candidates.map(c => (
                <div
                  key={c.code}
                  className={`rewind-modal-candidate ${c.selected ? 'selected' : ''}`}
                >
                  <span className="rewind-modal-candidate-code">{c.code}</span>
                  <span className="rewind-modal-candidate-label">{c.label}</span>
                  {c.selected && <span className="rewind-modal-selected-badge">selected</span>}
                </div>
              ))}
            </div>
            {selectedReasoning && (
              <div className="rewind-modal-reasoning">
                <h4>Previous Reasoning</h4>
                <p>{selectedReasoning}</p>
              </div>
            )}
          </div>
        )}

        {/* Feedback Input */}
        <div className="rewind-modal-feedback-section">
          <label htmlFor="rewind-feedback">Correction Feedback</label>
          <textarea
            id="rewind-feedback"
            value={feedback}
            onChange={e => setFeedback(e.target.value)}
            placeholder="e.g., 'Select E08.32 instead - patient has diabetic retinopathy'"
            disabled={isSubmitting}
            rows={3}
            autoFocus
          />
          <p className="rewind-modal-hint">
            Provide specific guidance for the LLM to make a different selection. Press Ctrl+Enter to submit.
          </p>
        </div>

        {/* Error Display */}
        {error && <div className="rewind-modal-error">{error}</div>}

        {/* Actions */}
        <div className="rewind-modal-actions">
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="rewind-modal-cancel-btn"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!feedback.trim() || isSubmitting}
            className="rewind-modal-submit-btn"
          >
            {isSubmitting ? 'Rewinding...' : 'Rewind & Re-traverse'}
          </button>
        </div>
      </div>
    </div>
  );
}
