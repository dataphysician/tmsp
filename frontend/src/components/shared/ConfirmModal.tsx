/**
 * Simple confirmation modal for destructive or important actions.
 */

interface ConfirmModalProps {
  /** Whether the modal is open */
  isOpen: boolean;

  /** Modal title */
  title: string;

  /** Message to display */
  message: string;

  /** Confirm button text */
  confirmText?: string;

  /** Cancel button text */
  cancelText?: string;

  /** Called when user confirms */
  onConfirm: () => void;

  /** Called when user cancels or closes */
  onCancel: () => void;

  /** Whether confirm action is in progress */
  isConfirming?: boolean;
}

export function ConfirmModal({
  isOpen,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  onConfirm,
  onCancel,
  isConfirming = false,
}: ConfirmModalProps) {
  if (!isOpen) return null;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && !isConfirming) {
      onCancel();
    }
    if (e.key === 'Enter' && !isConfirming) {
      onConfirm();
    }
  };

  return (
    <div className="confirm-modal-overlay" onClick={onCancel}>
      <div
        className="confirm-modal-content"
        onClick={e => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div className="confirm-modal-header">
          <h2>{title}</h2>
          <button
            className="confirm-modal-close-btn"
            onClick={onCancel}
            disabled={isConfirming}
            aria-label="Close modal"
          >
            &times;
          </button>
        </div>

        <div className="confirm-modal-body">
          <p>{message}</p>
        </div>

        <div className="confirm-modal-actions">
          <button
            onClick={onConfirm}
            disabled={isConfirming}
            className="confirm-modal-confirm-btn"
            autoFocus
          >
            {isConfirming ? 'Processing...' : confirmText}
          </button>
          <button
            onClick={onCancel}
            disabled={isConfirming}
            className="confirm-modal-cancel-btn"
          >
            {cancelText}
          </button>
        </div>
      </div>
    </div>
  );
}
