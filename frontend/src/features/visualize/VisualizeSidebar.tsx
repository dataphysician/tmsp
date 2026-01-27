import type { GraphNode, GraphEdge } from '../../lib/types';
import { CodeInput } from '../../components/shared/CodeInput';

interface VisualizeSidebarProps {
  inputCodes: Set<string>;
  codeInput: string;
  graphData: { nodes: GraphNode[]; edges: GraphEdge[] } | null;
  isLoading: boolean;
  onCodeInputChange: (value: string) => void;
  onAddCode: () => void;
  onRemoveCode: (code: string) => void;
  onClearCodes: () => void;
}

export function VisualizeSidebar({
  inputCodes,
  codeInput,
  graphData,
  isLoading,
  onCodeInputChange,
  onAddCode,
  onRemoveCode,
  onClearCodes,
}: VisualizeSidebarProps) {
  /**
   * Get display label for a code, handling 7th character codes specially.
   */
  const getDisplayLabel = (code: string): string => {
    if (!graphData) return '';

    const codeLen = code.replace(/\./g, '').length;

    if (codeLen === 7) {
      // Get the 7th char node's label
      const node = graphData.nodes.find(n => n.code === code);
      const nodeLabel = node?.label || '';
      // Extract value part from "Key: Value" format
      const labelValue = nodeLabel.includes(': ') ? nodeLabel.split(': ').slice(1).join(': ') : nodeLabel;

      // Traverse up hierarchy until finding non-placeholder node
      let ancestorLabel = '';
      let ancestorCode = code.slice(0, -1);
      while (ancestorCode.length > 0) {
        const ancestorNode = graphData.nodes.find(n => n.code === ancestorCode);
        if (ancestorNode && ancestorNode.category !== 'placeholder') {
          ancestorLabel = ancestorNode.label;
          break;
        }
        ancestorCode = ancestorCode.slice(0, -1);
        if (ancestorCode.endsWith('.')) {
          ancestorCode = ancestorCode.slice(0, -1);
        }
      }

      // Combine ancestor label with sevenChrDef value
      if (ancestorLabel && labelValue) {
        return `${ancestorLabel}, ${labelValue}`;
      } else if (ancestorLabel) {
        return ancestorLabel;
      }
      return nodeLabel;
    }

    const node = graphData.nodes.find(n => n.code === code);
    return node?.label || '';
  };

  return (
    <div className="sidebar-tab-content">
      <CodeInput
        value={codeInput}
        onChange={onCodeInputChange}
        onAdd={onAddCode}
        isLoading={isLoading}
      />

      <div className="input-group flex-grow">
        <div className="input-label-row">
          <label className="input-label">
            Input Codes{inputCodes.size > 0 ? ` (${inputCodes.size})` : ''}
          </label>
          {inputCodes.size > 0 && (
            <button onClick={onClearCodes} className="clear-btn">
              Clear
            </button>
          )}
        </div>
        <div
          className="input-codes-table"
          onClick={(e) => {
            // Focus textarea when clicking on table background (not on code rows)
            if (e.target === e.currentTarget || (e.target as HTMLElement).classList.contains('empty-hint')) {
              const textarea = document.querySelector<HTMLTextAreaElement>('textarea.code-input');
              textarea?.focus();
            }
          }}
        >
          {inputCodes.size === 0 ? (
            <span className="empty-hint">No codes added yet</span>
          ) : (
            <table>
              <tbody>
                {[...inputCodes].sort().map(code => (
                  <tr key={code} className="code-row" onClick={() => onRemoveCode(code)}>
                    <td className="code-cell">
                      <span className="code-badge">{code}</span>
                    </td>
                    <td className="label-cell">{getDisplayLabel(code)}</td>
                    <td className="remove-cell">&times;</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}