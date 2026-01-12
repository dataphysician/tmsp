import { useState, useMemo } from 'react';
import type { ReactNode } from 'react';

/**
 * Shared tree node structure for both Visualize and Traverse report views.
 */
export interface TreeNode {
  code: string;
  label: string;
  billable: boolean;
  isHighlighted: boolean;  // finalized (traverse) or input code (visualize)
  tag?: string;            // batchType or rule
  tagStyle?: 'default' | 'input';  // styling variant for tag
  children: TreeNode[];
  // Optional expandable details (used by Traverse view)
  details?: {
    candidates: { code: string; label: string; selected: boolean }[];
    reasoning: string;
  };
}

interface TreeNodeComponentProps {
  node: TreeNode;
  depth: number;
  expandedNodes: Set<string>;
  toggleNode: (code: string) => void;
  expandedDetails: Set<string>;
  toggleDetails: (code: string) => void;
  isLast?: boolean;
  showBillableIndicator?: boolean;  // Show BILLABLE/NOT BILLABLE vs just $
  detailsLabel?: string;  // Label for details section
}

function TreeNodeComponent({
  node,
  depth,
  expandedNodes,
  toggleNode,
  expandedDetails,
  toggleDetails,
  isLast = false,
  showBillableIndicator = false,
  detailsLabel = 'Candidates',
}: TreeNodeComponentProps) {
  const isExpanded = expandedNodes.has(node.code);
  const showDetails = expandedDetails.has(node.code);
  const hasChildren = node.children.length > 0;
  const hasDetails = node.details && node.details.candidates.length > 0;

  return (
    <div className={`tree-node ${isLast ? 'last-child' : ''}`}>
      <div className="tree-node-row">
        {/* Tree connector line */}
        {depth > 0 && (
          <div className={`tree-connector ${isLast ? 'corner' : 'branch'}`}>
            <div className="connector-vertical" />
            <div className="connector-horizontal" />
          </div>
        )}
        {/* Toggle area - always reserves space for alignment */}
        <div className={`tree-toggle-area ${hasChildren ? '' : 'no-toggle'}`}>
          {hasChildren && (
            <button
              className={`tree-toggle ${isExpanded ? 'expanded' : 'collapsed'}`}
              onClick={() => toggleNode(node.code)}
            >
              <span className="toggle-icon">{isExpanded ? '−' : '+'}</span>
            </button>
          )}
        </div>

        {/* Node content */}
        <div className={`tree-node-content ${node.isHighlighted ? 'finalized' : ''}`}>
          <div
            className={`tree-node-box ${showDetails ? 'expanded' : ''} ${hasChildren || hasDetails ? 'expandable' : ''}`}
            onClick={() => {
              if (hasChildren && !isExpanded) {
                toggleNode(node.code);
                // Auto-expand details when expanding node with details
                if (hasDetails && !showDetails) {
                  toggleDetails(node.code);
                }
              } else if (hasDetails) {
                toggleDetails(node.code);
              }
            }}
            role="button"
            tabIndex={0}
          >
            <div className="tree-node-header">
              <span className="tree-node-code">{node.code}</span>
              {node.code !== 'ROOT' && node.tag !== 'placeholder' && (
                showBillableIndicator ? (
                  <span className={`tree-node-billable ${node.billable ? 'billable' : 'not-billable'}`}>
                    {node.billable ? 'BILLABLE' : 'NOT BILLABLE'}
                  </span>
                ) : (
                  node.billable && <span className="tree-node-billable billable">$</span>
                )
              )}
              {node.label && (
                <span className="tree-node-label">{node.label}</span>
              )}
              {node.tag && node.tag !== 'children' && node.tag !== 'placeholder' && (
                <span className={`tree-node-type ${node.tagStyle === 'input' ? 'input' : ''}`}>
                  {node.tag}
                </span>
              )}
              {/* Inline hint text for expandable details */}
              {hasDetails && (isExpanded || !hasChildren) && (
                <span className="tree-node-hint">
                  {showDetails ? 'hide' : 'show more'}
                </span>
              )}
            </div>

            {showDetails && node.details && (
              <div className="tree-node-details">
                <div className="detail-section">
                  <div className="detail-label">
                    {detailsLabel} ({node.details.candidates.length})
                  </div>
                  <div className="detail-candidates">
                    {node.details.candidates.map(c => (
                      <div
                        key={c.code}
                        className={`detail-candidate ${c.selected ? 'selected' : ''}`}
                      >
                        <span className="candidate-code">{c.code}</span>
                        <span className="candidate-label">{c.label}</span>
                        {c.selected && (
                          <span className="candidate-check">✓</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {node.details?.reasoning && (
                  <div className="detail-section">
                    <div className="detail-label">Reasoning</div>
                    <div className="detail-reasoning">{node.details.reasoning}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Children */}
      {isExpanded && hasChildren && (
        <div className="tree-children">
          {node.children.map((child, index) => (
            <TreeNodeComponent
              key={child.code}
              node={child}
              depth={depth + 1}
              expandedNodes={expandedNodes}
              toggleNode={toggleNode}
              expandedDetails={expandedDetails}
              toggleDetails={toggleDetails}
              isLast={index === node.children.length - 1}
              showBillableIndicator={showBillableIndicator}
              detailsLabel={detailsLabel}
            />
          ))}
        </div>
      )}
    </div>
  );
}

type SortMode = 'default' | 'asc' | 'desc';

interface TreeViewProps {
  tree: TreeNode | null;
  highlightedCodes: string[];  // finalized codes or input codes
  showBillableIndicator?: boolean;
  emptyStateText?: string;
  emptyStateHint?: string;
  headerContent?: ReactNode;
  codesLabel?: string;  // Label for the codes bar (e.g., "Submitted Codes", "Extracted Codes")
  detailsLabel?: string;  // Label for details section (e.g., "Candidates", "Siblings")
}

export function TreeView({
  tree,
  highlightedCodes,
  showBillableIndicator = false,
  emptyStateText = 'No data yet',
  emptyStateHint = '',
  headerContent,
  codesLabel,
  detailsLabel = 'Candidates',
}: TreeViewProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['ROOT']));
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set());
  const [codeSortMode, setCodeSortMode] = useState<SortMode>('default');

  const sortedCodes = useMemo(() => {
    if (codeSortMode === 'default') return highlightedCodes;
    const sorted = [...highlightedCodes].sort((a, b) => a.localeCompare(b));
    return codeSortMode === 'desc' ? sorted.reverse() : sorted;
  }, [highlightedCodes, codeSortMode]);

  const toggleNode = (code: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(code)) {
        next.delete(code);
      } else {
        next.add(code);
      }
      return next;
    });
  };

  const toggleDetails = (code: string) => {
    setExpandedDetails(prev => {
      const next = new Set(prev);
      if (next.has(code)) {
        next.delete(code);
      } else {
        next.add(code);
      }
      return next;
    });
  };

  const expandAll = () => {
    if (!tree) return;
    const allCodes = new Set<string>();
    const collect = (node: TreeNode) => {
      allCodes.add(node.code);
      node.children.forEach(collect);
    };
    collect(tree);
    setExpandedNodes(allCodes);
  };

  const collapseAll = () => {
    setExpandedNodes(new Set(['ROOT']));
  };

  const showEmptyState = !tree || tree.children.length === 0;
  const showTree = tree && tree.children.length > 0;

  return (
    <div className="tree-view-container">
      {headerContent}

      {showEmptyState && (
        <div className="trajectory-empty">
          <div className="empty-text">{emptyStateText}</div>
          {emptyStateHint && <div className="empty-hint">{emptyStateHint}</div>}
        </div>
      )}

      {showTree && (
        <>
          <div className="trajectory-actions">
            <button onClick={expandAll} className="action-btn">Expand All</button>
            <button onClick={collapseAll} className="action-btn">Collapse All</button>
          </div>

          <div className="trajectory-tree">
            <TreeNodeComponent
              node={tree}
              depth={0}
              expandedNodes={expandedNodes}
              toggleNode={toggleNode}
              expandedDetails={expandedDetails}
              toggleDetails={toggleDetails}
              showBillableIndicator={showBillableIndicator}
              detailsLabel={detailsLabel}
            />
          </div>
        </>
      )}

      {highlightedCodes.length > 0 && (
        <div className="finalized-codes-bar">
          {codesLabel && <span className="codes-bar-label">{codesLabel}</span>}
          <div className="codes-list">
            {sortedCodes.map(code => (
              <span key={code} className="code-badge">{code}</span>
            ))}
          </div>
          <button
            className="sort-toggle"
            onClick={() => {
              if (codeSortMode === 'default') setCodeSortMode('asc');
              else if (codeSortMode === 'asc') setCodeSortMode('desc');
              else setCodeSortMode('default');
            }}
            title={codeSortMode === 'default' ? 'Unsorted' : codeSortMode === 'asc' ? 'Sorted A-Z' : 'Sorted Z-A'}
          >
            <span className="sort-text">Sort</span>
            <span className={`sort-indicator ${codeSortMode !== 'default' ? 'active' : ''}`}>
              {codeSortMode === 'asc' ? '▲' : codeSortMode === 'desc' ? '▼' : '-'}
            </span>
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Utility to count nodes in a tree (excluding ROOT)
 */
export function countTreeNodes(tree: TreeNode | null): number {
  if (!tree) return 0;
  let count = 0;
  const countNodes = (node: TreeNode) => {
    count++;
    node.children.forEach(countNodes);
  };
  countNodes(tree);
  return count - 1; // Exclude ROOT
}
