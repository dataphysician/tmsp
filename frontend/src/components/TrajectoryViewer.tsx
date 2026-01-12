import { useMemo } from 'react';
import type { DecisionPoint, TraversalStatus } from '../lib/types';
import { TreeView, countTreeNodes } from './TreeView';
import type { TreeNode } from './TreeView';

interface TrajectoryViewerProps {
  decisions: DecisionPoint[];
  finalizedCodes: string[];
  status?: TraversalStatus;
  currentStep?: string;
  errorMessage?: string | null;
}

/**
 * Build a tree from traversal decisions.
 * Backend handles all sevenChrDef X-padding and validation - codes are used as-is.
 */
function buildTreeFromDecisions(
  decisions: DecisionPoint[],
  finalizedCodes: string[]
): TreeNode | null {
  if (decisions.length === 0) return null;

  const finalizedSet = new Set(finalizedCodes);

  // Build maps for tree construction
  interface NodeInfo {
    parent: string;
    batchType: string;
    candidates: { code: string; label: string; billable?: boolean }[];
    reasoning: string;
    billable: boolean;
  }
  const childToParentInfo = new Map<string, NodeInfo>();

  // Process all decisions to build parent-child relationships
  decisions.forEach(decision => {
    const batchType = decision.current_label.replace(' batch', '');
    const parentNode = decision.current_node;

    // All batches handled uniformly - backend sends full validated codes
    decision.selected_codes.forEach(selectedCode => {
      // Only store if not already mapped (first occurrence wins - DFS order)
      if (!childToParentInfo.has(selectedCode)) {
        const selectedCandidate = decision.candidates.find(c => c.code === selectedCode);

        childToParentInfo.set(selectedCode, {
          parent: parentNode,
          batchType,
          candidates: decision.candidates.map(c => ({
            code: c.code,
            label: c.label,
            billable: c.billable,
          })),
          reasoning: selectedCandidate?.reasoning || '',
          billable: selectedCandidate?.billable ?? false,
        });
      }
    });
  });

  // Create tree nodes
  const nodeMap = new Map<string, TreeNode>();

  // Helper to find nearest non-placeholder ancestor label
  const findAncestorLabel = (code: string): string => {
    let currentCode = code;
    const visited = new Set<string>();
    while (currentCode && !visited.has(currentCode)) {
      visited.add(currentCode);
      const parentInfo = childToParentInfo.get(currentCode);
      if (!parentInfo || parentInfo.parent === 'ROOT') break;
      const parentCode = parentInfo.parent;
      const parentNodeInfo = childToParentInfo.get(parentCode);
      if (parentNodeInfo && parentNodeInfo.batchType !== 'placeholder') {
        const parentLabel = parentNodeInfo.candidates.find(c => c.code === parentCode)?.label;
        if (parentLabel) return parentLabel;
      }
      currentCode = parentCode;
    }
    return '';
  };

  const createNode = (code: string): TreeNode => {
    if (nodeMap.has(code)) {
      return nodeMap.get(code)!;
    }

    const info = childToParentInfo.get(code);
    const isPlaceholder = info?.batchType === 'placeholder';
    const isSevenChrDef = info?.batchType === 'sevenChrDef';

    // Get the raw label
    let label = isPlaceholder
      ? 'Placeholder'
      : info?.candidates.find(c => c.code === code)?.label || '';

    // For sevenChrDef, combine ancestor label with the 7th char description
    if (isSevenChrDef && label) {
      const ancestorLabel = findAncestorLabel(code);
      if (ancestorLabel) {
        // Extract just the description part from "X: description" format
        const colonIndex = label.indexOf(': ');
        const description = colonIndex >= 0 ? label.slice(colonIndex + 2) : label;
        label = `${ancestorLabel}, ${description}`;
      }
    }

    const node: TreeNode = {
      code,
      label,
      billable: info?.billable ?? false,
      isHighlighted: finalizedSet.has(code),
      tag: info?.batchType,
      children: [],
      details: info && !isPlaceholder && info.candidates.length > 0
        ? {
            candidates: info.candidates.map(c => ({
              code: c.code,
              label: c.label,
              selected: c.code === code,
            })),
            reasoning: info.reasoning,
          }
        : undefined,
    };
    nodeMap.set(code, node);
    return node;
  };

  // Create all nodes
  childToParentInfo.forEach((_, code) => {
    createNode(code);
  });

  // Build parent-child relationships
  nodeMap.forEach((node, code) => {
    const info = childToParentInfo.get(code);
    if (info?.parent && info.parent !== 'ROOT') {
      const parentNode = nodeMap.get(info.parent);
      if (parentNode && !parentNode.children.find(c => c.code === code)) {
        parentNode.children.push(node);
      }
    }
  });

  // Find root-level nodes (direct children of ROOT)
  const rootChildren: TreeNode[] = [];
  nodeMap.forEach((node, code) => {
    const info = childToParentInfo.get(code);
    if (info?.parent === 'ROOT') {
      rootChildren.push(node);
    }
  });

  // Sort children at each level for consistent display
  const sortChildren = (node: TreeNode) => {
    node.children.sort((a, b) => a.code.localeCompare(b.code));
    node.children.forEach(sortChildren);
  };
  rootChildren.forEach(sortChildren);
  rootChildren.sort((a, b) => a.code.localeCompare(b.code));

  // Create ROOT node
  return {
    code: 'ROOT',
    label: '',
    billable: false,
    isHighlighted: false,
    children: rootChildren,
  };
}

export function TrajectoryViewer({
  decisions,
  finalizedCodes,
  status = 'idle',
  currentStep = '',
  errorMessage = null,
}: TrajectoryViewerProps) {
  const tree = useMemo(
    () => buildTreeFromDecisions(decisions, finalizedCodes),
    [decisions, finalizedCodes]
  );

  const nodeCount = useMemo(() => countTreeNodes(tree), [tree]);

  // Determine content to show
  const showLoadingState = decisions.length === 0 && status === 'traversing';
  const showEmptyState = decisions.length === 0 && status !== 'traversing';
  const showBuildingState = decisions.length > 0 && (!tree || tree.children.length === 0);

  // Header content with status and stats
  const headerContent = (
    <div className="view-header-bar">
      <div className="view-status-section">
        <div className="status-line">
          <span className="status-label">Status:</span>
          {status === 'idle' ? (
            <span className="status-value status-idle">IDLE</span>
          ) : status === 'complete' ? (
            <span className="status-value status-complete">COMPLETE</span>
          ) : status === 'error' ? (
            <span className="status-value status-error">ERROR</span>
          ) : (
            <span className="status-value status-processing">PROCESSING</span>
          )}
          {status === 'error' && errorMessage && (
            <span className="status-message">{errorMessage}</span>
          )}
          {status === 'traversing' && currentStep && (
            <span className="status-message">{currentStep}</span>
          )}
        </div>
        {(nodeCount > 0 || finalizedCodes.length > 0 || decisions.length > 0) && (
          <div className="report-line">
            <span className="report-label">Report:</span>
            <span className="report-stats">
              {finalizedCodes.length > 0 && status === 'complete' && (
                <>
                  <strong>{finalizedCodes.length}</strong> codes finalized
                </>
              )}
              {finalizedCodes.length > 0 && status === 'complete' && nodeCount > 0 && (
                <span className="stat-separator">·</span>
              )}
              {nodeCount > 0 && (
                <>
                  <strong>{nodeCount}</strong> nodes explored
                </>
              )}
              {(nodeCount > 0 || (finalizedCodes.length > 0 && status === 'complete')) && decisions.length > 0 && (
                <span className="stat-separator">·</span>
              )}
              {decisions.length > 0 && (
                <>
                  <strong>{decisions.length}</strong> decisions made
                </>
              )}
            </span>
          </div>
        )}
      </div>
    </div>
  );

  // Handle loading/building states separately
  if (showLoadingState) {
    return (
      <div className="trajectory-container">
        {headerContent}
        <div className="trajectory-empty loading-state">
          <div className="spinner" />
          <span>Starting traversal...</span>
        </div>
      </div>
    );
  }

  if (showBuildingState) {
    return (
      <div className="trajectory-container">
        {headerContent}
        <div className="trajectory-empty loading-state">
          <div className="spinner" />
          <span>Traversal in progress...</span>
          <span className="empty-hint">{decisions.length} batches processed</span>
        </div>
      </div>
    );
  }

  return (
    <div className="trajectory-container">
      <TreeView
        tree={showEmptyState ? null : tree}
        highlightedCodes={finalizedCodes}
        showBillableIndicator={true}
        emptyStateText="No traversal data yet"
        emptyStateHint="Start a traversal to see the step-by-step decisions"
        headerContent={headerContent}
        codesLabel="Extracted Codes"
      />
    </div>
  );
}
