import { useMemo } from 'react';
import type { GraphNode, GraphEdge } from '../lib/types';
import { TreeView, countTreeNodes } from './TreeView';
import type { TreeNode } from './TreeView';

interface VisualizeReportViewerProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  inputCodes: Set<string>;
}

/**
 * Build a tree from graph nodes and edges.
 */
function buildTreeFromGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  inputCodes: Set<string>
): TreeNode | null {
  if (nodes.length === 0) return null;

  // Create node lookup map
  const nodeMap = new Map<string, GraphNode>();
  nodes.forEach(n => nodeMap.set(n.id, n));

  // Build parent -> children map from hierarchy edges AND lateral edges
  // Also track which lateral rule was used to reach each child
  const childrenMap = new Map<string, string[]>();
  const childRuleMap = new Map<string, string>(); // child -> rule name

  edges
    .filter(e => e.edge_type === 'hierarchy' || e.edge_type === 'lateral')
    .forEach(e => {
      const parent = String(e.source);
      const child = String(e.target);

      if (e.edge_type === 'hierarchy') {
        if (!childrenMap.has(parent)) {
          childrenMap.set(parent, []);
        }
        childrenMap.get(parent)!.push(child);
      } else if (e.edge_type === 'lateral') {
        // Track the rule for this edge
        if (e.rule) {
          childRuleMap.set(`${parent}->${child}`, e.rule);
        }
        // Include lateral edges as children to show cross-references
        if (!childrenMap.has(parent)) {
          childrenMap.set(parent, []);
        }
        if (!childrenMap.get(parent)!.includes(child)) {
          childrenMap.get(parent)!.push(child);
        }
      }
    });

  // Helper to find nearest non-placeholder ancestor label
  const findAncestorLabel = (code: string): string => {
    let currentCode = code;
    const visited = new Set<string>();
    // Walk up the hierarchy via parent map
    while (currentCode && !visited.has(currentCode)) {
      visited.add(currentCode);
      // Find parent from hierarchy edges
      let parentCode: string | null = null;
      for (const edge of edges) {
        if (String(edge.target) === currentCode &&
          (edge.edge_type === 'hierarchy' || (edge.edge_type === 'lateral' && edge.rule === 'sevenChrDef'))) {
          parentCode = String(edge.source);
          break;
        }
      }
      if (!parentCode || parentCode === 'ROOT') break;
      const parentNode = nodeMap.get(parentCode);
      if (parentNode && parentNode.category !== 'placeholder' && parentNode.label) {
        return parentNode.label;
      }
      currentCode = parentCode;
    }
    return '';
  };

  // Recursive function to build tree nodes
  const buildTreeNode = (code: string, parentCode: string | null = null): TreeNode => {
    const graphNode = nodeMap.get(code);
    const childCodes = childrenMap.get(code) || [];

    // Sort children alphabetically
    childCodes.sort((a, b) => a.localeCompare(b));

    // Get rule from parent->child key if this node was reached via lateral edge
    const ruleKey = parentCode ? `${parentCode}->${code}` : null;
    const rule = ruleKey ? childRuleMap.get(ruleKey) || null : null;

    // Determine tag: rule name or FINAL if it's an input code
    let tag: string | undefined;
    let tagStyle: 'default' | 'input' | undefined;
    if (rule) {
      tag = rule;
    } else if (inputCodes.has(code)) {
      tag = 'FINAL';
      tagStyle = 'input';
    }

    // Get the label, combining with ancestor for sevenChrDef nodes
    let label = graphNode?.label || '';
    if (rule === 'sevenChrDef' && label) {
      const ancestorLabel = findAncestorLabel(code);
      if (ancestorLabel) {
        // Extract just the description part from "X: description" format
        const colonIndex = label.indexOf(': ');
        const description = colonIndex >= 0 ? label.slice(colonIndex + 2) : label;
        label = `${ancestorLabel}, ${description}`;
      }
    }

    // Build details with siblings as candidates (for expandable row behavior)
    let details: TreeNode['details'] = undefined;
    if (parentCode && code !== 'ROOT') {
      const siblingCodes = childrenMap.get(parentCode) || [];
      if (siblingCodes.length > 1) {
        // Sort siblings alphabetically
        const sortedSiblings = [...siblingCodes].sort((a, b) => a.localeCompare(b));
        details = {
          candidates: sortedSiblings.map(sibCode => {
            const sibNode = nodeMap.get(sibCode);
            return {
              code: sibCode,
              label: sibNode?.label || '',
              selected: sibCode === code,
            };
          }),
          reasoning: '', // No reasoning for visualize view
        };
      }
    }

    return {
      code,
      label,
      billable: graphNode?.billable || false,
      isHighlighted: inputCodes.has(code),
      tag,
      tagStyle,
      children: childCodes.map(child => buildTreeNode(child, code)),
      details,
    };
  };

  // Build from ROOT
  return buildTreeNode('ROOT', null);
}

export function VisualizeReportViewer({
  nodes,
  edges,
  inputCodes,
}: VisualizeReportViewerProps) {
  const tree = useMemo(
    () => buildTreeFromGraph(nodes, edges, inputCodes),
    [nodes, edges, inputCodes]
  );

  const nodeCount = useMemo(() => countTreeNodes(tree), [tree]);
  const inputCodesArray = useMemo(() => [...inputCodes], [inputCodes]);

  // Header content with status and stats
  const headerContent = (
    <div className="view-header-bar">
      <div className="view-status-section">
        <div className="status-line">
          <span className="status-label">Status:</span>
          {nodes.length === 0 ? (
            <span className="status-value status-idle">IDLE</span>
          ) : (
            <span className="status-value status-complete">COMPLETE</span>
          )}
        </div>
        {nodeCount > 0 && (
          <div className="report-line">
            <span className="report-label">Report:</span>
            <span className="report-stats">
              <strong>{inputCodes.size}</strong> codes visualized
              <span className="stat-separator">·</span>
              <strong>{nodeCount}</strong> nodes in hierarchy
            </span>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="visualize-report-container">
      <TreeView
        tree={tree}
        highlightedCodes={inputCodesArray}
        showBillableIndicator={false}
        emptyStateText="No graph data yet"
        emptyStateHint="Add ICD codes to visualize the hierarchy"
        headerContent={headerContent}
        codesLabel="Submitted Codes"
        detailsLabel="Siblings"
      />
    </div>
  );
}