import { useState, useMemo } from 'react';
import type { BenchmarkMetrics, DecisionPoint, TraversalStatus, ExpectedCodeOutcome, GraphNode, GraphEdge, BenchmarkGraphNode } from '../lib/types';

// Highlight modes for the count grid buttons
// Column-level (Row 1): 'missed' | 'shared' | 'other' - expand interim, highlight column
// Code-level (Row 2): 'matched' | 'undershoot' | 'overshoot' - collapse interim, highlight codes
type HighlightMode = 'missed' | 'shared' | 'other' | 'matched' | 'undershoot' | 'overshoot';

interface BenchmarkReportViewerProps {
  metrics: BenchmarkMetrics | null;
  decisions: DecisionPoint[];
  status?: TraversalStatus;
  currentStep?: string;
  errorMessage?: string | null;
  onCodeClick?: (code: string) => void;
  expectedGraph?: { nodes: GraphNode[]; edges: GraphEdge[] } | null;
  expectedCodes?: Set<string>;
  combinedNodes?: GraphNode[] | BenchmarkGraphNode[];
  traversedNodes?: GraphNode[];  // Raw traversed nodes for interim computation
  // Elapsed time (managed by parent for persistence)
  elapsedTime?: number | null;
  // Hide overshoot/undershoot (for zero-shot without infer precursors)
  hideOvershootUndershoot?: boolean;
}

// Tree node interface for hierarchy display
interface TreeNode {
  code: string;
  label: string;
  depth: number;
  billable: boolean;
  isExpectedCode: boolean;
  rule: string | null;  // Lateral edge rule (e.g., 'sevenChrDef', 'codeFirst', 'codeAlso')
  children: TreeNode[];
}

// Simple Gauge Chart Component
function Gauge({ value, label }: { value: number; label: string }) {
  const radius = 35;
  const stroke = 8;
  const normalized = Math.min(Math.max(value, 0), 1);
  const arcLength = Math.PI * radius;

  return (
    <div className="gauge-item">
      <div className="gauge-chart-wrapper">
        <svg viewBox="0 0 100 55" className="gauge-svg">
          {/* Background Track (Missed/Red Zone) */}
          <path
            d="M 15 50 A 35 35 0 0 1 85 50"
            fill="none"
            stroke="#fee2e2"
            strokeWidth={stroke}
            strokeLinecap="round"
          />

          {/* Foreground Value (Green/Active Zone) */}
          <path
            d="M 15 50 A 35 35 0 0 1 85 50"
            fill="none"
            stroke="#22c55e"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${arcLength}`}
            strokeDashoffset={arcLength * (1 - normalized)}
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          />
        </svg>
        <div className="gauge-value-overlay">{(value * 100).toFixed(1)}%</div>
      </div>
      <div className="gauge-label">{label}</div>
    </div>
  );
}

function TreeNodeComponent({
  node,
  depth,
  expandedNodes,
  toggleNode,
  isLast = false,
}: {
  node: TreeNode;
  depth: number;
  expandedNodes: Set<string>;
  toggleNode: (code: string) => void;
  isLast?: boolean;
}) {
  const isExpanded = expandedNodes.has(node.code);
  const hasChildren = node.children.length > 0;

  return (
    <div className={`tree-node ${isLast ? 'last-child' : ''}`}>
      <div className="tree-node-row">
        {depth > 0 && (
          <div className={`tree-connector ${isLast ? 'corner' : 'branch'}`}>
            <div className="connector-vertical" />
            <div className="connector-horizontal" />
          </div>
        )}
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
        <div className={`tree-node-content ${node.isExpectedCode ? 'finalized' : ''}`}>
          <div
            className={`tree-node-box ${hasChildren ? 'expandable' : ''}`}
            onClick={() => hasChildren && toggleNode(node.code)}
            role="button"
            tabIndex={0}
          >
            <div className="tree-node-header">
              <span className="tree-node-code">{node.code}</span>
              {node.code !== 'ROOT' && node.billable && (
                <span className="tree-node-billable billable">$</span>
              )}
              {node.label && (
                <span className="tree-node-label">{node.label}</span>
              )}
              {node.rule && (
                <span className="tree-node-type">{node.rule}</span>
              )}
              {node.isExpectedCode && (
                <span className="tree-node-type input">TARGET</span>
              )}
            </div>
          </div>
        </div>
      </div>
      {isExpanded && hasChildren && (
        <div className="tree-children">
          {node.children.map((child, index) => (
            <TreeNodeComponent
              key={child.code}
              node={child}
              depth={depth + 1}
              expandedNodes={expandedNodes}
              toggleNode={toggleNode}
              isLast={index === node.children.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function BenchmarkReportViewer({
  metrics,
  decisions: _decisions,
  status = 'idle',
  currentStep = '',
  errorMessage = null,
  onCodeClick,
  expectedGraph = null,
  expectedCodes = new Set(),
  combinedNodes = [],
  traversedNodes = [],
  elapsedTime = null,
  hideOvershootUndershoot = false,
}: BenchmarkReportViewerProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['metrics', 'outcomes'])
  );
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['ROOT']));
  // highlightMode controls box muting, interim expansion, and code highlighting
  const [highlightMode, setHighlightMode] = useState<HighlightMode | null>(null);

  // Format elapsed time as seconds with 1 decimal
  const formatElapsedTime = (ms: number): string => {
    const seconds = ms / 1000;
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  // Handle clicks on Expected, Traversed, Matched, Missed count items
  const handleHighlightClick = (mode: HighlightMode, e: React.MouseEvent) => {
    e.stopPropagation();
    // Toggle off if clicking the same mode
    setHighlightMode(prev => prev === mode ? null : mode);
  };

  // Derive muted state for each venn box based on highlightMode
  const isBoxMuted = (box: 'target' | 'shared' | 'benchmark'): boolean => {
    if (!highlightMode) return false;

    // Column-level highlights (Row 1)
    if (highlightMode === 'missed') return box !== 'target';
    if (highlightMode === 'shared') return box !== 'shared';
    if (highlightMode === 'other') return box !== 'benchmark';

    // Code-level highlights (Row 2) - mute based on where codes live
    if (highlightMode === 'matched' || highlightMode === 'undershoot') return box !== 'shared';
    if (highlightMode === 'overshoot') return box !== 'benchmark';

    return false;
  };

  // Derive expanded state for interim sections based on highlightMode
  // Column-level (Row 1): expand interim for the associated column
  // Code-level (Row 2): collapse all interim (only highlights specific codes)
  const isInterimExpanded = (box: 'target' | 'shared' | 'benchmark'): boolean => {
    if (!highlightMode) return true; // Default: all expanded

    // Column-level highlights (Row 1): expand the associated column's interim
    if (highlightMode === 'missed') return box === 'target';
    if (highlightMode === 'shared') return box === 'shared';
    if (highlightMode === 'other') return box === 'benchmark';

    // Code-level highlights (Row 2): collapse all interim
    if (highlightMode === 'matched' || highlightMode === 'undershoot' || highlightMode === 'overshoot') {
      return false;
    }

    return true;
  };

  // Manual toggle for interim expansion (when clicking box directly)
  // Default: all columns expanded
  const [manualExpandedColumns, setManualExpandedColumns] = useState<Set<string>>(
    new Set(['target', 'shared', 'benchmark'])
  );

  const toggleColumn = (column: string) => {
    // Clear highlight mode when manually toggling
    setHighlightMode(null);
    setManualExpandedColumns(prev => {
      const next = new Set(prev);
      if (next.has(column)) next.delete(column);
      else next.add(column);
      return next;
    });
  };

  // Combined expansion check: highlightMode takes precedence, then manual
  const shouldShowInterim = (box: 'target' | 'shared' | 'benchmark'): boolean => {
    if (highlightMode) return isInterimExpanded(box);
    return manualExpandedColumns.has(box);
  };

  // Build tree from expected graph
  const tree = useMemo(() => {
    if (!expectedGraph || expectedGraph.nodes.length === 0) return null;

    const nodeMap = new Map<string, GraphNode>();
    expectedGraph.nodes.forEach(n => nodeMap.set(n.id, n));

    // Build parent -> children map from hierarchy edges AND lateral edges
    // Also track which lateral rule was used to reach each child
    const childrenMap = new Map<string, string[]>();
    const childRuleMap = new Map<string, string>();  // child -> rule name

    expectedGraph.edges
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
          // Track the rule for this edge (keyed by parent->child to handle multiple paths)
          if (e.rule) {
            childRuleMap.set(`${parent}->${child}`, e.rule);
          }
          // Include ALL lateral edges as children to show cross-references
          if (!childrenMap.has(parent)) {
            childrenMap.set(parent, []);
          }
          // Avoid duplicates if same child already added
          if (!childrenMap.get(parent)!.includes(child)) {
            childrenMap.get(parent)!.push(child);
          }
        }
      });

    // parentCode is used to look up the rule for this edge (parent->code)
    const buildTreeNode = (code: string, parentCode: string | null = null): TreeNode => {
      const graphNode = nodeMap.get(code);
      const childCodes = childrenMap.get(code) || [];
      childCodes.sort((a, b) => a.localeCompare(b));

      // Get rule from parent->child key if this node was reached via lateral edge
      const ruleKey = parentCode ? `${parentCode}->${code}` : null;
      const rule = ruleKey ? childRuleMap.get(ruleKey) || null : null;

      return {
        code,
        label: graphNode?.label || '',
        depth: graphNode?.depth || 0,
        billable: graphNode?.billable || false,
        isExpectedCode: expectedCodes.has(code),
        rule,
        children: childCodes.map(child => buildTreeNode(child, code)),
      };
    };

    return buildTreeNode('ROOT', null);
  }, [expectedGraph, expectedCodes]);

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

  // Count nodes (excluding ROOT)
  const nodeCount = useMemo(() => {
    if (!tree) return 0;
    let count = 0;
    const countNodes = (node: TreeNode) => {
      count++;
      node.children.forEach(countNodes);
    };
    countNodes(tree);
    return count - 1;
  }, [tree]);

  // Compute set of all nodes that have children (expandable nodes)
  const allExpandableNodes = useMemo(() => {
    if (!tree) return new Set<string>();
    const expandable = new Set<string>();
    const collect = (node: TreeNode) => {
      if (node.children.length > 0) {
        expandable.add(node.code);
      }
      node.children.forEach(collect);
    };
    collect(tree);
    return expandable;
  }, [tree]);

  // Expand All disabled when all expandable nodes are already expanded
  const isAllExpanded = allExpandableNodes.size > 0 &&
    [...allExpandableNodes].every(code => expandedNodes.has(code));

  // Collapse All disabled when only ROOT is expanded (default state)
  const isFullyCollapsed = expandedNodes.size === 1 && expandedNodes.has('ROOT');

  // Group outcomes by status
  // When hideOvershootUndershoot is true, treat overshoot/undershoot as missed
  const groupedOutcomes = metrics?.outcomes.reduce(
    (acc, outcome) => {
      if (hideOvershootUndershoot && (outcome.status === 'overshoot' || outcome.status === 'undershoot')) {
        acc.missed.push(outcome);
      } else {
        acc[outcome.status].push(outcome);
      }
      return acc;
    },
    {
      exact: [] as ExpectedCodeOutcome[],
      undershoot: [] as ExpectedCodeOutcome[],
      overshoot: [] as ExpectedCodeOutcome[],
      missed: [] as ExpectedCodeOutcome[],
    }
  );

  // Group INTERIM codes
  // - Target interim: from combinedNodes (expected nodes not traversed)
  // - Shared interim: from combinedNodes (traversed nodes in expected graph)
  // - Benchmark interim: from traversedNodes (traversed nodes NOT in expected graph)
  const { matches: interimTarget, shared: interimShared, benchmark: interimBenchmark } = useMemo(() => {
    const target: GraphNode[] = [];
    const shared: GraphNode[] = [];
    const benchmark: GraphNode[] = [];

    // Allow lookup of whether node is in expected graph
    const expectedNodeIds = new Set(expectedGraph?.nodes.map(n => n.id) || []);

    // Identify finalized codes (final codes in any outcome or otherCodes)
    const finalizedCodes = new Set<string>();
    metrics?.outcomes.forEach(o => {
      if (o.expectedCode) finalizedCodes.add(o.expectedCode);
      if (o.relatedCode) finalizedCodes.add(o.relatedCode);
    });
    metrics?.otherCodes.forEach(code => finalizedCodes.add(code));

    // Build set of traversed node IDs for quick lookup (authoritative source for what was visited)
    const traversedIds = new Set(traversedNodes?.map(n => n.id) || []);

    // 1. Target + Shared interim: from combinedNodes (expected graph with status overlays)
    const nodes = combinedNodes || [];
    nodes.forEach(n => {
      // Exclude ROOT and finalized nodes
      if (n.id === 'ROOT' || finalizedCodes.has(n.id)) return;

      const status = 'benchmarkStatus' in n ? n.benchmarkStatus : undefined;

      if (status === 'expected' && !traversedIds.has(n.id)) {
        // Target interim: expected nodes that were never traversed (missed path ancestors)
        target.push(n);
      } else if (traversedIds.has(n.id) && expectedNodeIds.has(n.id)) {
        // Shared interim: traversed nodes that are in expected graph
        shared.push(n);
      }
    });

    // 2. Benchmark interim: from traversedNodes (raw streamed nodes NOT in expected graph)
    const traversed = traversedNodes || [];
    traversed.forEach(n => {
      // Exclude ROOT and finalized nodes
      if (n.id === 'ROOT' || finalizedCodes.has(n.id)) return;
      // Only include nodes NOT in expected graph
      if (!expectedNodeIds.has(n.id)) {
        benchmark.push(n);
      }
    });

    // Sorting by depth/code to ensure logical order
    const sorter = (a: { depth: number; id: string }, b: { depth: number; id: string }) =>
      (a.depth - b.depth) || a.id.localeCompare(b.id);
    target.sort(sorter);
    shared.sort(sorter);
    benchmark.sort(sorter);

    return { matches: target, shared: shared, benchmark: benchmark };
  }, [combinedNodes, traversedNodes, expectedGraph, metrics]);

  // Node lookup for hover labels
  const nodeMap = useMemo(() => {
    if (!expectedGraph) return new Map<string, GraphNode>();
    const map = new Map<string, GraphNode>();
    expectedGraph.nodes.forEach(n => map.set(n.id, n));
    return map;
  }, [expectedGraph]);

  const getNodeLabel = (code: string): string => {
    const node = nodeMap.get(code);
    return node?.label || code;
  };


  const showTree = tree && tree.children.length > 0;
  const hasExpectedCodes = expectedCodes.size > 0;

  return (
    <div className="benchmark-report-container">
      {/* Header */}
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
            {(status === 'complete' || status === 'error') && elapsedTime !== null && (
              <span className="status-elapsed">({formatElapsedTime(elapsedTime)})</span>
            )}
            {status === 'error' && errorMessage && (
              <span className="status-message">{errorMessage}</span>
            )}
            {status === 'traversing' && currentStep && (
              <span className="status-message">{currentStep}</span>
            )}
          </div>
          {nodeCount > 0 && !metrics && (
            <div className="report-line">
              <span className="report-label">Report:</span>
              <span className="report-stats">
                <strong>{expectedCodes.size}</strong> target codes
                <span className="stat-separator">·</span>
                <strong>{nodeCount}</strong> nodes in hierarchy
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Expected Hierarchy Tree - Show when codes are added but no metrics yet */}
      {showTree && !metrics && (
        <>
          <div className="trajectory-actions">
            <button onClick={expandAll} className="action-btn" disabled={isAllExpanded}>Expand All</button>
            <button onClick={collapseAll} className="action-btn" disabled={isFullyCollapsed}>Collapse All</button>
          </div>
          <div className="trajectory-tree">
            <TreeNodeComponent
              node={tree}
              depth={0}
              expandedNodes={expandedNodes}
              toggleNode={toggleNode}
            />
          </div>
        </>
      )}

      {/* Metrics and Outcomes Container - updated layout */}
      {metrics && (
        <div className="report-sections-scroll">
          <div className="report-section">
            <button
              className={`section-header ${expandedSections.has('metrics') ? 'expanded' : ''}`}
              onClick={() => toggleSection('metrics')}
            >
              <span className="section-toggle">{expandedSections.has('metrics') ? '−' : '+'}</span>
              <span className="section-title">Benchmark Metrics</span>
            </button>
            {expandedSections.has('metrics') && (
              <div className="section-content metrics-panel">
                {/* Gauges (Left Side) */}
                <div className="metrics-gauges">
                  <Gauge value={metrics.traversalRecall} label="Traversal Recall" />
                  <Gauge value={metrics.finalCodesRecall} label="Final Codes Recall" />
                </div>

                {/* Count Summary (Right Side) */}
                <div className="metrics-counts compact">
                  <div className="count-grid two-row">
                    {/* Row 1: Column-level highlights (expand interim, highlight column) */}
                    <div className="count-row column-level">
                      <div
                        className={`count-item missed interactive ${highlightMode === 'missed' ? 'selected' : ''}`}
                        onClick={(e) => handleHighlightClick('missed', e)}
                        role="button"
                        tabIndex={0}
                      >
                        <span className="count-value">{hideOvershootUndershoot ? metrics.missedCount + metrics.overshootCount + metrics.undershootCount : metrics.missedCount}</span>
                        <span className="count-label">Missed</span>
                      </div>
                      <div
                        className={`count-item interactive ${highlightMode === 'shared' ? 'selected' : ''}`}
                        onClick={(e) => handleHighlightClick('shared', e)}
                        role="button"
                        tabIndex={0}
                      >
                        <span className="count-value">{metrics.exactCount + (hideOvershootUndershoot ? 0 : metrics.undershootCount)}</span>
                        <span className="count-label">Correct</span>
                      </div>
                      <div
                        className={`count-item interactive ${highlightMode === 'other' ? 'selected' : ''}`}
                        onClick={(e) => handleHighlightClick('other', e)}
                        role="button"
                        tabIndex={0}
                      >
                        <span className="count-value">{metrics.otherCount + (hideOvershootUndershoot ? 0 : metrics.overshootCount)}</span>
                        <span className="count-label">Extra</span>
                      </div>
                    </div>
                    {/* Row 2: Code-level highlights (collapse interim, highlight codes) */}
                    <div className="count-row code-level">
                      <div
                        className={`count-item exact interactive ${highlightMode === 'matched' ? 'selected' : ''}`}
                        onClick={(e) => handleHighlightClick('matched', e)}
                        role="button"
                        tabIndex={0}
                      >
                        <span className="count-value">{metrics.exactCount}</span>
                        <span className="count-label">Matched</span>
                      </div>
                      <div
                        className={`count-item undershoot interactive ${highlightMode === 'undershoot' ? 'selected' : ''} ${hideOvershootUndershoot ? 'disabled' : ''}`}
                        onClick={(e) => !hideOvershootUndershoot && handleHighlightClick('undershoot', e)}
                        role="button"
                        tabIndex={hideOvershootUndershoot ? -1 : 0}
                      >
                        <span className="count-value">{hideOvershootUndershoot ? 0 : metrics.undershootCount}</span>
                        <span className="count-label">Undershot</span>
                      </div>
                      <div
                        className={`count-item overshoot interactive ${highlightMode === 'overshoot' ? 'selected' : ''} ${hideOvershootUndershoot ? 'disabled' : ''}`}
                        onClick={(e) => !hideOvershootUndershoot && handleHighlightClick('overshoot', e)}
                        role="button"
                        tabIndex={hideOvershootUndershoot ? -1 : 0}
                      >
                        <span className="count-value">{hideOvershootUndershoot ? 0 : metrics.overshootCount}</span>
                        <span className="count-label">Overshot</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Venn Diagram Code Breakdown */}
          {groupedOutcomes && (
            <div className="benchmark-venn-container">
              <div className="venn-diagram">
                {/* TARGET CODES (Left) - Missed codes */}
                <div
                  className={`venn-region venn-expected ${isBoxMuted('target') ? 'muted' : ''}`}
                  onClick={() => toggleColumn('target')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="venn-region-label interactive">
                    Missed
                  </div>
                  <div className="venn-codes">
                    {groupedOutcomes.missed.map(outcome => (
                      <span
                        key={outcome.expectedCode}
                        className={`venn-code-tag expected-region ${highlightMode === 'missed' ? 'highlighted' : ''} ${highlightMode === 'matched' || highlightMode === 'undershoot' || highlightMode === 'overshoot' || highlightMode === 'shared' || highlightMode === 'other' ? 'muted' : ''}`}
                        title={`${getNodeLabel(outcome.expectedCode)}\nStatus: Missed`}
                        onClick={(e) => { e.stopPropagation(); onCodeClick?.(outcome.expectedCode); }}
                      >
                        {outcome.expectedCode}
                      </span>
                    ))}
                    {groupedOutcomes.missed.length === 0 && (
                      <span className="venn-empty">None</span>
                    )}
                  </div>

                  {/* EXPAND TOGGLE (shown when collapsed) */}
                  {!shouldShowInterim('target') && (
                    <div className="venn-expand-toggle">▲ Show Interim Nodes</div>
                  )}

                  {shouldShowInterim('target') && (
                    <div className="venn-interim-codes">
                      {/* COLLAPSE TOGGLE (shown when expanded, above header) */}
                      <div className="venn-expand-toggle">▼ Hide Interim Nodes</div>
                      <div className="venn-interim-header">Interim Nodes</div>
                      <div className="venn-codes">
                        {interimTarget.map(n => (
                          <span
                            key={n.id}
                            className={`venn-code-tag expected-region dimmed-interim ${highlightMode === 'missed' ? 'highlighted' : ''}`}
                            title={`${n.label}\nStatus: Expected Path`}
                            onClick={(e) => { e.stopPropagation(); onCodeClick?.(n.id); }}
                          >
                            {n.id}
                          </span>
                        ))}
                        {interimTarget.length === 0 && <span className="venn-empty">No interim nodes</span>}
                      </div>
                    </div>
                  )}
                </div>

                {/* SHARED CODES (Center) - Matched + Undershoot */}
                <div
                  className={`venn-region venn-intersection ${isBoxMuted('shared') ? 'muted' : ''}`}
                  onClick={() => toggleColumn('shared')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="venn-region-label interactive">
                    Correct
                  </div>
                  <div className="venn-codes">
                    {groupedOutcomes.exact.map(outcome => (
                      <span
                        key={outcome.expectedCode}
                        className={`venn-code-tag intersection-region exact ${highlightMode === 'matched' || highlightMode === 'shared' ? 'highlighted' : ''} ${highlightMode === 'undershoot' || highlightMode === 'overshoot' || highlightMode === 'missed' || highlightMode === 'other' ? 'muted' : ''}`}
                        title={`${getNodeLabel(outcome.expectedCode)}\nStatus: Matched`}
                        onClick={(e) => { e.stopPropagation(); onCodeClick?.(outcome.expectedCode); }}
                      >
                        {outcome.expectedCode}
                      </span>
                    ))}
                    {groupedOutcomes.undershoot.map(outcome => (
                      <span
                        key={outcome.expectedCode}
                        className={`venn-code-tag intersection-region undershoot ${highlightMode === 'undershoot' || highlightMode === 'shared' ? 'highlighted' : ''} ${highlightMode === 'matched' || highlightMode === 'overshoot' || highlightMode === 'missed' || highlightMode === 'other' ? 'muted' : ''}`}
                        title={`${getNodeLabel(outcome.relatedCode || outcome.expectedCode)}\nStatus: Undershot (expected ${outcome.expectedCode})`}
                        onClick={(e) => { e.stopPropagation(); onCodeClick?.(outcome.relatedCode ?? outcome.expectedCode); }}
                      >
                        {outcome.relatedCode || outcome.expectedCode}
                      </span>
                    ))}
                    {groupedOutcomes.exact.length === 0 && groupedOutcomes.undershoot.length === 0 && (
                      <span className="venn-empty">None</span>
                    )}
                  </div>

                  {/* EXPAND TOGGLE (shown when collapsed) */}
                  {!shouldShowInterim('shared') && (
                    <div className="venn-expand-toggle">▲ Show Interim Nodes</div>
                  )}

                  {shouldShowInterim('shared') && (
                    <div className="venn-interim-codes">
                      {/* COLLAPSE TOGGLE (shown when expanded, above header) */}
                      <div className="venn-expand-toggle">▼ Hide Interim Nodes</div>
                      <div className="venn-interim-header">Interim Nodes</div>
                      <div className="venn-codes">
                        {interimShared.map(n => (
                          <span
                            key={n.id}
                            className={`venn-code-tag intersection-region dimmed-interim ${highlightMode === 'shared' ? 'highlighted' : ''}`}
                            title={`${n.label}\nStatus: Traversed Path`}
                            onClick={(e) => { e.stopPropagation(); onCodeClick?.(n.id); }}
                          >
                            {n.id}
                          </span>
                        ))}
                        {interimShared.length === 0 && <span className="venn-empty">No interim nodes</span>}
                      </div>
                    </div>
                  )}
                </div>

                {/* BENCHMARK CODES (Right) - Overshoot + Alt Paths */}
                <div
                  className={`venn-region venn-benchmark ${isBoxMuted('benchmark') ? 'muted' : ''}`}
                  onClick={() => toggleColumn('benchmark')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="venn-region-label interactive">
                    Extra
                  </div>
                  <div className="venn-codes">
                    {groupedOutcomes.overshoot.map(outcome => (
                      <span
                        key={outcome.expectedCode}
                        className={`venn-code-tag benchmark-region overshoot ${highlightMode === 'overshoot' || highlightMode === 'other' ? 'highlighted' : ''} ${highlightMode === 'matched' || highlightMode === 'undershoot' || highlightMode === 'missed' || highlightMode === 'shared' ? 'muted' : ''}`}
                        title={`${getNodeLabel(outcome.relatedCode || outcome.expectedCode)}\nStatus: Overshot (expected ${outcome.expectedCode})`}
                        onClick={(e) => { e.stopPropagation(); onCodeClick?.(outcome.relatedCode ?? outcome.expectedCode); }}
                      >
                        {outcome.relatedCode || outcome.expectedCode}
                      </span>
                    ))}
                    {metrics && metrics.otherCodes.map(code => (
                      <span
                        key={code}
                        className={`venn-code-tag benchmark-region other ${highlightMode === 'other' ? 'highlighted' : ''} ${highlightMode === 'matched' || highlightMode === 'undershoot' || highlightMode === 'overshoot' || highlightMode === 'missed' || highlightMode === 'shared' ? 'muted' : ''}`}
                        title={`${getNodeLabel(code)}\nStatus: Extra`}
                        onClick={(e) => { e.stopPropagation(); onCodeClick?.(code); }}
                      >
                        {code}
                      </span>
                    ))}
                    {groupedOutcomes.overshoot.length === 0 && (!metrics || metrics.otherCodes.length === 0) && (
                      <span className="venn-empty">None</span>
                    )}
                  </div>

                  {/* EXPAND TOGGLE (shown when collapsed) */}
                  {!shouldShowInterim('benchmark') && (
                    <div className="venn-expand-toggle">▲ Show Interim Nodes</div>
                  )}

                  {shouldShowInterim('benchmark') && (
                    <div className="venn-interim-codes">
                      {/* COLLAPSE TOGGLE (shown when expanded, above header) */}
                      <div className="venn-expand-toggle">▼ Hide Interim Nodes</div>
                      <div className="venn-interim-header">Interim Nodes</div>
                      <div className="venn-codes">
                        {interimBenchmark.map(n => (
                          <span
                            key={n.id}
                            className={`venn-code-tag benchmark-region dimmed-interim ${highlightMode === 'other' ? 'highlighted' : ''}`}
                            title={`${n.label}\nStatus: Traversed Path`}
                            onClick={(e) => { e.stopPropagation(); onCodeClick?.(n.id); }}
                          >
                            {n.id}
                          </span>
                        ))}
                        {interimBenchmark.length === 0 && <span className="venn-empty">No interim nodes</span>}
                      </div>
                    </div>
                  )}
                </div>
              </div>

            </div>
          )}
        </div>
      )}

      {/* Decision History Panel - Removed as per user request */}

      {/* Empty State - only show when no expected codes and no metrics */}
      {!hasExpectedCodes && !metrics && status === 'idle' && (
        <div className="trajectory-empty">
          <div className="empty-text">No benchmark data yet</div>
          <div className="empty-hint">Add expected codes and run a benchmark to see results</div>
        </div>
      )}

      {/* Loading State */}
      {status === 'traversing' && !metrics && (
        <div className="trajectory-empty loading-state">
          <div className="spinner" />
          <span>Running benchmark...</span>
        </div>
      )}

    </div>
  );
}