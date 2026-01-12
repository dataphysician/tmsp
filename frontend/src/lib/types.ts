/**
 * Types matching the Python agent state models in agent_state.py
 */

export type TraversalStatus = 'idle' | 'traversing' | 'complete' | 'error';

export interface CandidateDecision {
  code: string;
  label: string;
  selected: boolean;
  confidence: number;
  evidence: string | null;
  reasoning: string;
  billable?: boolean;
}

export interface DecisionPoint {
  current_node: string;
  current_label: string;
  depth: number;
  candidates: CandidateDecision[];
  selected_codes: string[];
}

export interface GraphNode {
  id: string;
  code: string;
  label: string;
  depth: number;
  category: 'root' | 'finalized' | 'ancestor' | 'placeholder' | 'activator';
  billable: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  edge_type: 'hierarchy' | 'lateral';
  rule: string | null;
}

export interface TraversalState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  decision_history: DecisionPoint[];
  current_path: string[];
  finalized_codes: string[];
  status: TraversalStatus;
  current_step: string;
  error: string | null;
}

// Benchmark types

/**
 * Node highlighting states for benchmark visualization.
 * Applied during traversal streaming to show coverage of expected trajectory.
 *
 * Visual treatment:
 * - expected: Black dashed outline, no fill (waiting to be traversed)
 * - traversed: Green solid outline, no fill (in expected trajectory, traversed but not finalized)
 * - matched: Green solid outline, green fill (finalized at expected code - success!) 🏁
 * - undershoot: Green solid outline, amber fill (finalized too early - stopped at ancestor)
 * - overshoot: Red X marker (traversal went past expected - not rendered as node)
 * - other: Hidden from graph visualization, shown in metrics/report only
 */
export type BenchmarkNodeStatus =
  | 'expected'      // In expected trajectory, NOT yet traversed
  | 'traversed'     // In expected trajectory AND traversed (but not finalized here)
  | 'matched'       // Finalized at this expected code (exact match - success)
  | 'undershoot'    // Finalized here, but expected was deeper (stopped too early)
  | 'overshoot';    // Traversed beyond expected leaf depth (marker only, not a node status)

/**
 * Marker for overshoot visualization - rendered as red X instead of node.
 * Supports chaining for multi-level overshoots (rundown scenarios).
 */
export interface OvershootMarker {
  sourceNode: string;      // The node that was overshot from (expected leaf or previous overshoot)
  targetCode: string;      // The code that was traversed to (not rendered as node)
  depth: number;           // Depth level for positioning
}

/**
 * Marker for missed edge visualization - rendered as red X on the edge.
 * Used when expected node exists in graph but traversal took wrong path.
 */
export interface EdgeMissMarker {
  edgeSource: string;      // Parent node ID (source of the edge)
  edgeTarget: string;      // Expected node ID (target of the edge, the one that was missed)
  missedCode: string;      // The expected code that wasn't reached
}

export interface BenchmarkGraphNode extends GraphNode {
  benchmarkStatus?: BenchmarkNodeStatus;
}

/**
 * Endpoint comparison result for a single expected finalized code.
 *
 * Categories are determined by comparing the expected finalized code against
 * traversed finalized codes based on ancestry relationship:
 *
 * - exact: Traversed finalized exactly this code (regardless of path taken)
 * - undershoot: Traversed finalized an ancestor of this code (stopped too early)
 * - overshoot: Traversed finalized a descendant of this code (went too deep)
 * - missed: Neither this code, its ancestors, nor descendants were finalized
 *
 * Path taken (hierarchical vs lateral) does NOT affect the category -
 * only the endpoint relationship matters.
 */
export type ExpectedOutcomeStatus = 'exact' | 'undershoot' | 'overshoot' | 'missed';

export interface ExpectedCodeOutcome {
  expectedCode: string;
  status: ExpectedOutcomeStatus;
  relatedCode: string | null;  // The traversed code that matched (null for missed)
}

/**
 * Aggregate benchmark metrics comparing expected vs traversed finalized codes.
 */
export interface BenchmarkMetrics {
  // Input counts
  expectedCount: number;           // User-defined expected finalized codes
  traversedCount: number;          // Finalized codes from traversal
  expectedNodesCount: number;      // Total nodes in expected trajectory
  traversedNodesCount: number;     // Total nodes visited during traversal

  // Outcome breakdown for expected codes
  exactCount: number;         // Expected codes matched exactly (MATCHED)
  undershootCount: number;    // Expected codes where traversal stopped at ancestor
  overshootCount: number;     // Expected codes where traversal went to descendant
  missedCount: number;        // Expected codes not reached at all

  // Other: traversed finalized codes unrelated to any expected code
  // These represent alternative clinical pathways - hidden from graph, shown in report
  otherCount: number;

  // Accuracy scores
  traversalAccuracy: number;    // (Expected nodes traversed) / (Expected nodes) - coverage of expected trajectory
  finalCodesAccuracy: number;   // exact / expected - accuracy of finalization decisions

  // Detailed outcomes
  outcomes: ExpectedCodeOutcome[];
  otherCodes: string[];  // Traversed codes unrelated to any expected (hidden from graph)
}
