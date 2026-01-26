# TMSP Development Guidelines

## Critical: Graph Visualization Logic

The graph visualization in `graph/trace_tree.py` has been carefully debugged. **DO NOT modify the following without understanding the full implications:**

### 1. Lateral Link Depth Ordering (line ~319)

```python
candidates.sort(key=lambda x: (x["weight"], x["target_code"], x["anchor_code"]), reverse=True)
```

**Why `reverse=True` (descending order)?**
- Deeper links (higher weight) must be processed FIRST
- This ensures leaf-adjacent links are established before ancestors can create spurious links
- Processing order: deepest nodes first, then work upward
- **Bug if changed to ascending:** Ancestor nodes get anchored before their descendants, causing incorrect graph structure

### 2. 7th Character Code Resolution (lines ~30-104)

The `resolve_code()` and `build_placeholder_chain()` functions handle 7th character extensions:
- `T36.1X5D` -> `T36.1X5` (drop 7th char)
- `V29.9XXS` -> `V29.9` (drop 7th char AND X placeholders)
- `T88.XXXA` -> `T88` (drop 7th char, X placeholders, AND trailing dot)

**Critical behaviors:**
- Placeholder chains must be included in ancestry (V29.9XXS -> V29.9XX -> V29.9X -> V29.9)
- Trailing dots after stripping X placeholders must be removed (T88.XXX -> T88. -> T88)
- `sevenChrDef` source nodes are tracked as "activators" for styling
- Input code marked as visited for 7th char codes BEFORE walking ancestry

**Trailing dot handling (lines ~54-58, ~97-103):**
```python
# In resolve_code() and build_placeholder_chain():
# Handle trailing dot after stripping all X's (e.g., T88. -> T88)
if base.endswith("."):
    base = base[:-1]
    if base in index:
        return base
```

**Bug if trailing dot not handled:** Codes like T88.XXXA fail to resolve to T88, breaking the placeholder chain

### 3. Anchor Checking Order (lines ~457-470)

```python
# Handle 7th char codes: add placeholder chain first
if resolved != code:
    # For 7th char codes, mark the original code as visited
    visited_path_nodes.add(code)
    placeholder_chain = build_placeholder_chain(code, index)
    new_chain.extend(placeholder_chain)
    visited_path_nodes.update(placeholder_chain)
```

**Why this order matters:**
- 7th char codes must mark themselves visited FIRST
- Then add placeholder chain to visited set
- This prevents anchors from being checked incorrectly for 7th char codes
- Regular codes check anchors BEFORE skipping to parent

### 4. SevenChrDef Edge Type Determination (`server/app.py:256-280`)

The `/api/graph` endpoint determines edge types. SevenChrDef edges must ONLY be created from depth-6 to depth-7:

```python
elif child in seventh_char:
    parent_depth = node_depths.get(parent, 0)
    child_depth = node_depths.get(child, 0)
    if parent_depth == 6 and child_depth == 7:
        # Only the immediate parent (depth 6) creates sevenChrDef edge
        edges.append(GraphEdge(..., edge_type=EdgeType.LATERAL, rule="sevenChrDef"))
    else:
        # Ancestors create regular hierarchy edges
        edges.append(GraphEdge(..., edge_type=EdgeType.HIERARCHY, rule=None))
```

**Why depth check matters:**
- The "activator" node (has sevenChrDef metadata) may be at depth 3-5 (e.g., T88)
- Placeholder chain fills depths 4-6 (T88.X, T88.XX, T88.XXX)
- Only T88.XXX (depth 6) should create the sevenChrDef edge to T88.XXXA (depth 7)
- **Bug if not checked:** T88 links directly to T88.XXXA, causing vertical misalignment

### 5. SevenChrDef Target Positioning (`GraphViewer.tsx:2486-2489`)

The frontend positions sevenChrDef targets using the target's actual depth from the API:

```typescript
const targetNode = nodeMap.get(targetId);
const targetDepth = targetNode?.depth ?? (nodeDepth + 1);
const targetY = getYForDepth(targetDepth) + nodeHeight / 2;
```

**Why use API depth:**
- The API returns accurate depth for each node
- Calculating as `nodeDepth + 1` (source's depth + 1) fails when source isn't depth 6
- **Bug if calculated:** T88.XXXA appears at depth 4 instead of depth 7

## Shared Components: Visualize Tab & Benchmark Mode

Both modes use the **exact same backend pipeline**:

| Layer | Component | File |
|-------|-----------|------|
| Frontend API | `buildGraph()` | `frontend/src/lib/api.ts:332` |
| Backend Endpoint | `POST /api/graph` | `server/app.py` |
| Graph Builder | `build_graph()` | `graph/trace_tree.py:495` |
| Anchor Logic | `find_nearest_anchor()` | `graph/trace_tree.py:218` |
| Ancestry Tracing | `trace_with_nearest_anchor()` | `graph/trace_tree.py:398` |

**Any fix to `trace_tree.py` automatically applies to both:**
1. Visualize tab graph
2. Benchmark "Add" button's expected graph
3. Benchmark comparison visualization

The benchmark-specific logic (`computeFinalizedComparison()`, `computeBenchmarkVisualization()`) operates **on top of** the graph structure, not instead of it.

## Testing Graph Changes

Before modifying graph logic:

1. Test with 7th character codes: `T36.1X5D`, `V29.9XXS`, `S02.111A`, `T88.XXXA`
   - `T88.XXXA` specifically tests trailing dot handling (T88.XXX -> T88. -> T88)
2. Test with lateral links: codes that have `codeFirst`, `codeAlso`, `useAdditionalCode`
3. Test with multiple codes that share common ancestors
4. Verify both Visualize tab AND Benchmark mode show correct graphs
5. Verify only submitted codes show green "finalized" styling (not billable ancestors)

## Critical: Frontend Graph Positioning (`GraphViewer.tsx`)

### Purpose: Minimal Connected Graph Visualization

The graph visualization renders the **minimal set of nodes** that fully connect all submitted ICD-10-CM codes to a common ROOT. This means:
- Only nodes that are ancestors of submitted codes OR lateral link targets are rendered
- Submitted codes from different chapters share a single ROOT node
- Single-child chains form perfect vertical columns (collinear alignment)
- Lateral links (codeFirst, codeAlso, useAdditionalCode) branch off to the right without disrupting the main hierarchy
- No duplicate nodes - if a code appears in multiple paths, it's rendered once and edges point to it

The positioning algorithm ensures this minimal graph is laid out clearly with:
- Chapter containers that dynamically expand for lateral links
- No node overlaps at the same depth
- Proper spacing between sibling subtrees

### 1. Bounded Region Allocation (Key Architecture)

The positioning algorithm uses **bounded region allocation** to eliminate collisions by construction rather than fixing them reactively:

```typescript
// Two-pass algorithm:
// Pass 1 (bottom-up): Compute actual width requirements for each subtree
function computeSubtreeBounds(nodeId: string, visited = new Set<string>()): SubtreeBounds {
  // KEY CHANGE: Recurse into lateral targets to get their ACTUAL subtree width
  for (const targetId of effectiveLateralTargets) {
    const targetBounds = computeSubtreeBounds(targetId, new Set(visited));
    lateralTargetsWidth += targetBounds.requiredWidth + nodePadding;
  }
  // ...
}

// Pass 2 (top-down): Allocate non-overlapping regions to children
function positionWithRegion(nodeId: string, context: DFSPositionContext): { usedLeft: number; usedRight: number } {
  // Allocate bounded sub-regions to each child based on their requiredWidth
  for (const childId of regularChildren) {
    const childBounds = subtreeBounds.get(childId);
    const childRegion: AllocatedRegion = {
      left: regionX,
      right: regionX + childBounds.requiredWidth
    };
    positionWithRegion(childId, { region: childRegion, ... });
    regionX = childRegion.right + nodePadding;  // Use ALLOCATED region, not usedRight
  }
}
```

**Key invariants:**
1. **Region Containment**: A node's X position is always within its allocated region
2. **Non-Overlapping Regions**: Sibling regions are allocated sequentially, never overlap
3. **Lateral Expansion to Right**: Lateral targets get regions to the RIGHT of hierarchy children
4. **SevenChrDef Collinear**: These remain at same X as source (no additional width needed)

**Why bounded regions matter:**
- Old approach: Estimate lateral width as `count * LATERAL_EXTRA_WIDTH`, then react to collisions
- New approach: Compute ACTUAL lateral subtree widths recursively, allocate exact bounded regions
- Collisions are **eliminated by construction**, not fixed reactively with complex multi-phase resolution

### 2. Data Structures

```typescript
// Computed during bottom-up pass
interface SubtreeBounds {
  requiredWidth: number;           // Total width for subtree (hierarchy + laterals)
  hierarchyChildrenWidth: number;  // Width of hierarchy children portion
  lateralTargetsWidth: number;     // Width of lateral targets portion
  effectiveLateralTargets: string[]; // Lateral targets we position (not those with hierarchy parents)
  isSingleChain: boolean;          // No branching, no laterals - position collinearly
  sevenChrDefTargets: string[];    // Positioned collinearly (same X as source)
}

// Passed during top-down positioning
interface AllocatedRegion {
  left: number;
  right: number;
}
```

### 3. Spacing Constants

```typescript
const nodePadding = 10;           // Padding between sibling nodes
const CHAPTER_PADDING = 20;       // Padding between chapters
const minGap = nodeWidth + 10;    // Minimum gap between nodes at same depth
```

## Critical: Frontend Benchmark Performance

### 1. GraphViewer Bounded Region Allocation (`components/GraphViewer.tsx`)

The `calculatePositions` function uses **bounded region allocation** which is inherently O(n):

```typescript
// Phase 1: Bottom-up computation of SubtreeBounds - O(n)
// Phase 4: Top-down region allocation - O(n)
// No reactive collision resolution needed!
```

**Why bounded regions are fast:**
- Each node is visited exactly once in each pass (bottom-up and top-down)
- No iterative collision resolution loops
- Collisions are eliminated by construction, not detected and fixed reactively
- The old O(n²) collision resolution loops were removed entirely

### 2. handleBenchmarkEvent Dependencies (`App.tsx:~1104`)

```typescript
}, [benchmarkExpectedGraph, benchmarkExpectedCodes, llmConfig]);
```

**DO NOT add `benchmarkTraversedNodes` or `benchmarkTraversedEdges` to this dependency array:**
- These are accessed via refs (`benchmarkTraversedNodesRef`, `benchmarkTraversedEdgesRef`)
- Adding them as dependencies causes excessive callback recreation
- This cascade affects `handleBenchmarkRun` and can cause performance issues

### 3. Benchmark Flow

When benchmark runs:
1. `handleBenchmarkRun()` resets state and calls `streamTraversal()`
2. `setSidebarCollapsed(true)` triggers layout change
3. GraphViewer re-renders with expected graph nodes
4. Server streams events (possibly cached - very fast)
5. Events processed via `handleBenchmarkEvent`

Any blocking operation in steps 1-3 will cause "Page Unresponsive" errors.

### 4. Cached Replay Performance

**See "Cached Traversal Replay Architecture" section above for complete details.**

Key optimizations in `App.tsx` `handleBenchmarkEvent`:
- Skip visual progress state updates (`setStreamingTraversedIds`, `setBenchmarkCurrentStep`) for cached events
- Refs are updated immediately (no data loss), state is applied once at RUN_FINISHED
- `benchmarkIsCachedReplayRef.current` controls whether STATE_DELTA uses refs (fast) or setState (slow)

**Cache detection in handleBenchmarkEvent** (`App.tsx:849`):

```typescript
// In RUN_STARTED - single source of truth (server fix eliminated dual cache check issue):
benchmarkIsCachedReplayRef.current = event.metadata?.cached === true;
```

**Note:** The server now calls `build_app()` BEFORE emitting RUN_STARTED, so the `cached` flag is always accurate. No fallback detection in STEP_FINISHED is needed.

### 4. Node Reset on RUN_STARTED (`App.tsx` handleBenchmarkEvent)

Node reset to idle status now happens **synchronously in RUN_STARTED** (not RAF-deferred). This ensures expected nodes are visually reset to "freshly added" state BEFORE any traversal data arrives.

**Key points:**
- `resetNodesToIdle()` is called synchronously in RUN_STARTED handler
- This ensures users see the idle expected graph before results stream in
- Especially important for cached replays where events arrive very quickly
- `benchmarkIsCachedReplayRef` tracks whether current run is cached (for UI display)
- RAF cancellation in RUN_FINISHED is a safety net only

**Why synchronous reset is OK now:**
- The reset happens AFTER the network request starts (in event handler)
- User already clicked "Run", so brief UI pause is acceptable
- Cached replays benefit from seeing the reset before instant results
- The old RAF pattern caused race conditions where cached replays skipped the visual reset entirely

## Critical: Node Finalization Styling

### Finalized Codes = Submitted Codes Only (`GraphViewer.tsx:3195-3305`)

Only codes explicitly submitted by the user should be styled as "finalized" (green border). The rendered ancestry may include billable codes or depth-7 codes, but these should NOT be styled as finalized unless they were submitted.

**Key helper functions:**

```typescript
// CORRECT: Only check if code is in submitted codes
function isFinalizedNode(node: GraphNode, finalizedCodes: Set<string>): boolean {
  return finalizedCodes.has(node.code);
}

function isSevenChrDefFinalizedNode(node: GraphNode, finalizedCodes?: Set<string>): boolean {
  const code = node.code || node.id;
  if (finalizedCodes && code && finalizedCodes.has(code)) return true;
  return false;
}
```

**DO NOT add these conditions back:**
- `node.billable` - billable ancestors shouldn't auto-finalize
- `node.depth === 7` - unsubmitted depth-7 codes shouldn't auto-finalize
- `node.category === 'finalized'` - category is for behavior, not styling
- Pattern matching (`/X+[A-Z]$/`) - pattern alone doesn't mean submitted

**Separate concerns:**
- `isLeafNode()` - used for **behavior** (hide children batch on leaf nodes) - CAN check billable
- `isFinalizedNode()` - used for **styling** (green border) - ONLY check `finalizedCodes`

**Bug if billable/depth checked:** Ancestry nodes that happen to be billable get green styling even though user didn't submit them

## Critical: AG-UI Protocol Cached Replay Architecture

### Feature Overview

When a traversal has been run before with the same parameters, the backend serves results from SQLite cache instead of re-running LLM calls. This uses the **AG-UI STATE_SNAPSHOT pattern** for optimal performance:

**Previous approach (deprecated):** Server replayed 500+ individual events (STEP_STARTED, STATE_DELTA, STEP_FINISHED), requiring complex frontend accumulation.

**Current approach (AG-UI aligned):** Server emits a single STATE_SNAPSHOT with complete graph, then RUN_FINISHED with all decisions.

### AG-UI Protocol Alignment

From the [AG-UI documentation](https://docs.ag-ui.com):
- **STATE_SNAPSHOT** - "Delivers complete representation of agent's current state"
- Use case: "Significant state transformations" and "creating a fresh baseline"
- Frontend must "replace entire state model with snapshot contents"

A cached replay IS a significant state transformation - the complete graph is known instantly.

### Event Flow Comparison

**Live traversal (unchanged):**
```
RUN_STARTED
STEP_STARTED (batch 1)
STATE_DELTA (nodes/edges)
STEP_FINISHED (candidates, reasoning)
... repeat for each batch ...
RUN_FINISHED (final_nodes)
```

**Cached replay (AG-UI snapshot):**
```
RUN_STARTED (cached: true)
STATE_SNAPSHOT (complete nodes + edges)
RUN_FINISHED (final_nodes + decisions array)
```

### Server Implementation (`server/app.py:1220+`)

```python
if is_cached:
    # Build complete graph from cached state (single pass)
    snapshot_nodes = [{"id": "ROOT", ...}]
    snapshot_edges = []

    for batch_id, batch_info in sorted_batches:
        # Build nodes/edges from cached batch data
        # ... (same logic as live, but into lists not events)

    # Build decisions array for benchmark comparison
    all_decisions = [
        {
            'batch_id': batch_id,
            'node_id': node_id,
            'candidates': {...},
            'selected_ids': [...],
            'reasoning': '...',
        }
        for batch_id, batch_info in sorted_batches
    ]

    # Emit single STATE_SNAPSHOT
    yield AGUIEvent(type=STATE_SNAPSHOT, state=GraphState(nodes, edges))

    # Emit RUN_FINISHED with decisions
    yield AGUIEvent(type=RUN_FINISHED, metadata={
        'final_nodes': final_nodes,
        'cached': True,
        'decisions': all_decisions,
    })
```

### Frontend Handling (`App.tsx`)

**STATE_SNAPSHOT handler:**
```typescript
case 'STATE_SNAPSHOT':
    const nodes = event.state.nodes as GraphNode[];
    const edges = event.state.edges as GraphEdge[];
    // Complete state replacement - single render
    traverseNodesRef.current = nodes;
    traverseEdgesRef.current = edges;
    setState(prev => ({ ...prev, nodes, edges }));
    break;
```

**RUN_FINISHED with decisions:**
```typescript
case 'RUN_FINISHED':
    const snapshotDecisions = event.metadata?.decisions;
    if (snapshotDecisions) {
        // Convert server format to frontend DecisionPoint
        const decisions = snapshotDecisions.map(d => ({
            current_node: d.node_id,
            candidates: Object.entries(d.candidates).map(...),
            selected_codes: d.selected_ids,
        }));
        traverseDecisionsRef.current = decisions;
    }
    break;
```

### Node Reset in RUN_STARTED (Resolves RAF Race)

**Previous issue:** `handleBenchmarkRun()` used RAF to defer node reset, causing cached replays to skip the visual reset entirely (RAF cancelled before execution).

**Current solution:** Node reset happens **synchronously in RUN_STARTED handler**:

```typescript
case 'RUN_STARTED':
    // Track cached status
    benchmarkIsCachedReplayRef.current = event.metadata?.cached === true;

    // Cancel any pending RAF and reset nodes synchronously
    if (benchmarkResetRafRef.current !== null) {
        cancelAnimationFrame(benchmarkResetRafRef.current);
        benchmarkResetRafRef.current = null;
    }

    // Synchronously reset combined nodes to idle status
    if (benchmarkCombinedNodesRef.current.length > 0) {
        const idleNodes = resetNodesToIdle(benchmarkCombinedNodesRef.current);
        benchmarkCombinedNodesRef.current = idleNodes;
        setBenchmarkCombinedNodes(idleNodes);
    }

    setBenchmarkCurrentStep(isCached ? 'Loading cached results...' : 'Starting benchmark traversal');
    break;
```

**Benefits:**
- Users always see the "freshly added" (idle expected) state before results arrive
- Cached replays show proper visual transition: idle → final state
- No more RAF race condition - reset is guaranteed before data arrives
- `benchmarkIsCachedReplayRef` provides cache status for UI feedback

### SSE Processing (`sse.ts`)

Simplified after snapshot pattern - no accumulation needed:

```typescript
export async function processSSEStream(response, options) {
    // Events processed immediately as they arrive
    // For cached replays, server sends STATE_SNAPSHOT (3 events total)
    // instead of 500+ individual events
    for (const event of parseSSE(response)) {
        onEvent(event);
    }
}
```

### Benefits of Snapshot Pattern

| Aspect | Old (Event Accumulation) | New (AG-UI Snapshot) |
|--------|--------------------------|----------------------|
| Events sent | 500+ | 3 |
| Network payload | Large (redundant) | Minimal |
| Frontend complexity | Accumulation + batch processing | Direct state replacement |
| Render cycles | Multiple (with yielding) | Single |
| Protocol alignment | Custom workaround | AG-UI standard |

### Testing Cached Replay

1. Run a benchmark traversal (creates cache)
2. Run the SAME benchmark again (triggers cached replay)
3. Verify:
   - Console shows `[STATE_SNAPSHOT] Complete graph: N nodes, M edges`
   - Graph renders correctly with all nodes positioned
   - No "Page Unresponsive" errors
   - Final nodes match first run exactly
   - Network tab shows only 3 events (not 500+)

## Critical: Live Traversal Streaming Architecture (Benchmark Tab)

### Two-Phase Rendering Architecture

The Benchmark tab Graph View uses a **two-phase rendering architecture** for live traversal to maintain UI responsiveness while showing real-time progress:

**Phase 1 - Streaming (real-time visual feedback):**
- Expected graph is **pre-rendered** when user clicks "Add" (via `buildGraph()`)
- During streaming, only `streamingTraversedIds` Set is tracked
- GraphViewer updates **only node styles** (fill/stroke → green outline)
- Full node/edge data from STATE_DELTA goes into **refs only** (no re-renders)

**Phase 2 - Finalization (RUN_FINISHED):**
- All accumulated refs are applied to state
- `computeFinalizedComparison()` computes:
  - Red X markers (overshoot/undershoot)
  - Missed edge markers
  - Benchmark metrics (matched, missed, other)
  - Node status categorization

### AG-UI Event Flow for Live Traversal

```
RUN_STARTED
  └→ Reset all refs and state
STEP_STARTED (batch N)
  └→ Skip (no state updates during streaming)
STATE_DELTA (JSON Patch)
  └→ Update refs ONLY: benchmarkTraversedNodesRef, benchmarkTraversedEdgesRef
STEP_FINISHED (candidates, reasoning)
  └→ Track selected IDs in refs
  └→ THROTTLED state update for visual feedback (max 10/sec)
... repeat for each batch ...
RUN_FINISHED (final_nodes)
  └→ Apply all refs to state
  └→ Compute final metrics and markers
```

### Throttling Mechanism (`App.tsx`)

**Key refs and constants:**

```typescript
const streamingTraversedIdsRef = useRef<Set<string>>(new Set());
const lastVisualUpdateTimeRef = useRef<number>(0);
const VISUAL_UPDATE_THROTTLE_MS = 100; // Max 10 updates/sec
```

**STEP_FINISHED handler (throttled):**

```typescript
case 'STEP_FINISHED':
  // Track in refs (instant, no re-render)
  for (const id of selectedIds) {
    benchmarkStreamedIdsRef.current.add(id);
    streamingTraversedIdsRef.current.add(id);
  }

  // THROTTLED state updates for visual feedback
  const now = Date.now();
  if (now - lastVisualUpdateTimeRef.current >= VISUAL_UPDATE_THROTTLE_MS) {
    lastVisualUpdateTimeRef.current = now;
    setStreamingTraversedIds(new Set(streamingTraversedIdsRef.current));
    setBenchmarkBatchCount(benchmarkBatchCountRef.current);
  }
  // Skip setBenchmarkCurrentStep during streaming (too fast to read anyway)
```

**STATE_DELTA handler (refs only):**

```typescript
case 'STATE_DELTA':
  if (event.delta) {
    // Update refs only - no state updates during streaming
    const graphState = {
      nodes: [...benchmarkTraversedNodesRef.current],
      edges: [...benchmarkTraversedEdgesRef.current],
    };
    try {
      const result = applyPatch(graphState, event.delta as Operation[], true, false);
      benchmarkTraversedNodesRef.current = result.newDocument.nodes;
      benchmarkTraversedEdgesRef.current = result.newDocument.edges;
    } catch { /* ignore */ }
  }
  break;
```

### GraphViewer Memoization (`GraphViewer.tsx`)

**Custom comparison function:**

```typescript
function arePropsEqual(prev: GraphViewerProps, next: GraphViewerProps): boolean {
  // Core data changes - always re-render
  if (prev.nodes !== next.nodes) return false;
  if (prev.edges !== next.edges) return false;
  if (prev.status !== next.status) return false;

  // streamingTraversedIds - ALLOW through (throttled, lightweight update)
  if (prev.streamingTraversedIds !== next.streamingTraversedIds) return false;

  // Other props...
  return true;
}

export const GraphViewer = memo(GraphViewerInner, arePropsEqual);
```

**Lightweight style-update useEffect:**

```typescript
// Separate useEffect for streaming traversal style updates (avoids full re-render)
// Uses requestAnimationFrame to batch DOM updates
useEffect(() => {
  if (!svgRef.current || !benchmarkMode || !streamingTraversedIds) return;

  const svg = d3.select(svgRef.current);

  const rafId = requestAnimationFrame(() => {
    // Update only node styles without recreating them
    svg.selectAll<SVGRectElement, BenchmarkGraphNode>('.node-rect')
      .attr('fill', function() { /* getBenchmarkNodeFill */ })
      .attr('stroke', function() { /* getBenchmarkNodeStroke */ })
      .attr('stroke-width', function() { /* getBenchmarkNodeStrokeWidth */ })
      .attr('stroke-dasharray', function() { /* getBenchmarkNodeStrokeDasharray */ });
  });

  return () => cancelAnimationFrame(rafId);
}, [streamingTraversedIds, benchmarkMode]);
```

### Issues to Avoid

| Issue | Symptom | Solution |
|-------|---------|----------|
| State update on every event | UI freeze, "Page Unresponsive" | Use refs during streaming, throttle state updates |
| Rebuilding graph during streaming | Expensive re-renders | Only update D3 styles, not full graph structure |
| Missing ref resets | Stale visual state on re-run | Reset refs in BOTH `handleBenchmarkRun` AND `handleBenchmarkReset` |
| Unthrottled `setStreamingTraversedIds` | 100+ re-renders per traversal | Check `lastVisualUpdateTimeRef` before setState |
| GraphViewer not memoized | Parent re-render cascades | Use `React.memo` with custom comparison |
| D3 updates without RAF | Layout thrashing | Wrap in `requestAnimationFrame` |

### Required Ref Resets

**In `handleBenchmarkRun`:**

```typescript
streamingTraversedIdsRef.current = new Set();
lastVisualUpdateTimeRef.current = 0;
setStreamingTraversedIds(new Set());
```

**In `handleBenchmarkReset`:**

```typescript
streamingTraversedIdsRef.current = new Set();
lastVisualUpdateTimeRef.current = 0;
setStreamingTraversedIds(new Set());
```

**DO NOT forget either location** - missing resets cause stale visual state.

### Performance Comparison

| Metric | Before (unthrottled) | After (throttled + memoized) |
|--------|---------------------|------------------------------|
| State updates per traversal | 100+ | ~10 (throttled) |
| GraphViewer re-renders | 100+ | Only on nodes/edges change |
| D3 style updates | Every event | Batched via RAF |
| UI responsiveness | Freeze/crash | Smooth with progress |

### Testing Live Traversal

1. Run a **new** benchmark traversal (not cached - clear cache.db if needed)
2. Verify:
   - No "Page Unresponsive" warnings
   - Graph shows green borders appearing on traversed nodes progressively
   - Status bar shows batch count updating (throttled, not every event)
   - Final graph renders correctly at RUN_FINISHED with all markers
3. Console should NOT show excessive re-render warnings

---

## Critical: Cached Traversal Node Positioning

### Problem

Re-running cached traversals caused incorrect node positioning (widened spacing, misaligned parents). The positioning algorithm produces different results on cached replay due to rapid event processing triggering multiple intermediate renders.

**Note:** This is largely solved by the AG-UI snapshot pattern above - single STATE_SNAPSHOT means single render with correct positioning.

### Backend Cache Detection (`server/app.py`)

For scaffolded mode, `build_app()` is called BEFORE emitting `RUN_STARTED` - single source of truth:

```python
# Scaffolded mode: call build_app() directly (single source of truth for cache status)
scaffolded_burr_app, is_cached = await build_app(
    context=request.clinical_note,
    default_selector=request.selector,
    with_persistence=True,
    partition_key=scaffolded_partition_key,
)

# RUN_STARTED uses is_cached from build_app() - no separate cache check
run_started_event = AGUIEvent(
    type=AGUIEventType.RUN_STARTED,
    metadata={'clinical_note': request.clinical_note[:100], 'cached': is_cached}
)
yield f"data: {run_started_event.model_dump_json()}\n\n"
```

**Critical:** Build the AGUIEvent object first, then embed in f-string. Using `metadata={{'key': value}}` inside an f-string expression causes "unhashable type: 'dict'" error (Python interprets `{{}}` as a set literal inside expression blocks).

#### 3. Frontend Positioning Phases (`GraphViewer.tsx`)

The `calculatePositions` function uses bounded region allocation with simplified post-processing:

| Phase | Purpose |
|-------|---------|
| Phase 1 | Bottom-up: Compute `SubtreeBounds` with ACTUAL widths (including lateral subtrees) |
| Phase 2 | Initialize chapter boundaries from computed `requiredWidth` |
| Phase 3 | Position ROOT |
| Phase 4 | Top-down: Position nodes within bounded regions (`positionWithRegion`) |
| Phase 5 | Resolve chapter collisions (safety net) |
| Phase 6 | Helper functions only (collision resolution loops removed) |
| Phase 7 | Recalculate chapter boundaries from actual positions |
| Phase 7a | Compact chapters (shift to eliminate gaps) |
| Phase 7.5 | Re-center chapter nodes over immediate children |
| Phase 7.6 | Re-center ALL parent nodes (bottom-up) |
| Phase 8 | Force sevenChrDef alignment |
| Phase 9 | Final chapter centering |
| FINAL COMPACTION | Pack chapters tightly based on actual bounds |

**Note:** Phase 6 collision resolution loops and Phase 7.7 were removed. With bounded region allocation, collisions are eliminated by construction during Phase 4.

**FINAL COMPACTION is critical** - it runs last and:
1. Recalculates actual bounds for each chapter
2. Shifts chapters to pack tightly from left edge (x=50)
3. Re-aligns sevenChrDef targets after shifting

```typescript
// Step 5: Re-align sevenChrDef targets after compaction
for (const [sourceId, targets] of sevenChrDefTargetsPerSource.entries()) {
  const sourcePos = positions.get(sourceId);
  if (!sourcePos) continue;
  for (const { targetId } of targets) {
    const targetPos = positions.get(targetId);
    if (targetPos && Math.abs(targetPos.x - sourcePos.x) > 0.5) {
      shiftNodeAndDescendants(targetId, sourcePos.x - targetPos.x);
    }
  }
}
```

### Orphan-Rescued Nodes

Nodes reached only via lateral links (no hierarchy parent) are "orphan-rescued" and added to their lateral source's hierarchy. These must be tracked and included in `hierarchyParent`:

```typescript
// Track orphan-rescued nodes
const orphanRescuedNodes = new Set<string>();

// During orphan rescue pass
if (!hasHierarchyParent.has(targetId)) {
  hierarchyChildren.get(sourceId)!.push(targetId);
  hasHierarchyParent.add(targetId);
  orphanRescuedNodes.add(targetId);  // Track for hierarchyParent mapping
}

// When building hierarchyParent, include orphan-rescued nodes
if (!parentLateralTargets.has(childId) || orphanRescuedNodes.has(childId)) {
  hierarchyParent.set(childId, parentId);
}
```

**Bug if not tracked:** Orphan-rescued nodes get positioned as lateral targets (far right) instead of as hierarchy children, breaking parent-child alignment.

### Issues to Avoid

**Cached Replay Issues (see "AG-UI Protocol Cached Replay Architecture" above):**
1. **Server now uses AG-UI STATE_SNAPSHOT** - single snapshot instead of 500+ events, simplifying frontend
2. **Decisions are in RUN_FINISHED metadata** - check `event.metadata?.decisions` for cached replays
3. **Don't forget to cancel pending RAFs in RUN_FINISHED** - they will overwrite final state with stale idle nodes
4. **Don't use RAF for operations that must complete before streaming ends** - use synchronous updates or refs

**Graph Positioning Issues:**
6. **Must re-align sevenChrDef targets after ANY compaction/shifting** - depth 6→7 alignment breaks
7. **Must track orphan-rescued nodes** - they need hierarchyParent entries to position correctly
8. **Don't use `{{}}` for dicts inside f-string expressions** - causes Python unhashable type error
9. **FINAL COMPACTION must run last** - earlier compaction gets undone by subsequent phases
10. **Don't use fixed LATERAL_EXTRA_WIDTH for lateral targets** - recurse into lateral subtrees to compute actual width
11. **Don't allocate sibling regions based on usedRight** - use pre-computed bounded regions to guarantee non-overlap

## Critical: Rewind Mechanism

The Rewind feature allows users to correct LLM decisions during traversal by "rewinding" to a previous batch and providing feedback to guide a different selection.

### Feature Overview

**What it does:**
- User clicks "Rewind" on any batch node in the graph
- Opens a modal to provide feedback (natural language correction)
- Backend prunes all descendant state from that batch forward
- Re-runs selection with the feedback injected into the LLM prompt
- New traversal continues from the corrected decision point

**Use cases:**
- LLM selected wrong child code → rewind and specify correct one
- LLM stopped too early (undershoot) → rewind and request deeper traversal
- LLM selected wrong 7th character → rewind sevenChrDef batch

### Technical Architecture

#### 1. Backend Rewind Flow (`agent/traversal.py`)

```
User clicks Rewind → POST /api/rewind → retry_node() → prune_state_for_rewind()
                                           ↓
                                    Build retry app with inject_feedback entrypoint
                                           ↓
                                    Run from inject_feedback → select_candidates → ...
                                           ↓
                                    Update ROOT checkpoint with new results
```

**Key functions:**
- `retry_node(batch_id, feedback, selector)` - Main rewind entry point
- `prune_state_for_rewind(state_dict, rewind_batch_id)` - Removes descendant state
- `get_descendant_batch_ids(batch_id, batch_data)` - BFS to find all descendants

#### 2. State Pruning Logic (`agent/traversal.py:221-305`)

When rewinding to a batch, we must:
1. **Clear the rewind batch's selection** - Remove `selected_ids`, `reasoning`, `status`
2. **Remove all descendant batches** - Any batch spawned from selected children
3. **Remove descendant final_nodes** - Codes that were finalized via descendants

```python
def prune_state_for_rewind(state_dict: dict, rewind_batch_id: str) -> dict:
    # Find descendants via BFS through batch_data
    descendants = get_descendant_batch_ids(rewind_batch_id, batch_data)

    # Collect final_nodes to remove
    descendant_finals: set[str] = set()
    for desc_batch_id in descendants:
        desc_batch = batch_data.get(desc_batch_id, {})
        node_id = desc_batch.get("node_id")
        if node_id:
            descendant_finals.add(node_id)
        for sel in desc_batch.get("selected_ids", []):
            descendant_finals.add(sel)

    # Clear rewind batch selection
    # Remove descendant batches
    # Filter out descendant finals from final_nodes
```

#### 3. Frontend Rewind UI (`NodeRewindModal.tsx`)

The modal captures feedback and sends to `/api/rewind`:

```typescript
// POST /api/rewind with SSE response
const response = await fetch('/api/rewind', {
  method: 'POST',
  body: JSON.stringify({
    batch_id: batchId,
    feedback: userFeedback,
    selector: 'llm',
    // ... LLM config
  })
});

// Stream events same as initial traversal
for await (const event of streamSSE(response)) {
  onEvent(event);
}
```

### Bug Fix: SevenChrDef Codes Not Pruned During Rewind

**Problem:** When a sevenChrDef batch was rewound, the old 7th character code remained in `final_nodes` because the pruning logic collected:
- `node_id` (e.g., "T36.1X5")
- `selected_ids` (e.g., ["A"])

But `final_nodes` contains the **full code** created by `format_with_seventh_char()` (e.g., "T36.1X5A").

**Impact:**
- Old sevenChrDef code persisted alongside the corrected code
- Benchmark metrics showed the old code as "other" (unrelated to expected)
- Correction wasn't recognized properly

**Fix (`agent/traversal.py:252-275`):**

```python
# For sevenChrDef batches, also remove the full 7th char code
batch_type = desc_batch_id.rsplit("|", 1)[1] if "|" in desc_batch_id else "children"
if batch_type == "sevenChrDef" and node_id:
    selected_ids = desc_batch.get("selected_ids", [])
    if selected_ids:
        from agent.actions import format_with_seventh_char
        full_code = format_with_seventh_char(node_id, selected_ids[0])
        descendant_finals.add(full_code)

# Also handle the rewind batch itself if it's sevenChrDef
if rewind_batch_id in batch_data:
    rewind_batch_type = rewind_batch_id.rsplit("|", 1)[1] if "|" in rewind_batch_id else "children"
    if rewind_batch_type == "sevenChrDef" and rewind_node_id:
        rewind_selected = batch_data[rewind_batch_id].get("selected_ids", [])
        if rewind_selected:
            from agent.actions import format_with_seventh_char
            rewind_full_code = format_with_seventh_char(rewind_node_id, rewind_selected[0])
            descendant_finals.add(rewind_full_code)
```

### Rewind: Live vs Cached Runs

**Live runs:** Events stream in real-time, each processed immediately.

**Cached runs:** Server uses AG-UI STATE_SNAPSHOT pattern. **See "AG-UI Protocol Cached Replay Architecture" section for complete handling details**, including:
- STATE_SNAPSHOT with complete graph (single event)
- Decisions in RUN_FINISHED metadata
- RAF cancellation for cached replays

### Rewind Persists to Cache

After rewind completes, the corrected state is saved back to the ROOT checkpoint:

```python
# Use sequence_id + 1 to ensure this becomes the "latest" checkpoint
new_sequence_id = original_sequence_id + 1

await PERSISTER.save(
    partition_key=PARTITION_KEY,
    app_id="ROOT",
    sequence_id=new_sequence_id,
    position="finish",
    state=State(final_state_dict),
    status="completed",
)
```

This ensures subsequent cache hits return the corrected results.

## Critical: Benchmark Metrics After Rewind

### Feature Overview

The Benchmark Report view compares expected codes against traversed codes, calculating:
- **Exact Match**: Expected code was finalized
- **Undershoot**: Ancestor of expected was finalized (stopped too early)
- **Overshoot**: Descendant of expected was finalized (went too deep)
- **Missed**: Expected code not matched by any traversed code
- **Other**: Traversed code unrelated to any expected

### Bug Fix: Lateral Pathway Codes Marked as "Missed"

**Problem:** Codes reached via lateral edges (`codeFirst`, `codeAlso`, `useAdditionalCode`) were marked as "missed" even when their lateral source was finalized.

**Root cause:** The ancestor map only included hierarchy edges and `sevenChrDef` lateral edges:

```typescript
// OLD: Only hierarchy and sevenChrDef
if (isHierarchy || isSevenChrDef) {
  parentMap.set(edge.target, edge.source);
}
```

**Impact:**
- Codes finalized via lateral pathways counted as "missed decisions"
- Final Codes Recall was not 100% even when all expected codes matched
- Rewind corrections via lateral pathways weren't recognized

**Fix (`frontend/src/lib/benchmark.ts`):**

1. **Extended `buildAncestorMap()` to include ALL lateral edges:**
   ```typescript
   if (isHierarchy || isLateral) {
     parentMap.set(edge.target, edge.source);
   }
   ```

2. **Added `buildLateralParentMap()` for lateral relationship tracking:**
   ```typescript
   function buildLateralParentMap(edges: GraphEdge[]): Map<string, { parent: string; rule: string }> {
     const map = new Map<string, { parent: string; rule: string }>();
     for (const edge of edges) {
       if (edge.edge_type === 'lateral' && edge.rule) {
         map.set(String(edge.target), { parent: String(edge.source), rule: edge.rule });
       }
     }
     return map;
   }
   ```

3. **Updated `compareFinalizedCodes()` to check lateral relationships:**
   ```typescript
   // Check if expected code was reached via ANY lateral edge (non-sevenChrDef)
   const lateralInfo = lateralParentMap.get(expected);
   if (lateralInfo && lateralInfo.rule !== 'sevenChrDef' && traversedCodes.has(lateralInfo.parent)) {
     outcomes.push({
       expectedCode: expected,
       status: 'exact',
       relatedCode: lateralInfo.parent,
     });
     matchedTraversed.add(lateralInfo.parent);
     continue;
   }
   ```

4. **Updated `buildParentMap()` helper for consistent behavior:**
   ```typescript
   if (edge.edge_type === 'hierarchy' || edge.edge_type === 'lateral') {
     parentMap.set(String(edge.target), String(edge.source));
   }
   ```

### SevenChrDef Special Handling

SevenChrDef lateral edges are treated differently:
- If expected is a depth-7 code (e.g., T36.1X5D) and its sevenChrDef parent was finalized, this is **"missed"** not "exact"
- Rationale: The sevenChrDef batch WAS presented to the LLM (it's the children batch for depth-6→7), so failing to select the correct 7th character is a missed decision

```typescript
const sevenChrDefParent = sevenChrDefParentMap.get(expected);
if (sevenChrDefParent && traversedCodes.has(sevenChrDefParent)) {
  outcomes.push({
    expectedCode: expected,
    status: 'missed',  // Not 'exact' because the batch was presented
    relatedCode: sevenChrDefParent,
  });
}
```

### Testing Rewind and Benchmark Metrics

1. **Test sevenChrDef rewind:**
   - Run benchmark selecting sevenChrDef code A (e.g., T36.1X5A)
   - Rewind the sevenChrDef batch
   - Provide feedback to select code B (e.g., T36.1X5D)
   - Verify: Code A removed from `final_nodes`, Code B present
   - Verify: Code B marked "matched" if in expected codes

2. **Test lateral pathway matching:**
   - Add expected codes reachable via `codeFirst`, `codeAlso`, `useAdditionalCode`
   - Verify: Marked "exact" when lateral source was finalized

3. **Verify Final Codes Recall:**
   - Should be 100% when all expected codes are finalized (via any pathway)

4. **Visual check:**
   - Graph view shows matched codes as green "Matched Final Code"
   - No red X markers for correctly finalized codes
   - No "missed decisions" for lateral pathway codes

## Critical: Depth-6 SevenChrDef Finalization

### Problem

Depth-6 nodes with `sevenChrDef` requirements (like T36.1X5, T88.7XX) were being incorrectly finalized instead of continuing to the depth-7 sevenChrDef batch.

**Symptoms:**
- T36.1X5 finalized (green) alongside T36.1X5A
- T88.7XX finalized without creating T88.7XXA
- Only the depth-7 node should be finalized - depth-6 is just a transit point

**Root cause:** The `seven_chr_authority` wasn't always being propagated correctly through placeholder chains or lateral-linked nodes. When missing, the transition from `select_candidates` went to `finish_batch` instead of `spawn_seven_chr`.

### Fix: Two-Part Authority Self-Activation

#### Part 1: Authority Check in `select_candidates` (`actions.py:367-380`)

When selection is empty and no authority exists, check ancestry and set authority:

```python
# CRITICAL: If selection is empty and no seven_chr_authority, check ancestry
# and set authority if sevenChrDef exists. This ensures the transition to
# spawn_seven_chr is taken instead of finish_batch for depth-6 nodes.
if (not selected_ids and
    batch_type == "children" and
    batch_data[current_batch_id].get("seven_chr_authority") is None):
    node_id = batch_data[current_batch_id].get("node_id")
    if node_id:
        seven_chr_result = get_seventh_char_def(node_id, ICD_INDEX)
        if seven_chr_result is not None:
            _, ancestor_with_def = seven_chr_result
            batch_data[current_batch_id]["seven_chr_authority"] = {
                "batch_name": ancestor_with_def,
                "resolution_pattern": "sevenChrDef"
            }
```

**Why this works:** Setting authority BEFORE the transition decision ensures `spawn_seven_chr` is taken, not `finish_batch`.

#### Part 2: Safety Check in `finish_batch` (`actions.py:469-484`)

Even if authority wasn't set, prevent finalization if sevenChrDef exists in ancestry:

```python
# SAFETY CHECK: Even if authority wasn't propagated, check if this node
# has sevenChrDef in its ancestry. If so, it should NOT be finalized -
# the 7th character is mandatory and spawn_seven_chr should handle it.
has_seven_chr_in_ancestry = False
if not has_seven_chr_authority and node_id and batch_type == "children":
    seven_chr_result = get_seventh_char_def(node_id, ICD_INDEX)
    if seven_chr_result is not None:
        has_seven_chr_in_ancestry = True
        # Also set authority for logging
        batch_data[current_batch_id]["seven_chr_authority"] = {...}

should_report = batch_type == "children" and not has_seven_chr_authority and not has_seven_chr_in_ancestry
```

**Why two checks:** Part 1 ensures correct flow; Part 2 is a safety net if something bypasses Part 1.

### Key Invariant

**Depth-6 nodes with sevenChrDef in ancestry MUST NEVER be finalized.** Only depth-7 nodes (the actual 7th character codes like T36.1X5A) should be in `final_nodes`.

### Testing

1. **Test T36 hierarchy:**
   - Input code T36.1X5D → should finalize T36.1X5D only, NOT T36.1X5

2. **Test T88 placeholder chain:**
   - Input code T88.7XXA → should finalize T88.7XXA only
   - T88.7X and T88.7XX should NOT be in final_nodes

3. **Test lateral-linked nodes:**
   - Node reached via codeFirst with sevenChrDef → should NOT finalize at depth-6

## Critical: SevenChrDef Rewind Updates final_nodes

### Why finish_batch is Required in Retry App

When rewinding a sevenChrDef batch, the new 7th character selection must be finalized (added to `final_nodes`). This requires `finish_batch` in the retry app.

**Key insight:** The retry app has a DIFFERENT structure than the main app's MapStates subgraphs:
- Main app: MapStates classes handle `finish_batch` internally in their subgraphs
- Retry app: Runs at the TOP LEVEL, not inside MapStates, so needs explicit `finish_batch`

### Correct Retry App Pattern

```python
retry_app = await (
    ApplicationBuilder()
    .with_actions(
        inject_feedback=inject_feedback,
        load_node=load_node,
        select_candidates=select_candidates,
        spawn_parallel=SpawnParallelBatches(),
        spawn_seven_chr=SpawnSevenChr(),
        finish_batch=finish_batch,  # REQUIRED for sevenChrDef finalization
        finish=finish,
    )
    .with_transitions(
        ("inject_feedback", "select_candidates"),
        ("load_node", "select_candidates"),
        # Empty selection WITH authority -> spawn sevenChrDef
        ("select_candidates", "spawn_seven_chr", ...),
        # Empty selection WITHOUT authority -> finish_batch
        ("select_candidates", "finish_batch", ...),
        # Has selections -> spawn parallel
        ("select_candidates", "spawn_parallel", ...),
        # After sevenChrDef -> finish_batch (finalize the new code)
        ("spawn_seven_chr", "finish_batch"),
        # After parallel -> finish_batch
        ("spawn_parallel", "finish_batch"),
        # After finish_batch -> finish
        ("finish_batch", "finish"),
    )
```

### What Happens Without finish_batch

If `finish_batch` is removed from the retry app:
1. User rewinds sevenChrDef batch from T36.1X5A to T36.1X5D
2. LLM selects "D" as the new 7th character
3. **BUG:** The new code T36.1X5D is never added to `final_nodes`
4. Finalized codes bar doesn't update
5. Benchmark metrics show the correction as "missed"

### After Rewind

After reverting to include `finish_batch`, delete `cache.db` to clear any corrupted state:

```bash
rm cache.db
```

## Critical: Benchmark Overshoot Marker Bug (Lateral Targets)

### Feature Overview

The Benchmark tab compares expected codes against traversed codes, displaying visual markers:
- **Green border**: Matched (expected code was finalized)
- **Red X marker**: Indicates a problem (undershoot, overshoot, or missed)
- **Red arrow**: Points from problematic node to related code

**Overshoot** specifically means: The traversal went DEEPER in the hierarchy than the expected code. For example, if expected was I13 but traversal finalized I13.0, that's overshoot.

### The Bug: Lateral Targets Incorrectly Marked as Overshoot

**Symptoms observed:**
- I13.0 was expected AND finalized (correct trajectory)
- But the graph showed a red X marker on I13.0
- Red arrow pointed from I13.0 to N18.9

**Debug output revealed:**
```
[DEBUG] Overshoot markers: [
  {
    "sourceNode": "I13.0",
    "targetCode": "N18.9",
    "depth": 4
  }
]

[DEBUG] Expected edges involving I13*: [
  {
    "source": "I13.0",
    "target": "N18.9",
    "edge_type": "lateral",
    "rule": "useAdditionalCode"
  }
]
```

**Root Cause:**

The overshoot detection used `buildAncestorMap()` which includes ALL lateral edges:

```typescript
// buildAncestorMap included ALL edges (hierarchy + lateral)
if (isHierarchy || isLateral) {
  parentMap.set(edge.target, edge.source);
}
```

This caused N18.9 to have I13.0 in its "ancestors" via the `useAdditionalCode` lateral edge. When checking for overshoot:

```typescript
// OLD: Used ancestor map that includes lateral edges
const ancestors = ancestorMap.get(finalized) ?? new Set();
for (const leaf of expectedLeaves) {
  if (ancestors.has(leaf)) {
    // N18.9 has I13.0 as "ancestor" via lateral edge
    // Incorrectly flagged as overshoot!
    overshootMarkers.push({ sourceNode: leaf, targetCode: finalized });
  }
}
```

**The semantic error:** Lateral edges represent "use additional code" or "code first" relationships, NOT hierarchy depth. Going to a lateral target is a SEPARATE pathway, not "going deeper" in the hierarchy.

### The Fix: Hierarchy-Only Ancestor Map for Overshoot

**Added `buildHierarchyAncestorMap()` function** (`frontend/src/lib/benchmark.ts`):

```typescript
/**
 * Build hierarchy-only ancestor map for overshoot detection.
 * Overshoot means "went deeper in the HIERARCHY than expected".
 * Lateral targets should NOT be considered overshoots - they are separate pathways.
 */
function buildHierarchyAncestorMap(edges: GraphEdge[]): Map<string, Set<string>> {
  // Build parent map from hierarchy edges ONLY
  const parentMap = new Map<string, string>();
  for (const edge of edges) {
    if (edge.edge_type === 'hierarchy') {
      parentMap.set(String(edge.target), String(edge.source));
    }
  }

  // Build ancestor set for each node
  const ancestorMap = new Map<string, Set<string>>();
  for (const [child] of parentMap) {
    const ancestors = new Set<string>();
    let current = child;
    while (parentMap.has(current)) {
      const parent = parentMap.get(current)!;
      ancestors.add(parent);
      current = parent;
    }
    ancestorMap.set(child, ancestors);
  }

  return ancestorMap;
}
```

**Updated overshoot detection** in both `computeFinalizedComparison()` and `computeBenchmarkVisualization()`:

```typescript
// Build hierarchy-only ancestor map for overshoot detection
const hierarchyAncestorMap = buildHierarchyAncestorMap(expectedEdges);

// Compute overshoot markers using hierarchy-only ancestors
const overshootMarkers: OvershootMarker[] = [];
for (const finalized of finalizedCodes) {
  const hierarchyAncestors = hierarchyAncestorMap.get(finalized) ?? new Set();
  for (const leaf of expectedLeaves) {
    if (hierarchyAncestors.has(leaf)) {
      overshootMarkers.push({
        sourceNode: leaf,
        targetCode: finalized,
        depth: calculateDepthFromCode(finalized),
      });
      break;
    }
  }
}
```

### Key Distinction: Undershoot vs Overshoot Ancestor Maps

| Detection | Ancestor Map Used | Why |
|-----------|------------------|-----|
| **Undershoot** | `buildAncestorMap()` (includes lateral) | If you stopped at a lateral source instead of going to the target, that IS undershoot |
| **Overshoot** | `buildHierarchyAncestorMap()` (hierarchy only) | Going to a lateral target is NOT overshoot - it's a separate pathway |

### Files Modified

- `frontend/src/lib/benchmark.ts`:
  - Added `buildHierarchyAncestorMap()` function
  - Updated `computeFinalizedComparison()` to use hierarchy-only map for overshoot
  - Updated `computeBenchmarkVisualization()` to use hierarchy-only map for overshoot

### Testing the Fix

1. **Test case: I13.0 with useAdditionalCode lateral**
   - Expected: I13.0
   - Traversal finalizes: I13.0 AND N18.9 (lateral target via useAdditionalCode)
   - **Before fix:** Red X on I13.0, red arrow to N18.9 (WRONG)
   - **After fix:** Green border on I13.0, no red markers (CORRECT)

2. **Verify real overshoot still detected:**
   - Expected: I13
   - Traversal finalizes: I13.0 (hierarchy child)
   - Should show: Red X on I13, red arrow to I13.0 (overshoot)

3. **Verify lateral relationships don't hide real overshoots:**
   - Expected: I13
   - Traversal finalizes: I13.0 AND N18.9
   - Should show: Red X on I13 → I13.0 (overshoot), N18.9 treated as "other"

### Debug Logging (Temporary)

Debug logging was added to `App.tsx` RUN_FINISHED handler for investigation:

```typescript
// DEBUG: Investigate red X marker bug
console.log('[DEBUG] Expected codes:', [...benchmarkExpectedCodes]);
console.log('[DEBUG] Final nodes:', [...finalNodes]);
console.log('[DEBUG] Missed edge markers:', JSON.stringify(missedEdgeMarkers, null, 2));
console.log('[DEBUG] Overshoot markers:', JSON.stringify(overshootMarkers, null, 2));
// ... additional I13-specific checks
```

**Remove these debug logs** after confirming the fix works in production.

## Code Style

- Use Python 3.10+ type annotations: `str | None` not `Optional[str]`, `list[str]` not `List[str]`
- Keep the graph algorithms in `trace_tree.py` - don't split across files
- The `data` variable is the global ICD-10-CM index loaded at module level
