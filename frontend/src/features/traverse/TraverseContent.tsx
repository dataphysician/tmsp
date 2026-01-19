import { GraphViewer } from '../../components/GraphViewer';
import { TrajectoryViewer } from '../../components/TrajectoryViewer';
import { VIEW_TABS, type ViewTab } from '../../lib/constants';
import type { TraversalState } from '../../lib/types';

interface TraverseContentProps {
  viewTab: ViewTab;
  state: TraversalState;
  selectedNode: string | null;
  onNodeClick: (nodeId: string) => void;
  elapsedTime: number | null;
}

export function TraverseContent({
  viewTab,
  state,
  selectedNode,
  onNodeClick,
  elapsedTime,
}: TraverseContentProps) {
  if (viewTab === VIEW_TABS.GRAPH) {
    return (
      <GraphViewer
        nodes={state.nodes}
        edges={state.edges}
        selectedNode={selectedNode}
        onNodeClick={onNodeClick}
        finalizedCodes={state.finalized_codes}
        isTraversing={state.status === 'traversing'}
        currentStep={state.current_step}
        decisionCount={state.decision_history.length}
        status={state.status}
        errorMessage={state.error}
        decisions={state.decision_history}
        codesBarLabel="Extracted Codes"
        elapsedTime={elapsedTime}
      />
    );
  }

  return (
    <TrajectoryViewer
      decisions={state.decision_history}
      finalizedCodes={state.finalized_codes}
      status={state.status}
      currentStep={state.current_step}
      errorMessage={state.error}
    />
  );
}