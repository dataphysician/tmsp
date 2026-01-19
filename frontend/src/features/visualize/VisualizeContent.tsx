import { GraphViewer } from '../../components/GraphViewer';
import { VisualizeReportViewer } from '../../components/VisualizeReportViewer';
import { VIEW_TABS, type ViewTab } from '../../lib/constants';
import type { GraphNode, GraphEdge } from '../../lib/types';

interface VisualizeContentProps {
  viewTab: ViewTab;
  graphData: { nodes: GraphNode[]; edges: GraphEdge[] } | null;
  inputCodes: Set<string>;
  isLoading: boolean;
  error: string | null;
  selectedNode: string | null;
  onNodeClick: (nodeId: string) => void;
}

export function VisualizeContent({
  viewTab,
  graphData,
  inputCodes,
  isLoading,
  error,
  selectedNode,
  onNodeClick,
}: VisualizeContentProps) {
  if (viewTab === VIEW_TABS.GRAPH) {
    return (
      <GraphViewer
        nodes={graphData?.nodes ?? []}
        edges={graphData?.edges ?? []}
        selectedNode={selectedNode}
        onNodeClick={onNodeClick}
        finalizedCodes={[...inputCodes]}
        isTraversing={isLoading}
        status={isLoading ? 'traversing' : graphData ? 'complete' : 'idle'}
        errorMessage={error}
        codesBarLabel="Submitted Codes"
      />
    );
  }

  return (
    <VisualizeReportViewer
      nodes={graphData?.nodes ?? []}
      edges={graphData?.edges ?? []}
      inputCodes={inputCodes}
    />
  );
}