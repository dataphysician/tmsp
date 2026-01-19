import { useState, useCallback } from 'react';
import type { GraphNode, GraphEdge } from '../../lib/types';
import { buildGraph } from '../../lib/api';
import { parseCodeInput } from '../../lib/graphUtils';
import type { ViewTab } from '../../lib/constants';

export function useVisualizeState() {
  const [inputCodes, setInputCodes] = useState<Set<string>>(new Set());
  const [codeInput, setCodeInput] = useState('');
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewTab, setViewTab] = useState<ViewTab>('graph');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const handleAddCode = useCallback(async () => {
    const codes = parseCodeInput(codeInput);

    if (codes.length > 0) {
      const newSet = new Set(inputCodes);
      codes.forEach(code => newSet.add(code));
      setInputCodes(newSet);
      setCodeInput('');

      // Auto-build graph with all codes
      setIsLoading(true);
      setError(null);
      try {
        const result = await buildGraph([...newSet]);
        setGraphData({ nodes: result.nodes, edges: result.edges });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to build graph');
      } finally {
        setIsLoading(false);
      }
    }
  }, [codeInput, inputCodes]);

  const handleRemoveCode = useCallback(async (code: string) => {
    const newCodes = new Set(inputCodes);
    newCodes.delete(code);
    setInputCodes(newCodes);

    // Auto-refresh graph with remaining codes
    if (newCodes.size > 0) {
      setIsLoading(true);
      try {
        const result = await buildGraph([...newCodes]);
        setGraphData({ nodes: result.nodes, edges: result.edges });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to build graph');
      } finally {
        setIsLoading(false);
      }
    } else {
      setGraphData(null);
    }
  }, [inputCodes]);

  const handleClearCodes = useCallback(() => {
    setInputCodes(new Set());
    setGraphData(null);
    setError(null);
  }, []);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(prev => (prev === nodeId ? null : nodeId));
  }, []);

  return {
    // State
    inputCodes,
    codeInput,
    graphData,
    isLoading,
    error,
    viewTab,
    selectedNode,

    // Setters
    setCodeInput,
    setViewTab,

    // Handlers
    handleAddCode,
    handleRemoveCode,
    handleClearCodes,
    handleNodeClick,
  };
}