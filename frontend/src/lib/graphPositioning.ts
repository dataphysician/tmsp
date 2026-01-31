/**
 * Graph positioning algorithm for GraphViewer.
 *
 * Implements a DFS-based chapter container boundary system for laying out
 * ICD-10-CM code hierarchy graphs. Uses bounded region allocation to
 * eliminate collisions by construction.
 *
 * Key features:
 * - Chapter containers that dynamically expand for lateral links
 * - Bounded region allocation (no reactive collision resolution)
 * - Single-child chain collinear alignment
 * - sevenChrDef target alignment with source nodes
 */

import type { GraphNode, GraphEdge } from './types';

// ============================================================================
// Type Definitions
// ============================================================================

interface ChapterBoundary {
  chapterId: string;
  left: number;
  right: number;
  centerX: number;
}

/** Computed during bottom-up pass, contains actual width requirements */
interface SubtreeBounds {
  /** Total width for subtree (hierarchy + laterals) */
  requiredWidth: number;
  /** Width of hierarchy children portion (includes their laterals) */
  hierarchyChildrenWidth: number;
  /** Pure hierarchy width (EXCLUDES all nested laterals) */
  hierarchyOnlyWidth: number;
  /** Width of THIS node's direct lateral targets */
  lateralTargetsWidth: number;
  /** Lateral targets we position (not those with hierarchy parents) */
  effectiveLateralTargets: string[];
  /** No branching, no laterals - position collinearly */
  isSingleChain: boolean;
  /** Positioned collinearly (same X as source) */
  sevenChrDefTargets: string[];
}

/** Passed during top-down positioning, defines the region a node can use */
interface AllocatedRegion {
  left: number;
  right: number;
}

interface DFSPositionContext {
  /** The bounded region this subtree can use */
  region: AllocatedRegion;
  /** X position to inherit for single-chain collinear positioning */
  parentX: number;
  depth: number;
  chapterId: string;
}

// ============================================================================
// Main Positioning Function
// ============================================================================

/**
 * Calculate positions for all nodes using chapter boundary algorithm.
 *
 * Chapters (depth 1 nodes) act as containers - their boundaries expand based on
 * the number of children and lateral link targets within each chapter.
 *
 * @param _nodes - All nodes to position
 * @param hierarchyChildren - Map of parent -> children (hierarchy + connectivity lateral)
 * @param _allChildren - All edges (unused, kept for API compat)
 * @param containerWidth - Width of the container
 * @param nodeMap - Map of node ID -> node
 * @param lateralEdges - All lateral edges
 * @param nodeWidth - Width of a node
 * @param nodeHeight - Height of a node
 * @param levelHeight - Vertical spacing between levels
 * @param _finalizedCodesSet - Finalized codes (unused, kept for API compat)
 * @param orphanRescuedNodes - Nodes rescued via lateral edges
 * @returns Map of node ID -> position
 */
export function calculatePositions(
  _nodes: GraphNode[],
  hierarchyChildren: Map<string, string[]>,
  _allChildren: Map<string, string[]>,
  containerWidth: number,
  nodeMap: Map<string, GraphNode>,
  lateralEdges: GraphEdge[],
  nodeWidth: number,
  nodeHeight: number,
  levelHeight: number,
  _finalizedCodesSet: Set<string> = new Set(),
  orphanRescuedNodes: Set<string> = new Set()
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const layoutWidth = nodeWidth;
  const nodePadding = 10;
  const CHAPTER_PADDING = 20;
  const minGap = nodeWidth + 10;

  // ===== HELPER FUNCTIONS =====

  const getYForDepth = (depth: number): number => depth * levelHeight + 50;

  const isPlaceholderNode = (childId: string): boolean => {
    const child = nodeMap.get(childId);
    return child?.category === 'placeholder' || (childId.endsWith('X') && !nodeMap.has(childId));
  };

  const getParentCode = (code: string): string | null => {
    if (!code || code === 'ROOT') return null;
    if (code.includes('.')) {
      const [category, subcategory] = code.split('.');
      if (subcategory.length > 1) {
        return `${category}.${subcategory.slice(0, -1)}`;
      } else {
        return category;
      }
    }
    return null;
  };

  // Build sevenChrDef target map: source -> targets
  const sevenChrDefTargetsPerSource = new Map<string, Array<{ targetId: string; edge: GraphEdge }>>();
  for (const edge of lateralEdges) {
    if (edge.rule === 'sevenChrDef') {
      const sourceId = String(edge.source);
      const targetId = String(edge.target);
      if (!sevenChrDefTargetsPerSource.has(sourceId)) {
        sevenChrDefTargetsPerSource.set(sourceId, []);
      }
      sevenChrDefTargetsPerSource.get(sourceId)!.push({ targetId, edge });
    }
  }

  // Build non-sevenChrDef lateral edges map
  const lateralTargetsPerSource = new Map<string, Array<{ targetId: string; edge: GraphEdge }>>();
  for (const edge of lateralEdges) {
    if (edge.rule !== 'sevenChrDef' && edge.rule !== null) {
      const sourceId = String(edge.source);
      const targetId = String(edge.target);
      if (!lateralTargetsPerSource.has(sourceId)) {
        lateralTargetsPerSource.set(sourceId, []);
      }
      lateralTargetsPerSource.get(sourceId)!.push({ targetId, edge });
    }
  }

  // Build TRUE hierarchy parent map
  const hierarchyParent = new Map<string, string>();
  for (const [parentId, children] of hierarchyChildren.entries()) {
    const parentLateralTargets = new Set(
      (lateralTargetsPerSource.get(parentId) || []).map(t => t.targetId)
    );
    for (const childId of children) {
      if (!parentLateralTargets.has(childId) || orphanRescuedNodes.has(childId)) {
        hierarchyParent.set(childId, parentId);
      }
    }
  }

  // ===== PHASE 1: Pre-compute subtree bounds (bottom-up) =====
  const subtreeBounds = new Map<string, SubtreeBounds>();

  function computeSubtreeBounds(nodeId: string, visited = new Set<string>()): SubtreeBounds {
    if (subtreeBounds.has(nodeId)) return subtreeBounds.get(nodeId)!;

    if (visited.has(nodeId)) {
      return {
        requiredWidth: layoutWidth + nodePadding,
        hierarchyChildrenWidth: 0,
        hierarchyOnlyWidth: layoutWidth + nodePadding,
        lateralTargetsWidth: 0,
        effectiveLateralTargets: [],
        isSingleChain: true,
        sevenChrDefTargets: []
      };
    }
    visited.add(nodeId);

    const allChildIds = hierarchyChildren.get(nodeId) || [];
    const sevenChrTargets = sevenChrDefTargetsPerSource.get(nodeId) || [];
    const lateralTargets = lateralTargetsPerSource.get(nodeId) || [];

    const effectiveLateralTargets = lateralTargets
      .filter(({ targetId }) => {
        if (orphanRescuedNodes.has(targetId)) return true;
        const targetHierarchyParent = hierarchyParent.get(targetId);
        return !targetHierarchyParent || !nodeMap.has(targetHierarchyParent);
      })
      .map(({ targetId }) => targetId);

    const lateralTargetIds = new Set(lateralTargets.map(t => t.targetId));
    const childIds = allChildIds.filter(id => !lateralTargetIds.has(id));
    const regularChildren = childIds.filter(id => !isPlaceholderNode(id));
    const allPlaceholderChildren = childIds.filter(id => isPlaceholderNode(id));

    const placeholdersWithDescendants: string[] = [];
    const placeholdersWithoutDescendants: string[] = [];
    for (const placeholderId of allPlaceholderChildren) {
      const hasSevenChrTargets = (sevenChrDefTargetsPerSource.get(placeholderId) || []).length > 0;
      const hasChildren = (hierarchyChildren.get(placeholderId) || []).length > 0;
      if (hasSevenChrTargets || hasChildren) {
        placeholdersWithDescendants.push(placeholderId);
      } else {
        placeholdersWithoutDescendants.push(placeholderId);
      }
    }

    const sevenChrDefTargetIds = sevenChrTargets.map(({ targetId }) => targetId);

    let hierarchyChildrenWidth = 0;
    let hierarchyOnlyWidth = 0;
    let allChildrenSingleChain = true;
    const totalRenderedChildren = regularChildren.length + placeholdersWithDescendants.length;

    for (const childId of regularChildren) {
      const childBounds = computeSubtreeBounds(childId, new Set(visited));
      hierarchyChildrenWidth += childBounds.requiredWidth;
      hierarchyOnlyWidth += childBounds.hierarchyOnlyWidth;
      if (!childBounds.isSingleChain || totalRenderedChildren > 1) {
        allChildrenSingleChain = false;
      }
    }

    for (const childId of placeholdersWithDescendants) {
      const childBounds = computeSubtreeBounds(childId, new Set(visited));
      hierarchyChildrenWidth += childBounds.requiredWidth;
      hierarchyOnlyWidth += childBounds.hierarchyOnlyWidth;
      if (!childBounds.isSingleChain || totalRenderedChildren > 1) {
        allChildrenSingleChain = false;
      }
    }

    if (totalRenderedChildren > 1) {
      hierarchyChildrenWidth += (totalRenderedChildren - 1) * nodePadding;
      hierarchyOnlyWidth += (totalRenderedChildren - 1) * nodePadding;
    }

    if (totalRenderedChildren === 0) {
      for (const childId of placeholdersWithoutDescendants) {
        computeSubtreeBounds(childId, new Set(visited));
      }
    }

    for (const targetId of sevenChrDefTargetIds) {
      computeSubtreeBounds(targetId, new Set(visited));
    }

    let lateralTargetsWidth = 0;
    for (const targetId of effectiveLateralTargets) {
      const targetBounds = computeSubtreeBounds(targetId, new Set(visited));
      lateralTargetsWidth += targetBounds.requiredWidth + nodePadding;
    }

    const hasEffectiveLaterals = effectiveLateralTargets.length > 0;
    const effectiveChildCount = regularChildren.length + placeholdersWithDescendants.length + sevenChrDefTargetIds.length;
    const isSingleChain = effectiveChildCount <= 1 &&
                          allChildrenSingleChain &&
                          placeholdersWithoutDescendants.length <= 1 &&
                          !hasEffectiveLaterals;

    const selfWidth = layoutWidth + nodePadding;
    const requiredWidth = Math.max(selfWidth, hierarchyChildrenWidth) + lateralTargetsWidth;
    const finalHierarchyOnlyWidth = Math.max(selfWidth, hierarchyOnlyWidth);

    const bounds: SubtreeBounds = {
      requiredWidth,
      hierarchyChildrenWidth,
      hierarchyOnlyWidth: finalHierarchyOnlyWidth,
      lateralTargetsWidth,
      effectiveLateralTargets,
      isSingleChain,
      sevenChrDefTargets: sevenChrDefTargetIds
    };
    subtreeBounds.set(nodeId, bounds);
    return bounds;
  }

  const chapters: string[] = hierarchyChildren.get('ROOT') || [];
  for (const chapterId of chapters) {
    computeSubtreeBounds(chapterId, new Set<string>());
  }

  // ===== PHASE 2: Initialize chapter boundaries =====
  const sortedChapters = [...chapters].sort();
  const chapterBoundaries = new Map<string, ChapterBoundary>();

  let totalRequiredWidth = 0;
  for (const chapterId of sortedChapters) {
    const bounds = subtreeBounds.get(chapterId);
    totalRequiredWidth += bounds?.requiredWidth ?? (layoutWidth + nodePadding);
  }
  totalRequiredWidth += (sortedChapters.length - 1) * CHAPTER_PADDING;

  let currentX = Math.max(50, (containerWidth - totalRequiredWidth) / 2);

  for (const chapterId of sortedChapters) {
    const bounds = subtreeBounds.get(chapterId);
    const width = bounds?.requiredWidth ?? (layoutWidth + nodePadding);
    const left = currentX;
    const right = currentX + width;
    const centerX = (left + right) / 2;
    chapterBoundaries.set(chapterId, { chapterId, left, right, centerX });
    currentX = right + CHAPTER_PADDING;
  }

  // ===== PHASE 3: Position ROOT =====
  positions.set('ROOT', { x: containerWidth / 2, y: 25 });

  // ===== PHASE 4: Bounded Region Positioning =====
  const positionedNodes = new Set<string>();
  const nodePositionedUnderChapter = new Map<string, string>();

  function positionWithRegion(
    nodeId: string,
    context: DFSPositionContext
  ): { usedLeft: number; usedRight: number } {
    if (positionedNodes.has(nodeId)) {
      const existingPos = positions.get(nodeId);
      if (existingPos) {
        return { usedLeft: existingPos.x - layoutWidth / 2, usedRight: existingPos.x + layoutWidth / 2 };
      }
    }
    positionedNodes.add(nodeId);
    nodePositionedUnderChapter.set(nodeId, context.chapterId);

    const node = nodeMap.get(nodeId);
    const nodeDepth = node?.depth ?? context.depth;
    const y = getYForDepth(nodeDepth) + nodeHeight / 2;

    const bounds = subtreeBounds.get(nodeId);
    const isSingleChain = bounds?.isSingleChain ?? false;
    const effectiveLateralTargets = bounds?.effectiveLateralTargets ?? [];
    const sevenChrDefTargets = bounds?.sevenChrDefTargets ?? [];

    const allChildIds = hierarchyChildren.get(nodeId) || [];
    const lateralTargets = lateralTargetsPerSource.get(nodeId) || [];
    const lateralTargetIds = new Set(lateralTargets.map(t => t.targetId));
    const childIds = allChildIds.filter(id => !lateralTargetIds.has(id));
    const regularChildren = childIds.filter(id => !isPlaceholderNode(id)).sort();
    const allPlaceholderChildren = childIds.filter(id => isPlaceholderNode(id)).sort();

    const placeholdersWithDescendants: string[] = [];
    const placeholdersWithoutDescendants: string[] = [];
    for (const placeholderId of allPlaceholderChildren) {
      const hasSevenChrTargets = (sevenChrDefTargetsPerSource.get(placeholderId) || []).length > 0;
      const hasChildren = (hierarchyChildren.get(placeholderId) || []).length > 0;
      if (hasSevenChrTargets || hasChildren) {
        placeholdersWithDescendants.push(placeholderId);
      } else {
        placeholdersWithoutDescendants.push(placeholderId);
      }
    }

    const renderedChildren = [...regularChildren, ...placeholdersWithDescendants].sort();

    let myX: number;
    let usedLeft: number;
    let usedRight: number;

    // CASE 1: Single-chain collinear positioning
    if (isSingleChain) {
      myX = context.parentX;
      positions.set(nodeId, { x: myX, y });
      usedLeft = myX - layoutWidth / 2;
      usedRight = myX + layoutWidth / 2;

      if (renderedChildren.length === 1) {
        const childId = renderedChildren[0];
        const childResult = positionWithRegion(childId, {
          region: context.region,
          parentX: myX,
          depth: nodeDepth + 1,
          chapterId: context.chapterId
        });
        usedLeft = Math.min(usedLeft, childResult.usedLeft);
        usedRight = Math.max(usedRight, childResult.usedRight);
      }

      if (renderedChildren.length === 0) {
        for (const placeholderId of placeholdersWithoutDescendants) {
          const childResult = positionWithRegion(placeholderId, {
            region: context.region,
            parentX: myX,
            depth: nodeDepth + 1,
            chapterId: context.chapterId
          });
          usedLeft = Math.min(usedLeft, childResult.usedLeft);
          usedRight = Math.max(usedRight, childResult.usedRight);
        }
      }

      for (const targetId of sevenChrDefTargets) {
        if (!positionedNodes.has(targetId)) {
          positionedNodes.add(targetId);
          const targetNode = nodeMap.get(targetId);
          const targetDepth = targetNode?.depth ?? (nodeDepth + 1);
          const targetY = getYForDepth(targetDepth) + nodeHeight / 2;
          positions.set(targetId, { x: myX, y: targetY });

          const targetResult = positionWithRegion(targetId, {
            region: context.region,
            parentX: myX,
            depth: targetDepth,
            chapterId: context.chapterId
          });
          usedLeft = Math.min(usedLeft, targetResult.usedLeft);
          usedRight = Math.max(usedRight, targetResult.usedRight);
        }
      }
    }
    // CASE 2: Multiple rendered children
    else if (renderedChildren.length > 1) {
      let totalHierarchyWidth = 0;
      for (const childId of renderedChildren) {
        const childBounds = subtreeBounds.get(childId);
        totalHierarchyWidth += childBounds?.requiredWidth ?? (layoutWidth + nodePadding);
      }
      totalHierarchyWidth += (renderedChildren.length - 1) * nodePadding;

      const lateralWidth = bounds?.lateralTargetsWidth ?? 0;
      const hierarchyRegionRight = context.region.right - lateralWidth;
      const hierarchyRegionWidth = hierarchyRegionRight - context.region.left;
      const centeringOffset = Math.max(0, (hierarchyRegionWidth - totalHierarchyWidth) / 2);
      let regionX = context.region.left + centeringOffset;

      const childAnchors: number[] = [];
      usedLeft = Infinity;
      usedRight = -Infinity;

      for (const childId of renderedChildren) {
        const childBounds = subtreeBounds.get(childId);
        const childWidth = childBounds?.requiredWidth ?? (layoutWidth + nodePadding);
        const childRegion: AllocatedRegion = {
          left: regionX,
          right: regionX + childWidth
        };

        const childHierarchyWidth = childBounds?.hierarchyChildrenWidth ?? 0;
        const childSelfWidth = layoutWidth + nodePadding;
        const hierarchyPortionWidth = Math.max(childSelfWidth, childHierarchyWidth);
        const childParentX = childRegion.left + hierarchyPortionWidth / 2;

        childAnchors.push(childParentX);

        const childResult = positionWithRegion(childId, {
          region: childRegion,
          parentX: childParentX,
          depth: nodeDepth + 1,
          chapterId: context.chapterId
        });

        usedLeft = Math.min(usedLeft, childResult.usedLeft);
        usedRight = Math.max(usedRight, childResult.usedRight);
        usedRight = Math.max(usedRight, childResult.usedRight, childRegion.right);
        regionX = Math.max(childRegion.right, childResult.usedRight) + nodePadding;
      }

      const allocatedRegionEnd = regionX - nodePadding;
      usedRight = Math.max(usedRight, allocatedRegionEnd);

      myX = (Math.min(...childAnchors) + Math.max(...childAnchors)) / 2;
      positions.set(nodeId, { x: myX, y });

      usedLeft = Math.min(usedLeft, myX - layoutWidth / 2);
      usedRight = Math.max(usedRight, myX + layoutWidth / 2);

      for (const targetId of sevenChrDefTargets) {
        if (!positionedNodes.has(targetId)) {
          positionedNodes.add(targetId);
          const targetNode = nodeMap.get(targetId);
          const targetDepth = targetNode?.depth ?? (nodeDepth + 1);
          const targetY = getYForDepth(targetDepth) + nodeHeight / 2;
          positions.set(targetId, { x: myX, y: targetY });

          const targetResult = positionWithRegion(targetId, {
            region: { left: usedLeft, right: usedRight },
            parentX: myX,
            depth: targetDepth,
            chapterId: context.chapterId
          });
          usedLeft = Math.min(usedLeft, targetResult.usedLeft);
          usedRight = Math.max(usedRight, targetResult.usedRight);
        }
      }
    }
    // CASE 3: Leaf or single rendered child with lateral targets
    else {
      myX = context.parentX;
      positions.set(nodeId, { x: myX, y });
      usedLeft = myX - layoutWidth / 2;
      usedRight = myX + layoutWidth / 2;

      if (renderedChildren.length === 1) {
        const childId = renderedChildren[0];
        const childRegion: AllocatedRegion = {
          left: context.region.left,
          right: context.region.right
        };

        const childResult = positionWithRegion(childId, {
          region: childRegion,
          parentX: myX,
          depth: nodeDepth + 1,
          chapterId: context.chapterId
        });
        usedLeft = Math.min(usedLeft, childResult.usedLeft);
        usedRight = Math.max(usedRight, childResult.usedRight);

        const nodeHierarchyOnlyWidth = bounds?.hierarchyOnlyWidth ?? (layoutWidth + nodePadding);
        usedRight = Math.max(usedRight, context.region.left + nodeHierarchyOnlyWidth);
      }

      if (renderedChildren.length === 0) {
        for (const placeholderId of placeholdersWithoutDescendants) {
          const childResult = positionWithRegion(placeholderId, {
            region: context.region,
            parentX: myX,
            depth: nodeDepth + 1,
            chapterId: context.chapterId
          });
          usedLeft = Math.min(usedLeft, childResult.usedLeft);
          usedRight = Math.max(usedRight, childResult.usedRight);
        }
      }

      for (const targetId of sevenChrDefTargets) {
        if (!positionedNodes.has(targetId)) {
          positionedNodes.add(targetId);
          const targetNode = nodeMap.get(targetId);
          const targetDepth = targetNode?.depth ?? (nodeDepth + 1);
          const targetY = getYForDepth(targetDepth) + nodeHeight / 2;
          positions.set(targetId, { x: myX, y: targetY });

          const targetResult = positionWithRegion(targetId, {
            region: context.region,
            parentX: myX,
            depth: targetDepth,
            chapterId: context.chapterId
          });
          usedLeft = Math.min(usedLeft, targetResult.usedLeft);
          usedRight = Math.max(usedRight, targetResult.usedRight);
        }
      }
    }

    // Position effective lateral targets
    if (effectiveLateralTargets.length > 0) {
      let lateralX = usedRight + nodePadding + layoutWidth / 2;
      const sourceRight = myX + layoutWidth / 2;
      const minLateralXFromSource = sourceRight + nodePadding + layoutWidth / 2;
      if (lateralX < minLateralXFromSource) {
        lateralX = minLateralXFromSource;
      }

      for (const targetId of effectiveLateralTargets) {
        if (positionedNodes.has(targetId)) continue;

        const targetNode = nodeMap.get(targetId);
        const targetDepth = targetNode?.depth ?? nodeDepth;
        const targetBounds = subtreeBounds.get(targetId);
        const targetWidth = targetBounds?.requiredWidth ?? (layoutWidth + nodePadding);

        const targetCode = targetNode?.code || targetId;
        const parentCode = getParentCode(targetCode);
        const parentPos = parentCode ? positions.get(parentCode) : undefined;

        let targetX: number;
        let targetRegion: AllocatedRegion;
        let alignedWithParent = false;

        if (parentPos) {
          const parentColumnRight = parentPos.x + targetWidth / 2;
          if (parentColumnRight <= usedRight + nodePadding) {
            targetX = lateralX;
          } else {
            targetX = parentPos.x;
            alignedWithParent = true;
          }
        } else {
          targetX = lateralX;
        }

        if (targetDepth <= nodeDepth) {
          const nodesToCheck: string[] = [nodeId];
          let currentAncestor = hierarchyParent.get(nodeId) || '';
          while (currentAncestor && currentAncestor !== 'ROOT') {
            const ancestorNode = nodeMap.get(currentAncestor);
            if (ancestorNode) {
              nodesToCheck.push(currentAncestor);
              if (ancestorNode.depth <= targetDepth) break;
            }
            currentAncestor = hierarchyParent.get(currentAncestor) || '';
          }

          for (const checkNodeId of nodesToCheck) {
            const checkPos = positions.get(checkNodeId);
            if (checkPos) {
              const minLateralX = checkPos.x + minGap + layoutWidth / 2;
              if (targetX < minLateralX) {
                targetX = minLateralX;
                alignedWithParent = false;
              }
            }
          }
        }

        targetRegion = {
          left: targetX - layoutWidth / 2,
          right: targetX - layoutWidth / 2 + targetWidth
        };

        const lateralResult = positionWithRegion(targetId, {
          region: targetRegion,
          parentX: targetX,
          depth: targetDepth,
          chapterId: context.chapterId
        });

        usedRight = Math.max(usedRight, lateralResult.usedRight);

        if (!alignedWithParent) {
          lateralX = lateralResult.usedRight + nodePadding + layoutWidth / 2;
        }
      }
    }

    return { usedLeft, usedRight };
  }

  // Position each chapter
  for (const chapterId of sortedChapters) {
    const boundary = chapterBoundaries.get(chapterId)!;
    const chapterBounds = subtreeBounds.get(chapterId);
    const hierarchyOnlyWidth = chapterBounds?.hierarchyOnlyWidth ?? (layoutWidth + nodePadding);
    const hierarchyCenterX = boundary.left + hierarchyOnlyWidth / 2;
    positionWithRegion(chapterId, {
      region: { left: boundary.left, right: boundary.right },
      parentX: hierarchyCenterX,
      depth: 1,
      chapterId
    });
  }

  // ===== PHASE 5: Resolve Chapter Collisions =====
  const wasPositionedUnderChapter = (nodeId: string, chapterId: string): boolean => {
    return nodePositionedUnderChapter.get(nodeId) === chapterId;
  };

  const resolveChapterCollisions = (): boolean => {
    let anyShifted = false;
    const sortedBoundaries = sortedChapters.map(id => chapterBoundaries.get(id)!);

    for (let i = 0; i < sortedBoundaries.length - 1; i++) {
      const leftChapter = sortedBoundaries[i];
      const rightChapter = sortedBoundaries[i + 1];

      const overlap = leftChapter.right + CHAPTER_PADDING - rightChapter.left;
      if (overlap > 0) {
        anyShifted = true;
        const shiftAmount = overlap;

        rightChapter.left += shiftAmount;
        rightChapter.right += shiftAmount;
        rightChapter.centerX += shiftAmount;

        for (const [nodeId, pos] of positions.entries()) {
          if (nodeId === 'ROOT') continue;
          if (wasPositionedUnderChapter(nodeId, rightChapter.chapterId)) {
            positions.set(nodeId, { x: pos.x + shiftAmount, y: pos.y });
          }
        }

        for (let j = i + 2; j < sortedBoundaries.length; j++) {
          const cascadeChapter = sortedBoundaries[j];
          cascadeChapter.left += shiftAmount;
          cascadeChapter.right += shiftAmount;
          cascadeChapter.centerX += shiftAmount;

          for (const [nodeId, pos] of positions.entries()) {
            if (nodeId === 'ROOT') continue;
            if (wasPositionedUnderChapter(nodeId, cascadeChapter.chapterId)) {
              positions.set(nodeId, { x: pos.x + shiftAmount, y: pos.y });
            }
          }
        }
      }
    }
    return anyShifted;
  };

  for (let pass = 0; pass < 5; pass++) {
    if (!resolveChapterCollisions()) break;
  }

  // ===== PHASE 6: Helper Functions =====
  const getAllDescendants = (nodeId: string): Set<string> => {
    const descendants = new Set<string>();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop()!;
      const children = hierarchyChildren.get(current) || [];
      for (const childId of children) {
        if (!descendants.has(childId)) {
          descendants.add(childId);
          stack.push(childId);
        }
      }
      const sevenChrTargets = sevenChrDefTargetsPerSource.get(current) || [];
      for (const { targetId } of sevenChrTargets) {
        if (!descendants.has(targetId)) {
          descendants.add(targetId);
          stack.push(targetId);
        }
      }
    }
    return descendants;
  };

  const shiftNodeAndDescendants = (nodeId: string, deltaX: number) => {
    const pos = positions.get(nodeId);
    if (pos) positions.set(nodeId, { x: pos.x + deltaX, y: pos.y });

    const descendants = getAllDescendants(nodeId);
    for (const descId of descendants) {
      const descPos = positions.get(descId);
      if (descPos) positions.set(descId, { x: descPos.x + deltaX, y: descPos.y });
    }
  };

  // ===== PHASE 7: Recalculate boundaries =====
  for (const chapterId of sortedChapters) {
    const boundary = chapterBoundaries.get(chapterId)!;
    let actualLeft = Infinity;
    let actualRight = -Infinity;

    for (const [nodeId, pos] of positions.entries()) {
      if (nodeId === 'ROOT') continue;
      if (wasPositionedUnderChapter(nodeId, chapterId)) {
        actualLeft = Math.min(actualLeft, pos.x - layoutWidth / 2);
        actualRight = Math.max(actualRight, pos.x + layoutWidth / 2);
      }
    }

    if (actualLeft !== Infinity && actualRight !== -Infinity) {
      boundary.left = actualLeft;
      boundary.right = actualRight;
      boundary.centerX = (actualLeft + actualRight) / 2;
    }
  }

  // ===== PHASE 7a: Compact chapters =====
  for (let i = 1; i < sortedChapters.length; i++) {
    const prevChapterId = sortedChapters[i - 1];
    const currChapterId = sortedChapters[i];
    const prevBoundary = chapterBoundaries.get(prevChapterId)!;
    const currBoundary = chapterBoundaries.get(currChapterId)!;

    const desiredLeft = prevBoundary.right + CHAPTER_PADDING;
    const shiftAmount = currBoundary.left - desiredLeft;

    if (shiftAmount > 1) {
      for (const [nodeId, pos] of positions.entries()) {
        if (nodeId === 'ROOT') continue;
        if (wasPositionedUnderChapter(nodeId, currChapterId)) {
          positions.set(nodeId, { x: pos.x - shiftAmount, y: pos.y });
        }
      }

      currBoundary.left -= shiftAmount;
      currBoundary.right -= shiftAmount;
      currBoundary.centerX -= shiftAmount;
    }
  }

  for (let pass = 0; pass < 5; pass++) {
    if (!resolveChapterCollisions()) break;
  }

  // ===== PHASE 7.5: Re-center chapter nodes =====
  for (const chapterId of sortedChapters) {
    const chapterPos = positions.get(chapterId);
    if (!chapterPos) continue;

    const chapterNode = nodeMap.get(chapterId);
    const chapterDepth = chapterNode?.depth ?? 1;
    const immediateChildren = (hierarchyChildren.get(chapterId) || [])
      .filter(childId => {
        const childNode = nodeMap.get(childId);
        return childNode && childNode.depth === chapterDepth + 1;
      });

    const trueHierarchyChildren = immediateChildren.filter(id => !orphanRescuedNodes.has(id));

    if (trueHierarchyChildren.length >= 2) {
      let minChildX = Infinity;
      let maxChildX = -Infinity;
      for (const childId of trueHierarchyChildren) {
        const childPos = positions.get(childId);
        if (childPos) {
          minChildX = Math.min(minChildX, childPos.x);
          maxChildX = Math.max(maxChildX, childPos.x);
        }
      }

      if (minChildX !== Infinity && maxChildX !== -Infinity) {
        const targetX = (minChildX + maxChildX) / 2;
        positions.set(chapterId, { x: targetX, y: chapterPos.y });
      }
    } else if (trueHierarchyChildren.length === 1) {
      const childPos = positions.get(trueHierarchyChildren[0]);
      if (childPos) {
        positions.set(chapterId, { x: childPos.x, y: chapterPos.y });
      }
    }
  }

  // ===== PHASE 7.6: Re-center ALL parent nodes =====
  const centerParentsBottomUp = () => {
    const nodesByDepth: Array<{ id: string; depth: number }> = [];
    for (const node of _nodes) {
      if (node.id === 'ROOT') continue;
      nodesByDepth.push({ id: node.id, depth: node.depth ?? 0 });
    }
    nodesByDepth.sort((a, b) => b.depth - a.depth);

    for (const { id: nodeId } of nodesByDepth) {
      if (sortedChapters.includes(nodeId)) continue;

      const children = hierarchyChildren.get(nodeId) || [];
      const renderedChildren = children.filter(id => {
        if (!isPlaceholderNode(id)) return true;
        const hasSevenChrTargets = (sevenChrDefTargetsPerSource.get(id) || []).length > 0;
        const hasChildren = (hierarchyChildren.get(id) || []).length > 0;
        return hasSevenChrTargets || hasChildren;
      });

      const trueHierarchyChildren = renderedChildren.filter(id => !orphanRescuedNodes.has(id));

      if (trueHierarchyChildren.length < 2) continue;

      const nodePos = positions.get(nodeId);
      if (!nodePos) continue;

      let minChildX = Infinity;
      let maxChildX = -Infinity;
      for (const childId of trueHierarchyChildren) {
        const childPos = positions.get(childId);
        if (childPos) {
          minChildX = Math.min(minChildX, childPos.x);
          maxChildX = Math.max(maxChildX, childPos.x);
        }
      }

      if (minChildX !== Infinity && maxChildX !== -Infinity) {
        const targetX = (minChildX + maxChildX) / 2;
        if (Math.abs(nodePos.x - targetX) > 0.5) {
          positions.set(nodeId, { x: targetX, y: nodePos.y });
        }
      }
    }
  };
  centerParentsBottomUp();

  // ===== PHASE 8: Force sevenChrDef alignment =====
  for (const [sourceId, targets] of sevenChrDefTargetsPerSource.entries()) {
    const sourcePos = positions.get(sourceId);
    if (!sourcePos) continue;

    for (const { targetId } of targets) {
      const targetPos = positions.get(targetId);
      if (targetPos && Math.abs(targetPos.x - sourcePos.x) > 0.5) {
        const deltaX = sourcePos.x - targetPos.x;
        shiftNodeAndDescendants(targetId, deltaX);
      }
    }
  }

  // ===== PHASE 9: Final chapter centering =====
  for (const chapterId of sortedChapters) {
    const chapterPos = positions.get(chapterId);
    if (!chapterPos) continue;

    const chapterNode = nodeMap.get(chapterId);
    const chapterDepth = chapterNode?.depth ?? 1;
    const immediateChildren = (hierarchyChildren.get(chapterId) || [])
      .filter(childId => {
        const childNode = nodeMap.get(childId);
        return childNode && childNode.depth === chapterDepth + 1;
      });

    const trueHierarchyChildren = immediateChildren.filter(id => !orphanRescuedNodes.has(id));

    if (trueHierarchyChildren.length >= 2) {
      let minChildX = Infinity;
      let maxChildX = -Infinity;
      for (const childId of trueHierarchyChildren) {
        const childPos = positions.get(childId);
        if (childPos) {
          minChildX = Math.min(minChildX, childPos.x);
          maxChildX = Math.max(maxChildX, childPos.x);
        }
      }

      if (minChildX !== Infinity && maxChildX !== -Infinity) {
        const targetX = (minChildX + maxChildX) / 2;
        positions.set(chapterId, { x: targetX, y: chapterPos.y });
      }
    } else if (trueHierarchyChildren.length === 1) {
      const childPos = positions.get(trueHierarchyChildren[0]);
      if (childPos) {
        positions.set(chapterId, { x: childPos.x, y: chapterPos.y });
      }
    }
  }

  // ===== FINAL COMPACTION =====
  const actualChapterBounds = new Map<string, { left: number; right: number }>();
  for (const chapterId of sortedChapters) {
    let actualLeft = Infinity;
    let actualRight = -Infinity;

    for (const [nodeId, pos] of positions.entries()) {
      if (nodeId === 'ROOT') continue;
      if (wasPositionedUnderChapter(nodeId, chapterId)) {
        actualLeft = Math.min(actualLeft, pos.x - layoutWidth / 2);
        actualRight = Math.max(actualRight, pos.x + layoutWidth / 2);
      }
    }

    if (actualLeft !== Infinity && actualRight !== -Infinity) {
      actualChapterBounds.set(chapterId, { left: actualLeft, right: actualRight });
    }
  }

  let targetLeft = 50;
  const chapterShifts = new Map<string, number>();

  for (const chapterId of sortedChapters) {
    const bounds = actualChapterBounds.get(chapterId);
    if (!bounds) continue;

    const shift = bounds.left - targetLeft;
    chapterShifts.set(chapterId, shift);
    targetLeft = (bounds.right - shift) + CHAPTER_PADDING;
  }

  for (const [nodeId, pos] of positions.entries()) {
    if (nodeId === 'ROOT') continue;
    const chapterId = nodePositionedUnderChapter.get(nodeId);
    if (chapterId) {
      const shift = chapterShifts.get(chapterId);
      if (shift && Math.abs(shift) > 0.5) {
        positions.set(nodeId, { x: pos.x - shift, y: pos.y });
      }
    }
  }

  for (const chapterId of sortedChapters) {
    const bounds = actualChapterBounds.get(chapterId);
    const shift = chapterShifts.get(chapterId);
    const boundary = chapterBoundaries.get(chapterId);
    if (bounds && shift !== undefined && boundary) {
      boundary.left = bounds.left - shift;
      boundary.right = bounds.right - shift;
      boundary.centerX = (boundary.left + boundary.right) / 2;
    }
  }

  for (const [sourceId, targets] of sevenChrDefTargetsPerSource.entries()) {
    const sourcePos = positions.get(sourceId);
    if (!sourcePos) continue;

    for (const { targetId } of targets) {
      const targetPos = positions.get(targetId);
      if (targetPos && Math.abs(targetPos.x - sourcePos.x) > 0.5) {
        const deltaX = sourcePos.x - targetPos.x;
        shiftNodeAndDescendants(targetId, deltaX);
      }
    }
  }

  // ===== FINAL: Re-center ROOT =====
  // Center ROOT over its DIRECT children (Chapter nodes only),
  // not over their lateral node expansions
  if (sortedChapters.length > 0) {
    const chapterPositions = sortedChapters
      .map(chapterId => positions.get(chapterId))
      .filter((pos): pos is { x: number; y: number } => pos !== undefined);

    if (chapterPositions.length > 0) {
      let minChapterX = Infinity;
      let maxChapterX = -Infinity;

      for (const pos of chapterPositions) {
        minChapterX = Math.min(minChapterX, pos.x);
        maxChapterX = Math.max(maxChapterX, pos.x);
      }

      if (minChapterX !== Infinity && maxChapterX !== -Infinity) {
        const midpointX = (minChapterX + maxChapterX) / 2;
        const rootPos = positions.get('ROOT');
        if (rootPos) {
          positions.set('ROOT', { x: midpointX, y: rootPos.y });
        }
      }
    }
  }

  // ===== PHASE 10: Same-row collision resolution =====
  // After all positioning, check for any nodes at the same Y that overlap horizontally
  // This handles edge cases where lateral links or rewind operations cause collisions
  const resolveRowCollisions = () => {
    // Group nodes by Y position (with small tolerance for floating point)
    const nodesByRow = new Map<number, Array<{ id: string; x: number }>>();

    for (const [nodeId, pos] of positions.entries()) {
      if (nodeId === 'ROOT') continue;

      // Round Y to handle floating point precision
      const rowY = Math.round(pos.y);
      if (!nodesByRow.has(rowY)) {
        nodesByRow.set(rowY, []);
      }
      nodesByRow.get(rowY)!.push({ id: nodeId, x: pos.x });
    }

    let anyShifted = false;

    // For each row, sort by X and resolve overlaps
    for (const [_rowY, nodesInRow] of nodesByRow.entries()) {
      if (nodesInRow.length < 2) continue;

      // Sort by X position
      nodesInRow.sort((a, b) => a.x - b.x);

      // Check each pair for collision
      for (let i = 0; i < nodesInRow.length - 1; i++) {
        const leftNode = nodesInRow[i];
        const rightNode = nodesInRow[i + 1];

        // Calculate minimum required gap (node width + padding)
        const minGapRequired = layoutWidth + nodePadding;
        const actualGap = rightNode.x - leftNode.x;

        if (actualGap < minGapRequired) {
          // Collision detected - shift ALL remaining nodes on this row (not just rightNode)
          // This prevents cascading collisions within the same row
          const shiftNeeded = minGapRequired - actualGap;
          anyShifted = true;

          // Shift ALL remaining nodes on this row plus their descendants
          for (let j = i + 1; j < nodesInRow.length; j++) {
            const nodeToShift = nodesInRow[j];
            const currentPos = positions.get(nodeToShift.id)!;
            positions.set(nodeToShift.id, { x: currentPos.x + shiftNeeded, y: currentPos.y });
            nodeToShift.x = currentPos.x + shiftNeeded;

            // Also shift all descendants
            const descendants = getAllDescendants(nodeToShift.id);
            for (const descId of descendants) {
              const descPos = positions.get(descId);
              if (descPos) {
                positions.set(descId, { x: descPos.x + shiftNeeded, y: descPos.y });
              }
            }
          }

          // After shifting all remaining nodes on this row, no need to check
          // more pairs in this row - they all moved together maintaining gaps
          break;
        }
      }
    }

    return anyShifted;
  };

  // Run collision resolution up to 5 passes (shifts may cascade)
  for (let pass = 0; pass < 5; pass++) {
    if (!resolveRowCollisions()) break;
  }

  // Log unpositioned nodes
  for (const node of _nodes) {
    if (!positions.has(node.id)) {
      console.warn(`[POSITION] Node ${node.id} has no position`);
    }
  }

  return positions;
}
