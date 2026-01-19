"""Agent state types for ICD-10-CM traversal

These types define the agent's decision-making process during DFS traversal:
- CandidateDecision: Individual code selection decision
- DecisionPoint: A point where the agent made choices
- TraversalState: Complete state during traversal (for Burr persistence)
"""

from pydantic import BaseModel, Field

from graph import GraphEdge, GraphNode, GraphStatus


class CandidateDecision(BaseModel):
    """Agent's selection decision for a candidate code.

    Records the decision, confidence, and reasoning for each
    candidate code presented at a decision point.
    """

    code: str
    label: str
    selected: bool  # Whether to include this candidate
    confidence: float  # 0.0-1.0
    evidence: str | None = None  # Clinical text snippet supporting selection
    reasoning: str  # Explanation of the decision


class DecisionPoint(BaseModel):
    """A decision point during DFS traversal.

    Represents a single node in the traversal where the agent
    evaluated candidates and made selections. Streamed to frontend.
    """

    current_node: str
    current_label: str
    depth: int
    candidates: list[CandidateDecision]
    selected_codes: list[str]  # Codes selected for further traversal


class TraversalState(BaseModel):
    """Complete state during DFS traversal.

    This state is:
    - Persisted by Burr for checkpoint/resume
    - Synchronized to frontend via AG-UI
    - Updated incrementally as the agent traverses

    Uses core.GraphNode and core.GraphEdge for consistent serialization.
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    decision_history: list[DecisionPoint] = Field(default_factory=list)
    current_path: list[str] = Field(default_factory=list)  # DFS stack
    finalized_codes: list[str] = Field(default_factory=list)
    status: GraphStatus = GraphStatus.IDLE
    current_step: str = ""
    error: str | None = None
