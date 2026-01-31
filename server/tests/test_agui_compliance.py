"""AG-UI Protocol Compliance Tests

These tests verify that the SSE events conform to the AG-UI protocol v0.1 spec.
They ensure that a stock AG-UI client/SDK can consume the stream without adapters.

AG-UI Protocol Requirements:
- RUN_STARTED/RUN_FINISHED/RUN_ERROR: must include threadId, runId
- STEP_STARTED/STEP_FINISHED: must use stepName (not step_id)
- STATE_SNAPSHOT: must use snapshot field (not state)
- RUN_ERROR: dedicated event for errors (not RUN_FINISHED.metadata.error)
"""

import json
import pytest

from server.agui_enums import AGUIEventType
from server.events import AGUIEvent, GraphState, JsonPatchOp
from graph import GraphNode, GraphEdge, NodeCategory, EdgeType


class TestAGUIEventTypes:
    """Test that all required AG-UI event types are defined."""

    def test_run_lifecycle_events_exist(self):
        """RUN_STARTED, RUN_FINISHED, and RUN_ERROR must be defined."""
        assert hasattr(AGUIEventType, 'RUN_STARTED')
        assert hasattr(AGUIEventType, 'RUN_FINISHED')
        assert hasattr(AGUIEventType, 'RUN_ERROR')

    def test_step_lifecycle_events_exist(self):
        """STEP_STARTED and STEP_FINISHED must be defined."""
        assert hasattr(AGUIEventType, 'STEP_STARTED')
        assert hasattr(AGUIEventType, 'STEP_FINISHED')

    def test_state_events_exist(self):
        """STATE_SNAPSHOT and STATE_DELTA must be defined."""
        assert hasattr(AGUIEventType, 'STATE_SNAPSHOT')
        assert hasattr(AGUIEventType, 'STATE_DELTA')


class TestRunLifecycleEvents:
    """Test run lifecycle events (RUN_STARTED, RUN_FINISHED, RUN_ERROR)."""

    def test_run_started_includes_thread_and_run_id(self):
        """RUN_STARTED must include threadId and runId in JSON output."""
        event = AGUIEvent(
            type=AGUIEventType.RUN_STARTED,
            threadId="thread-123",
            runId="run-456",
            metadata={"cached": False}
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "RUN_STARTED"
        assert data["threadId"] == "thread-123"
        assert data["runId"] == "run-456"
        assert "metadata" in data

    def test_run_finished_includes_thread_and_run_id(self):
        """RUN_FINISHED must include threadId and runId in JSON output."""
        event = AGUIEvent(
            type=AGUIEventType.RUN_FINISHED,
            threadId="thread-123",
            runId="run-456",
            metadata={"final_nodes": ["A00", "A01"]}
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "RUN_FINISHED"
        assert data["threadId"] == "thread-123"
        assert data["runId"] == "run-456"

    def test_run_error_includes_thread_run_id_and_error(self):
        """RUN_ERROR must include threadId, runId, and error field."""
        event = AGUIEvent(
            type=AGUIEventType.RUN_ERROR,
            threadId="thread-123",
            runId="run-456",
            error="API Error: Rate limit exceeded"
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "RUN_ERROR"
        assert data["threadId"] == "thread-123"
        assert data["runId"] == "run-456"
        assert data["error"] == "API Error: Rate limit exceeded"

    def test_rewind_run_includes_parent_run_id(self):
        """Rewind runs must include parentRunId."""
        event = AGUIEvent(
            type=AGUIEventType.RUN_STARTED,
            threadId="thread-123",
            runId="run-789",
            parentRunId="ROOT|children",  # batch_id of parent run
            metadata={"rewind_from": "ROOT|children"}
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "RUN_STARTED"
        assert data["parentRunId"] == "ROOT|children"


class TestStepLifecycleEvents:
    """Test step lifecycle events (STEP_STARTED, STEP_FINISHED)."""

    def test_step_started_uses_step_name(self):
        """STEP_STARTED must use stepName (not step_id)."""
        event = AGUIEvent(
            type=AGUIEventType.STEP_STARTED,
            stepName="ROOT|children"
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "STEP_STARTED"
        assert data["stepName"] == "ROOT|children"
        assert "step_id" not in data

    def test_step_finished_uses_step_name(self):
        """STEP_FINISHED must use stepName (not step_id)."""
        event = AGUIEvent(
            type=AGUIEventType.STEP_FINISHED,
            stepName="ROOT|children",
            metadata={
                "node_id": "ROOT",
                "batch_type": "children",
                "selected_ids": ["A00", "A01"]
            }
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "STEP_FINISHED"
        assert data["stepName"] == "ROOT|children"
        assert "step_id" not in data


class TestStateEvents:
    """Test state events (STATE_SNAPSHOT, STATE_DELTA)."""

    def test_state_snapshot_uses_snapshot_field(self):
        """STATE_SNAPSHOT must use 'snapshot' field (not 'state')."""
        graph_state = GraphState(
            nodes=[
                GraphNode(
                    id="ROOT",
                    code="ROOT",
                    label="ICD-10-CM",
                    depth=0,
                    category=NodeCategory.ROOT
                ),
                GraphNode(
                    id="A00",
                    code="A00",
                    label="Cholera",
                    depth=1,
                    category=NodeCategory.ANCESTOR
                )
            ],
            edges=[
                GraphEdge(
                    source="ROOT",
                    target="A00",
                    edge_type=EdgeType.HIERARCHY
                )
            ]
        )

        event = AGUIEvent(
            type=AGUIEventType.STATE_SNAPSHOT,
            snapshot=graph_state
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "STATE_SNAPSHOT"
        assert "snapshot" in data
        assert "state" not in data
        assert len(data["snapshot"]["nodes"]) == 2
        assert len(data["snapshot"]["edges"]) == 1

    def test_state_delta_uses_delta_field(self):
        """STATE_DELTA must use 'delta' field with JSON Patch ops."""
        event = AGUIEvent(
            type=AGUIEventType.STATE_DELTA,
            delta=[
                JsonPatchOp(op="add", path="/nodes/-", value={"id": "A01", "code": "A01"}),
                JsonPatchOp(op="add", path="/edges/-", value={"source": "ROOT", "target": "A01"})
            ]
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "STATE_DELTA"
        assert "delta" in data
        assert len(data["delta"]) == 2
        assert data["delta"][0]["op"] == "add"


class TestFieldNamingConventions:
    """Test that field names follow AG-UI conventions."""

    def test_no_step_id_in_any_event(self):
        """Ensure 'step_id' is never used (replaced by 'stepName')."""
        # Test all event types
        events = [
            AGUIEvent(type=AGUIEventType.RUN_STARTED, threadId="t", runId="r"),
            AGUIEvent(type=AGUIEventType.RUN_FINISHED, threadId="t", runId="r"),
            AGUIEvent(type=AGUIEventType.RUN_ERROR, threadId="t", runId="r", error="err"),
            AGUIEvent(type=AGUIEventType.STEP_STARTED, stepName="s"),
            AGUIEvent(type=AGUIEventType.STEP_FINISHED, stepName="s"),
            AGUIEvent(type=AGUIEventType.STATE_SNAPSHOT, snapshot=GraphState(nodes=[], edges=[])),
            AGUIEvent(type=AGUIEventType.STATE_DELTA, delta=[]),
        ]

        for event in events:
            json_str = event.model_dump_json()
            assert "step_id" not in json_str, f"Found 'step_id' in {event.type}"

    def test_no_state_field_in_snapshot_event(self):
        """Ensure 'state' is never used (replaced by 'snapshot')."""
        event = AGUIEvent(
            type=AGUIEventType.STATE_SNAPSHOT,
            snapshot=GraphState(nodes=[], edges=[])
        )
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        # Check that 'state' key doesn't exist at top level
        assert "state" not in data, "Found 'state' field in STATE_SNAPSHOT event"


class TestErrorHandling:
    """Test error handling follows AG-UI protocol."""

    def test_run_error_is_separate_from_run_finished(self):
        """Errors should emit RUN_ERROR, not just RUN_FINISHED with error in metadata."""
        # Create a RUN_ERROR event
        error_event = AGUIEvent(
            type=AGUIEventType.RUN_ERROR,
            threadId="t",
            runId="r",
            error="Connection timeout"
        )
        json_str = error_event.model_dump_json()
        data = json.loads(json_str)

        # Verify it's a proper RUN_ERROR event
        assert data["type"] == "RUN_ERROR"
        assert data["error"] == "Connection timeout"

        # Verify RUN_FINISHED doesn't need error in metadata for normal completion
        finished_event = AGUIEvent(
            type=AGUIEventType.RUN_FINISHED,
            threadId="t",
            runId="r",
            metadata={"final_nodes": ["A00"]}
        )
        finished_json = finished_event.model_dump_json()
        finished_data = json.loads(finished_json)

        assert finished_data["type"] == "RUN_FINISHED"
        assert "error" not in finished_data.get("metadata", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
