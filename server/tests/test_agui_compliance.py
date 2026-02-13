"""AG-UI Protocol Compliance Tests

These tests verify that the SSE events conform to the AG-UI protocol v0.1 spec
using the official ag-ui-protocol package types and EventEncoder.

AG-UI Protocol Requirements:
- Events use typed classes from ag_ui.core (not a flat envelope)
- RunStartedEvent/RunFinishedEvent: must include threadId, runId
- RunStartedEvent.input: carries the original RunAgentInput for observability
- RunErrorEvent: must include message (no threadId/runId per spec)
- StepStartedEvent/StepFinishedEvent: must use stepName (not step_id)
- StateSnapshotEvent: must use snapshot field
- StateDeltaEvent: must use delta field with JSON Patch ops
- Reasoning events: carry LLM reasoning text per step
- CustomEvent: carries domain-specific data with name + value
- RunAgentInput: standard request format with domain config in state
- EventEncoder: produces SSE format "data: {...}\n\n" with camelCase keys
- Timestamps: millisecond-precision timestamps on all events
"""

import json
import time

import pytest

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
    ReasoningStartEvent,
    ReasoningMessageStartEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningEndEvent,
    CustomEvent,
)
from ag_ui.encoder import EventEncoder

from server.app import encode_event

encoder = EventEncoder()


class TestEventEncoder:
    """Test that EventEncoder produces correct SSE format."""

    def test_encode_produces_sse_format(self):
        """EventEncoder.encode() must produce 'data: {...}\\n\\n' format."""
        event = RunStartedEvent(thread_id="t-1", run_id="r-1")
        encoded = encoder.encode(event)
        assert encoded.startswith("data: ")
        assert encoded.endswith("\n\n")

    def test_encode_uses_camel_case_keys(self):
        """Encoded JSON must use camelCase keys (threadId, not thread_id)."""
        event = RunStartedEvent(thread_id="t-1", run_id="r-1")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())
        assert "threadId" in data
        assert "runId" in data
        assert "thread_id" not in data
        assert "run_id" not in data

    def test_encode_excludes_none_fields(self):
        """Encoded JSON must exclude None fields (exclude_none behavior)."""
        event = StepStartedEvent(step_name="ROOT|children")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())
        # stepName should be present, but threadId/runId should NOT
        assert "stepName" in data
        assert "threadId" not in data


class TestEncodeEventHelper:
    """Test the encode_event helper that injects timestamps."""

    def test_encode_event_adds_timestamp(self):
        """encode_event() must inject a millisecond timestamp."""
        before = int(time.time() * 1000)
        event = RunStartedEvent(thread_id="t-1", run_id="r-1")
        encoded = encode_event(event)
        after = int(time.time() * 1000)

        data = json.loads(encoded.removeprefix("data: ").rstrip())
        assert "timestamp" in data
        assert before <= data["timestamp"] <= after

    def test_encode_event_preserves_sse_format(self):
        """encode_event() must still produce valid SSE format."""
        event = StepStartedEvent(step_name="ROOT|children")
        encoded = encode_event(event)
        assert encoded.startswith("data: ")
        assert encoded.endswith("\n\n")

    def test_encode_event_does_not_mutate_original(self):
        """encode_event() must not mutate the original event."""
        event = RunStartedEvent(thread_id="t-1", run_id="r-1")
        assert event.timestamp is None
        encode_event(event)
        assert event.timestamp is None


class TestRunLifecycleEvents:
    """Test run lifecycle events using ag_ui.core typed classes."""

    def test_run_started_serialization(self):
        """RunStartedEvent must serialize with threadId, runId, and type."""
        event = RunStartedEvent(thread_id="thread-123", run_id="run-456")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "RUN_STARTED"
        assert data["threadId"] == "thread-123"
        assert data["runId"] == "run-456"
        # No metadata field on standard events
        assert "metadata" not in data

    def test_run_started_with_input(self):
        """RunStartedEvent.input carries the original RunAgentInput."""
        run_input = RunAgentInput(
            thread_id="t-1",
            run_id="r-1",
            state={"clinical_note": "test"},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )
        event = RunStartedEvent(thread_id="t-1", run_id="r-1", input=run_input)
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "RUN_STARTED"
        assert "input" in data
        assert data["input"]["threadId"] == "t-1"
        assert data["input"]["state"]["clinical_note"] == "test"

    def test_run_finished_with_result(self):
        """RunFinishedEvent uses 'result' field (not 'metadata') for domain data."""
        event = RunFinishedEvent(
            thread_id="thread-123",
            run_id="run-456",
            result={"final_nodes": ["A00", "A01"], "batch_count": 3},
        )
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "RUN_FINISHED"
        assert data["threadId"] == "thread-123"
        assert data["result"]["final_nodes"] == ["A00", "A01"]
        assert "metadata" not in data

    def test_run_error_uses_message_field(self):
        """RunErrorEvent uses 'message' field (not 'error'), no threadId/runId."""
        event = RunErrorEvent(message="API Error: Rate limit exceeded")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "RUN_ERROR"
        assert data["message"] == "API Error: Rate limit exceeded"
        assert "error" not in data
        # RunErrorEvent does NOT have threadId/runId per AG-UI spec
        assert "threadId" not in data
        assert "runId" not in data


class TestStepLifecycleEvents:
    """Test step lifecycle events using ag_ui.core typed classes."""

    def test_step_started_uses_step_name(self):
        """StepStartedEvent must serialize with stepName in camelCase."""
        event = StepStartedEvent(step_name="ROOT|children")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "STEP_STARTED"
        assert data["stepName"] == "ROOT|children"
        assert "step_id" not in data
        assert "step_name" not in data

    def test_step_finished_has_no_metadata(self):
        """StepFinishedEvent has only stepName — no metadata field."""
        event = StepFinishedEvent(step_name="ROOT|children")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "STEP_FINISHED"
        assert data["stepName"] == "ROOT|children"
        assert "metadata" not in data


class TestReasoningEvents:
    """Test REASONING events for LLM reasoning (AG-UI standard)."""

    def test_reasoning_start_serialization(self):
        """ReasoningStartEvent must serialize with messageId."""
        event = ReasoningStartEvent(message_id="msg-1")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "REASONING_START"
        assert data["messageId"] == "msg-1"

    def test_reasoning_message_start_serialization(self):
        """ReasoningMessageStartEvent must serialize with role='reasoning'.

        The TypeScript @ag-ui/core v0.0.45 schema requires role='reasoning',
        but the Python model restricts to Literal['assistant']. The server uses
        _reasoning_message_start() helper to bypass this via model_construct().
        """
        from server.app import _reasoning_message_start
        event = _reasoning_message_start("msg-1")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "REASONING_MESSAGE_START"
        assert data["messageId"] == "msg-1"
        assert data["role"] == "reasoning"

    def test_reasoning_message_content_serialization(self):
        """ReasoningMessageContentEvent must serialize with messageId and delta."""
        event = ReasoningMessageContentEvent(message_id="msg-1", delta="Selected A00 based on clinical note")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "REASONING_MESSAGE_CONTENT"
        assert data["messageId"] == "msg-1"
        assert data["delta"] == "Selected A00 based on clinical note"

    def test_reasoning_message_end_serialization(self):
        """ReasoningMessageEndEvent must serialize with messageId."""
        event = ReasoningMessageEndEvent(message_id="msg-1")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "REASONING_MESSAGE_END"
        assert data["messageId"] == "msg-1"

    def test_reasoning_end_serialization(self):
        """ReasoningEndEvent must serialize with messageId."""
        event = ReasoningEndEvent(message_id="msg-1")
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "REASONING_END"
        assert data["messageId"] == "msg-1"

    def test_full_reasoning_sequence(self):
        """Full 5-event reasoning sequence must encode correctly."""
        events = [
            ReasoningStartEvent(message_id="msg-1"),
            ReasoningMessageStartEvent(message_id="msg-1", role="assistant"),
            ReasoningMessageContentEvent(message_id="msg-1", delta="reasoning text"),
            ReasoningMessageEndEvent(message_id="msg-1"),
            ReasoningEndEvent(message_id="msg-1"),
        ]
        types = [
            "REASONING_START",
            "REASONING_MESSAGE_START",
            "REASONING_MESSAGE_CONTENT",
            "REASONING_MESSAGE_END",
            "REASONING_END",
        ]
        for event, expected_type in zip(events, types):
            encoded = encoder.encode(event)
            data = json.loads(encoded.removeprefix("data: ").rstrip())
            assert data["type"] == expected_type
            assert data["messageId"] == "msg-1"


class TestCustomEvents:
    """Test CustomEvent for domain-specific data."""

    def test_step_metadata_custom_event(self):
        """step_metadata CUSTOM event carries domain data (no reasoning — it's in REASONING events)."""
        event = CustomEvent(
            name="step_metadata",
            value={
                "node_id": "ROOT",
                "batch_type": "children",
                "selected_ids": ["A00", "A01"],
                "candidates": {"A00": "Cholera", "A01": "Typhoid"},
                "error": False,
                "selected_details": {},
            },
        )
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "CUSTOM"
        assert data["name"] == "step_metadata"
        assert data["value"]["node_id"] == "ROOT"
        assert data["value"]["selected_ids"] == ["A00", "A01"]
        # Reasoning is no longer in step_metadata — it's in REASONING events
        assert "reasoning" not in data["value"]

    def test_run_metadata_custom_event(self):
        """RUN_STARTED metadata moves to CustomEvent(name='run_metadata')."""
        event = CustomEvent(
            name="run_metadata",
            value={"clinical_note": "Patient presents with...", "cached": False},
        )
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "CUSTOM"
        assert data["name"] == "run_metadata"
        assert data["value"]["cached"] is False


class TestStateEvents:
    """Test state events using ag_ui.core typed classes."""

    def test_state_snapshot_uses_snapshot_field(self):
        """StateSnapshotEvent must use 'snapshot' field."""
        snapshot = {
            "nodes": [
                {"id": "ROOT", "code": "ROOT", "label": "ICD-10-CM", "depth": 0},
                {"id": "A00", "code": "A00", "label": "Cholera", "depth": 1},
            ],
            "edges": [
                {"source": "ROOT", "target": "A00", "edge_type": "hierarchy"},
            ],
        }
        event = StateSnapshotEvent(snapshot=snapshot)
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "STATE_SNAPSHOT"
        assert "snapshot" in data
        assert "state" not in data
        assert len(data["snapshot"]["nodes"]) == 2

    def test_state_delta_uses_delta_field(self):
        """StateDeltaEvent must use 'delta' field with JSON Patch ops."""
        delta = [
            {"op": "add", "path": "/nodes/-", "value": {"id": "A01", "code": "A01"}},
            {"op": "add", "path": "/edges/-", "value": {"source": "ROOT", "target": "A01"}},
        ]
        event = StateDeltaEvent(delta=delta)
        encoded = encoder.encode(event)
        data = json.loads(encoded.removeprefix("data: ").rstrip())

        assert data["type"] == "STATE_DELTA"
        assert "delta" in data
        assert len(data["delta"]) == 2
        assert data["delta"][0]["op"] == "add"


class TestRunAgentInput:
    """Test RunAgentInput acceptance for streaming endpoints."""

    def test_run_agent_input_with_domain_state(self):
        """RunAgentInput.state carries domain config (clinical_note, provider, etc.)."""
        run_input = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            state={
                "clinical_note": "Patient presents with fever",
                "provider": "openai",
                "model": "gpt-4o",
                "scaffolded": True,
            },
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )
        assert run_input.thread_id == "thread-123"
        assert run_input.run_id == "run-456"
        assert run_input.state["clinical_note"] == "Patient presents with fever"
        assert run_input.state["provider"] == "openai"

    def test_run_agent_input_serializes_camel_case(self):
        """RunAgentInput serialization uses camelCase (threadId, runId)."""
        run_input = RunAgentInput(
            thread_id="t-1",
            run_id="r-1",
            state={"clinical_note": "test"},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )
        data = json.loads(run_input.model_dump_json(by_alias=True, exclude_none=True))
        assert "threadId" in data
        assert "runId" in data
        assert "forwardedProps" in data


class TestEventTypeEnum:
    """Test that ag_ui.core.EventType has all required event types."""

    def test_run_lifecycle_events_exist(self):
        assert EventType.RUN_STARTED.value == "RUN_STARTED"
        assert EventType.RUN_FINISHED.value == "RUN_FINISHED"
        assert EventType.RUN_ERROR.value == "RUN_ERROR"

    def test_step_lifecycle_events_exist(self):
        assert EventType.STEP_STARTED.value == "STEP_STARTED"
        assert EventType.STEP_FINISHED.value == "STEP_FINISHED"

    def test_state_events_exist(self):
        assert EventType.STATE_SNAPSHOT.value == "STATE_SNAPSHOT"
        assert EventType.STATE_DELTA.value == "STATE_DELTA"

    def test_reasoning_events_exist(self):
        assert EventType.REASONING_START.value == "REASONING_START"
        assert EventType.REASONING_MESSAGE_START.value == "REASONING_MESSAGE_START"
        assert EventType.REASONING_MESSAGE_CONTENT.value == "REASONING_MESSAGE_CONTENT"
        assert EventType.REASONING_MESSAGE_END.value == "REASONING_MESSAGE_END"
        assert EventType.REASONING_END.value == "REASONING_END"

    def test_custom_event_type_exists(self):
        assert EventType.CUSTOM.value == "CUSTOM"


class TestProtocolCompliance:
    """End-to-end compliance tests for the full event stream pattern."""

    def test_full_traversal_event_sequence(self):
        """Verify the event sequence for a live traversal is protocol-compliant."""
        events = [
            RunStartedEvent(thread_id="t", run_id="r"),
            CustomEvent(name="run_metadata", value={"cached": False}),
            StateSnapshotEvent(snapshot={"nodes": [{"id": "ROOT"}], "edges": []}),
            # Per-step sequence: STEP_STARTED → STATE_DELTA → REASONING → STEP_FINISHED → CUSTOM
            StepStartedEvent(step_name="ROOT|children"),
            StateDeltaEvent(delta=[{"op": "add", "path": "/nodes/-", "value": {"id": "A00"}}]),
            ReasoningStartEvent(message_id="msg-1"),
            ReasoningMessageStartEvent(message_id="msg-1", role="assistant"),
            ReasoningMessageContentEvent(message_id="msg-1", delta="Selected A00 based on clinical note"),
            ReasoningMessageEndEvent(message_id="msg-1"),
            ReasoningEndEvent(message_id="msg-1"),
            StepFinishedEvent(step_name="ROOT|children"),
            CustomEvent(name="step_metadata", value={"node_id": "ROOT", "selected_ids": ["A00"]}),
            RunFinishedEvent(thread_id="t", run_id="r", result={"final_nodes": ["A00"]}),
        ]

        for event in events:
            encoded = encoder.encode(event)
            assert encoded.startswith("data: ")
            assert encoded.endswith("\n\n")
            data = json.loads(encoded.removeprefix("data: ").rstrip())
            assert "type" in data

    def test_cached_replay_event_sequence(self):
        """Cached replay uses STATE_SNAPSHOT pattern: no per-step events."""
        events = [
            RunStartedEvent(thread_id="t", run_id="r"),
            CustomEvent(name="run_metadata", value={"cached": True}),
            StateSnapshotEvent(snapshot={"nodes": [], "edges": []}),
            RunFinishedEvent(thread_id="t", run_id="r", result={
                "final_nodes": ["A00"], "decisions": [],
            }),
        ]

        encoded_all = [encoder.encode(e) for e in events]
        assert len(encoded_all) == 4
        # Verify each is valid SSE
        for enc in encoded_all:
            assert enc.startswith("data: ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
