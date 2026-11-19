
from threading import Event
from types import SimpleNamespace

import pytest

from agentops.agents.developer import DeveloperAgent
from agentops.tools.cancellation import TaskCancelledError


class TrackingStream:
    """A small mock stream that records whether it was closed."""

    def __init__(self, chunks, cancel_event=None):
        self.chunks = iter(chunks)
        self.cancel_event = cancel_event
        self.close_called = False

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.chunks)

        if self.cancel_event is not None:
            self.cancel_event.set()

        return chunk

    def close(self):
        self.close_called = True


def create_agent_with_stream(stream):
    """Create a DeveloperAgent without initializing a real LLM."""

    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.llm = SimpleNamespace(stream=lambda messages: stream)

    return agent


def test_stream_response_combines_model_chunks():
    stream = TrackingStream(
        [
            SimpleNamespace(content="Hello "),
            SimpleNamespace(content="from "),
            SimpleNamespace(content="AgentOps."),
        ]
    )

    agent = create_agent_with_stream(stream)

    result = agent._stream_response(
        messages=[],
    )

    assert result == "Hello from AgentOps."
    assert stream.close_called is True


def test_stream_response_returns_empty_string_for_empty_stream():
    stream = TrackingStream([])

    agent = create_agent_with_stream(stream)

    result = agent._stream_response(
        messages=[],
    )

    assert result == ""
    assert stream.close_called is True


def test_stream_response_ignores_none_content_chunks():
    stream = TrackingStream(
        [
            SimpleNamespace(content="First"),
            SimpleNamespace(content=None),
            SimpleNamespace(content="Second"),
        ]
    )

    agent = create_agent_with_stream(stream)

    result = agent._stream_response(
        messages=[],
    )

    assert result == "FirstSecond"
    assert stream.close_called is True


def test_stream_response_rejects_unexpected_content_type():
    stream = TrackingStream(
        [
            SimpleNamespace(content="Valid"),
            SimpleNamespace(content={"unexpected": "data"}),
        ]
    )

    agent = create_agent_with_stream(stream)

    with pytest.raises(
        ValueError,
        match="unexpected response format",
    ):
        agent._stream_response(messages=[])

    assert stream.close_called is True


def test_stream_response_stops_when_cancelled_before_streaming():
    stream = TrackingStream(
        [
            SimpleNamespace(content="This should not be consumed."),
        ]
    )

    agent = create_agent_with_stream(stream)

    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        agent._stream_response(
            messages=[],
            cancel_event=cancel_event,
        )

    assert stream.close_called is False


def test_stream_response_closes_stream_when_cancelled_during_generation():
    cancel_event = Event()

    stream = TrackingStream(
        [
            SimpleNamespace(content="Generation started."),
            SimpleNamespace(content="This chunk should not be processed."),
        ],
        cancel_event=cancel_event,
    )

    agent = create_agent_with_stream(stream)

    with pytest.raises(TaskCancelledError):
        agent._stream_response(
            messages=[],
            cancel_event=cancel_event,
        )

    assert stream.close_called is True


def test_stream_response_closes_stream_when_iteration_raises():
    class FailingStream:
        def __init__(self):
            self.close_called = False

        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("Simulated model stream failure.")

        def close(self):
            self.close_called = True

    stream = FailingStream()
    agent = create_agent_with_stream(stream)

    with pytest.raises(
        RuntimeError,
        match="Simulated model stream failure",
    ):
        agent._stream_response(messages=[])

    assert stream.close_called is True
