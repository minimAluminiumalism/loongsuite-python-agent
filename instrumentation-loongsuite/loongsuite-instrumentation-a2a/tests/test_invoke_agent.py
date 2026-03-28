# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for A2A SDK instrumentation."""

import pytest
from conftest import (
    MockAgentCard,
    MockBaseClient,
    MockDefaultRequestHandler,
    MockMessage,
    MockMessageSendParams,
)

from opentelemetry.trace import SpanKind, StatusCode

# ---------------------------------------------------------------------------
# Client-side: BaseClient.send_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_message_creates_span(instrument, span_exporter):
    """send_message should create a CLIENT span named 'invoke_agent {name}'."""
    card = MockAgentCard(name="my-agent")
    client = MockBaseClient(card=card)

    events = []
    async for event in client.send_message(MockMessage()):
        events.append(event)

    assert len(events) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "invoke_agent my-agent"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_send_message_genai_attributes(instrument, span_exporter):
    """send_message span should carry GenAI semantic convention attributes."""
    card = MockAgentCard(
        name="agent-x",
        description="Does X things",
        version="2.0.0",
        url="http://remote:9000",
    )
    client = MockBaseClient(card=card)
    message = MockMessage(
        message_id="msg-123",
        task_id="task-456",
        context_id="ctx-789",
    )

    async for _ in client.send_message(message):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.system"] == "a2a"
    assert attrs["gen_ai.agent.name"] == "agent-x"
    assert attrs["gen_ai.agent.description"] == "Does X things"
    assert attrs["gen_ai.agent.version"] == "2.0.0"
    assert attrs["server.address"] == "http://remote:9000"
    assert attrs["a2a.message.id"] == "msg-123"
    assert attrs["a2a.task.id"] == "task-456"
    assert attrs["gen_ai.conversation.id"] == "ctx-789"
    assert attrs["a2a.message.role"] == "user"


@pytest.mark.asyncio
async def test_send_message_missing_card(instrument, span_exporter):
    """send_message should handle a missing AgentCard gracefully."""
    client = MockBaseClient(card=None)

    async for _ in client.send_message(MockMessage()):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "invoke_agent unknown"
    assert spans[0].attributes.get("gen_ai.operation.name") == "invoke_agent"
    assert spans[0].attributes.get("gen_ai.system") == "a2a"
    # Agent-specific attrs should not be present when card is None
    assert "gen_ai.agent.name" not in spans[0].attributes


@pytest.mark.asyncio
async def test_send_message_partial_card(instrument, span_exporter):
    """send_message should handle an AgentCard with only some fields set."""
    card = MockAgentCard(name="partial")
    card.description = None
    card.version = None
    card.url = None
    client = MockBaseClient(card=card)

    async for _ in client.send_message(MockMessage()):
        pass

    spans = span_exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.agent.name"] == "partial"
    assert "gen_ai.agent.description" not in attrs
    assert "gen_ai.agent.version" not in attrs
    assert "server.address" not in attrs


@pytest.mark.asyncio
async def test_send_message_error_handling(instrument, span_exporter):
    """send_message span should record exceptions and set ERROR status."""
    card = MockAgentCard(name="fail-agent")
    client = MockBaseClient(
        card=card, error_after=RuntimeError("connection lost")
    )

    with pytest.raises(RuntimeError, match="connection lost"):
        async for _ in client.send_message(MockMessage()):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "invoke_agent fail-agent"
    assert span.status.status_code == StatusCode.ERROR
    assert "connection lost" in span.status.description

    error_events = [e for e in span.events if e.name == "exception"]
    assert len(error_events) == 1
    assert error_events[0].attributes["exception.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_send_message_collects_all_events(instrument, span_exporter):
    """All yielded events should pass through the wrapper unchanged."""
    client = MockBaseClient(card=MockAgentCard(name="multi"))
    events = [e async for e in client.send_message(MockMessage())]

    assert len(events) == 2
    assert events[0].data == "event_1"
    assert events[1].data == "event_2"


# ---------------------------------------------------------------------------
# Server-side: DefaultRequestHandler.on_message_send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_message_send_creates_span(instrument, span_exporter):
    """on_message_send should create a SERVER span with message attributes."""
    handler = MockDefaultRequestHandler()
    params = MockMessageSendParams(
        message=MockMessage(message_id="srv-msg-001", task_id="srv-task-001")
    )
    result = await handler.on_message_send(params)

    assert result is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "invoke_agent"
    assert span.kind == SpanKind.SERVER
    assert span.status.status_code == StatusCode.OK
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.system"] == "a2a"
    assert span.attributes["a2a.message.id"] == "srv-msg-001"
    assert span.attributes["a2a.task.id"] == "srv-task-001"


@pytest.mark.asyncio
async def test_on_message_send_error(instrument, span_exporter):
    """on_message_send should record errors."""
    handler = MockDefaultRequestHandler(error=ValueError("invalid params"))

    with pytest.raises(ValueError, match="invalid params"):
        await handler.on_message_send(MockMessageSendParams())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Server-side: DefaultRequestHandler.on_message_send_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_message_send_stream_creates_span(instrument, span_exporter):
    """on_message_send_stream should create a SERVER span with message attributes."""
    handler = MockDefaultRequestHandler()
    params = MockMessageSendParams(
        message=MockMessage(
            message_id="stream-msg-001", context_id="stream-ctx"
        )
    )

    events = []
    async for event in handler.on_message_send_stream(params):
        events.append(event)

    assert len(events) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "invoke_agent"
    assert span.kind == SpanKind.SERVER
    assert span.status.status_code == StatusCode.OK
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.system"] == "a2a"


@pytest.mark.asyncio
async def test_on_message_send_stream_error(instrument, span_exporter):
    """on_message_send_stream should record errors."""
    handler = MockDefaultRequestHandler(error=RuntimeError("stream error"))

    with pytest.raises(RuntimeError, match="stream error"):
        async for _ in handler.on_message_send_stream(MockMessageSendParams()):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Instrument / uninstrument lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uninstrument_removes_wrapping(tracer_provider, span_exporter):
    """After uninstrument(), no spans should be created."""
    from opentelemetry.instrumentation.a2a import (  # noqa: PLC0415
        A2AInstrumentor,
    )

    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )

    # Verify instrumentation is active
    client = MockBaseClient(card=MockAgentCard(name="temp"))
    async for _ in client.send_message(MockMessage()):
        pass
    assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()

    # Uninstrument
    instrumentor.uninstrument()

    # Verify no spans are created after uninstrumenting
    client2 = MockBaseClient(card=MockAgentCard(name="temp2"))
    async for _ in client2.send_message(MockMessage()):
        pass
    assert len(span_exporter.get_finished_spans()) == 0


# ---------------------------------------------------------------------------
# Metrics: gen_ai.client.operation.duration
# ---------------------------------------------------------------------------


def _get_duration_metrics(metric_reader):
    """Extract duration histogram data points from the metric reader."""
    data = metric_reader.get_metrics_data()
    points = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        points.append(dp)
    return points


@pytest.mark.asyncio
async def test_send_message_records_duration(
    instrument, span_exporter, metric_reader
):
    """send_message should record gen_ai.client.operation.duration."""
    client = MockBaseClient(card=MockAgentCard(name="metrics-agent"))
    async for _ in client.send_message(MockMessage()):
        pass

    points = _get_duration_metrics(metric_reader)
    assert len(points) == 1
    assert points[0].sum > 0
    attrs = dict(points[0].attributes)
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.system"] == "a2a"


@pytest.mark.asyncio
async def test_on_message_send_records_duration(
    instrument, span_exporter, metric_reader
):
    """on_message_send should record gen_ai.client.operation.duration."""
    handler = MockDefaultRequestHandler()
    await handler.on_message_send(MockMessageSendParams())

    points = _get_duration_metrics(metric_reader)
    assert len(points) == 1
    assert points[0].sum > 0


@pytest.mark.asyncio
async def test_on_message_send_stream_records_duration(
    instrument, span_exporter, metric_reader
):
    """on_message_send_stream should record gen_ai.client.operation.duration."""
    handler = MockDefaultRequestHandler()
    async for _ in handler.on_message_send_stream(MockMessageSendParams()):
        pass

    points = _get_duration_metrics(metric_reader)
    assert len(points) == 1
    assert points[0].sum > 0


@pytest.mark.asyncio
async def test_error_still_records_duration(
    instrument, span_exporter, metric_reader
):
    """Duration should be recorded even when the operation fails."""
    client = MockBaseClient(
        card=MockAgentCard(name="err"),
        error_after=RuntimeError("fail"),
    )
    with pytest.raises(RuntimeError):
        async for _ in client.send_message(MockMessage()):
            pass

    points = _get_duration_metrics(metric_reader)
    assert len(points) == 1
    assert points[0].sum > 0
