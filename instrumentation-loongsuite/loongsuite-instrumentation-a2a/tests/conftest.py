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

"""Test fixtures for A2A instrumentation tests.

Sets up mock A2A SDK modules in sys.modules so that the instrumentor can
wrap them without requiring the real a2a-sdk to be installed.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock A2A SDK module hierarchy
# ---------------------------------------------------------------------------


class MockAgentCard:
    """Mock of a2a.types.AgentCard."""

    def __init__(
        self,
        name="test-agent",
        description="A test agent",
        version="1.0.0",
        url="http://localhost:8080",
    ):
        self.name = name
        self.description = description
        self.version = version
        self.url = url
        self.capabilities = MagicMock(streaming=True)


class MockClientEvent:
    """Mock of a2a.client.client.ClientEvent."""

    def __init__(self, data=None):
        self.data = data


class MockRole:
    """Mock of a2a.types.Role enum."""

    def __init__(self, value="user"):
        self.value = value


class MockMessage:
    """Mock of a2a.types.Message."""

    def __init__(
        self,
        message_id="msg-001",
        task_id=None,
        context_id=None,
        role=None,
    ):
        self.message_id = message_id
        self.task_id = task_id
        self.context_id = context_id
        self.role = role or MockRole("user")
        self.parts = []


class MockMessageSendParams:
    """Mock of a2a.types.MessageSendParams."""

    def __init__(self, message=None, configuration=None):
        self.message = message or MockMessage()
        self.configuration = configuration
        self.metadata = None


class MockBaseClient:
    """Mock of a2a.client.base_client.BaseClient.

    Args:
        card: AgentCard instance, or None for no-card tests.
        error_after: If set, raise this exception after the first yield
            in send_message.
    """

    def __init__(self, card=None, error_after=None):
        self._card = card
        self._error_after = error_after

    async def send_message(self, request, **kwargs):
        yield MockClientEvent(data="event_1")
        if self._error_after is not None:
            raise self._error_after
        yield MockClientEvent(data="event_2")


class MockDefaultRequestHandler:
    """Mock of DefaultRequestHandler.

    Args:
        error: If set, raise this exception during on_message_send /
            on_message_send_stream.
    """

    def __init__(self, error=None):
        self._error = error

    async def on_message_send(self, params, context=None):
        if self._error is not None:
            raise self._error
        return MagicMock(name="task_result")

    async def on_message_send_stream(self, params, context=None):
        yield MagicMock(name="stream_event_1")
        if self._error is not None:
            raise self._error
        yield MagicMock(name="stream_event_2")


def _register_mock_modules():
    """Register mock A2A SDK modules in sys.modules."""
    created = {}

    def _make(name, parent=None):
        mod = types.ModuleType(name)
        created[name] = mod
        sys.modules[name] = mod
        if parent is not None:
            attr = name.rsplit(".", 1)[-1]
            setattr(parent, attr, mod)
        return mod

    a2a_mod = _make("a2a")
    a2a_client = _make("a2a.client", a2a_mod)
    a2a_base_client = _make("a2a.client.base_client", a2a_client)
    _make("a2a.client.client", a2a_client)
    a2a_server = _make("a2a.server", a2a_mod)
    a2a_rh = _make("a2a.server.request_handlers", a2a_server)
    a2a_drh = _make(
        "a2a.server.request_handlers.default_request_handler", a2a_rh
    )
    a2a_types = _make("a2a.types", a2a_mod)

    a2a_base_client.BaseClient = MockBaseClient
    a2a_drh.DefaultRequestHandler = MockDefaultRequestHandler
    a2a_types.AgentCard = MockAgentCard

    return created


_mock_modules = _register_mock_modules()

# Now safe to import the instrumentor (after mock modules are registered)
from opentelemetry.instrumentation.a2a import A2AInstrumentor  # noqa: E402
from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import (  # noqa: E402
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture(name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(name="metric_reader")
def fixture_metric_reader():
    return InMemoryMetricReader()


@pytest.fixture(name="meter_provider")
def fixture_meter_provider(metric_reader):
    return MeterProvider(metric_readers=[metric_reader])


@pytest.fixture()
def instrument(tracer_provider, meter_provider):
    instrumentor = A2AInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    yield instrumentor
    instrumentor.uninstrument()
