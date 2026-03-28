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

"""
OpenTelemetry A2A SDK Instrumentation
=====================================

This package provides automatic instrumentation for the A2A (Agent-to-Agent)
Python SDK, capturing telemetry data for agent-to-agent communication.

Usage
-----

Basic instrumentation::

    from opentelemetry.instrumentation.a2a import A2AInstrumentor

    A2AInstrumentor().instrument()

    from a2a.client import A2AClient

    async with A2AClient(agent_card=card) as client:
        async for event in client.send_message(message):
            print(event)

The instrumentation automatically captures:

- Client-side invoke_agent spans (BaseClient.send_message)
- Server-side invoke_agent spans (on_message_send, on_message_send_stream)
- Agent card metadata (name, description, version, URL)
- Message metadata (message ID, task ID, conversation ID, role)
"""

import logging
import time
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import context, metrics, trace
from opentelemetry.instrumentation.a2a.package import _instruments
from opentelemetry.instrumentation.a2a.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

_TRACER_NAME = "opentelemetry.instrumentation.a2a"

_CLIENT_MODULE = "a2a.client.base_client"
_SERVER_HANDLER_MODULE = "a2a.server.request_handlers.default_request_handler"


def _get_agent_attrs(card):
    """Extract span attributes from an AgentCard."""
    attrs = {}
    name = getattr(card, "name", None)
    if name:
        attrs["gen_ai.agent.name"] = name
    desc = getattr(card, "description", None)
    if desc:
        attrs["gen_ai.agent.description"] = desc
    version = getattr(card, "version", None)
    if version:
        attrs["gen_ai.agent.version"] = version
    url = getattr(card, "url", None)
    if url:
        attrs["server.address"] = str(url)
    return attrs


def _get_message_attrs(message):
    """Extract span attributes from a Message object."""
    attrs = {}
    msg_id = getattr(message, "message_id", None)
    if msg_id:
        attrs["a2a.message.id"] = msg_id
    task_id = getattr(message, "task_id", None)
    if task_id:
        attrs["a2a.task.id"] = task_id
    # context_id maps to gen_ai.conversation.id (semconv standard attribute
    # for grouping related interactions — matches A2A's context_id semantics)
    context_id = getattr(message, "context_id", None)
    if context_id:
        attrs["gen_ai.conversation.id"] = context_id
    role = getattr(message, "role", None)
    if role:
        attrs["a2a.message.role"] = (
            str(role.value) if hasattr(role, "value") else str(role)
        )
    return attrs


def _get_params_attrs(params):
    """Extract span attributes from MessageSendParams."""
    message = getattr(params, "message", None)
    if message:
        return _get_message_attrs(message)
    return {}


class _SendMessageWrapper:
    """Wraps BaseClient.send_message (async generator) with a CLIENT span.

    The span stays open until the async iterator is fully consumed, ensuring
    all streamed events are children of the invoke_agent span.
    """

    def __init__(self, tracer, duration_histogram):
        self._tracer = tracer
        self._duration_histogram = duration_histogram

    async def __call__(self, wrapped, instance, args, kwargs):
        card = getattr(instance, "_card", None)
        agent_name = getattr(card, "name", "unknown") if card else "unknown"

        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": "a2a",
        }
        if card:
            attributes.update(_get_agent_attrs(card))

        # Extract message metadata from the first positional arg (request: Message)
        request = args[0] if args else kwargs.get("request")
        if request:
            attributes.update(_get_message_attrs(request))

        span = self._tracer.start_span(
            name=f"invoke_agent {agent_name}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        )
        ctx = trace.set_span_in_context(span)
        token = context.attach(ctx)
        start_time = time.monotonic()
        try:
            async for item in wrapped(*args, **kwargs):
                yield item
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            duration = time.monotonic() - start_time
            self._duration_histogram.record(
                duration,
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.system": "a2a",
                },
            )
            span.end()
            context.detach(token)


class _OnMessageSendWrapper:
    """Wraps DefaultRequestHandler.on_message_send with a SERVER span."""

    def __init__(self, tracer, duration_histogram):
        self._tracer = tracer
        self._duration_histogram = duration_histogram

    async def __call__(self, wrapped, instance, args, kwargs):
        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": "a2a",
        }

        params = args[0] if args else kwargs.get("params")
        if params:
            attributes.update(_get_params_attrs(params))

        start_time = time.monotonic()
        with self._tracer.start_as_current_span(
            name="invoke_agent",
            kind=SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                duration = time.monotonic() - start_time
                self._duration_histogram.record(
                    duration,
                    attributes={
                        "gen_ai.operation.name": "invoke_agent",
                        "gen_ai.system": "a2a",
                    },
                )


class _OnMessageSendStreamWrapper:
    """Wraps DefaultRequestHandler.on_message_send_stream (async generator)
    with a SERVER span."""

    def __init__(self, tracer, duration_histogram):
        self._tracer = tracer
        self._duration_histogram = duration_histogram

    async def __call__(self, wrapped, instance, args, kwargs):
        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system": "a2a",
        }

        params = args[0] if args else kwargs.get("params")
        if params:
            attributes.update(_get_params_attrs(params))

        span = self._tracer.start_span(
            name="invoke_agent",
            kind=SpanKind.SERVER,
            attributes=attributes,
        )
        ctx = trace.set_span_in_context(span)
        token = context.attach(ctx)
        start_time = time.monotonic()
        try:
            async for item in wrapped(*args, **kwargs):
                yield item
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            duration = time.monotonic() - start_time
            self._duration_histogram.record(
                duration,
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.system": "a2a",
                },
            )
            span.end()
            context.detach(token)


class A2AInstrumentor(BaseInstrumentor):
    """Instrumentor for the A2A (Agent-to-Agent) Python SDK.

    Patches ``BaseClient.send_message`` (client-side) and
    ``DefaultRequestHandler.on_message_send`` /
    ``on_message_send_stream`` (server-side) to emit OpenTelemetry
    spans with GenAI semantic convention attributes.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")

        tracer = trace.get_tracer(
            _TRACER_NAME,
            __version__,
            tracer_provider=tracer_provider,
        )

        meter = metrics.get_meter(
            _TRACER_NAME,
            __version__,
            meter_provider=meter_provider,
        )
        duration_histogram = meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI operations",
        )

        try:
            wrap_function_wrapper(
                module=_CLIENT_MODULE,
                name="BaseClient.send_message",
                wrapper=_SendMessageWrapper(tracer, duration_histogram),
            )
        except Exception as e:
            logger.warning(
                "Failed to instrument BaseClient.send_message: %s", e
            )

        try:
            wrap_function_wrapper(
                module=_SERVER_HANDLER_MODULE,
                name="DefaultRequestHandler.on_message_send",
                wrapper=_OnMessageSendWrapper(tracer, duration_histogram),
            )
        except Exception as e:
            logger.warning(
                "Failed to instrument "
                "DefaultRequestHandler.on_message_send: %s",
                e,
            )

        try:
            wrap_function_wrapper(
                module=_SERVER_HANDLER_MODULE,
                name="DefaultRequestHandler.on_message_send_stream",
                wrapper=_OnMessageSendStreamWrapper(
                    tracer, duration_histogram
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to instrument "
                "DefaultRequestHandler.on_message_send_stream: %s",
                e,
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import a2a.client.base_client as _base_client  # noqa: PLC0415

            unwrap(_base_client.BaseClient, "send_message")
        except Exception as e:
            logger.debug("Failed to unwrap BaseClient.send_message: %s", e)

        try:
            import a2a.server.request_handlers.default_request_handler as _handler  # noqa: PLC0415

            unwrap(_handler.DefaultRequestHandler, "on_message_send")
        except Exception as e:
            logger.debug(
                "Failed to unwrap DefaultRequestHandler.on_message_send: %s",
                e,
            )

        try:
            import a2a.server.request_handlers.default_request_handler as _handler  # noqa: PLC0415

            unwrap(_handler.DefaultRequestHandler, "on_message_send_stream")
        except Exception as e:
            logger.debug(
                "Failed to unwrap "
                "DefaultRequestHandler.on_message_send_stream: %s",
                e,
            )


__all__ = [
    "__version__",
    "A2AInstrumentor",
]
