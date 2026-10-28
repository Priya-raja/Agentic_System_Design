from unittest.mock import MagicMock
import httpx
import pytest
from observability import observability as obs


@pytest.fixture(autouse=True)
def reset_tracing(monkeypatch):
    monkeypatch.setattr(obs, '_setup_attempted', False)
    monkeypatch.setattr(obs, '_tracing_initialized', False)
    monkeypatch.setattr(obs, 'PHOENIX_ENABLED', True)


def test_offline_is_nonfatal_and_only_probed_once(monkeypatch, caplog):
    probe = MagicMock(side_effect=httpx.ConnectError('private-url'))
    monkeypatch.setattr(obs.httpx, 'get', probe)
    assert obs.setup_tracing() is None
    assert obs.setup_tracing() is None
    assert probe.call_count == 1
    assert 'Phoenix not detected, tracing disabled.' in caplog.text
    assert 'private-url' not in caplog.text
    assert obs.current_trace_metadata() == {}


def test_register_failure_is_nonfatal(monkeypatch):
    import phoenix.otel
    monkeypatch.setattr(obs.httpx, 'get', MagicMock())
    monkeypatch.setattr(phoenix.otel, 'register', MagicMock(side_effect=RuntimeError()))
    assert obs.setup_tracing() is None
    assert not obs._tracing_initialized


def test_disabled_skips_network(monkeypatch):
    monkeypatch.setattr(obs, 'PHOENIX_ENABLED', False)
    probe = MagicMock()
    monkeypatch.setattr(obs.httpx, 'get', probe)
    assert obs.setup_tracing() is None
    probe.assert_not_called()


def test_successful_setup_and_current_trace_link(monkeypatch):
    import phoenix.otel
    from opentelemetry.sdk.trace import TracerProvider
    provider = TracerProvider()
    register = MagicMock(return_value=provider)
    monkeypatch.setattr(obs.httpx, 'get', MagicMock())
    monkeypatch.setattr(phoenix.otel, 'register', register)
    monkeypatch.setattr(obs, 'PHOENIX_PUBLIC_URL', 'http://localhost:6006')
    assert obs.setup_tracing() is provider
    assert obs.setup_tracing() is None
    register.assert_called_once()
    with provider.get_tracer('test').start_as_current_span('request') as span:
        metadata = obs.current_trace_metadata()
        assert metadata['trace_id'] == format(span.get_span_context().trace_id, '032x')
        assert metadata['phoenix_url'] == f"http://localhost:6006/redirects/traces/{metadata['trace_id']}"
    assert obs.current_trace_metadata() == {}
    provider.shutdown()


def test_parallel_tool_spans_share_root_and_propagate_into_threads(monkeypatch):
    import asyncio
    import threading
    from pydantic import BaseModel
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from agent import executor
    from agent.route_types import ToolPlan, ToolCall
    from agent.tools.base import Tool, ToolRegistry

    class Args(BaseModel):
        pass

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    monkeypatch.setattr(executor, 'tracer', tracer)
    barrier = threading.Barrier(2)
    thread_parents = []

    def retrieve():
        thread_parents.append(trace.get_current_span().get_span_context().span_id)
        barrier.wait(timeout=2)
        return {'chunks': []}

    registry = ToolRegistry().register(Tool('retrieve_policy', Args, retrieve))
    plan = ToolPlan(risk_category='safe', calls=[ToolCall(tool='retrieve_policy'), ToolCall(tool='retrieve_policy')])
    with tracer.start_as_current_span('aeronova.request') as root:
        assert all(result.ok for result in asyncio.run(executor.execute_plan(plan, registry)))
    lanes = [span for span in exporter.get_finished_spans() if span.name.startswith('aeronova.tool_')]
    assert len(lanes) == 2
    assert all(span.parent.span_id == root.get_span_context().span_id for span in lanes)
    assert set(thread_parents) == {span.context.span_id for span in lanes}
    assert max(span.start_time for span in lanes) < min(span.end_time for span in lanes)
    provider.shutdown()
