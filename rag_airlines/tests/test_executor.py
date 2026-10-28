import asyncio
import threading
import time
from pydantic import BaseModel, ConfigDict
import pytest
from agent.route_types import ToolCall, ToolPlan
from agent.tools.base import Tool, ToolRegistry
from agent.executor import ExecutionContext, execute_plan


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "ok"


def test_stable_slots_and_references_across_dependency_waves():
    registry = ToolRegistry().register(Tool("flight_status", Args, lambda value: {"value": value}))
    plan = ToolPlan(
        risk_category="safe",
        calls=[
            ToolCall(tool="flight_status", args={"value": "root"}),
            ToolCall(tool="flight_status", args={"value": "$0.data.value"}, depends_on=[0]),
            ToolCall(tool="flight_status", args={"value": "independent"}),
            ToolCall(tool="flight_status", args={"value": "$1.data.value"}, depends_on=[1]),
        ],
    )
    results = asyncio.run(execute_plan(plan, registry))
    assert [r.data["value"] for r in results] == ["root", "root", "independent", "root"]
    assert [r.call_id for r in results] == [0, 1, 2, 3]


def test_independent_calls_really_run_concurrently():
    barrier = threading.Barrier(2)

    def handler(value):
        barrier.wait(timeout=2)
        return {"value": value}

    registry = ToolRegistry().register(Tool("flight_status", Args, handler))
    results = asyncio.run(
        execute_plan(
            ToolPlan(
                risk_category="safe", calls=[ToolCall(tool="flight_status"), ToolCall(tool="flight_status")]
            ),
            registry,
        )
    )
    assert all(result.ok for result in results)


def test_auth_confirmation_schema_and_failed_dependencies():
    executed = []
    registry = ToolRegistry().register(
        Tool(
            "refund_calc",
            Args,
            lambda value: executed.append(value) or {"value": value},
            requires_auth=True,
            requires_confirmation=True,
        )
    )
    plan = ToolPlan(risk_category="safe", calls=[ToolCall(tool="refund_calc")])
    assert not asyncio.run(execute_plan(plan, registry, authed=True))[0].ok
    assert not asyncio.run(execute_plan(plan, registry, context=ExecutionContext(customer_id="demo")))[0].ok
    assert not executed
    result = asyncio.run(
        execute_plan(
            plan,
            registry,
            context=ExecutionContext(customer_id="demo", confirmed_tools=frozenset({"refund_calc"})),
        )
    )
    assert result[0].ok
    registry = ToolRegistry().register(Tool("flight_status", Args, lambda **kwargs: {"value": "ok"}))
    bad = ToolPlan(
        risk_category="safe",
        calls=[
            ToolCall(tool="flight_status", args={"unknown": "bad"}),
            ToolCall(tool="flight_status", depends_on=[0]),
        ],
    )
    results = asyncio.run(execute_plan(bad, registry))
    assert all(not result.ok for result in results)
    assert results[1].error == "dependency failed"


def test_unknown_tool_timeout_and_undeclared_reference_fail_closed():
    missing = asyncio.run(
        execute_plan(ToolPlan(risk_category="safe", calls=[ToolCall(tool="flight_status")]), ToolRegistry())
    )
    assert not missing[0].ok
    slow = ToolRegistry().register(
        Tool("flight_status", Args, lambda value: time.sleep(0.05) or {}, timeout=0.001)
    )
    assert (
        asyncio.run(
            execute_plan(ToolPlan(risk_category="safe", calls=[ToolCall(tool="flight_status")]), slow)
        )[0].error
        == "tool timeout"
    )
    normal = ToolRegistry().register(Tool("flight_status", Args, lambda value: {"value": value}))
    plan = ToolPlan(
        risk_category="safe",
        calls=[
            ToolCall(tool="flight_status"),
            ToolCall(tool="flight_status", args={"value": "$0.data.value"}),
        ],
    )
    assert not asyncio.run(execute_plan(plan, normal))[1].ok


def test_cycles_forward_refs_and_budget_rejected():
    for calls in [[ToolCall(tool="flight_status", depends_on=[0])], [ToolCall(tool="flight_status")] * 5]:
        with pytest.raises(ValueError):
            ToolPlan(risk_category="safe", calls=calls)


def test_request_context_propagates_to_tool_worker_threads():
    from observability.logging import request_id

    token = request_id.set("correlated-request")
    try:
        registry = ToolRegistry().register(
            Tool("flight_status", Args, lambda value: {"request_id": request_id.get()})
        )
        result = asyncio.run(
            execute_plan(ToolPlan(risk_category="safe", calls=[ToolCall(tool="flight_status")]), registry)
        )
        assert result[0].data["request_id"] == "correlated-request"
    finally:
        request_id.reset(token)
