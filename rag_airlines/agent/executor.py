"""Topological batches with stable result slots, auth gates, refs and deadlines."""

import asyncio
import contextvars
import re
import functools
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
from concurrent.futures import ThreadPoolExecutor

_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="aeronova-tool")
from dataclasses import dataclass, field
from agent.tools.base import ToolResult


@dataclass(frozen=True)
class ExecutionContext:
    customer_id: str | None = None
    confirmed_tools: frozenset[str] = field(default_factory=frozenset)
    cancelled: object = None


def _resolve(value, call, results):
    if isinstance(value, dict):
        return {k: _resolve(v, call, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, call, results) for v in value]
    if isinstance(value, str) and value.startswith("$"):
        match = re.fullmatch(r"\$(\d+)\.data\.([a-zA-Z0-9_.]+)", value)
        if not match:
            raise ValueError("Invalid reference")
        index = int(match[1])
        if index not in call.depends_on or index not in results or not results[index].ok:
            raise ValueError("Reference must target a successful declared dependency")
        node = results[index].data
        for key in match[2].split("."):
            node = node[key]
        return node
    return value


async def execute_plan(plan, registry, *, context=None, authed=False, emit=None):
    context = context or ExecutionContext()
    results: dict[int, ToolResult] = {}
    pending = set(range(len(plan.calls)))

    def event(name, **data):
        if emit:
            emit(name, **data)

    async def run(index):
        with tracer.start_as_current_span(
            f"aeronova.tool_{index + 1}", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("tool.name", plan.calls[index].tool)
            span.set_attribute("tool.call_id", index)
            output = await run_tool(index)
            span.set_attribute("tool.ok", output[1].ok)
            return output

    async def run_tool(index):
        call = plan.calls[index]
        event("tool_start", call_id=index, tool=call.tool, depends_on=call.depends_on)
        if context.cancelled is not None and context.cancelled.is_set():
            result = ToolResult.failure(call.tool, "request cancelled", index)
        elif any(not results[dep].ok for dep in call.depends_on):
            result = ToolResult.failure(call.tool, "dependency failed", index)
        else:
            try:
                tool = registry.get(call.tool)
                if tool.requires_auth and not context.customer_id:
                    raise PermissionError("bound customer required")
                if tool.customer_id and tool.customer_id != context.customer_id:
                    raise PermissionError("customer binding mismatch")
                if tool.requires_confirmation and tool.name not in context.confirmed_tools:
                    raise PermissionError("explicit confirmation required")
                args = _resolve(call.args, call, results)
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        _pool, contextvars.copy_context().run, functools.partial(tool.run, **args)
                    ),
                    timeout=tool.timeout,
                )
                result.call_id = index
            except PermissionError as error:
                result = ToolResult.failure(call.tool, str(error), index)
            except TimeoutError:
                result = ToolResult.failure(call.tool, "tool timeout", index)
            except Exception as error:
                result = ToolResult.failure(call.tool, f"tool failed: {type(error).__name__}", index)
        event(
            "tool_end",
            call_id=index,
            tool=call.tool,
            ok=result.ok,
            error=result.error,
            **(
                {"chunk_count": len(result.data.get("chunks", []))}
                if call.tool == "retrieve_policy" and not result.private
                else {}
            ),
        )
        return index, result

    while pending:
        ready = sorted(i for i in pending if all(dep in results for dep in plan.calls[i].depends_on))
        if not ready:
            raise ValueError("Invalid dependency graph")
        for index, result in await asyncio.gather(*(run(i) for i in ready)):
            results[index] = result
        pending.difference_update(ready)
    return [results[i] for i in range(len(plan.calls))]
