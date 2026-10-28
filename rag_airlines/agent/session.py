"""The shared CLI/API pipeline: bounded plan -> execute -> verify -> synthesize."""

import asyncio
import logging
import re
import threading
import time
import uuid
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
from dataclasses import dataclass, field
from typing import Any
from agent.answers import GroundedAnswer, generate_answer, deserialize_results
from agent.answers import missing_policy_reason
from agent.route_types import ToolCall, ToolPlan
from agent.tools import ToolRegistry, RetrievePolicyTool, FlightStatusTool, MetadataTool, customer_tools
from agent.verifier import Verifier
from context.memory.answer_cache import (
    get_cached_answer,
    set_cached_answer,
)
from context.memory.semantic_answer_cache import SemanticAnswerCache
from observability.logging import request_id, sanitize_for_log
from observability.observability import current_trace_metadata

log = logging.getLogger("aeronova.session")


@dataclass
class ChatSession:
    agent: Any = None
    registry: Any = None
    semantic_cache: Any = None
    refund_agent: Any = None
    synthesizer: Any = generate_answer
    cache_get: Any = get_cached_answer
    cache_set: Any = set_cached_answer
    _init_lock: Any = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def build(cls, refund_agent=None):
        return cls(refund_agent=refund_agent)

    def _initialize(self):
        with self._init_lock:
            self._initialize_resources()

    def _initialize_resources(self):
        if self.agent is None:
            from agent.factory import create_agent

            self.agent = create_agent()
        if self.registry is None:
            self.registry = (
                ToolRegistry()
                .register(RetrievePolicyTool(self.agent.chunks, self.agent.vector_store))
                .register(MetadataTool(self.agent.chunks))
                .register(FlightStatusTool())
            )
            if self.refund_agent:
                for tool in customer_tools(self.refund_agent):
                    self.registry.register(tool)
        if self.semantic_cache is None:
            self.semantic_cache = SemanticAnswerCache()

    @tracer.start_as_current_span("aeronova.request", record_exception=False, set_status_on_exception=False)
    def ask(self, question, *, emit=None, cancelled=None):
        identity = uuid.uuid4().hex
        token = request_id.set(identity)
        trace.get_current_span().set_attribute("aeronova.request_id", identity)
        trace.get_current_span().set_attribute("request_id", identity)
        started = time.perf_counter()
        events = []

        def event(name, **data):
            payload = sanitize_for_log(
                {
                    "event": name,
                    "request_id": identity,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                    **data,
                }
            )
            events.append(payload)
            if name not in {"token", "result", "sources"}:
                log.info(name, extra={"details": payload})
            if emit:
                emit(payload)

        def respond(
            answer, kind="answer", cache="bypass", sources=None, private_data=None, original_execution=None
        ):
            answer = self._deduplicate_answer_paragraphs(answer)
            out = {
                "answer": answer,
                "kind": kind,
                "answerable": kind in {"answer", "tool_result"},
                "refusal_reason": answer if kind not in {"answer", "tool_result"} else None,
                "cache": cache,
                "sources": sources or [],
                "request_id": identity,
                **current_trace_metadata(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            check = next(
                (
                    item
                    for item in reversed(events)
                    if item["event"] == "verify" and item.get("stage") in {"cache", "synthesis", "final"}
                ),
                None,
            )
            if check is not None:
                out["verification"] = {
                    "passed": check["passed"],
                    "citation_errors": check.get("citation_errors", 0),
                    "measurement_errors": check.get("measurement_errors", 0),
                    "safety_errors": check.get("safety_errors", 0),
                    "format_errors": check.get("format_errors", 0),
                }
            if original_execution is not None:
                out["original_execution"] = original_execution
            if private_data is not None:
                out["private_data"] = private_data
            # Never publish private tool facts in shared traces/token events.
            event("sources", sources=out["sources"])
            if private_data is None:
                for word in answer.split(" "):
                    event("token", text=word + " ")
            event("result", result={key: value for key, value in out.items() if key != "private_data"})
            out["trace"] = events.copy()
            return out

        try:
            from agent.graph import build_request_graph

            graph = build_request_graph(
                self, event=event, respond=respond, events=events, cancelled=cancelled
            )
            state = graph.invoke({"question": question}, config={"recursion_limit": 16})
            return state["result"]
        except Exception as error:
            log.warning("request_failed", extra={"details": {"type": type(error).__name__}})
            event("error", message="A required service is unavailable.")
            return respond(
                "A required service is unavailable. Please retry or check the local service configuration.",
                "error",
            )
        finally:
            request_id.reset(token)

    def _private_plan(self, question):
        claim = re.search(r"\bREF-DEMO-\d{3}\b", question, re.I)
        pnr = re.search(r"\bAN\d{4}K\b", question, re.I)
        if any(
            word in question.casefold()
            for word in ["ignore", "forge", "hide", "hack", "bypass", "another customer"]
        ):
            return None
        if claim and re.search(r"\b(check|status|claim)\b", question, re.I):
            return ToolPlan(
                risk_category="safe",
                request_scope="customer_specific",
                calls=[ToolCall(tool="claim_check", args={"claim_id": claim[0].upper()})],
            )
        if pnr and re.search(r"\b(booking|itinerary)\b", question, re.I):
            return ToolPlan(
                risk_category="safe",
                request_scope="customer_specific",
                calls=[ToolCall(tool="booking_lookup", args={"pnr": pnr[0].upper()})],
            )
        if re.fullmatch(r"(?:list|show)(?: my)? claims[.!?]*", question, re.I):
            return ToolPlan(
                risk_category="safe", request_scope="customer_specific", calls=[ToolCall(tool="list_claims")]
            )
        return None

    @staticmethod
    def _deduplicate_answer_paragraphs(answer):
        """Collapse repeated full paragraphs before UI/stream serialization."""
        paragraphs = (answer or "").strip().split("\n\n")
        unique = []
        seen = set()
        for paragraph in paragraphs:
            normalized = " ".join(paragraph.casefold().split())
            if normalized and normalized in seen:
                continue
            if normalized:
                seen.add(normalized)
                unique.append(paragraph.strip())
        return "\n\n".join(unique)

    def _cached(self, payload, question=""):
        try:
            response = GroundedAnswer.model_validate(payload["response"])
            verifier = Verifier(deserialize_results(payload["results"]))
            if not response.answerable:
                response = response.model_copy(
                    update={
                        "answer": "",
                        "refusal_reason": missing_policy_reason(
                            question, verifier.results, response.refusal_reason
                        ),
                    }
                )
            if not verifier.check(response).passed:
                return None
            return (
                response.answer if response.answerable else response.refusal_reason or "No supported answer.",
                "answer" if response.answerable else "refusal",
                self._sources(response, verifier),
            )
        except Exception:
            return None

    @staticmethod
    def _sources(response, verifier):
        if not response.answerable:
            return []
        cited = set(response.citation_chunk_ids)
        return [
            {
                **{
                    key: d.metadata.get(key)
                    for key in ["document_id", "source_file", "page", "authority", "chunk_id"]
                },
                "excerpt": d.page_content,
            }
            for d, _ in verifier.results
            if d.metadata.get("chunk_id") in cited
        ]

    async def astream(self, question):
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()

        def emit(event):
            if not cancelled.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, event)

        def worker():
            try:
                self.ask(question, emit=emit, cancelled=cancelled)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, worker)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            cancelled.set()
            # Calls already executing in worker threads cannot be force-killed;
            # tools are read-only and synthesis/provider timeouts bound completion.
            task.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
