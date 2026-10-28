"""Linear, deterministic RAG stages; approvals use the separate refund graph.

Every early result ends the graph. Public retrieval is dispatched directly by
department, and only verified generated answers are published.
"""

import asyncio
from typing import Any, TypedDict
from langgraph.graph import StateGraph, START, END
from opentelemetry import trace
from agent.answers import (
    serialize_results,
    missing_policy_reason,
    is_ticket_refund_handoff,
    is_ticket_refund_email_request,
    TICKET_REFUND_GUIDANCE,
    TICKET_REFUND_EMAIL_DRAFT,
)
from agent.cascade import route_public_intent, needs_customer_workflow
from agent.execution_trace import record_execution, read_execution
from agent.executor import ExecutionContext, execute_plan
from agent.tools.base import ToolResult
from agent.tools import ToolRegistry, customer_tools
from agent.verifier import Verifier
from config import ANSWER_MODEL, NO_CONTEXT_CACHE_TTL_SECONDS
from context.memory.answer_cache import create_answer_cache_key, get_index_hash
from context.memory.semantic_answer_cache import semantic_cache_allowed
from safety.heuristics import assess_user_input, is_standalone_insult, BLOCK, REVIEW

tracer = trace.get_tracer(__name__)


class RequestState(TypedDict, total=False):
    question: str
    key: str
    handoff: bool
    department: str | None
    route_kind: str
    model_name: str
    metadata: dict
    semantic_allowed: bool
    outputs: list
    verifier: Any
    response: Any
    result: dict


def build_request_graph(session, *, event, respond, events, cancelled=None):
    # These callbacks are request-scoped and never placed in a durable checkpoint.
    self = session

    def safety(state: RequestState):
        question = state["question"]
        question = (question or "").strip()
        if not question or len(question) > 2000:
            return {"result": respond("Enter a question of up to 2,000 characters.", "clarify")}
        assessment = assess_user_input(question)
        event("safety", action=assessment.action, pii_types=assessment.pii.pii_types)
        if assessment.action in {BLOCK, REVIEW}:
            return {
                "result": respond(
                    "This request needs safety review. Please ask an airline question without unsafe instructions or links.",
                    "safety",
                )
            }
        if is_standalone_insult(question):
            return {
                "result": respond(
                    "I'm here to help. Tell me what went wrong or what you need help with.", "support"
                )
            }
        private_plan = self._private_plan(question) if self.refund_agent else None
        if assessment.pii.detected and private_plan is None:
            return {
                "result": respond(
                    "Please use the dedicated secure workflow for requests containing personal identifiers.",
                    "secure",
                )
            }
        if private_plan:
            private_registry = ToolRegistry()
            for tool in customer_tools(self.refund_agent):
                private_registry.register(tool)
            event("route", scope="customer_specific", classifier="local_bound_workflow")
            output = asyncio.run(
                execute_plan(
                    private_plan,
                    private_registry,
                    context=ExecutionContext(customer_id=self.refund_agent.customer_id, cancelled=cancelled),
                    emit=lambda name, **data: event(name, **data),
                )
            )
            if any((not item.ok for item in output)):
                return {
                    "result": respond("That record is unavailable for this bound demo customer.", "refusal")
                }
            return {
                "result": respond(
                    "The customer-scoped demo lookup is complete. No payment was sent.",
                    "tool_result",
                    private_data=[item.data for item in output],
                )
            }
        question = assessment.pii.redacted_text
        return {"question": question}

    def exact_cache(state: RequestState):
        question = state["question"]
        if is_ticket_refund_email_request(question):
            event("route", scope="customer_specific", department="refunds_claims", classifier="local_email_draft")
            return {"result": respond(TICKET_REFUND_EMAIL_DRAFT, "handoff", "bypass")}
        if is_ticket_refund_handoff(question):
            event("route", scope="customer_specific", department="refunds_claims", classifier="local_handoff")
            return {"result": respond(TICKET_REFUND_GUIDANCE, "handoff", "bypass")}
        self._initialize()
        key = create_answer_cache_key(question, self.agent.prompt.content_hash)
        cached = self.cache_get(key)
        event("cache", kind="exact", hit=cached is not None)
        if cached:
            if cached.get("cache_kind") == "no_context":
                return {
                    "result": respond(
                        cached["message"], "no_context", "exact", original_execution=read_execution(cached)
                    )
                }
            hit = self._cached(cached, question)
            if hit:
                event("verify", stage="cache", passed=True)
                return {
                    "result": respond(
                        hit[0], hit[1], "exact", hit[2], original_execution=read_execution(cached)
                    )
                }
        return {"key": key}

    def route(state: RequestState):
        question = state["question"]
        if needs_customer_workflow(question):
            event("route", classifier="deterministic", scope="customer_specific")
            return {
                "result": respond(
                    "This request needs a customer record or live flight data. Please use AeroNova's "
                    "authenticated flight or rewards service.",
                    "handoff",
                )
            }
        route = route_public_intent(question)
        classifier = "deterministic" if route else "unmatched"
        event(
            "route",
            classifier=classifier,
            scope="public_information" if route else "unknown",
            department=route or "",
        )
        if route is None:
            return {
                "result": respond(
                    "Please specify a supported airline policy area so I can search the right documents.", "clarify"
                )
            }
        route_kind = "metadata" if route == "inspect_metadata" else "policy"
        event("route_selected", destination=route_kind)
        return {
            "department": None if route_kind == "metadata" else route,
            "route_kind": route_kind,
            "model_name": ANSWER_MODEL,
        }

    def semantic_cache(state: RequestState):
        question = state["question"]
        key = state["key"]
        department = state.get("department")
        cacheable = state["route_kind"] == "policy" and department is not None
        metadata = dict(
            topic=department,
            policy_year=None,
            prompt_version=self.agent.prompt.version,
            prompt_hash=self.agent.prompt.content_hash,
            corpus_version=get_index_hash(),
        )
        from context.indexers.retrieve import extract_policy_year

        metadata["policy_year"] = extract_policy_year(question)
        semantic_allowed = cacheable and semantic_cache_allowed(question, metadata["topic"])
        if semantic_allowed:
            cached = self.semantic_cache.lookup(question=question, **metadata)
            event("cache", kind="semantic", hit=cached is not None)
            hit = self._cached(cached, question) if cached else None
            if hit:
                event("verify", stage="cache", passed=True)
                self.cache_set(key, cached)
                return {
                    "result": respond(
                        hit[0], hit[1], "semantic", hit[2], original_execution=read_execution(cached)
                    )
                }
        return {"metadata": metadata, "semantic_allowed": semantic_allowed}

    def retrieve(state: RequestState):
        outputs = []
        if cancelled is not None and cancelled.is_set():
            return {"result": respond("Request cancelled.", "cancelled")}
        department = state.get("department")
        tool_name = "inspect_metadata" if state["route_kind"] == "metadata" else "retrieve_policy"
        arguments = {"query": state["question"]}
        if department:
            arguments["department"] = department
        event("tool_start", tool=tool_name, call_id=0, depends_on=[])
        try:
            result = self.registry.get(tool_name).run(**arguments)
            result.call_id = 0
            outputs.append(result)
            event(
                "tool_end",
                tool=tool_name,
                call_id=0,
                ok=True,
                chunk_count=len(result.data.get("chunks", [])),
            )
        except Exception:
            outputs.append(ToolResult.failure(tool_name, "retrieval failed", call_id=0))
            event("tool_end", tool=tool_name, call_id=0, ok=False, chunk_count=0)
        if cancelled is not None and cancelled.is_set():
            return {"result": respond("Request cancelled.", "cancelled")}
        if any((not result.ok for result in outputs)):
            return {
                "result": respond(
                    "A required lookup failed; I cannot safely answer the complete request.", "tool_error"
                )
            }
        return {"outputs": outputs}

    def evidence(state: RequestState):
        question = state["question"]
        outputs = state["outputs"]
        metadata = state["metadata"]
        key = state["key"]
        if is_ticket_refund_handoff(question):
            return {"result": respond(TICKET_REFUND_GUIDANCE, "handoff", "miss")}
        verifier = Verifier(tool_results=outputs)
        if not verifier.results:
            message = missing_policy_reason(question, verifier.results)
            year = metadata["policy_year"]
            message = (
                f"No applicable policy was found for {year}." if year else "No relevant policy was found."
            )
            self.cache_set(
                key,
                {
                    "cache_kind": "no_context",
                    "message": message,
                    "original_execution": record_execution(events),
                },
                ttl_seconds=NO_CONTEXT_CACHE_TTL_SECONDS,
            )
            return {"result": respond(message, "no_context", "miss")}
        message = missing_policy_reason(question, verifier.results)
        evidence_errors = verifier.evidence_errors()
        event("verify", stage="evidence", passed=not evidence_errors)
        if evidence_errors:
            return {"result": respond("The available evidence cannot support a verified answer.", "refusal")}
        return {"verifier": verifier}

    def synthesize(state: RequestState):
        question = state["question"]
        verifier = state["verifier"]
        event("synth_start")
        with tracer.start_as_current_span(
            "aeronova.synthesize", record_exception=False, set_status_on_exception=False
        ):
            response = self.synthesizer(
                question, verifier.results, self.agent.prompt, state.get("model_name", ANSWER_MODEL)
            )
        if not response.answerable:
            response = response.model_copy(
                update={
                    "answer": "",
                    "refusal_reason": missing_policy_reason(
                        question, verifier.results, response.refusal_reason
                    ),
                }
            )
        return {"response": response}

    def verify(state: RequestState):
        verifier = state["verifier"]
        response = state["response"]
        for stage in ["synthesis", "final"]:
            report = verifier.check(response)
            event(
                "verify",
                stage=stage,
                passed=report.passed,
                citation_errors=len(report.citation_errors),
                measurement_errors=len(report.measurement_errors),
                safety_errors=int(report.safety_error is not None),
                format_errors=len(report.format_errors),
            )
            if not report.passed:
                return {
                    "result": respond(
                        "The generated answer failed verification; no unverified answer was released.",
                        "verification_failed",
                    )
                }
        if cancelled is not None and cancelled.is_set():
            return {"result": respond("Request cancelled.", "cancelled")}
        return {}

    def publish(state: RequestState):
        question = state["question"]
        key = state["key"]
        metadata = state["metadata"]
        semantic_allowed = state["semantic_allowed"]
        verifier = state["verifier"]
        response = state["response"]
        payload = {
            "response": response.model_dump(mode="json"),
            "results": serialize_results(verifier.results),
            "original_execution": record_execution(events),
        }
        if state["route_kind"] == "policy":
            self.cache_set(key, payload)
            if semantic_allowed:
                self.semantic_cache.store(question=question, answer=payload, metadata=metadata)
            event("cache", kind="write", hit=False)
        return {
            "result": respond(
                response.answer if response.answerable else response.refusal_reason or "No supported answer.",
                "answer" if response.answerable else "refusal",
                "miss",
                self._sources(response, verifier),
            )
        }

    graph = StateGraph(RequestState)
    nodes = {
        "safety": safety,
        "route": route,
        "exact_cache": exact_cache,
        "semantic_cache": semantic_cache,
        "retrieve": retrieve,
        "evidence": evidence,
        "synthesize": synthesize,
        "verify": verify,
        "publish": publish,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    names = list(nodes)
    graph.add_edge(START, names[0])
    for current, following in zip(names, names[1:]):
        graph.add_conditional_edges(
            current,
            lambda state: "done" if "result" in state else "continue",
            {"done": END, "continue": following},
        )
    graph.add_edge("publish", END)
    return graph.compile(name="aeronova_request")
