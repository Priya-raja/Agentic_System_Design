from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.answers import GroundedAnswer
from agent.execution_trace import record_execution, read_execution
from agent.route_types import ToolCall, ToolPlan
from agent.prompt_manager import PromptVersion
from agent.session import ChatSession
from agent.tools.base import Tool, ToolRegistry
from agent.tools.policy_tool import PolicyArgs


def test_original_execution_keeps_only_public_operational_metadata():
    events = [
        {"event": "route", "ms": 1, "scope": "customer_specific", "department": "private"},
        {"event": "token", "ms": 2, "text": "private generated text"},
        {"event": "tool_start", "ms": 3, "tool": "claim_check", "call_id": 0, "args": {"id": "private"}},
        {
            "event": "plan",
            "ms": 4,
            "calls": [{"tool": "retrieve_policy", "depends_on": [], "args": {"query": "private question"}}],
        },
        {
            "event": "tool_end",
            "ms": 50,
            "tool": "retrieve_policy",
            "call_id": 0,
            "ok": True,
            "chunk_count": 4,
            "data": {"private": "record"},
            "request_id": "secret",
        },
        {"event": "verify", "ms": 60, "stage": "final", "passed": True},
        {"event": "result", "ms": 61, "result": {"answer": "private text"}},
    ]
    original = record_execution(events)
    assert [event["event"] for event in original["events"]] == ["plan", "tool_end", "verify"]
    assert original["latency_ms"] == 60
    assert original["events"][1]["chunk_count"] == 4
    assert all(word not in str(original) for word in ["private", "request_id", "secret", "args"])
    assert read_execution({"original_execution": original}) == original
    assert read_execution({}) is None
    assert read_execution({"original_execution": {"version": 1, "events": "invalid"}}) is None


def test_ticket_refund_policy_question_gets_contact_guidance_without_retrieval():
    cache = {}
    registry = ToolRegistry().register(
        Tool(
            "retrieve_policy",
            PolicyArgs,
            lambda **kw: {
                "chunks": [
                    {
                        "content": "Delayed baggage essentials may be reviewed with itemised receipts.",
                        "metadata": {
                            "chunk_id": f"CLAIM-{i}",
                            "document_id": "POL-CLAIM-2026",
                            "authority": "primary",
                            "department": "refunds_claims",
                        },
                        "score": 1,
                    }
                    for i in range(4)
                ],
            },
        )
    )
    semantic = MagicMock()
    semantic.lookup.return_value = None
    synth = MagicMock(
        return_value=GroundedAnswer(
            answerable=False,
            answer="",
            refusal_reason="The documents cover baggage claims, not cancelled-ticket refunds.",
            citation_chunk_ids=[f"CLAIM-{i}" for i in range(4)],
        )
    )
    session = ChatSession(
        SimpleNamespace(prompt=PromptVersion("test", "v4", "fixture", "hash")),
        registry,
        semantic,
        synthesizer=synth,
        cache_get=cache.get,
        cache_set=lambda key, value, **kw: cache.__setitem__(key, value),
    )
    question = "Refund policy for a cancelled flight"
    first = session.ask(question)
    assert first["kind"] == "handoff"
    assert first["sources"] == []
    assert not first["answerable"]
    assert "official website" in first["answer"]
    assert "booking reference" in first["answer"]
    second = session.ask(question)
    assert second["cache"] == "bypass"
    assert second["kind"] == "handoff"
    assert synth.call_count == 0
    assert not any(item["event"] == "tool_start" for item in second["trace"])


def test_claim_expense_starter_uses_claims_not_baggage_allowances():
    from agent.cascade import route_public_intent

    assert route_public_intent("What expenses can I claim for delayed baggage?") == "refunds_claims"


def test_missing_ticket_refund_reason_uses_support_contact_guidance():
    from langchain_core.documents import Document
    from agent.answers import missing_policy_reason

    evidence = [(Document(page_content="Baggage essentials", metadata={"document_id": "POL-CLAIM-2026"}), 1)]
    assert "official website" in missing_policy_reason("Refund policy for a cancelled flight", evidence)
    assert "ticket-refund" not in missing_policy_reason("Lost baggage in 2025?", evidence)
    other = [(Document(page_content="Other policy", metadata={"document_id": "OTHER"}), 1)]
    assert "booking reference" in missing_policy_reason("Refund policy for a cancelled flight", other)
