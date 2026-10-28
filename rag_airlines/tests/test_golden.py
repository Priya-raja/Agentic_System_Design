"""Offline recorded-response contract tests; no models, network or RAGAS."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent.session import ChatSession
from agent.answers import GroundedAnswer
from agent.prompt_manager import PromptVersion
from agent.tools.base import ToolRegistry, Tool
from agent.tools.policy_tool import PolicyArgs


def test_golden_replay_pipeline_and_cache():
    for case in json.loads((Path(__file__).parent / "fixtures/golden_replay.json").read_text()):
        cache = {}

        def retrieve(department, query):
            return {
                "department": department,
                "chunks": [
                    {
                        "content": d["content"],
                        "metadata": {
                            "chunk_id": d["chunk"],
                            "document_id": d["id"],
                            "authority": "primary",
                            "department": department,
                        },
                        "score": 1,
                    }
                    for d in case["documents"]
                ],
            }

        registry = ToolRegistry().register(Tool("retrieve_policy", PolicyArgs, retrieve))
        synth = MagicMock(
            return_value=GroundedAnswer.model_validate(case["response"]) if case["response"] else None
        )
        semantic = MagicMock()
        semantic.lookup.return_value = None
        session = ChatSession(
            SimpleNamespace(prompt=PromptVersion("test", "v4", "fixture", "hash")),
            registry,
            semantic,
            synthesizer=synth,
            cache_get=cache.get,
            cache_set=lambda key, value, **kw: cache.__setitem__(key, value),
        )
        result = session.ask(case["question"])
        assert result["kind"] == case["expected_kind"]
        assert set(source["document_id"] for source in result["sources"]) == set(case["expected_ids"])
        documents = {document["chunk"]: document["content"] for document in case["documents"]}
        for source in result["sources"]:
            assert source["excerpt"] == documents[source["chunk_id"]]
        if result["kind"] == "answer":
            assert result["verification"] == {
                "passed": True,
                "citation_errors": 0,
                "measurement_errors": 0,
                "safety_errors": 0,
                "format_errors": 0,
            }
            retrieval = next(event for event in result["trace"] if event["event"] == "tool_end")
            assert retrieval["tool"] == "retrieve_policy"
        events = [event["event"] for event in result["trace"]]
        if case["response"]:
            verifies = [event for event in result["trace"] if event["event"] == "verify"]
            assert [event["stage"] for event in verifies] == ["evidence", "synthesis", "final"]
            assert events.index("token") > max(i for i, name in enumerate(events) if name == "verify")
        else:
            synth.assert_not_called()
        hit = session.ask(case["question"])
        assert hit["cache"] == "exact"
        assert hit["original_execution"]["events"]
        assert any(item["event"] == "tool_start" for item in hit["original_execution"]["events"])
        assert not any(item["event"] == "tool_start" for item in hit["trace"])
        assert "request_id" not in str(hit["original_execution"])
        assert synth.call_count == (1 if case["response"] else 0)


def test_failed_synthesis_emits_no_unverified_tokens_and_no_cache():
    cache = {}
    semantic = MagicMock()
    semantic.lookup.return_value = None
    registry = ToolRegistry().register(
        Tool(
            "retrieve_policy",
            PolicyArgs,
            lambda **kw: {
                "chunks": [
                    {
                        "content": "25 kg",
                        "metadata": {"chunk_id": "B", "authority": "primary", "department": "baggage"},
                        "score": 1,
                    }
                ]
            },
        )
    )
    response = GroundedAnswer(answerable=True, answer="100 kg", citation_chunk_ids=["B"])
    session = ChatSession(
        SimpleNamespace(prompt=PromptVersion("test", "v4", "fixture", "hash")),
        registry,
        semantic,
        synthesizer=lambda *a: response,
        cache_get=lambda _: None,
        cache_set=lambda *a, **kw: cache.update({"bad": True}),
    )
    result = session.ask("What is the baggage allowance in 2026?")
    assert result["kind"] == "verification_failed"
    assert result["verification"]["passed"] is False
    assert result["verification"]["measurement_errors"] > 0
    assert result["verification"]["citation_errors"] == 0
    assert "100 kg" not in "".join(event.get("text", "") for event in result["trace"])
    assert cache == {}


def test_multi_department_request_does_not_answer_from_partial_evidence():
    semantic = MagicMock()
    semantic.lookup.return_value = None
    registry = ToolRegistry().register(
        Tool(
            "retrieve_policy",
            PolicyArgs,
            lambda department, query: {
                "department": department,
                "chunks": []
                if department == "upgrades"
                else [
                    {
                        "content": "25 kg",
                        "metadata": {"chunk_id": "B", "authority": "primary", "department": "baggage"},
                        "score": 1,
                    }
                ],
            },
        )
    )
    synth = MagicMock()
    session = ChatSession(
        SimpleNamespace(prompt=PromptVersion("test", "v4", "fixture", "hash")),
        registry,
        semantic,
        synthesizer=synth,
        cache_get=lambda _: None,
    )
    result = session.ask("Can I upgrade and bring extra baggage?")
    assert result["kind"] == "clarify"
    synth.assert_not_called()


def test_deterministic_route_preserves_the_original_policy_question():
    question = "Please explain checked baggage in November 2025."
    handler = MagicMock(return_value={"chunks": []})
    semantic = MagicMock()
    semantic.lookup.return_value = None
    registry = ToolRegistry().register(Tool("retrieve_policy", PolicyArgs, handler))
    session = ChatSession(
        SimpleNamespace(prompt=PromptVersion("test", "v4", "fixture", "hash")),
        registry,
        semantic,
        cache_get=lambda _: None,
        cache_set=lambda *a, **kw: None,
    )
    assert session.ask(question)["kind"] == "no_context"
    assert handler.call_args.kwargs["query"] == question


def test_public_rag_hands_flight_status_to_the_live_service():
    from agent.tools.flight_tool import FlightArgs

    handler = MagicMock()
    registry = ToolRegistry().register(Tool("flight_status", FlightArgs, handler))
    session = ChatSession(
        SimpleNamespace(prompt=PromptVersion("test", "v4", "fixture", "hash")),
        registry,
        object(),
        cache_get=lambda _: None,
    )
    assert session.ask("What is my flight status?")["kind"] == "handoff"
    handler.assert_not_called()


def test_session_collapses_duplicate_answer_paragraph_before_releasing_it():
    from agent.session import ChatSession

    paragraph = "Passengers must be offered assistance under the regulation."
    assert ChatSession._deduplicate_answer_paragraphs(f"{paragraph}\n\n{paragraph}") == paragraph
