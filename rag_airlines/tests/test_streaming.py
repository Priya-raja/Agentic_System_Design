import asyncio
from unittest.mock import MagicMock
from agent.session import ChatSession
from agent.route_types import ToolPlan
import pytest


def test_early_safety_paths_always_emit_final_result_without_sensitive_tokens():
    session = ChatSession()

    async def collect():
        return [event async for event in session.astream("My email is person@example.com")]

    events = asyncio.run(collect())
    assert any(event["event"] == "result" for event in events)
    assert all("person@example.com" not in str(event) for event in events)


def test_missing_model_safety_signal_cannot_default_to_safe():
    with pytest.raises(ValueError):
        ToolPlan.model_validate({"calls": []})
