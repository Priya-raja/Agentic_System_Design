from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    WAITING_FOR_APPROVAL = "waiting_for_approval"


class AgentStep(BaseModel):
    """One reasoning or execution step taken by the agent."""

    step_number: int = Field(ge=1)
    description: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str | None = None


class AgentResponse(BaseModel):
    """Final result returned by the agent layer."""

    status: AgentStatus
    answer: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)