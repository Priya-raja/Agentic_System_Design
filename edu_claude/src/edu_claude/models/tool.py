from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ToolResult(BaseModel):
    """Standard result returned by every tool."""

    tool_name: str
    status: ToolStatus
    output: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == ToolStatus.SUCCESS