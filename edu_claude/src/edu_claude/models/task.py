from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    CREATED = "created"
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    RETRYING = "retrying"
    PAUSED = "paused"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskDefinition(BaseModel):
    """Immutable description of work to execute."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    dependencies: set[UUID] = Field(default_factory=set)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = Field(default=300, gt=0)
    max_retries: int = Field(default=3, ge=0)
    requires_approval: bool = False
    input: dict[str, Any] = Field(default_factory=dict)


class TaskState(BaseModel):
    """Mutable runtime state for a task."""

    task_id: UUID
    status: TaskStatus = TaskStatus.CREATED
    attempt: int = Field(default=0, ge=0)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
    heartbeat_at: datetime | None = None