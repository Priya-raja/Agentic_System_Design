from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from edu_claude.models.task import TaskDefinition, TaskState


class WorkflowStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowDefinition(BaseModel):
    """A complete DAG representing a user goal."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    description: str
    tasks: list[TaskDefinition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dependencies_exist(self) -> "WorkflowDefinition":
        task_ids = {task.id for task in self.tasks}

        for task in self.tasks:
            missing = task.dependencies - task_ids

            if missing:
                missing_ids = ", ".join(str(item) for item in missing)
                raise ValueError(
                    f"Task '{task.name}' has missing dependencies: {missing_ids}"
                )

            if task.id in task.dependencies:
                raise ValueError(f"Task '{task.name}' cannot depend on itself")

        return self


class WorkflowState(BaseModel):
    """Mutable runtime state of a workflow."""

    workflow_id: UUID
    status: WorkflowStatus = WorkflowStatus.CREATED
    task_states: dict[UUID, TaskState] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
