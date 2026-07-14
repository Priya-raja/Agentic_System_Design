from uuid import uuid4

import pytest
from pydantic import ValidationError

from edu_claude.models import TaskDefinition, WorkflowDefinition


def test_workflow_accepts_valid_dependencies() -> None:
    first = TaskDefinition(
        name="Read repository",
        description="Inspect repository structure",
    )

    second = TaskDefinition(
        name="Explain repository",
        description="Generate an explanation",
        dependencies={first.id},
    )

    workflow = WorkflowDefinition(
        name="Explain codebase",
        description="Inspect and explain a repository",
        tasks=[first, second],
    )

    assert len(workflow.tasks) == 2
    assert first.id in second.dependencies


def test_workflow_rejects_missing_dependency() -> None:
    task = TaskDefinition(
        name="Invalid task",
        description="Depends on a task outside the workflow",
        dependencies={uuid4()},
    )

    with pytest.raises(ValidationError):
        WorkflowDefinition(
            name="Invalid workflow",
            description="Contains a missing dependency",
            tasks=[task],
        )


def test_task_cannot_depend_on_itself() -> None:
    task = TaskDefinition(
        name="Self-dependent task",
        description="Invalid task",
    )

    task.dependencies.add(task.id)

    with pytest.raises(ValidationError):
        WorkflowDefinition(
            name="Invalid workflow",
            description="Contains a self-dependency",
            tasks=[task],
        )
