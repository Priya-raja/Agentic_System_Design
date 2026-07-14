import pytest


from edu_claude.models import (
    TaskDefinition,
    TaskState,
    TaskStatus,
    WorkflowDefinition,
)
from edu_claude.workflows import CycleDetectedError, DAG


def build_sample_dag() -> tuple[
    DAG,
    TaskDefinition,
    TaskDefinition,
    TaskDefinition,
    TaskDefinition,
]:
    task_a = TaskDefinition(
        name="A",
        description="Root task",
    )

    task_b = TaskDefinition(
        name="B",
        description="Depends on A",
        dependencies={task_a.id},
    )

    task_c = TaskDefinition(
        name="C",
        description="Depends on A",
        dependencies={task_a.id},
    )

    task_d = TaskDefinition(
        name="D",
        description="Depends on B and C",
        dependencies={task_b.id, task_c.id},
    )

    workflow = WorkflowDefinition(
        name="Sample workflow",
        description="A diamond-shaped DAG",
        tasks=[task_a, task_b, task_c, task_d],
    )

    return DAG(workflow), task_a, task_b, task_c, task_d


def test_root_tasks() -> None:
    dag, task_a, _, _, _ = build_sample_dag()

    roots = dag.root_tasks()

    assert roots == [task_a]


def test_leaf_tasks() -> None:
    dag, _, _, _, task_d = build_sample_dag()

    leaves = dag.leaf_tasks()

    assert leaves == [task_d]


def test_dependencies_and_dependents() -> None:
    dag, task_a, task_b, task_c, _ = build_sample_dag()

    assert dag.dependencies_of(task_b.id) == {task_a.id}
    assert dag.dependents_of(task_a.id) == {
        task_b.id,
        task_c.id,
    }


def test_topological_order_respects_dependencies() -> None:
    dag, task_a, task_b, task_c, task_d = build_sample_dag()

    ordered = dag.topological_order()
    positions = {task.id: index for index, task in enumerate(ordered)}

    assert positions[task_a.id] < positions[task_b.id]
    assert positions[task_a.id] < positions[task_c.id]
    assert positions[task_b.id] < positions[task_d.id]
    assert positions[task_c.id] < positions[task_d.id]


def test_ready_tasks_initially_contains_only_roots() -> None:
    dag, task_a, task_b, task_c, task_d = build_sample_dag()

    states = {
        task.id: TaskState(
            task_id=task.id,
            status=TaskStatus.PENDING,
        )
        for task in [task_a, task_b, task_c, task_d]
    }

    ready = dag.ready_tasks(states)

    assert ready == [task_a]


def test_ready_tasks_after_root_completes() -> None:
    dag, task_a, task_b, task_c, task_d = build_sample_dag()

    states = {
        task_a.id: TaskState(
            task_id=task_a.id,
            status=TaskStatus.COMPLETED,
        ),
        task_b.id: TaskState(
            task_id=task_b.id,
            status=TaskStatus.PENDING,
        ),
        task_c.id: TaskState(
            task_id=task_c.id,
            status=TaskStatus.PENDING,
        ),
        task_d.id: TaskState(
            task_id=task_d.id,
            status=TaskStatus.PENDING,
        ),
    }

    ready_ids = {task.id for task in dag.ready_tasks(states)}

    assert ready_ids == {task_b.id, task_c.id}


def test_task_is_blocked_when_dependency_fails() -> None:
    dag, task_a, task_b, _, _ = build_sample_dag()

    states = {
        task_a.id: TaskState(
            task_id=task_a.id,
            status=TaskStatus.FAILED,
        ),
        task_b.id: TaskState(
            task_id=task_b.id,
            status=TaskStatus.PENDING,
        ),
    }

    blocked_ids = {task.id for task in dag.blocked_tasks(states)}

    assert task_b.id in blocked_ids


def test_descendants() -> None:
    dag, task_a, task_b, task_c, task_d = build_sample_dag()

    descendants = dag.descendants_of(task_a.id)

    assert descendants == {
        task_b.id,
        task_c.id,
        task_d.id,
    }


def test_ancestors() -> None:
    dag, task_a, task_b, task_c, task_d = build_sample_dag()

    ancestors = dag.ancestors_of(task_d.id)

    assert ancestors == {
        task_a.id,
        task_b.id,
        task_c.id,
    }


def test_cycle_is_rejected() -> None:
    task_a = TaskDefinition(
        name="A",
        description="Task A",
    )

    task_b = TaskDefinition(
        name="B",
        description="Task B",
        dependencies={task_a.id},
    )

    # Pydantic models validate assignment only when configured to do so,
    # so we can construct a cycle for the DAG validation test.
    task_a.dependencies.add(task_b.id)

    workflow = WorkflowDefinition(
        name="Cyclic workflow",
        description="Contains a dependency cycle",
        tasks=[task_a, task_b],
    )

    with pytest.raises(CycleDetectedError):
        DAG(workflow)
