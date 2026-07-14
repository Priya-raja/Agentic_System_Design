from collections import deque
from collections.abc import Iterable
from uuid import UUID

from edu_claude.models import (
    TaskDefinition,
    TaskState,
    TaskStatus,
    WorkflowDefinition,
)
from edu_claude.workflows.exceptions import (
    CycleDetectedError,
    DuplicateTaskError,
    MissingDependencyError,
)


class DAG:
    """Validated directed acyclic graph of workflow tasks."""

    def __init__(self, workflow: WorkflowDefinition) -> None:
        self.workflow = workflow

        self._tasks: dict[UUID, TaskDefinition] = {}
        self._dependencies: dict[UUID, set[UUID]] = {}
        self._dependents: dict[UUID, set[UUID]] = {}

        self._build_graph()
        self._validate_acyclic()

    def _build_graph(self) -> None:
        """Build dependency and reverse-dependency indexes."""

        for task in self.workflow.tasks:
            if task.id in self._tasks:
                raise DuplicateTaskError(task.id)

            self._tasks[task.id] = task
            self._dependencies[task.id] = set(task.dependencies)
            self._dependents[task.id] = set()

        for task_id, dependencies in self._dependencies.items():
            for dependency_id in dependencies:
                if dependency_id not in self._tasks:
                    raise MissingDependencyError(task_id, dependency_id)

                self._dependents[dependency_id].add(task_id)

    def _validate_acyclic(self) -> None:
        """Detect dependency cycles using depth-first search."""

        unvisited = 0
        visiting = 1
        visited = 2

        states: dict[UUID, int] = {task_id: unvisited for task_id in self._tasks}

        path: list[UUID] = []

        def visit(task_id: UUID) -> None:
            state = states[task_id]

            if state == visited:
                return

            if state == visiting:
                cycle_start = path.index(task_id)
                cycle = [*path[cycle_start:], task_id]
                raise CycleDetectedError(cycle)

            states[task_id] = visiting
            path.append(task_id)

            for dependency_id in self._dependencies[task_id]:
                visit(dependency_id)

            path.pop()
            states[task_id] = visited

        for task_id in self._tasks:
            if states[task_id] == unvisited:
                visit(task_id)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def get_task(self, task_id: UUID) -> TaskDefinition:
        """Return a task definition by ID."""

        return self._tasks[task_id]

    def dependencies_of(self, task_id: UUID) -> frozenset[UUID]:
        """Return the direct dependencies of a task."""

        return frozenset(self._dependencies[task_id])

    def dependents_of(self, task_id: UUID) -> frozenset[UUID]:
        """Return tasks that directly depend on the given task."""

        return frozenset(self._dependents[task_id])

    def root_tasks(self) -> list[TaskDefinition]:
        """Return tasks with no dependencies."""

        return [
            self._tasks[task_id]
            for task_id, dependencies in self._dependencies.items()
            if not dependencies
        ]

    def leaf_tasks(self) -> list[TaskDefinition]:
        """Return tasks that have no dependents."""

        return [
            self._tasks[task_id]
            for task_id, dependents in self._dependents.items()
            if not dependents
        ]

    def topological_order(self) -> list[TaskDefinition]:
        """Return tasks in dependency-safe execution order."""

        indegree = {
            task_id: len(dependencies)
            for task_id, dependencies in self._dependencies.items()
        }

        ready = deque(task_id for task_id, degree in indegree.items() if degree == 0)

        ordered: list[TaskDefinition] = []

        while ready:
            task_id = ready.popleft()
            ordered.append(self._tasks[task_id])

            for dependent_id in self._dependents[task_id]:
                indegree[dependent_id] -= 1

                if indegree[dependent_id] == 0:
                    ready.append(dependent_id)

        if len(ordered) != len(self._tasks):
            raise CycleDetectedError([])

        return ordered

    def ready_tasks(
        self,
        task_states: dict[UUID, TaskState],
    ) -> list[TaskDefinition]:
        """
        Return tasks that can run now.

        A task is ready when:
        - its current state is CREATED, PENDING or RETRYING
        - every dependency has completed successfully
        """

        runnable_statuses = {
            TaskStatus.CREATED,
            TaskStatus.PENDING,
            TaskStatus.RETRYING,
        }

        ready: list[TaskDefinition] = []

        for task_id, task in self._tasks.items():
            state = task_states.get(task_id)

            if state is None:
                continue

            if state.status not in runnable_statuses:
                continue

            dependencies = self._dependencies[task_id]

            if all(
                task_states.get(dependency_id) is not None
                and task_states[dependency_id].status == TaskStatus.COMPLETED
                for dependency_id in dependencies
            ):
                ready.append(task)

        return ready

    def blocked_tasks(
        self,
        task_states: dict[UUID, TaskState],
    ) -> list[TaskDefinition]:
        """
        Return tasks blocked by failed or cancelled dependencies.
        """

        blocking_statuses = {
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.BLOCKED,
        }

        blocked: list[TaskDefinition] = []

        for task_id, task in self._tasks.items():
            state = task_states.get(task_id)

            if state is None:
                continue

            if state.status in {
                TaskStatus.COMPLETED,
                TaskStatus.RUNNING,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.BLOCKED,
            }:
                continue

            dependency_states = (
                task_states.get(dependency_id)
                for dependency_id in self._dependencies[task_id]
            )

            if any(
                dependency_state is not None
                and dependency_state.status in blocking_statuses
                for dependency_state in dependency_states
            ):
                blocked.append(task)

        return blocked

    def descendants_of(self, task_id: UUID) -> set[UUID]:
        """Return all direct and indirect dependents of a task."""

        descendants: set[UUID] = set()
        pending = list(self._dependents[task_id])

        while pending:
            current = pending.pop()

            if current in descendants:
                continue

            descendants.add(current)
            pending.extend(self._dependents[current])

        return descendants

    def ancestors_of(self, task_id: UUID) -> set[UUID]:
        """Return all direct and indirect dependencies of a task."""

        ancestors: set[UUID] = set()
        pending = list(self._dependencies[task_id])

        while pending:
            current = pending.pop()

            if current in ancestors:
                continue

            ancestors.add(current)
            pending.extend(self._dependencies[current])

        return ancestors

    def tasks(self) -> Iterable[TaskDefinition]:
        """Iterate over every task in the DAG."""

        return self._tasks.values()
