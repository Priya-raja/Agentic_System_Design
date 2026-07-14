from uuid import UUID


class DAGError(Exception):
    """Base exception for DAG-related errors."""


class DuplicateTaskError(DAGError):
    """Raised when a workflow contains duplicate task IDs."""

    def __init__(self, task_id: UUID) -> None:
        super().__init__(f"Duplicate task ID found: {task_id}")
        self.task_id = task_id


class MissingDependencyError(DAGError):
    """Raised when a task depends on a task outside the workflow."""

    def __init__(self, task_id: UUID, dependency_id: UUID) -> None:
        super().__init__(f"Task {task_id} depends on missing task {dependency_id}")
        self.task_id = task_id
        self.dependency_id = dependency_id


class CycleDetectedError(DAGError):
    """Raised when the workflow dependency graph contains a cycle."""

    def __init__(self, cycle: list[UUID]) -> None:
        cycle_path = " -> ".join(str(task_id) for task_id in cycle)
        super().__init__(f"Cycle detected in workflow: {cycle_path}")
        self.cycle = cycle
