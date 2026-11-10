
"""
AgentOps - Execution History Manager

Maintains structured execution history for AgentOps tasks.

Current storage:
- In-memory records.
- Thread-safe access.
- Task-specific execution history.
- Unique execution identifiers.
- Serializable history output.
- Defensive copies to prevent external mutation.

Future integration:
- PostgreSQL persistence.
- Dashboard execution timelines.
- Historical test comparisons.
- Workflow recovery.
"""

import copy
import threading
import uuid

from datetime import datetime, timezone
from typing import Any, Optional


class ExecutionHistoryManager:
    """Thread-safe manager for test execution history."""

    def __init__(self, max_records_per_task: int = 100):
        if (
            not isinstance(max_records_per_task, int)
            or isinstance(max_records_per_task, bool)
            or max_records_per_task < 1
        ):
            raise ValueError(
                "Maximum records per task must be a positive integer."
            )

        self.max_records_per_task = max_records_per_task

        self._history: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def record(
        self,
        task_id: str,
        execution_result,
        retry_count: int = 0,
        agent: str = "qa",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add an execution result to task history.

        Accepts either an ExecutionResult instance or a compatible
        object exposing the expected execution result attributes.
        """

        if not task_id or not isinstance(task_id, str):
            raise ValueError("A valid task ID is required.")

        if (
            not isinstance(retry_count, int)
            or isinstance(retry_count, bool)
            or retry_count < 0
        ):
            raise ValueError(
                "Retry count must be a non-negative integer."
            )

        if not agent or not isinstance(agent, str):
            raise ValueError("Agent name must be provided.")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Execution metadata must be a dictionary.")

        if hasattr(execution_result, "to_dict"):
            result_data = execution_result.to_dict()

        elif isinstance(execution_result, dict):
            result_data = execution_result

        elif hasattr(execution_result, "__dict__"):
            result_data = execution_result.__dict__

        else:
            raise TypeError("Unsupported execution result type.")

        if not isinstance(result_data, dict):
            raise TypeError(
                "Execution result must serialize to a dictionary."
            )

        # Copy incoming data so callers cannot mutate stored records.
        result_data = copy.deepcopy(result_data)

        record = {
            "execution_id": str(uuid.uuid4()),
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "retry_count": retry_count,
            "status": result_data.get("status", "UNKNOWN"),
            "exit_code": result_data.get("exit_code"),
            "duration_seconds": result_data.get(
                "duration_seconds",
                0.0,
            ),
            "tests_executed": result_data.get(
                "tests_executed",
                False,
            ),
            "test_count": result_data.get("test_count"),
            "execution_environment": result_data.get(
                "execution_environment",
                "unknown",
            ),
            "stdout": result_data.get("stdout", ""),
            "stderr": result_data.get("stderr", ""),
            "error_message": result_data.get("error_message"),
            "metadata": copy.deepcopy(metadata or {}),
        }

        with self._lock:
            task_records = self._history.setdefault(task_id, [])

            task_records.append(record)

            if len(task_records) > self.max_records_per_task:
                excess_records = (
                    len(task_records) - self.max_records_per_task
                )

                del task_records[:excess_records]

        return copy.deepcopy(record)

    def get_history(
        self,
        task_id: str,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        """Return execution history for a specific task."""

        with self._lock:
            records = copy.deepcopy(
                self._history.get(task_id, [])
            )

        if newest_first:
            records.reverse()

        return records

    def get_latest(
        self,
        task_id: str,
    ) -> Optional[dict[str, Any]]:
        """Return the most recent execution record."""

        with self._lock:
            records = self._history.get(task_id, [])

            if not records:
                return None

            return copy.deepcopy(records[-1])

    def get_all_history(self) -> dict[str, list[dict[str, Any]]]:
        """Return a snapshot of all task execution history."""

        with self._lock:
            return copy.deepcopy(self._history)

    def clear_task_history(self, task_id: str) -> bool:
        """Remove all in-memory execution history for a task."""

        with self._lock:
            return self._history.pop(task_id, None) is not None

    def get_execution_count(self, task_id: str) -> int:
        """Return the number of retained executions for a task."""

        with self._lock:
            return len(self._history.get(task_id, []))
