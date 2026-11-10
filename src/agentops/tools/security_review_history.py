
"""
AgentOps - Security Review History

Maintains structured security review history for AgentOps tasks.

Current storage:
- In-memory review records.
- Thread-safe access.
- Task-specific review history.
- Sequential review identifiers.
- Serializable review output.

Future integration:
- PostgreSQL persistence.
- Dashboard security timelines.
- Historical security comparisons.
- Audit and compliance reporting.
"""

import threading

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from agentops.models.security_review import SecurityReview


class SecurityReviewHistoryManager:
    """Thread-safe storage for structured security reviews."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._next_id = 1

    def record(
        self,
        task_id: str,
        review: SecurityReview,
    ) -> dict[str, Any]:
        """Store a security review and return its history record."""

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string.")

        if not isinstance(review, SecurityReview):
            raise TypeError(
                "review must be a SecurityReview instance."
            )

        review_data = review.to_dict()

        with self._lock:
            record_id = self._next_id
            self._next_id += 1

            record = {
                "record_id": record_id,
                "task_id": task_id,
                "recorded_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "review": deepcopy(review_data),
            }

            self._history.setdefault(task_id, []).append(record)

            return deepcopy(record)

    def get_history(
        self,
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Return all security review records for one task."""

        with self._lock:
            records = self._history.get(task_id, [])
            return deepcopy(records)

    def get_latest(
        self,
        task_id: str,
    ) -> dict[str, Any] | None:
        """Return the most recent security review for a task."""

        with self._lock:
            records = self._history.get(task_id, [])

            if not records:
                return None

            return deepcopy(records[-1])

    def get_all_history(self) -> list[dict[str, Any]]:
        """Return all security review records across all tasks."""

        with self._lock:
            records = []

            for task_records in self._history.values():
                records.extend(task_records)

            records.sort(
                key=lambda item: item["record_id"]
            )

            return deepcopy(records)

    def get_review_count(
        self,
        task_id: str | None = None,
    ) -> int:
        """Return the number of stored reviews."""

        with self._lock:

            if task_id is not None:
                return len(self._history.get(task_id, []))

            return sum(
                len(records)
                for records in self._history.values()
            )

    def clear_task_history(self, task_id: str) -> int:
        """Remove all security review history for one task."""

        with self._lock:
            records = self._history.pop(task_id, [])
            return len(records)

    def clear_all_history(self) -> int:
        """Remove all stored security review history."""

        with self._lock:
            count = sum(
                len(records)
                for records in self._history.values()
            )

            self._history.clear()

            return count
