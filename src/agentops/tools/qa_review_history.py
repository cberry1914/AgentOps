
"""
AgentOps - QA Review History

Maintains structured QA review history by task.

This manager is currently in-memory. Persistent database
storage will be added as part of the platform's history
and persistence implementation.
"""

from copy import deepcopy
from threading import RLock
from typing import Optional

from agentops.models.qa_review import QAReview


class QAReviewHistoryManager:
    """Thread-safe storage for structured QA reviews."""

    def __init__(self) -> None:
        self._history: dict[str, list[dict]] = {}
        self._lock = RLock()

    def record(
        self,
        task_id: str,
        review: QAReview,
    ) -> dict:
        """Record a QA review and return its stored representation."""

        if not task_id or not task_id.strip():
            raise ValueError("Task ID cannot be empty.")

        if not isinstance(review, QAReview):
            raise TypeError(
                "Review must be a QAReview instance."
            )

        entry = review.to_dict()

        with self._lock:
            task_history = self._history.setdefault(
                task_id,
                [],
            )

            entry["history_index"] = len(task_history) + 1

            task_history.append(
                deepcopy(entry)
            )

            return deepcopy(entry)

    def get_history(
        self,
        task_id: str,
    ) -> list[dict]:
        """Return all QA reviews for one task, oldest first."""

        with self._lock:
            return deepcopy(
                self._history.get(task_id, [])
            )

    def get_latest(
        self,
        task_id: str,
    ) -> Optional[dict]:
        """Return the most recent QA review for a task."""

        with self._lock:
            task_history = self._history.get(
                task_id,
                [],
            )

            if not task_history:
                return None

            return deepcopy(task_history[-1])

    def get_all_history(self) -> dict[str, list[dict]]:
        """Return a copy of all recorded QA review history."""

        with self._lock:
            return deepcopy(self._history)

    def get_review_count(
        self,
        task_id: Optional[str] = None,
    ) -> int:
        """Return the number of recorded reviews."""

        with self._lock:
            if task_id is not None:
                return len(
                    self._history.get(task_id, [])
                )

            return sum(
                len(reviews)
                for reviews in self._history.values()
            )

    def clear_task_history(
        self,
        task_id: str,
    ) -> int:
        """Clear review history for one task."""

        with self._lock:
            removed = self._history.pop(
                task_id,
                [],
            )

            return len(removed)

    def clear_all_history(self) -> int:
        """Clear all QA review history."""

        with self._lock:
            removed_count = sum(
                len(reviews)
                for reviews in self._history.values()
            )

            self._history.clear()

            return removed_count
