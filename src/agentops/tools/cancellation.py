"""
AgentOps - Task Cancellation

Provides a shared cancellation signal for agents
and workflow operations.

The workflow owns the cancellation event and passes
it to agents when they execute.
"""

from threading import Event
from typing import Optional


class TaskCancelledError(Exception):
    """
    Raised when an AgentOps task has been cancelled.

    This is a controlled workflow condition, not an
    unexpected application failure.
    """


def check_cancelled(
    cancel_event: Optional[Event] = None,
) -> None:
    """
    Check whether a task has been cancelled.

    Args:
        cancel_event: Shared cancellation signal.

    Raises:
        TaskCancelledError: If cancellation was requested.
    """

    if cancel_event is not None and cancel_event.is_set():
        raise TaskCancelledError(
            "Task cancellation was requested."
        )
