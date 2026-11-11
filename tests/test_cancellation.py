
import pytest

from threading import Event

from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)


def test_task_cancelled_error_is_exception():
    error = TaskCancelledError()

    assert isinstance(error, Exception)


def test_check_cancelled_without_event_does_not_raise():
    check_cancelled()


def test_check_cancelled_with_none_does_not_raise():
    check_cancelled(cancel_event=None)


def test_check_cancelled_with_unset_event_does_not_raise():
    cancel_event = Event()

    check_cancelled(cancel_event)


def test_check_cancelled_with_set_event_raises():
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        check_cancelled(cancel_event)


def test_cancellation_error_has_expected_message():
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(
        TaskCancelledError,
        match="Task cancellation was requested.",
    ):
        check_cancelled(cancel_event)


def test_check_cancelled_detects_event_set_after_creation():
    cancel_event = Event()

    check_cancelled(cancel_event)

    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        check_cancelled(cancel_event)


def test_check_cancelled_detects_repeated_cancellation_checks():
    cancel_event = Event()
    cancel_event.set()

    for _ in range(3):
        with pytest.raises(TaskCancelledError):
            check_cancelled(cancel_event)
