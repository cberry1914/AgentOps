
"""
AgentOps - Execution History Tests

Tests execution result serialization, validation, history
ordering, retention, isolation, and concurrent access.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import UUID

import pytest

from agentops.models.execution import ExecutionResult
from agentops.tools.execution_history import ExecutionHistoryManager


@pytest.fixture
def manager():
    return ExecutionHistoryManager()


@pytest.fixture
def execution_result():
    return ExecutionResult(
        status="PASSED",
        exit_code=0,
        duration_seconds=1.25,
        stdout="3 passed",
        stderr="",
        environment="docker",
        tests_executed=True,
        test_count=3,
        metadata={"source": "pytest"},
    )


def make_result(**overrides):
    result = {
        "status": "PASSED",
        "exit_code": 0,
        "duration_seconds": 1.25,
        "stdout": "3 passed",
        "stderr": "",
        "tests_executed": True,
        "test_count": 3,
        "execution_environment": "docker",
        "error_message": None,
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


def test_manager_initializes_with_default_retention():
    manager = ExecutionHistoryManager()

    assert manager.max_records_per_task == 100
    assert manager.get_all_history() == {}


def test_manager_accepts_custom_retention_limit():
    manager = ExecutionHistoryManager(max_records_per_task=5)

    assert manager.max_records_per_task == 5
    assert manager.get_execution_count("task-1") == 0


@pytest.mark.parametrize("value", [0, -1, -10])
def test_manager_rejects_non_positive_retention_limits(value):
    with pytest.raises(ValueError):
        ExecutionHistoryManager(max_records_per_task=value)


@pytest.mark.parametrize("value", [True, False, 1.5, "10", None])
def test_manager_rejects_non_integer_retention_limits(value):
    with pytest.raises(ValueError):
        ExecutionHistoryManager(max_records_per_task=value)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("task_id", ["", None, 123, [], {}])
def test_record_rejects_invalid_task_id(manager, task_id):
    with pytest.raises(ValueError):
        manager.record(task_id, make_result())


@pytest.mark.parametrize("retry_count", [-1, -10])
def test_record_rejects_negative_retry_count(manager, retry_count):
    with pytest.raises(ValueError):
        manager.record("task-1", make_result(), retry_count=retry_count)


@pytest.mark.parametrize("retry_count", [True, False, 1.5, "1", None])
def test_record_rejects_non_integer_retry_count(manager, retry_count):
    with pytest.raises(ValueError):
        manager.record("task-1", make_result(), retry_count=retry_count)


@pytest.mark.parametrize("agent", ["", None, 123, [], {}])
def test_record_rejects_invalid_agent(manager, agent):
    with pytest.raises(ValueError):
        manager.record("task-1", make_result(), agent=agent)


@pytest.mark.parametrize("metadata", ["invalid", 123, [], True])
def test_record_rejects_non_dictionary_metadata(manager, metadata):
    with pytest.raises(TypeError):
        manager.record("task-1", make_result(), metadata=metadata)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_record_execution_result(manager, execution_result):
    record = manager.record("task-1", execution_result)

    assert record["task_id"] == "task-1"
    assert record["status"] == "PASSED"
    assert record["exit_code"] == 0
    assert record["duration_seconds"] == 1.25
    assert record["stdout"] == "3 passed"
    assert record["stderr"] == ""
    assert record["tests_executed"] is True
    assert record["test_count"] == 3
    assert record["execution_environment"] == "docker"


def test_record_dictionary_result(manager):
    record = manager.record("task-1", make_result())

    assert record["status"] == "PASSED"
    assert record["duration_seconds"] == 1.25


def test_record_object_with_dict_attribute(manager):
    class CompatibleResult:
        def __init__(self):
            self.status = "FAILED"
            self.exit_code = 1
            self.duration_seconds = 2.5
            self.stdout = ""
            self.stderr = "Test failed"

    record = manager.record("task-1", CompatibleResult())

    assert record["status"] == "FAILED"
    assert record["exit_code"] == 1
    assert record["duration_seconds"] == 2.5


def test_to_dict_serialization_is_used(manager, execution_result):
    record = manager.record("task-1", execution_result)

    assert record["status"] == execution_result.status
    assert record["duration_seconds"] == execution_result.duration_seconds


def test_record_uses_default_values(manager):
    record = manager.record(
        "task-1",
        {"status": "PASSED"},
    )

    assert record["exit_code"] is None
    assert record["duration_seconds"] == 0.0
    assert record["tests_executed"] is False
    assert record["test_count"] is None
    assert record["execution_environment"] == "unknown"
    assert record["stdout"] == ""
    assert record["stderr"] == ""
    assert record["error_message"] is None
    assert record["metadata"] == {}


def test_record_uses_default_status_when_missing(manager):
    record = manager.record("task-1", {})

    assert record["status"] == "UNKNOWN"


def test_record_preserves_retry_count(manager):
    record = manager.record(
        "task-1",
        make_result(),
        retry_count=2,
    )

    assert record["retry_count"] == 2


def test_record_preserves_agent(manager):
    record = manager.record(
        "task-1",
        make_result(),
        agent="security",
    )

    assert record["agent"] == "security"


def test_record_preserves_metadata(manager):
    metadata = {
        "branch": "feature/test",
        "attempt": 2,
    }

    record = manager.record(
        "task-1",
        make_result(),
        metadata=metadata,
    )

    assert record["metadata"] == metadata


def test_record_generates_unique_execution_ids(manager):
    first = manager.record("task-1", make_result())
    second = manager.record("task-1", make_result())

    assert first["execution_id"] != second["execution_id"]


def test_execution_id_is_valid_uuid(manager):
    record = manager.record("task-1", make_result())

    parsed = UUID(record["execution_id"])

    assert str(parsed) == record["execution_id"]


def test_record_timestamp_is_valid_iso_format(manager):
    record = manager.record("task-1", make_result())

    timestamp = datetime.fromisoformat(record["timestamp"])

    assert timestamp.tzinfo is not None


def test_record_contains_task_id(manager):
    record = manager.record("my-task", make_result())

    assert record["task_id"] == "my-task"


def test_record_contains_all_expected_fields(manager):
    record = manager.record("task-1", make_result())

    expected_fields = {
        "execution_id",
        "task_id",
        "timestamp",
        "agent",
        "retry_count",
        "status",
        "exit_code",
        "duration_seconds",
        "tests_executed",
        "test_count",
        "execution_environment",
        "stdout",
        "stderr",
        "error_message",
        "metadata",
    }

    assert set(record) == expected_fields


def test_record_rejects_unsupported_result_type(manager):
    with pytest.raises(TypeError):
        manager.record("task-1", object())


def test_record_rejects_non_dictionary_serialized_result(manager):
    class InvalidResult:
        def to_dict(self):
            return "not a dictionary"

    with pytest.raises(TypeError):
        manager.record("task-1", InvalidResult())


# ---------------------------------------------------------------------
# Deep-copy isolation
# ---------------------------------------------------------------------


def test_record_returns_independent_copy(manager):
    record = manager.record("task-1", make_result())

    record["status"] = "MODIFIED"

    stored = manager.get_latest("task-1")

    assert stored["status"] == "PASSED"


def test_record_deepcopies_metadata(manager):
    metadata = {"details": {"attempt": 1}}

    record = manager.record(
        "task-1",
        make_result(),
        metadata=metadata,
    )

    metadata["details"]["attempt"] = 99

    assert record["metadata"]["details"]["attempt"] == 1


def test_record_deepcopies_input_result(manager):
    result = make_result(
        metadata={"details": {"value": "original"}}
    )

    record = manager.record(
        "task-1",
        result,
        metadata=result["metadata"],
    )

    result["metadata"]["details"]["value"] = "changed"

    assert record["metadata"]["details"]["value"] == "original"


def test_get_history_returns_independent_copy(manager):
    manager.record("task-1", make_result())

    history = manager.get_history("task-1")
    history[0]["status"] = "MODIFIED"

    assert manager.get_history("task-1")[0]["status"] == "PASSED"


def test_get_latest_returns_independent_copy(manager):
    manager.record("task-1", make_result())

    latest = manager.get_latest("task-1")
    latest["status"] = "MODIFIED"

    assert manager.get_latest("task-1")["status"] == "PASSED"


def test_get_all_history_returns_independent_copy(manager):
    manager.record("task-1", make_result())

    history = manager.get_all_history()
    history["task-1"][0]["status"] = "MODIFIED"

    assert manager.get_all_history()["task-1"][0]["status"] == "PASSED"


# ---------------------------------------------------------------------
# Task-specific history
# ---------------------------------------------------------------------


def test_get_history_returns_empty_list_for_unknown_task(manager):
    assert manager.get_history("missing-task") == []


def test_get_history_returns_only_requested_task(manager):
    manager.record("task-1", make_result())
    manager.record("task-2", make_result())

    history = manager.get_history("task-1")

    assert len(history) == 1
    assert history[0]["task_id"] == "task-1"


def test_get_history_preserves_insertion_order(manager):
    manager.record("task-1", make_result(status="PASSED"))
    manager.record("task-1", make_result(status="FAILED"))
    manager.record("task-1", make_result(status="TIMEOUT"))

    history = manager.get_history("task-1")

    assert [item["status"] for item in history] == [
        "PASSED",
        "FAILED",
        "TIMEOUT",
    ]


def test_get_history_newest_first(manager):
    manager.record("task-1", make_result(status="PASSED"))
    manager.record("task-1", make_result(status="FAILED"))
    manager.record("task-1", make_result(status="TIMEOUT"))

    history = manager.get_history("task-1", newest_first=True)

    assert [item["status"] for item in history] == [
        "TIMEOUT",
        "FAILED",
        "PASSED",
    ]


def test_get_history_newest_first_does_not_modify_stored_order(manager):
    manager.record("task-1", make_result(status="PASSED"))
    manager.record("task-1", make_result(status="FAILED"))

    manager.get_history("task-1", newest_first=True)

    history = manager.get_history("task-1")

    assert [item["status"] for item in history] == [
        "PASSED",
        "FAILED",
    ]


def test_get_latest_returns_most_recent_record(manager):
    manager.record("task-1", make_result(status="PASSED"))
    manager.record("task-1", make_result(status="FAILED"))

    latest = manager.get_latest("task-1")

    assert latest["status"] == "FAILED"


def test_get_latest_returns_none_for_unknown_task(manager):
    assert manager.get_latest("missing-task") is None


def test_get_all_history_returns_records_for_all_tasks(manager):
    manager.record("task-1", make_result())
    manager.record("task-2", make_result())
    manager.record("task-3", make_result())

    history = manager.get_all_history()

    assert len(history) == 3
    assert set(history) == {"task-1", "task-2", "task-3"}

    for task_id, records in history.items():
        assert len(records) == 1
        assert records[0]["task_id"] == task_id


def test_get_all_history_returns_empty_dictionary_when_empty(manager):
    assert manager.get_all_history() == {}


# ---------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------


def test_retention_limit_is_applied_per_task():
    manager = ExecutionHistoryManager(max_records_per_task=2)

    manager.record("task-1", make_result(status="PASSED"))
    manager.record("task-1", make_result(status="FAILED"))
    manager.record("task-1", make_result(status="TIMEOUT"))

    history = manager.get_history("task-1")

    assert len(history) == 2
    assert [item["status"] for item in history] == [
        "FAILED",
        "TIMEOUT",
    ]


def test_retention_does_not_affect_other_tasks():
    manager = ExecutionHistoryManager(max_records_per_task=2)

    manager.record("task-1", make_result())
    manager.record("task-1", make_result())
    manager.record("task-1", make_result())

    manager.record("task-2", make_result())

    assert manager.get_execution_count("task-1") == 2
    assert manager.get_execution_count("task-2") == 1


def test_retention_preserves_newest_records():
    manager = ExecutionHistoryManager(max_records_per_task=3)

    for number in range(5):
        manager.record(
            "task-1",
            make_result(test_count=number),
        )

    history = manager.get_history("task-1")

    assert [item["test_count"] for item in history] == [2, 3, 4]


def test_retention_limit_of_one():
    manager = ExecutionHistoryManager(max_records_per_task=1)

    manager.record("task-1", make_result(status="PASSED"))
    manager.record("task-1", make_result(status="FAILED"))

    history = manager.get_history("task-1")

    assert len(history) == 1
    assert history[0]["status"] == "FAILED"


# ---------------------------------------------------------------------
# Clearing and counting
# ---------------------------------------------------------------------


def test_clear_task_history_removes_existing_history(manager):
    manager.record("task-1", make_result())

    result = manager.clear_task_history("task-1")

    assert result is True
    assert manager.get_history("task-1") == []


def test_clear_task_history_returns_false_for_unknown_task(manager):
    assert manager.clear_task_history("missing-task") is False


def test_clear_task_history_does_not_affect_other_tasks(manager):
    manager.record("task-1", make_result())
    manager.record("task-2", make_result())

    manager.clear_task_history("task-1")

    assert manager.get_execution_count("task-1") == 0
    assert manager.get_execution_count("task-2") == 1


def test_execution_count_returns_zero_for_unknown_task(manager):
    assert manager.get_execution_count("missing-task") == 0


def test_execution_count_tracks_records(manager):
    manager.record("task-1", make_result())
    manager.record("task-1", make_result())
    manager.record("task-1", make_result())

    assert manager.get_execution_count("task-1") == 3


def test_execution_count_respects_retention_limit():
    manager = ExecutionHistoryManager(max_records_per_task=2)

    for _ in range(4):
        manager.record("task-1", make_result())

    assert manager.get_execution_count("task-1") == 2


# ---------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------


def test_concurrent_records_are_preserved():
    manager = ExecutionHistoryManager(max_records_per_task=100)

    def record_execution(number):
        return manager.record(
            "task-1",
            make_result(test_count=number),
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        records = list(executor.map(record_execution, range(50)))

    history = manager.get_history("task-1")

    assert len(records) == 50
    assert len(history) == 50
    assert len({item["execution_id"] for item in history}) == 50


def test_concurrent_records_remain_within_retention_limit():
    manager = ExecutionHistoryManager(max_records_per_task=10)

    def record_execution(number):
        manager.record(
            "task-1",
            make_result(test_count=number),
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(record_execution, range(50)))

    assert manager.get_execution_count("task-1") == 10


def test_concurrent_records_for_separate_tasks():
    manager = ExecutionHistoryManager(max_records_per_task=20)

    def record_execution(task_number):
        task_id = f"task-{task_number}"

        for attempt in range(5):
            manager.record(
                task_id,
                make_result(test_count=attempt),
            )

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(record_execution, range(10)))

    history = manager.get_all_history()

    assert len(history) == 10
    assert sum(len(records) for records in history.values()) == 50

    for task_number in range(10):
        task_id = f"task-{task_number}"
        assert len(history[task_id]) == 5
