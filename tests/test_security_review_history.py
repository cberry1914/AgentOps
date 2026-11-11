
"""
AgentOps - Security Review History Tests

Tests review storage, retrieval, record isolation,
sequential identifiers, clearing, and thread safety.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from agentops.models.security_review import (
    SecurityFinding,
    SecurityReview,
)
from agentops.tools.security_review_history import (
    SecurityReviewHistoryManager,
)


@pytest.fixture
def manager():
    return SecurityReviewHistoryManager()


def make_review(title="Test security finding"):
    finding = SecurityFinding(
        severity="HIGH",
        category="Authentication",
        title=title,
        description="Authentication configuration may be insecure.",
        file_path="src/auth.py",
        line_number=25,
        evidence="authentication_enabled = False",
        recommendation="Review authentication configuration.",
        confidence="HIGH",
        source="static_analysis",
    )

    return SecurityReview(
        decision="NEEDS_REVIEW",
        status="COMPLETED",
        summary="Security review completed.",
        findings=[finding],
        files_scanned=10,
        files_skipped=2,
        static_analysis_completed=True,
        ai_review_completed=True,
        requires_human_review=True,
        limitations=["Dynamic analysis was not performed."],
        metadata={"scanner_version": "1.0"},
    )


# Record validation


def test_record_rejects_empty_task_id(manager):
    with pytest.raises(ValueError):
        manager.record("", make_review())


def test_record_rejects_whitespace_task_id(manager):
    with pytest.raises(ValueError):
        manager.record("   ", make_review())


def test_record_rejects_non_string_task_id(manager):
    with pytest.raises(ValueError):
        manager.record(None, make_review())


def test_record_rejects_invalid_review(manager):
    with pytest.raises(TypeError):
        manager.record("task-1", {"decision": "NEEDS_REVIEW"})


# Record creation


def test_record_returns_structured_record(manager):
    record = manager.record("task-1", make_review())

    assert record["record_id"] == 1
    assert record["task_id"] == "task-1"
    assert isinstance(record["recorded_at"], str)
    assert isinstance(record["review"], dict)


def test_record_contains_serialized_review(manager):
    record = manager.record("task-1", make_review())

    assert record["review"]["decision"] == "NEEDS_REVIEW"
    assert record["review"]["status"] == "COMPLETED"
    assert record["review"]["files_scanned"] == 10


def test_record_timestamp_is_timezone_aware(manager):
    record = manager.record("task-1", make_review())

    timestamp = datetime.fromisoformat(record["recorded_at"])

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None


def test_record_assigns_sequential_ids(manager):
    first = manager.record("task-1", make_review())
    second = manager.record("task-1", make_review())
    third = manager.record("task-2", make_review())

    assert first["record_id"] == 1
    assert second["record_id"] == 2
    assert third["record_id"] == 3


def test_record_stores_separate_tasks(manager):
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())

    assert len(manager.get_history("task-1")) == 1
    assert len(manager.get_history("task-2")) == 1


# Task-specific retrieval


def test_get_history_returns_all_task_records(manager):
    manager.record("task-1", make_review("First finding"))
    manager.record("task-1", make_review("Second finding"))

    records = manager.get_history("task-1")

    assert len(records) == 2
    assert records[0]["record_id"] == 1
    assert records[1]["record_id"] == 2


def test_get_history_returns_empty_list_for_unknown_task(manager):
    assert manager.get_history("unknown-task") == []


def test_get_latest_returns_most_recent_record(manager):
    manager.record("task-1", make_review("First finding"))
    manager.record("task-1", make_review("Second finding"))

    latest = manager.get_latest("task-1")

    assert latest is not None
    assert latest["record_id"] == 2
    assert (
        latest["review"]["findings"][0]["title"]
        == "Second finding"
    )


def test_get_latest_returns_none_for_unknown_task(manager):
    assert manager.get_latest("unknown-task") is None


def test_get_all_history_returns_records_in_id_order(manager):
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())
    manager.record("task-1", make_review())

    records = manager.get_all_history()

    assert [record["record_id"] for record in records] == [1, 2, 3]


# Defensive copying


def test_record_return_value_cannot_modify_stored_history(manager):
    record = manager.record("task-1", make_review())

    record["review"]["summary"] = "Modified externally"
    record["review"]["findings"][0]["title"] = "Modified finding"

    stored = manager.get_latest("task-1")

    assert stored["review"]["summary"] == "Security review completed."
    assert (
        stored["review"]["findings"][0]["title"]
        == "Test security finding"
    )


def test_get_history_returns_independent_copy(manager):
    manager.record("task-1", make_review())

    records = manager.get_history("task-1")
    records[0]["review"]["summary"] = "Modified externally"

    stored = manager.get_latest("task-1")

    assert stored["review"]["summary"] == "Security review completed."


def test_get_latest_returns_independent_copy(manager):
    manager.record("task-1", make_review())

    latest = manager.get_latest("task-1")
    latest["review"]["summary"] = "Modified externally"

    stored = manager.get_latest("task-1")

    assert stored["review"]["summary"] == "Security review completed."


def test_get_all_history_returns_independent_copy(manager):
    manager.record("task-1", make_review())

    records = manager.get_all_history()
    records[0]["review"]["summary"] = "Modified externally"

    stored = manager.get_latest("task-1")

    assert stored["review"]["summary"] == "Security review completed."


# Review counts


def test_get_review_count_returns_total_count(manager):
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())
    manager.record("task-1", make_review())

    assert manager.get_review_count() == 3


def test_get_review_count_returns_task_count(manager):
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())
    manager.record("task-1", make_review())

    assert manager.get_review_count("task-1") == 2
    assert manager.get_review_count("task-2") == 1


def test_get_review_count_returns_zero_for_unknown_task(manager):
    assert manager.get_review_count("unknown-task") == 0


# Clearing history


def test_clear_task_history_returns_removed_count(manager):
    manager.record("task-1", make_review())
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())

    removed = manager.clear_task_history("task-1")

    assert removed == 2
    assert manager.get_review_count("task-1") == 0
    assert manager.get_review_count("task-2") == 1


def test_clear_task_history_returns_zero_for_unknown_task(manager):
    assert manager.clear_task_history("unknown-task") == 0


def test_clear_task_history_preserves_other_tasks(manager):
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())

    manager.clear_task_history("task-1")

    assert len(manager.get_all_history()) == 1
    assert manager.get_all_history()[0]["task_id"] == "task-2"


def test_clear_all_history_returns_removed_count(manager):
    manager.record("task-1", make_review())
    manager.record("task-2", make_review())
    manager.record("task-1", make_review())

    removed = manager.clear_all_history()

    assert removed == 3
    assert manager.get_review_count() == 0
    assert manager.get_all_history() == []


def test_clear_all_history_on_empty_manager_returns_zero(manager):
    assert manager.clear_all_history() == 0


def test_record_ids_continue_after_clearing_history(manager):
    first = manager.record("task-1", make_review())

    manager.clear_all_history()

    second = manager.record("task-1", make_review())

    assert first["record_id"] == 1
    assert second["record_id"] == 2


# Thread safety


def test_concurrent_records_receive_unique_ids(manager):
    review = make_review()

    with ThreadPoolExecutor(max_workers=10) as executor:
        records = list(
            executor.map(
                lambda _: manager.record("task-1", review),
                range(50),
            )
        )

    record_ids = [record["record_id"] for record in records]

    assert len(record_ids) == 50
    assert len(set(record_ids)) == 50
    assert sorted(record_ids) == list(range(1, 51))
    assert manager.get_review_count("task-1") == 50


def test_concurrent_records_preserve_all_tasks(manager):
    review = make_review()

    task_ids = [f"task-{index % 5}" for index in range(50)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(
            executor.map(
                lambda task_id: manager.record(task_id, review),
                task_ids,
            )
        )

    assert manager.get_review_count() == 50

    for task_number in range(5):
        assert manager.get_review_count(f"task-{task_number}") == 10
