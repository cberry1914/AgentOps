
"""
AgentOps - Security Review Service Tests

Tests pipeline coordination, validation, finding aggregation,
policy decisions, reporting, history, limitations, and cancellation.
"""

from threading import Event
from unittest.mock import Mock

import pytest

from agentops.models.security_review import (
    SecurityFinding,
    SecurityReview,
)
from agentops.tools.security_review_service import (
    SecurityReviewService,
    SecurityReviewServiceResult,
)


@pytest.fixture
def service():
    return SecurityReviewService()


def make_finding(
    title="Test finding",
    severity="HIGH",
    source="static_analysis",
):
    return SecurityFinding(
        severity=severity,
        category="Authentication",
        title=title,
        description="Authentication configuration may be insecure.",
        file_path="src/auth.py",
        line_number=25,
        evidence="authentication_enabled = False",
        recommendation="Review authentication configuration.",
        confidence="HIGH",
        source=source,
    )


def valid_ai_response():
    return """
    {
        "summary": "AI security review completed.",
        "findings": [
            {
                "severity": "MEDIUM",
                "category": "Configuration",
                "title": "Missing security header",
                "description": "A recommended security header is missing.",
                "file_path": "src/app.py",
                "line_number": 12,
                "evidence": "No security header configuration found.",
                "recommendation": "Configure the recommended header.",
                "confidence": "MEDIUM"
            }
        ],
        "limitations": []
    }
    """


# Result model


def test_service_result_has_independent_default_lists():
    first = SecurityReviewServiceResult(
        review=Mock(spec=SecurityReview),
        report="Report",
    )
    second = SecurityReviewServiceResult(
        review=Mock(spec=SecurityReview),
        report="Report",
    )

    first.static_findings.append(make_finding())

    assert second.static_findings == []


# Dependency initialization


def test_service_initializes_default_dependencies(service):
    assert service.scanner is not None
    assert service.parser is not None
    assert service.policy_engine is not None
    assert service.report_formatter is not None
    assert service.history_manager is not None


def test_service_accepts_injected_dependencies():
    scanner = Mock()
    parser = Mock()
    policy = Mock()
    formatter = Mock()
    history = Mock()

    service = SecurityReviewService(
        scanner=scanner,
        parser=parser,
        policy_engine=policy,
        report_formatter=formatter,
        history_manager=history,
    )

    assert service.scanner is scanner
    assert service.parser is parser
    assert service.policy_engine is policy
    assert service.report_formatter is formatter
    assert service.history_manager is history


# Input validation


@pytest.mark.parametrize("task_id", ["", "   ", None, 123])
def test_review_rejects_invalid_task_id(service, task_id):
    with pytest.raises(ValueError):
        service.review(
            task_id=task_id,
            workspace_files={},
            ai_response="{}",
        )


@pytest.mark.parametrize("workspace_files", [None, [], "files"])
def test_review_rejects_non_dictionary_workspace(service, workspace_files):
    with pytest.raises(TypeError):
        service.review(
            task_id="task-1",
            workspace_files=workspace_files,
            ai_response="{}",
        )


@pytest.mark.parametrize("ai_response", [None, {}, 123])
def test_review_rejects_non_string_ai_response(service, ai_response):
    with pytest.raises(TypeError):
        service.review(
            task_id="task-1",
            workspace_files={},
            ai_response=ai_response,
        )


@pytest.mark.parametrize("files_skipped", [True, False, "1", 1.5, None])
def test_review_rejects_non_integer_files_skipped(service, files_skipped):
    with pytest.raises(TypeError):
        service.review(
            task_id="task-1",
            workspace_files={},
            ai_response="{}",
            files_skipped=files_skipped,
        )


def test_review_rejects_negative_files_skipped(service):
    with pytest.raises(ValueError):
        service.review(
            task_id="task-1",
            workspace_files={},
            ai_response="{}",
            files_skipped=-1,
        )


@pytest.mark.parametrize(
    "workspace_files",
    [
        {1: "content"},
        {"src/app.py": None},
        {"src/app.py": 123},
    ],
)
def test_review_rejects_invalid_workspace_entries(service, workspace_files):
    with pytest.raises(TypeError):
        service.review(
            task_id="task-1",
            workspace_files=workspace_files,
            ai_response="{}",
        )


# Cancellation


def test_review_stops_when_already_cancelled(service):
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(Exception) as exc_info:
        service.review(
            task_id="task-1",
            workspace_files={},
            ai_response="{}",
            cancel_event=cancel_event,
        )

    assert type(exc_info.value).__name__ == "TaskCancelledError"
    assert service.history_manager.get_review_count() == 0


def test_review_checks_cancellation_before_scanning():
    scanner = Mock()
    service = SecurityReviewService(scanner=scanner)

    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(Exception):
        service.review(
            task_id="task-1",
            workspace_files={},
            ai_response="{}",
            cancel_event=cancel_event,
        )

    scanner.scan_files.assert_not_called()


# Pipeline execution


def test_review_returns_service_result(service):
    result = service.review(
        task_id="task-1",
        workspace_files={"src/app.py": "print('hello')"},
        ai_response=valid_ai_response(),
    )

    assert isinstance(result, SecurityReviewServiceResult)
    assert isinstance(result.review, SecurityReview)
    assert isinstance(result.report, str)


def test_review_runs_static_scanner(service):
    result = service.review(
        task_id="task-1",
        workspace_files={
            "src/auth.py": 'database_password = "SuperSecretPassword123"'
        },
        ai_response='{"summary": "Reviewed", "findings": []}',
    )

    assert len(result.static_findings) >= 1
    assert any(
        finding.source == "static_analysis"
        for finding in result.static_findings
    )


def test_review_parses_ai_findings(service):
    result = service.review(
        task_id="task-1",
        workspace_files={},
        ai_response=valid_ai_response(),
    )

    assert len(result.ai_findings) == 1
    assert result.ai_findings[0].title == "Missing security header"


def test_review_combines_static_and_ai_findings(service):
    result = service.review(
        task_id="task-1",
        workspace_files={
            "src/auth.py": 'database_password = "SuperSecretPassword123"'
        },
        ai_response=valid_ai_response(),
    )

    combined = result.review.findings

    assert len(combined) == (
        len(result.static_findings) + len(result.ai_findings)
    )


def test_review_preserves_both_finding_sources(service):
    result = service.review(
        task_id="task-1",
        workspace_files={
            "src/auth.py": 'database_password = "SuperSecretPassword123"'
        },
        ai_response=valid_ai_response(),
    )

    sources = {finding.source for finding in result.review.findings}

    assert "static_analysis" in sources
    assert "ai_review" in sources


def test_review_reports_completed_status(service):
    result = service.review(
        task_id="task-1",
        workspace_files={"src/app.py": "print('hello')"},
        ai_response=valid_ai_response(),
    )

    assert result.review.status == "COMPLETED"


def test_review_reports_partial_status_for_invalid_ai_json(service):
    result = service.review(
        task_id="task-1",
        workspace_files={"src/app.py": "print('hello')"},
        ai_response="This is not valid JSON.",
    )

    assert result.review.status == "PARTIAL"
    assert result.review.ai_review_completed is False


def test_review_records_ai_parsing_warnings(service):
    result = service.review(
        task_id="task-1",
        workspace_files={},
        ai_response="This is not valid JSON.",
    )

    assert result.parsing_warnings
    assert result.review.limitations


# File limits and limitations


def test_review_accounts_for_files_skipped(service):
    result = service.review(
        task_id="task-1",
        workspace_files={
            "src/app.py": "print('hello')",
            "src/auth.py": "print('hello')",
        },
        ai_response='{"summary": "Reviewed", "findings": []}',
        files_skipped=3,
    )

    assert result.review.files_scanned == 2
    assert result.review.files_skipped == 3


def test_review_deduplicates_limitations(service):
    result = service.review(
        task_id="task-1",
        workspace_files={},
        ai_response='''{
            "summary": "Reviewed",
            "findings": [],
            "limitations": ["Dynamic analysis was not performed."]
        }''',
        limitations=[
            "Dynamic analysis was not performed.",
            "Manual review is required.",
            "Manual review is required.",
        ],
    )

    assert result.review.limitations.count(
        "Dynamic analysis was not performed."
    ) == 1

    assert result.review.limitations.count(
        "Manual review is required."
    ) == 1


def test_review_ignores_invalid_limitation_values(service):
    result = service.review(
        task_id="task-1",
        workspace_files={},
        ai_response='{"summary": "Reviewed", "findings": []}',
        limitations=[
            "Valid limitation",
            "",
            "   ",
            None,
            123,
        ],
    )

    assert "Valid limitation" in result.review.limitations
    assert "" not in result.review.limitations
    assert "   " not in result.review.limitations


def test_review_uses_scanner_file_limit():
    scanner = Mock()
    scanner.MAX_FILES = 1
    scanner.scan_files.return_value = []

    service = SecurityReviewService(scanner=scanner)

    result = service.review(
        task_id="task-1",
        workspace_files={
            "src/one.py": "print(1)",
            "src/two.py": "print(2)",
        },
        ai_response='{"summary": "Reviewed", "findings": []}',
    )

    assert result.review.files_scanned == 1
    assert result.review.files_skipped == 1
    assert result.review.status == "PARTIAL"
    assert any(
        "scanner file limit" in limitation.lower()
        for limitation in result.review.limitations
    )


# Policy and reporting integration


def test_review_uses_policy_engine():
    policy = Mock()
    expected_review = Mock(spec=SecurityReview)
    policy.evaluate.return_value = expected_review

    formatter = Mock()
    formatter.format.return_value = "Formatted report"

    service = SecurityReviewService(
        policy_engine=policy,
        report_formatter=formatter,
    )

    result = service.review(
        task_id="task-1",
        workspace_files={},
        ai_response='{"summary": "Reviewed", "findings": []}',
    )

    assert result.review is expected_review
    policy.evaluate.assert_called_once()
    formatter.format.assert_called_once_with(expected_review)
    assert result.report == "Formatted report"


def test_review_passes_combined_findings_to_policy(service):
    result = service.review(
        task_id="task-1",
        workspace_files={
            "src/auth.py": 'database_password = "SuperSecretPassword123"'
        },
        ai_response=valid_ai_response(),
    )

    assert result.review.metadata["combined_findings"] == len(
        result.review.findings
    )


# History integration


def test_review_records_history(service):
    result = service.review(
        task_id="task-1",
        workspace_files={"src/app.py": "print('hello')"},
        ai_response=valid_ai_response(),
    )

    assert result.history_record is not None
    assert result.history_record["task_id"] == "task-1"
    assert result.history_record["record_id"] == 1


def test_review_history_can_be_retrieved(service):
    service.review(
        task_id="task-1",
        workspace_files={},
        ai_response=valid_ai_response(),
    )

    history = service.get_review_history("task-1")

    assert len(history) == 1
    assert history[0]["task_id"] == "task-1"


def test_latest_review_can_be_retrieved(service):
    service.review(
        task_id="task-1",
        workspace_files={},
        ai_response='{"summary": "First review", "findings": []}',
    )
    service.review(
        task_id="task-1",
        workspace_files={},
        ai_response='{"summary": "Second review", "findings": []}',
    )

    latest = service.get_latest_review("task-1")

    assert latest is not None
    assert latest["record_id"] == 2
