
"""
AgentOps - QA Execution Manager Tests

Tests discovery, sandbox coordination, pytest result
interpretation, cancellation, error handling, and history.
"""

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentops.models.execution import ExecutionResult
from agentops.tools.cancellation import TaskCancelledError
from agentops.tools.qa_execution_manager import QAExecutionManager


@pytest.fixture
def dependencies():
    discovery = Mock()
    sandbox = Mock()
    parser = Mock()
    history = Mock()

    manager = QAExecutionManager(
        test_discovery=discovery,
        sandbox_manager=sandbox,
        report_parser=parser,
        history_manager=history,
    )

    return manager, discovery, sandbox, parser, history


def make_discovery(
    test_files=None,
    discovery_complete=True,
    warnings=None,
    error_message=None,
):
    return SimpleNamespace(
        test_files=test_files if test_files is not None else [
            "tests/test_example.py"
        ],
        discovery_complete=discovery_complete,
        warnings=warnings or [],
        error_message=error_message,
        configuration_files=["pytest.ini"],
        test_directories=["tests"],
    )


def make_sandbox_result(
    status="PASSED",
    exit_code=0,
    stdout="1 passed in 0.10s",
    stderr="",
    duration_seconds=0.10,
    tests_executed=True,
    error_message=None,
    environment="docker",
):
    return SimpleNamespace(
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration_seconds,
        tests_executed=tests_executed,
        error_message=error_message,
        environment=environment,
        execution_environment=environment,
    )


def make_parsed_report(
    total=1,
    passed=1,
    failed=0,
    errors=0,
    skipped=0,
    collection_errors=0,
    no_tests_collected=False,
    successful=True,
    warnings=None,
):
    return SimpleNamespace(
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        collection_errors=collection_errors,
        no_tests_collected=no_tests_collected,
        successful=successful,
        has_failures=failed > 0 or errors > 0,
        warnings=warnings or [],
    )


def configure_successful_execution(
    discovery,
    sandbox,
    parser,
    workspace,
):
    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result()
    parser.parse.return_value = make_parsed_report()

    return workspace


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_example.py").write_text(
        "def test_example():\n    assert True\n",
        encoding="utf-8",
    )
    return tmp_path


# Dependency initialization


def test_manager_initializes_default_dependencies():
    manager = QAExecutionManager()

    assert manager.test_discovery is not None
    assert manager.sandbox_manager is not None
    assert manager.report_parser is not None
    assert manager.history_manager is not None


def test_manager_accepts_injected_dependencies(dependencies):
    manager, discovery, sandbox, parser, history = dependencies

    assert manager.test_discovery is discovery
    assert manager.sandbox_manager is sandbox
    assert manager.report_parser is parser
    assert manager.history_manager is history


# History recording


def test_record_result_stores_execution_history(dependencies):
    manager, _, _, _, history = dependencies

    result = ExecutionResult(status="PASSED")

    returned = manager._record_result("task-1", result)

    history.record.assert_called_once_with(
        task_id="task-1",
        result=result,
    )
    assert returned is result


# Workspace validation


def test_execute_rejects_missing_workspace(dependencies, tmp_path):
    manager, discovery, sandbox, _, history = dependencies

    missing = tmp_path / "missing"

    result = manager.execute("task-1", missing)

    assert result.status == "INVALID_WORKSPACE"
    assert "does not exist" in result.error_message
    assert result.metadata["discovery_complete"] is False
    discovery.discover.assert_not_called()
    sandbox.run_tests.assert_not_called()
    history.record.assert_called_once()


def test_execute_rejects_file_as_workspace(dependencies, tmp_path):
    manager, discovery, sandbox, _, _ = dependencies

    file_path = tmp_path / "not_a_directory"
    file_path.write_text("not a directory", encoding="utf-8")

    result = manager.execute("task-1", file_path)

    assert result.status == "INVALID_WORKSPACE"
    discovery.discover.assert_not_called()
    sandbox.run_tests.assert_not_called()


def test_execute_rejects_unresolvable_workspace(dependencies):
    manager, discovery, sandbox, _, _ = dependencies

    result = manager.execute("task-1", None)

    assert result.status == "INVALID_WORKSPACE"
    discovery.discover.assert_not_called()
    sandbox.run_tests.assert_not_called()


# Test discovery


def test_execute_runs_test_discovery(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    configure_successful_execution(
        discovery,
        sandbox,
        parser,
        workspace,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "PASSED"
    discovery.discover.assert_called_once_with(workspace)


def test_execute_passes_workspace_to_sandbox(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    configure_successful_execution(
        discovery,
        sandbox,
        parser,
        workspace,
    )

    manager.execute("task-1", workspace)

    sandbox.run_tests.assert_called_once()
    assert sandbox.run_tests.call_args.kwargs["workspace_path"] == workspace


def test_execute_passes_cancellation_event_to_sandbox(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    configure_successful_execution(
        discovery,
        sandbox,
        parser,
        workspace,
    )

    cancel_event = Event()

    manager.execute(
        "task-1",
        workspace,
        cancel_event=cancel_event,
    )

    assert (
        sandbox.run_tests.call_args.kwargs["cancel_event"]
        is cancel_event
    )


def test_execute_records_discovery_metadata(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery(
        warnings=["Discovery warning"]
    )
    sandbox.run_tests.return_value = make_sandbox_result()
    parser.parse.return_value = make_parsed_report()

    result = manager.execute("task-1", workspace)

    assert result.metadata["discovery_complete"] is True
    assert result.metadata["discovered_test_files"] == [
        "tests/test_example.py"
    ]
    assert result.metadata["tests_discovered"] == 1
    assert result.metadata["configuration_files"] == ["pytest.ini"]
    assert result.metadata["test_directories"] == ["tests"]
    assert result.metadata["discovery_warnings"] == [
        "Discovery warning"
    ]


def test_execute_records_retry_count(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    configure_successful_execution(
        discovery,
        sandbox,
        parser,
        workspace,
    )

    result = manager.execute(
        "task-1",
        workspace,
        retry_count=2,
    )

    assert result.metadata["retry_count"] == 2


def test_execute_handles_discovery_exception(dependencies, workspace):
    manager, discovery, sandbox, _, _ = dependencies

    discovery.discover.side_effect = RuntimeError("Discovery failed")

    result = manager.execute("task-1", workspace)

    assert result.status == "EXECUTION_ERROR"
    assert "Discovery failed" in result.error_message
    assert result.metadata["discovery_complete"] is False
    assert result.metadata["discovery_error"] == "Discovery failed"
    sandbox.run_tests.assert_not_called()


def test_execute_handles_discovery_reported_error(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, _, _ = dependencies

    discovery.discover.return_value = make_discovery(
        error_message="Unable to read test directory"
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "EXECUTION_ERROR"
    assert "Unable to read test directory" in result.error_message
    sandbox.run_tests.assert_not_called()


def test_execute_returns_no_tests_when_discovery_is_complete(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, _, _ = dependencies

    discovery.discover.return_value = make_discovery(
        test_files=[],
        discovery_complete=True,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "NO_TESTS_FOUND"
    assert result.metadata["discovery_complete"] is True
    sandbox.run_tests.assert_not_called()


def test_execute_does_not_assume_no_tests_when_discovery_incomplete(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, _, _ = dependencies

    discovery.discover.return_value = make_discovery(
        test_files=[],
        discovery_complete=False,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "EXECUTION_ERROR"
    assert "incomplete" in result.error_message.lower()
    sandbox.run_tests.assert_not_called()


# Cancellation


def test_execute_records_cancellation_before_discovery(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, _, history = dependencies

    cancel_event = Event()
    cancel_event.set()

    result = manager.execute(
        "task-1",
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.metadata["cancellation_phase"] == (
        "before test discovery"
    )
    discovery.discover.assert_not_called()
    sandbox.run_tests.assert_not_called()
    history.record.assert_called_once()


def test_execute_records_cancellation_during_discovery(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, _, _ = dependencies

    def cancel_during_discovery(path):
        cancel_event.set()
        return make_discovery()

    cancel_event = Event()
    discovery.discover.side_effect = cancel_during_discovery

    result = manager.execute(
        "task-1",
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.metadata["cancellation_phase"] == "test discovery"
    sandbox.run_tests.assert_not_called()


def test_execute_records_cancellation_before_sandbox(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.side_effect = lambda path: (
        cancel_event.set() or make_discovery()
    )
    cancel_event = Event()

    result = manager.execute(
        "task-1",
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.metadata["cancellation_phase"] == (
        "test discovery"
    )
    sandbox.run_tests.assert_not_called()


def test_execute_preserves_sandbox_cancellation_result(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        status="CANCELLED",
        exit_code=None,
        stdout="",
        stderr="Execution cancelled",
        tests_executed=False,
        error_message="Container stopped after cancellation.",
    )
    parser.parse.return_value = make_parsed_report(
        total=0,
        passed=0,
        successful=False,
        no_tests_collected=True,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "CANCELLED"
    assert result.metadata["sandbox_status"] == "CANCELLED"


def test_execute_handles_sandbox_cancellation_exception(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, _, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.side_effect = TaskCancelledError()

    result = manager.execute("task-1", workspace)

    assert result.status == "CANCELLED"
    assert result.metadata["cancellation_phase"] == "sandbox execution"


# Sandbox errors


def test_execute_handles_sandbox_exception(dependencies, workspace):
    manager, discovery, sandbox, _, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.side_effect = RuntimeError("Docker failed")

    result = manager.execute("task-1", workspace)

    assert result.status == "EXECUTION_ERROR"
    assert "Docker failed" in result.error_message
    assert result.metadata["sandbox_status"] == "EXECUTION_ERROR"
    assert result.metadata["execution_environment"] == "docker"


@pytest.mark.parametrize(
    ("sandbox_status", "expected_status"),
    [
        ("DOCKER_UNAVAILABLE", "DOCKER_UNAVAILABLE"),
        ("IMAGE_NOT_FOUND", "IMAGE_NOT_FOUND"),
        ("INVALID_WORKSPACE", "INVALID_WORKSPACE"),
        ("SANDBOX_UNAVAILABLE", "SANDBOX_UNAVAILABLE"),
        ("EXECUTION_ERROR", "EXECUTION_ERROR"),
        ("TIMEOUT", "TIMEOUT"),
        ("CANCELLED", "CANCELLED"),
    ],
)
def test_execute_preserves_sandbox_status(
    dependencies,
    workspace,
    sandbox_status,
    expected_status,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        status=sandbox_status,
        exit_code=None,
        stdout="",
        stderr="Sandbox output",
        tests_executed=False,
        error_message="Sandbox did not complete.",
    )
    parser.parse.return_value = make_parsed_report(
        total=0,
        passed=0,
        successful=False,
        no_tests_collected=True,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == expected_status


# Pytest report parsing


def test_execute_passes_combined_output_to_parser(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        stdout="stdout content",
        stderr="stderr content",
    )
    parser.parse.return_value = make_parsed_report()

    manager.execute("task-1", workspace)

    parser.parse.assert_called_once_with(
        "stdout content\nstderr content",
        exit_code=0,
    )


def test_execute_records_test_case_counts(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery(
        test_files=[
            "tests/test_one.py",
            "tests/test_two.py",
        ]
    )
    sandbox.run_tests.return_value = make_sandbox_result()
    parser.parse.return_value = make_parsed_report(
        total=8,
        passed=5,
        failed=1,
        errors=1,
        skipped=1,
        successful=False,
    )

    result = manager.execute("task-1", workspace)

    assert result.test_count == 8
    assert result.metadata["tests_discovered"] == 2
    assert result.metadata["test_cases_reported"] == 8
    assert result.metadata["tests_passed"] == 5
    assert result.metadata["tests_failed"] == 2
    assert result.metadata["tests_skipped"] == 1


def test_execute_preserves_collection_errors(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        status="FAILED",
        exit_code=2,
    )
    parser.parse.return_value = make_parsed_report(
        total=0,
        passed=0,
        errors=1,
        collection_errors=1,
        successful=False,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "FAILED"
    assert result.metadata["collection_errors"] == 1
    assert "collection errors" in result.error_message.lower()


def test_execute_handles_no_tests_collected(dependencies, workspace):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        status="NO_TESTS_FOUND",
        exit_code=5,
    )
    parser.parse.return_value = make_parsed_report(
        total=0,
        passed=0,
        successful=False,
        no_tests_collected=True,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "NO_TESTS_FOUND"


def test_execute_handles_report_parser_exception(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result()
    parser.parse.side_effect = ValueError("Malformed pytest output")

    result = manager.execute("task-1", workspace)

    assert result.status == "EXECUTION_ERROR"
    assert "Malformed pytest output" in result.error_message
    assert result.metadata["report_parsing_error"] == (
        "Malformed pytest output"
    )


# Incomplete discovery


def test_incomplete_discovery_prevents_overall_pass(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery(
        discovery_complete=False
    )
    sandbox.run_tests.return_value = make_sandbox_result()
    parser.parse.return_value = make_parsed_report()

    result = manager.execute("task-1", workspace)

    assert result.status == "EXECUTION_ERROR"
    assert result.metadata["pytest_report_status"] == "PASSED"
    assert "incomplete" in result.error_message.lower()


def test_incomplete_discovery_with_failures_remains_failed(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery(
        discovery_complete=False
    )
    sandbox.run_tests.return_value = make_sandbox_result(
        status="FAILED",
        exit_code=1,
    )
    parser.parse.return_value = make_parsed_report(
        passed=0,
        failed=1,
        successful=False,
    )

    result = manager.execute("task-1", workspace)

    assert result.status == "FAILED"


# Execution evidence and cleanup


def test_execute_preserves_execution_environment(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        environment="docker"
    )
    parser.parse.return_value = make_parsed_report()

    result = manager.execute("task-1", workspace)

    assert result.environment == "docker"
    assert result.metadata["execution_environment"] == "docker"


def test_execute_extracts_container_cleanup_warning(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result(
        error_message=(
            "Test execution completed. "
            "Container cleanup warning: Container removal failed."
        )
    )
    parser.parse.return_value = make_parsed_report()

    result = manager.execute("task-1", workspace)

    assert result.metadata["container_cleanup_warning"] == (
        "Container removal failed."
    )
    assert result.metadata["container_cleanup_status"] == "WARNING"


def test_execute_reports_no_cleanup_warning(
    dependencies,
    workspace,
):
    manager, discovery, sandbox, parser, _ = dependencies

    discovery.discover.return_value = make_discovery()
    sandbox.run_tests.return_value = make_sandbox_result()
    parser.parse.return_value = make_parsed_report()

    result = manager.execute("task-1", workspace)

    assert result.metadata["container_cleanup_warning"] is None
    assert result.metadata["container_cleanup_status"] == (
        "NO_WARNING_REPORTED"
    )


# Status determination unit tests


@pytest.mark.parametrize(
    ("sandbox_status", "exit_code", "expected"),
    [
        ("CANCELLED", None, "CANCELLED"),
        ("TIMEOUT", None, "TIMEOUT"),
        ("DOCKER_UNAVAILABLE", None, "DOCKER_UNAVAILABLE"),
        ("IMAGE_NOT_FOUND", None, "IMAGE_NOT_FOUND"),
        ("INVALID_WORKSPACE", None, "INVALID_WORKSPACE"),
        ("SANDBOX_UNAVAILABLE", None, "SANDBOX_UNAVAILABLE"),
        ("EXECUTION_ERROR", None, "EXECUTION_ERROR"),
    ],
)
def test_determine_status_prioritizes_sandbox_status(
    sandbox_status,
    exit_code,
    expected,
):
    parsed = make_parsed_report(
        failed=1,
        successful=False,
    )

    result = QAExecutionManager._determine_status(
        sandbox_status=sandbox_status,
        parsed_report=parsed,
        exit_code=exit_code,
    )

    assert result == expected


def test_determine_status_prioritizes_collection_errors():
    parsed = make_parsed_report(
        collection_errors=1,
        no_tests_collected=True,
        successful=False,
    )

    result = QAExecutionManager._determine_status(
        sandbox_status="COMPLETED",
        parsed_report=parsed,
        exit_code=5,
    )

    assert result == "FAILED"


def test_determine_status_marks_successful_tests_passed():
    parsed = make_parsed_report()

    result = QAExecutionManager._determine_status(
        sandbox_status="PASSED",
        parsed_report=parsed,
        exit_code=0,
    )

    assert result == "PASSED"


def test_determine_status_rejects_pass_when_discovery_incomplete():
    parsed = make_parsed_report()

    result = QAExecutionManager._determine_status(
        sandbox_status="PASSED",
        parsed_report=parsed,
        exit_code=0,
        discovery_complete=False,
    )

    assert result == "EXECUTION_ERROR"


def test_determine_status_returns_failed_for_failed_tests():
    parsed = make_parsed_report(
        passed=0,
        failed=1,
        successful=False,
    )

    result = QAExecutionManager._determine_status(
        sandbox_status="FAILED",
        parsed_report=parsed,
        exit_code=1,
    )

    assert result == "FAILED"


# Report error explanation


@pytest.mark.parametrize(
    ("status", "parsed", "expected_text"),
    [
        ("PASSED", make_parsed_report(), None),
        (
            "NO_TESTS_FOUND",
            make_parsed_report(),
            "did not collect",
        ),
        (
            "FAILED",
            make_parsed_report(
                passed=0,
                failed=1,
                successful=False,
            ),
            "failed test cases",
        ),
        ("TIMEOUT", make_parsed_report(), "timed out"),
        ("CANCELLED", make_parsed_report(), "cancelled"),
        (
            "DOCKER_UNAVAILABLE",
            make_parsed_report(),
            "environment problem",
        ),
        ("UNKNOWN", make_parsed_report(), "sufficient evidence"),
    ],
)
def test_report_error_explains_execution_status(
    status,
    parsed,
    expected_text,
):
    message = QAExecutionManager._report_error(
        parsed,
        status,
    )

    if expected_text is None:
        assert message is None
    else:
        assert expected_text in message.lower()


def test_report_error_explains_incomplete_discovery():
    parsed = make_parsed_report()

    message = QAExecutionManager._report_error(
        parsed,
        "EXECUTION_ERROR",
        discovery_complete=False,
    )

    assert "incomplete" in message.lower()


# Cleanup warning extraction


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (None, None),
        ("Execution completed.", None),
        (
            "Container cleanup warning: removal failed",
            "removal failed",
        ),
        (
            "Execution failed. Container cleanup warning: "
            "container still running.",
            "container still running.",
        ),
    ],
)
def test_extract_cleanup_warning(message, expected):
    assert (
        QAExecutionManager._extract_cleanup_warning(message)
        == expected
    )
