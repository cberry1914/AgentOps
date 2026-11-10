
"""
AgentOps - QA Execution Manager

Coordinates test discovery, Docker sandbox execution,
pytest report parsing, and execution history.

Responsibilities:
- Discover supported pytest files.
- Execute tests through the restricted Docker sandbox.
- Parse pytest output into structured evidence.
- Preserve cancellation and timeout results.
- Record execution and container cleanup information.
- Maintain accurate distinctions between test files
  and individual test cases.
- Preserve test collection errors as independent evidence.
- Track incomplete test discovery accurately.

This component never treats static review as a
substitute for actual automated test execution.
"""

from pathlib import Path
from threading import Event
from typing import Optional

from agentops.models.execution import ExecutionResult
from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)
from agentops.tools.execution_history import ExecutionHistoryManager
from agentops.tools.sandbox_manager import SandboxManager
from agentops.tools.test_discovery import TestDiscovery
from agentops.tools.test_report_parser import TestReportParser


class QAExecutionManager:
    """Manage the automated test execution lifecycle."""

    INFRASTRUCTURE_STATUSES = {
        "DOCKER_UNAVAILABLE",
        "IMAGE_NOT_FOUND",
        "INVALID_WORKSPACE",
        "SANDBOX_UNAVAILABLE",
        "EXECUTION_ERROR",
    }

    def __init__(
        self,
        test_discovery: Optional[TestDiscovery] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        report_parser: Optional[TestReportParser] = None,
        history_manager: Optional[ExecutionHistoryManager] = None,
    ) -> None:
        self.test_discovery = (
            test_discovery or TestDiscovery()
        )

        self.sandbox_manager = (
            sandbox_manager or SandboxManager()
        )

        self.report_parser = (
            report_parser or TestReportParser()
        )

        self.history_manager = (
            history_manager or ExecutionHistoryManager()
        )

    # ---------------------------------------------------------
    # MAIN EXECUTION
    # ---------------------------------------------------------

    def execute(
        self,
        task_id: str,
        workspace_path: str | Path,
        retry_count: int = 0,
        cancel_event: Optional[Event] = None,
    ) -> ExecutionResult:
        """
        Discover and execute tests for one task workspace.

        Cancellation is recorded as a structured execution result
        at every phase rather than escaping without history.

        Cancellation returned by the sandbox is preserved rather
        than discarded by raising another cancellation exception.
        """

        try:
            check_cancelled(cancel_event)

        except TaskCancelledError:
            return self._record_cancelled_result(
                task_id=task_id,
                retry_count=retry_count,
                phase="before test discovery",
            )

        try:
            workspace = Path(workspace_path).resolve()

        except (OSError, TypeError, ValueError) as exc:
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="INVALID_WORKSPACE",
                    error_message=(
                        "Unable to resolve the task workspace: "
                        f"{exc}"
                    ),
                    metadata={
                        "retry_count": retry_count,
                        "discovery_complete": False,
                    },
                ),
            )

        if not workspace.exists() or not workspace.is_dir():
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="INVALID_WORKSPACE",
                    error_message=(
                        "The task workspace does not exist "
                        "or is not a directory."
                    ),
                    metadata={
                        "retry_count": retry_count,
                        "discovery_complete": False,
                    },
                ),
            )

        # -----------------------------------------------------
        # Test discovery
        # -----------------------------------------------------

        try:
            discovery = self.test_discovery.discover(workspace)

            check_cancelled(cancel_event)

        except TaskCancelledError:
            return self._record_cancelled_result(
                task_id=task_id,
                retry_count=retry_count,
                phase="test discovery",
            )

        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                return self._record_cancelled_result(
                    task_id=task_id,
                    retry_count=retry_count,
                    phase="test discovery",
                )

            return self._record_result(
                task_id,
                ExecutionResult(
                    status="EXECUTION_ERROR",
                    error_message=(
                        "Test discovery failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    metadata={
                        "retry_count": retry_count,
                        "discovery_complete": False,
                        "discovered_test_files": [],
                        "tests_discovered": 0,
                        "discovery_error": str(exc),
                    },
                ),
            )

        discovery_complete = bool(
            discovery.discovery_complete
        )

        discovery_warnings = list(
            discovery.warnings or []
        )

        discovered_test_files = list(
            discovery.test_files or []
        )

        discovery_metadata = {
            "discovery_complete": discovery_complete,
            "discovered_test_files": discovered_test_files,
            "tests_discovered": len(discovered_test_files),
            "configuration_files": list(
                discovery.configuration_files or []
            ),
            "test_directories": list(
                discovery.test_directories or []
            ),
            "discovery_warnings": discovery_warnings,
        }

        if discovery.error_message:
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="EXECUTION_ERROR",
                    error_message=(
                        "Test discovery reported an error: "
                        f"{discovery.error_message}"
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                        "discovery_error": discovery.error_message,
                    },
                ),
            )

        # An incomplete scan cannot establish that no tests exist.
        if not discovered_test_files and not discovery_complete:
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="EXECUTION_ERROR",
                    error_message=(
                        "Test discovery was incomplete and did not "
                        "identify any test files. AgentOps cannot "
                        "confirm that the workspace contains no tests."
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                    },
                ),
            )

        if not discovered_test_files:
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="NO_TESTS_FOUND",
                    error_message=(
                        "No supported pytest test files "
                        "were discovered in the workspace."
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                    },
                ),
            )

        # -----------------------------------------------------
        # Docker sandbox execution
        # -----------------------------------------------------

        try:
            check_cancelled(cancel_event)

        except TaskCancelledError:
            return self._record_cancelled_result(
                task_id=task_id,
                retry_count=retry_count,
                phase="before sandbox execution",
                additional_metadata=discovery_metadata,
            )

        try:
            sandbox_result = self.sandbox_manager.run_tests(
                workspace_path=workspace,
                cancel_event=cancel_event,
            )

        except TaskCancelledError:
            return self._record_cancelled_result(
                task_id=task_id,
                retry_count=retry_count,
                phase="sandbox execution",
                additional_metadata=discovery_metadata,
            )

        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                return self._record_cancelled_result(
                    task_id=task_id,
                    retry_count=retry_count,
                    phase="sandbox execution",
                    additional_metadata=discovery_metadata,
                )

            return self._record_result(
                task_id,
                ExecutionResult(
                    status="EXECUTION_ERROR",
                    error_message=(
                        "Sandbox execution raised an unexpected "
                        f"exception: {type(exc).__name__}: {exc}"
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                        "sandbox_exception": str(exc),
                        "sandbox_status": "EXECUTION_ERROR",
                        "execution_environment": "docker",
                    },
                ),
            )

        # -----------------------------------------------------
        # Pytest report parsing
        # -----------------------------------------------------

        combined_output = "\n".join(
            part
            for part in (
                sandbox_result.stdout,
                sandbox_result.stderr,
            )
            if part
        )

        try:
            parsed_report = self.report_parser.parse(
                combined_output,
                exit_code=sandbox_result.exit_code,
            )

        except Exception as exc:
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="EXECUTION_ERROR",
                    exit_code=sandbox_result.exit_code,
                    duration_seconds=sandbox_result.duration_seconds,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    environment=(
                        getattr(sandbox_result, "environment", None)
                        or getattr(
                            sandbox_result,
                            "execution_environment",
                            "docker",
                        )
                    ),
                    tests_executed=sandbox_result.tests_executed,
                    error_message=(
                        "Unable to parse pytest execution output: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                        "sandbox_status": sandbox_result.status,
                        "report_parsing_error": str(exc),
                    },
                ),
            )

        # -----------------------------------------------------
        # Interpret execution evidence
        # -----------------------------------------------------

        status = self._determine_status(
            sandbox_status=sandbox_result.status,
            parsed_report=parsed_report,
            exit_code=sandbox_result.exit_code,
            discovery_complete=discovery_complete,
        )

        # Individual test cases are counted from pytest's report.
        # Discovered test files are recorded separately.
        test_count = parsed_report.total

        tests_passed = parsed_report.passed
        tests_failed = (
            parsed_report.failed + parsed_report.errors
        )
        tests_skipped = parsed_report.skipped

        sandbox_error = sandbox_result.error_message or None

        cleanup_warning = self._extract_cleanup_warning(
            sandbox_error
        )

        execution_environment = (
            getattr(
                sandbox_result,
                "environment",
                None,
            )
            or getattr(
                sandbox_result,
                "execution_environment",
                "docker",
            )
        )

        report_error = self._report_error(
            parsed_report,
            status,
            discovery_complete=discovery_complete,
        )

        # Preserve pytest's actual outcome independently from
        # the overall execution status.
        pytest_report_status = (
            "PASSED"
            if (
                parsed_report.successful
                and sandbox_result.exit_code == 0
            )
            else (
                "FAILED"
                if parsed_report.has_failures
                else "INCOMPLETE"
            )
        )

        execution_result = ExecutionResult(
            status=status,
            exit_code=sandbox_result.exit_code,
            duration_seconds=sandbox_result.duration_seconds,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            environment=execution_environment,
            tests_executed=sandbox_result.tests_executed,
            test_count=test_count,
            error_message=(
                sandbox_error
                or report_error
            ),
            metadata={
                "retry_count": retry_count,

                **discovery_metadata,

                # Test files and test cases are different metrics.
                "test_cases_reported": test_count,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,

                # Preserve the actual pytest outcome even when
                # incomplete discovery prevents overall approval.
                "pytest_report_status": pytest_report_status,

                # Collection errors are independent evidence.
                "collection_errors": (
                    parsed_report.collection_errors
                ),

                "warnings": parsed_report.warnings,
                "no_tests_collected": (
                    parsed_report.no_tests_collected
                ),

                "sandbox_status": sandbox_result.status,
                "execution_environment": execution_environment,

                "container_cleanup_warning": cleanup_warning,
                "container_cleanup_status": (
                    "WARNING"
                    if cleanup_warning
                    else "NO_WARNING_REPORTED"
                ),
            },
        )

        return self._record_result(
            task_id,
            execution_result,
        )

    # ---------------------------------------------------------
    # STATUS DETERMINATION
    # ---------------------------------------------------------

    @classmethod
    def _determine_status(
        cls,
        sandbox_status: str,
        parsed_report,
        exit_code: Optional[int],
        discovery_complete: bool = True,
    ) -> str:
        """
        Map sandbox and pytest evidence to an execution status.

        Infrastructure and interruption statuses take priority.

        Collection errors are checked before the no-tests condition.

        A passing pytest report cannot produce an overall PASSED
        result if the separate workspace discovery was incomplete.
        """

        if sandbox_status == "CANCELLED":
            return "CANCELLED"

        if sandbox_status == "TIMEOUT":
            return "TIMEOUT"

        if sandbox_status in cls.INFRASTRUCTURE_STATUSES:
            return sandbox_status

        if parsed_report.collection_errors > 0:
            return "FAILED"

        if parsed_report.failed > 0 or parsed_report.errors > 0:
            return "FAILED"

        if parsed_report.no_tests_collected:
            return (
                "NO_TESTS_FOUND"
                if discovery_complete
                else "EXECUTION_ERROR"
            )

        if parsed_report.successful and exit_code == 0:
            return (
                "PASSED"
                if discovery_complete
                else "EXECUTION_ERROR"
            )

        return "EXECUTION_ERROR"

    # ---------------------------------------------------------
    # CANCELLATION
    # ---------------------------------------------------------

    def _record_cancelled_result(
        self,
        task_id: str,
        retry_count: int,
        phase: str,
        additional_metadata: Optional[dict] = None,
    ) -> ExecutionResult:
        """Record cancellation that occurs before a result is returned."""

        metadata = {
            "retry_count": retry_count,
            "discovery_complete": False,
            "cancellation_phase": phase,
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        result = ExecutionResult(
            status="CANCELLED",
            error_message=(
                f"Automated QA was cancelled during {phase}."
            ),
            metadata=metadata,
        )

        return self._record_result(
            task_id,
            result,
        )

    # ---------------------------------------------------------
    # CONTAINER CLEANUP
    # ---------------------------------------------------------

    @staticmethod
    def _extract_cleanup_warning(
        error_message: Optional[str],
    ) -> Optional[str]:
        """
        Extract a Docker cleanup warning from the sandbox result.

        Cleanup warnings are currently embedded in the sandbox
        error message rather than exposed as a separate field.
        """

        if not error_message:
            return None

        marker = "Container cleanup warning:"

        if marker not in error_message:
            return None

        return error_message.split(
            marker,
            1,
        )[1].strip()

    # ---------------------------------------------------------
    # ERROR REPORTING
    # ---------------------------------------------------------

    @staticmethod
    def _report_error(
        parsed_report,
        status: str,
        discovery_complete: bool = True,
    ) -> Optional[str]:
        """Explain non-successful or incomplete pytest results."""

        if status == "PASSED":
            return None

        if status == "NO_TESTS_FOUND":
            return (
                "Pytest did not collect any executable test cases."
            )

        if status == "FAILED":
            if parsed_report.collection_errors > 0:
                return (
                    "Pytest encountered one or more test collection "
                    "errors. Some tests may not have been imported "
                    "or executed."
                )

            if parsed_report.failed > 0 or parsed_report.errors > 0:
                return (
                    "Pytest reported one or more failed test cases "
                    "or execution errors."
                )

            return "Pytest reported a failure."

        if status == "TIMEOUT":
            return "Automated test execution timed out."

        if status == "CANCELLED":
            return "Automated test execution was cancelled."

        if not discovery_complete:
            if (
                parsed_report.successful
                and parsed_report.total > 0
            ):
                return (
                    "Pytest reported passing test cases, but "
                    "workspace test discovery was incomplete. "
                    "The overall execution cannot be marked PASSED."
                )

            return (
                "Workspace test discovery was incomplete. "
                "The overall test execution result requires review."
            )

        if status in QAExecutionManager.INFRASTRUCTURE_STATUSES:
            return (
                "Automated test execution could not complete "
                "because of an execution environment problem."
            )

        return (
            "The test execution did not produce sufficient "
            "evidence for a confirmed result."
        )

    # ---------------------------------------------------------
    # EXECUTION HISTORY
    # ---------------------------------------------------------

    def _record_result(
        self,
        task_id: str,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Record execution evidence in task history."""

        self.history_manager.record(
            task_id=task_id,
            result=result,
        )

        return result
