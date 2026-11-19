
"""
AgentOps - QA Execution Manager

Coordinates Python and Node.js test discovery, Docker sandbox
execution, pytest report parsing, and execution history.

Responsibilities:
- Discover supported pytest files and Node.js test projects.
- Execute tests through restricted Docker sandboxes.
- Parse pytest output into structured evidence.
- Evaluate Node.js results independently from pytest.
- Aggregate results across Python and Node.js environments.
- Preserve cancellation and timeout results.
- Record execution and container cleanup information.
- Maintain accurate distinctions between test files,
  Node.js projects, and individual Python test cases.
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
        "DEPENDENCY_PREPARATION_FAILED",
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
        Discover and execute supported tests for one workspace.

        Python and Node.js results are evaluated independently.

        Python output is parsed only by TestReportParser.
        Node.js output is evaluated using its sandbox status,
        exit code, and execution evidence.

        For mixed projects, every discovered test environment
        must pass before the overall result can be PASSED.
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
        # TEST DISCOVERY
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
                        "node_project_directories": [],
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

        node_project_directories = list(
            getattr(
                discovery,
                "node_project_directories",
                [],
            ) or []
        )

        python_tests_discovered = bool(
            discovered_test_files
        )

        node_tests_discovered = bool(
            node_project_directories
        )

        supported_tests_discovered = (
            python_tests_discovered
            or node_tests_discovered
        )

        discovery_metadata = {
            "discovery_complete": discovery_complete,
            "discovered_test_files": discovered_test_files,
            "tests_discovered": len(discovered_test_files),
            "node_project_directories": (
                node_project_directories
            ),
            "python_tests_discovered": (
                python_tests_discovered
            ),
            "node_tests_discovered": (
                node_tests_discovered
            ),
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

        if (
            not supported_tests_discovered
            and not discovery_complete
        ):
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="EXECUTION_ERROR",
                    error_message=(
                        "Test discovery was incomplete and did not "
                        "identify any supported test environments. "
                        "AgentOps cannot confirm that the workspace "
                        "contains no tests."
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                    },
                ),
            )

        if not supported_tests_discovered:
            return self._record_result(
                task_id,
                ExecutionResult(
                    status="NO_TESTS_FOUND",
                    error_message=(
                        "No supported Python pytest files or "
                        "Node.js test projects were discovered."
                    ),
                    metadata={
                        "retry_count": retry_count,
                        **discovery_metadata,
                    },
                ),
            )

        # -----------------------------------------------------
        # SANDBOX EXECUTION
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
        # INDIVIDUAL ENVIRONMENT RESULTS
        # -----------------------------------------------------

        python_result = getattr(
            sandbox_result,
            "python_result",
            None,
        )

        node_result = getattr(
            sandbox_result,
            "node_result",
            None,
        )

        # -----------------------------------------------------
        # PYTHON / PYTEST REPORT PARSING
        # -----------------------------------------------------

        parsed_report = None
        python_status = None
        python_report_error = None

        if python_tests_discovered:
            if python_result is None:
                python_report_error = (
                    "Python tests were discovered, but the sandbox "
                    "manager did not return a Python execution result."
                )
                python_status = "EXECUTION_ERROR"

            else:
                python_output = "\n".join(
                    part
                    for part in (
                        python_result.stdout,
                        python_result.stderr,
                    )
                    if part
                )

                try:
                    parsed_report = self.report_parser.parse(
                        python_output,
                        exit_code=python_result.exit_code,
                    )

                    python_status = self._determine_status(
                        sandbox_status=python_result.status,
                        parsed_report=parsed_report,
                        exit_code=python_result.exit_code,
                        discovery_complete=discovery_complete,
                    )

                except Exception as exc:
                    python_status = "EXECUTION_ERROR"
                    python_report_error = (
                        "Unable to parse pytest execution output: "
                        f"{type(exc).__name__}: {exc}"
                    )

        # -----------------------------------------------------
        # NODE.JS RESULT EVALUATION
        # -----------------------------------------------------

        node_status = None
        node_error = None

        if node_tests_discovered:
            if node_result is None:
                node_status = "EXECUTION_ERROR"
                node_error = (
                    "Node.js test projects were discovered, but "
                    "the sandbox manager did not return a "
                    "Node.js execution result."
                )

            else:
                node_status = self._determine_node_status(
                    sandbox_status=node_result.status,
                    exit_code=node_result.exit_code,
                    tests_executed=node_result.tests_executed,
                )

                node_error = node_result.error_message or None

                if (
                    node_status != "PASSED"
                    and not node_error
                ):
                    node_error = (
                        "Node.js automated tests did not produce "
                        "a confirmed passing result."
                    )

        # -----------------------------------------------------
        # AGGREGATE EXECUTION STATUS
        # -----------------------------------------------------

        sandbox_status = getattr(
            sandbox_result,
            "status",
            "EXECUTION_ERROR",
        )

        status = self._aggregate_status(
            sandbox_status=sandbox_status,
            python_status=python_status,
            node_status=node_status,
            discovery_complete=discovery_complete,
        )

        # -----------------------------------------------------
        # OUTPUT AND EXECUTION EVIDENCE
        # -----------------------------------------------------

        stdout_parts = []
        stderr_parts = []

        if python_result is not None:
            if python_result.stdout:
                stdout_parts.append(
                    "[Python / pytest]\n"
                    + python_result.stdout
                )

            if python_result.stderr:
                stderr_parts.append(
                    "[Python / pytest]\n"
                    + python_result.stderr
                )

        if node_result is not None:
            if node_result.stdout:
                stdout_parts.append(
                    "[Node.js / npm]\n"
                    + node_result.stdout
                )

            if node_result.stderr:
                stderr_parts.append(
                    "[Node.js / npm]\n"
                    + node_result.stderr
                )

        # Preserve manager-level output when execution failed
        # before individual environment results were available.
        if not python_result and not node_result:
            if sandbox_result.stdout:
                stdout_parts.append(sandbox_result.stdout)

            if sandbox_result.stderr:
                stderr_parts.append(sandbox_result.stderr)

        stdout = "\n".join(stdout_parts)
        stderr = "\n".join(stderr_parts)

        duration_seconds = getattr(
            sandbox_result,
            "duration_seconds",
            0.0,
        )

        execution_environment = getattr(
            sandbox_result,
            "environment",
            None,
        ) or getattr(
            sandbox_result,
            "execution_environment",
            "docker",
        )

        tests_executed = any(
            bool(getattr(result, "tests_executed", False))
            for result in (
                python_result,
                node_result,
            )
            if result is not None
        )

        # Python's parser provides individual test-case counts.
        # Node.js test counts remain unknown unless a future
        # framework-specific parser supplies structured counts.
        test_count = (
            parsed_report.total
            if parsed_report is not None
            else None
        )

        tests_passed = (
            parsed_report.passed
            if parsed_report is not None
            else None
        )

        tests_failed = (
            parsed_report.failed + parsed_report.errors
            if parsed_report is not None
            else None
        )

        tests_skipped = (
            parsed_report.skipped
            if parsed_report is not None
            else None
        )

        collection_errors = (
            parsed_report.collection_errors
            if parsed_report is not None
            else 0
        )

        pytest_report_status = (
            self._pytest_report_status(
                parsed_report,
                python_result,
            )
            if python_tests_discovered
            else "NOT_APPLICABLE"
        )

        cleanup_warning = self._extract_cleanup_warning(
            getattr(sandbox_result, "error_message", None)
        )

        if python_result is not None:
            cleanup_warning = (
                cleanup_warning
                or self._extract_cleanup_warning(
                    python_result.error_message
                )
            )

        if node_result is not None:
            cleanup_warning = (
                cleanup_warning
                or self._extract_cleanup_warning(
                    node_result.error_message
                )
            )

        report_error = self._report_error(
            parsed_report=parsed_report,
            status=status,
            discovery_complete=discovery_complete,
        )

        error_message = (
            getattr(sandbox_result, "error_message", None)
            or python_report_error
            or node_error
            or report_error
        )

        environment_results = {
            "python": {
                "discovered": python_tests_discovered,
                "status": python_status,
                "exit_code": (
                    python_result.exit_code
                    if python_result is not None
                    else None
                ),
                "tests_executed": (
                    python_result.tests_executed
                    if python_result is not None
                    else False
                ),
                "duration_seconds": (
                    python_result.duration_seconds
                    if python_result is not None
                    else None
                ),
            },
            "node": {
                "discovered": node_tests_discovered,
                "status": node_status,
                "exit_code": (
                    node_result.exit_code
                    if node_result is not None
                    else None
                ),
                "tests_executed": (
                    node_result.tests_executed
                    if node_result is not None
                    else False
                ),
                "duration_seconds": (
                    node_result.duration_seconds
                    if node_result is not None
                    else None
                ),
            },
        }

        execution_result = ExecutionResult(
            status=status,
            exit_code=getattr(
                sandbox_result,
                "exit_code",
                None,
            ),
            duration_seconds=duration_seconds,
            stdout=stdout,
            stderr=stderr,
            environment=execution_environment,
            tests_executed=tests_executed,
            test_count=test_count,
            error_message=error_message,
            metadata={
                "retry_count": retry_count,

                **discovery_metadata,

                "supported_test_environments": [
                    name
                    for name, discovered in (
                        ("python", python_tests_discovered),
                        ("node", node_tests_discovered),
                    )
                    if discovered
                ],

                "environment_results": environment_results,

                # Python test files and individual test cases
                # remain separate metrics.
                "test_cases_reported": test_count,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,

                "pytest_report_status": pytest_report_status,
                "collection_errors": collection_errors,

                "warnings": (
                    parsed_report.warnings
                    if parsed_report is not None
                    else []
                ),

                "no_tests_collected": (
                    parsed_report.no_tests_collected
                    if parsed_report is not None
                    else False
                ),

                "sandbox_status": sandbox_status,
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
        Map Python sandbox and pytest evidence to an execution status.

        Collection errors are checked before the no-tests condition.

        A passing pytest report cannot produce an overall PASSED
        result if workspace discovery was incomplete.
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

    @classmethod
    def _determine_node_status(
        cls,
        sandbox_status: str,
        exit_code: Optional[int],
        tests_executed: bool,
    ) -> str:
        """
        Evaluate Node.js execution without using the pytest parser.

        A Node.js result passes only when the sandbox reports
        PASSED, the process exits successfully, and execution
        evidence confirms that tests were run.
        """

        if sandbox_status == "CANCELLED":
            return "CANCELLED"

        if sandbox_status == "TIMEOUT":
            return "TIMEOUT"

        if sandbox_status in cls.INFRASTRUCTURE_STATUSES:
            return sandbox_status

        if sandbox_status == "NO_TESTS_FOUND":
            return "NO_TESTS_FOUND"

        if (
            sandbox_status == "PASSED"
            and exit_code == 0
            and tests_executed
        ):
            return "PASSED"

        if (
            sandbox_status == "FAILED"
            or (
                exit_code is not None
                and exit_code != 0
            )
        ):
            return "FAILED"

        return "EXECUTION_ERROR"

    @classmethod
    def _aggregate_status(
        cls,
        sandbox_status: str,
        python_status: Optional[str],
        node_status: Optional[str],
        discovery_complete: bool,
    ) -> str:
        """
        Aggregate all discovered environment results.

        Every discovered environment must pass.

        Infrastructure and interruption statuses take priority
        over ordinary test failures.
        """

        environment_statuses = [
            status
            for status in (
                python_status,
                node_status,
            )
            if status is not None
        ]

        all_statuses = [
            sandbox_status,
            *environment_statuses,
        ]

        if "CANCELLED" in all_statuses:
            return "CANCELLED"

        if "TIMEOUT" in all_statuses:
            return "TIMEOUT"

        for infrastructure_status in cls.INFRASTRUCTURE_STATUSES:
            if infrastructure_status in all_statuses:
                return infrastructure_status

        if "NO_TESTS_FOUND" in environment_statuses:
            return "NO_TESTS_FOUND"

        if "FAILED" in environment_statuses:
            return "FAILED"

        if any(
            status != "PASSED"
            for status in environment_statuses
        ):
            return "EXECUTION_ERROR"

        if not environment_statuses:
            return (
                sandbox_status
                if sandbox_status != "PASSED"
                else "EXECUTION_ERROR"
            )

        if not discovery_complete:
            return "EXECUTION_ERROR"

        if sandbox_status != "PASSED":
            return (
                "FAILED"
                if sandbox_status == "FAILED"
                else "EXECUTION_ERROR"
            )

        return "PASSED"

    # ---------------------------------------------------------
    # REPORT EVIDENCE
    # ---------------------------------------------------------

    @staticmethod
    def _pytest_report_status(
        parsed_report,
        python_result,
    ) -> str:
        """Preserve the Python pytest outcome independently."""

        if parsed_report is None or python_result is None:
            return "INCOMPLETE"

        if (
            parsed_report.successful
            and python_result.exit_code == 0
        ):
            return "PASSED"

        if parsed_report.has_failures:
            return "FAILED"

        return "INCOMPLETE"

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
        """Explain non-successful or incomplete QA results."""

        if status == "PASSED":
            return None

        if status == "NO_TESTS_FOUND":
            return (
                "No executable test cases were collected "
                "by the discovered test environment."
            )

        if status == "FAILED":
            if (
                parsed_report is not None
                and parsed_report.collection_errors > 0
            ):
                return (
                    "Pytest encountered one or more test collection "
                    "errors. Some tests may not have been imported "
                    "or executed."
                )

            if (
                parsed_report is not None
                and (
                    parsed_report.failed > 0
                    or parsed_report.errors > 0
                )
            ):
                return (
                    "Pytest reported one or more failed test cases "
                    "or execution errors."
                )

            return (
                "One or more discovered test environments "
                "reported a failure."
            )

        if status == "TIMEOUT":
            return "Automated test execution timed out."

        if status == "CANCELLED":
            return "Automated test execution was cancelled."

        if not discovery_complete:
            if (
                parsed_report is not None
                and parsed_report.successful
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
                "because of an execution environment or "
                "dependency preparation problem."
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
            execution_result=result,
        )

        return result
