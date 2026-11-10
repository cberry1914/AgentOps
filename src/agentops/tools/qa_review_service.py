
"""
AgentOps - QA Review Service

Coordinates the complete QA evidence and reporting process.

Responsibilities:
- Execute automated tests through QAExecutionManager.
- Parse structured findings from the QA agent.
- Apply deterministic QA policy.
- Generate a human-readable QA report.
- Record completed reviews in QA review history.
- Preserve cancellation and execution limitations.

The LLM never assigns the final QA decision.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

from agentops.models.execution import ExecutionResult
from agentops.models.qa_review import QAReview
from agentops.tools.cancellation import check_cancelled
from agentops.tools.qa_execution_manager import (
    QAExecutionManager,
)
from agentops.tools.qa_finding_parser import (
    QAFindingParser,
)
from agentops.tools.qa_policy import QAPolicyEngine
from agentops.tools.qa_report_formatter import (
    QAReportFormatter,
)
from agentops.tools.qa_review_history import (
    QAReviewHistoryManager,
)


@dataclass
class QAReviewServiceResult:
    """Contains the complete output of one QA review."""

    review: QAReview
    report: str
    execution_result: Optional[ExecutionResult]
    parsing_warnings: list[str]
    history_entry: Optional[dict] = None

    @property
    def decision(self) -> str:
        return self.review.decision

    @property
    def passed(self) -> bool:
        return self.review.passed

    @property
    def requires_human_review(self) -> bool:
        return self.review.requires_human_review


class QAReviewService:
    """Coordinate QA execution, evaluation, history, and reporting."""

    def __init__(
        self,
        execution_manager: Optional[QAExecutionManager] = None,
        finding_parser: Optional[QAFindingParser] = None,
        policy_engine: Optional[QAPolicyEngine] = None,
        report_formatter: Optional[QAReportFormatter] = None,
        history_manager: Optional[QAReviewHistoryManager] = None,
    ) -> None:
        self.execution_manager = (
            execution_manager or QAExecutionManager()
        )

        self.finding_parser = (
            finding_parser or QAFindingParser()
        )

        self.policy_engine = (
            policy_engine or QAPolicyEngine()
        )

        self.report_formatter = (
            report_formatter or QAReportFormatter()
        )

        self.history_manager = (
            history_manager or QAReviewHistoryManager()
        )

    # ---------------------------------------------------------
    # REVIEW LIFECYCLE
    # ---------------------------------------------------------

    def review(
        self,
        task_id: str,
        workspace_path: str | Path,
        static_review_response: str,
        static_review_completed: bool,
        retry_count: int = 0,
        cancel_event: Optional[Event] = None,
        static_review_summary: str = "",
    ) -> QAReviewServiceResult:
        """
        Execute a complete evidence-based QA review.

        Cancellation before automated execution is propagated.

        If QAExecutionManager returns a structured CANCELLED result,
        the service completes deterministic policy evaluation,
        report generation, and history recording. This preserves
        cancellation as review evidence instead of discarding it.

        Args:
            task_id: AgentOps task identifier.
            workspace_path: Task workspace directory.
            static_review_response: Raw QA agent findings.
            static_review_completed: Whether static review
                actually completed.
            retry_count: Current developer correction attempt.
            cancel_event: Optional cancellation event.
            static_review_summary: Summary from static review.

        Returns:
            QAReviewServiceResult containing the structured
            review, formatted report, and execution evidence.
        """

        check_cancelled(cancel_event)

        if not task_id or not task_id.strip():
            raise ValueError("Task ID cannot be empty.")

        if not isinstance(static_review_response, str):
            raise TypeError(
                "Static review response must be a string."
            )

        if not isinstance(static_review_completed, bool):
            raise TypeError(
                "static_review_completed must be a boolean."
            )

        if (
            not isinstance(retry_count, int)
            or isinstance(retry_count, bool)
            or retry_count < 0
        ):
            raise ValueError(
                "Retry count must be a nonnegative integer."
            )

        # -----------------------------------------------------
        # Parse static review findings
        # -----------------------------------------------------

        findings, parsing_warnings = (
            self.finding_parser.parse(
                static_review_response
            )
        )

        check_cancelled(cancel_event)

        # -----------------------------------------------------
        # Execute automated tests
        # -----------------------------------------------------

        execution_result = self.execution_manager.execute(
            task_id=task_id,
            workspace_path=str(workspace_path),
            retry_count=retry_count,
            cancel_event=cancel_event,
        )

        # A structured cancellation result is already recorded
        # by QAExecutionManager. Do not raise another cancellation
        # exception and discard the QA review that must summarize it.
        execution_was_cancelled = (
            execution_result is not None
            and execution_result.status == "CANCELLED"
        )

        if not execution_was_cancelled:
            check_cancelled(cancel_event)

        execution_metadata = (
            execution_result.metadata
            if execution_result is not None
            else {}
        ) or {}

        tests_discovered = self._safe_count(
            execution_metadata.get(
                "tests_discovered",
                0,
            )
        )

        limitations = list(parsing_warnings)

        if parsing_warnings:
            limitations.append(
                "One or more QA findings could not be parsed. "
                "Review the parsing warnings before approval."
            )

        # Preserve test discovery limitations as review evidence.
        discovery_warnings = execution_metadata.get(
            "discovery_warnings",
            [],
        )

        if isinstance(discovery_warnings, list):
            for warning in discovery_warnings:
                if isinstance(warning, str) and warning.strip():
                    limitations.append(
                        f"Test discovery: {warning}"
                    )

        if execution_metadata.get("discovery_complete") is False:
            limitations.append(
                "Test discovery was incomplete. "
                "The discovered test files may not represent "
                "the entire workspace."
            )

        if execution_was_cancelled:
            limitations.append(
                "Automated test execution was cancelled. "
                "The QA decision is based on incomplete execution evidence."
            )

        # -----------------------------------------------------
        # Deterministic policy evaluation
        # -----------------------------------------------------

        review = self.policy_engine.evaluate(
            execution_result=execution_result,
            findings=findings,
            static_review_completed=static_review_completed,
            static_review_summary=static_review_summary,
            tests_discovered=tests_discovered,
            limitations=limitations,
        )

        if not execution_was_cancelled:
            check_cancelled(cancel_event)

        # -----------------------------------------------------
        # Defensive approval safeguards
        # -----------------------------------------------------

        if parsing_warnings and review.decision == "PASS":
            review.decision = "NEEDS_REVIEW"
            review.requires_human_review = True
            review.summary = (
                "QA decision: NEEDS_REVIEW. "
                "Approval was withheld because the finding "
                "response contained parsing problems."
            )

            review.limitations.append(
                "QA approval was withheld because the finding "
                "response contained parsing problems."
            )

        # A review must not be approved when workspace discovery
        # was incomplete, regardless of individual test results.
        if (
            execution_metadata.get("discovery_complete") is False
            and review.decision == "PASS"
        ):
            review.decision = "NEEDS_REVIEW"
            review.requires_human_review = True
            review.summary = (
                "QA decision: NEEDS_REVIEW. "
                "Approval was withheld because workspace "
                "test discovery was incomplete."
            )

            review.limitations.append(
                "QA approval was withheld because workspace "
                "test discovery was incomplete."
            )

        # -----------------------------------------------------
        # Preserve execution evidence in review metadata
        # -----------------------------------------------------

        review.metadata.update({
            "retry_count": retry_count,
            "execution_status": (
                execution_result.status
                if execution_result is not None
                else "NOT_AVAILABLE"
            ),
            "execution_cancelled": execution_was_cancelled,
            "tests_discovered": tests_discovered,
            "test_cases_reported": self._safe_count(
                execution_metadata.get(
                    "test_cases_reported",
                    0,
                )
            ),
            "tests_passed": self._safe_count(
                execution_metadata.get(
                    "tests_passed",
                    0,
                )
            ),
            "tests_failed": self._safe_count(
                execution_metadata.get(
                    "tests_failed",
                    0,
                )
            ),
            "tests_skipped": self._safe_count(
                execution_metadata.get(
                    "tests_skipped",
                    0,
                )
            ),
            "collection_errors": self._safe_count(
                execution_metadata.get(
                    "collection_errors",
                    0,
                )
            ),
            "discovery_complete": execution_metadata.get(
                "discovery_complete",
                False,
            ),
            "pytest_report_status": execution_metadata.get(
                "pytest_report_status",
                "NOT_AVAILABLE",
            ),
            "execution_environment": execution_metadata.get(
                "execution_environment",
                "NOT_AVAILABLE",
            ),
            "container_cleanup_status": execution_metadata.get(
                "container_cleanup_status",
                "NOT_AVAILABLE",
            ),
        })

        if not execution_was_cancelled:
            check_cancelled(cancel_event)

        # -----------------------------------------------------
        # Format report before recording completed review
        # -----------------------------------------------------

        report = self.report_formatter.format(review)

        if not execution_was_cancelled:
            check_cancelled(cancel_event)

        # Store the final structured decision after successful
        # report generation so history represents completed reviews.
        history_entry = self.history_manager.record(
            task_id=task_id,
            review=review,
        )

        return QAReviewServiceResult(
            review=review,
            report=report,
            execution_result=execution_result,
            parsing_warnings=parsing_warnings,
            history_entry=history_entry,
        )

    # ---------------------------------------------------------
    # HISTORY ACCESS
    # ---------------------------------------------------------

    def get_review_history(
        self,
        task_id: str,
    ) -> list[dict]:
        """Return QA review history for a task."""

        return self.history_manager.get_history(task_id)

    def get_latest_review(
        self,
        task_id: str,
    ) -> Optional[dict]:
        """Return the most recent QA review for a task."""

        return self.history_manager.get_latest(task_id)

    # ---------------------------------------------------------
    # VALIDATION HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _safe_count(value) -> int:
        """Convert a count into a nonnegative integer safely."""

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
