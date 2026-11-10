
"""
AgentOps - QA Policy Engine

Makes deterministic QA decisions using structured evidence.

The LLM may identify findings, but it cannot assign
the final QA decision.

Policy:
- Critical or high findings produce FAIL.
- Confirmed automated test failures produce FAIL.
- Pytest collection errors produce FAIL.
- Missing tests or incomplete execution require review.
- Infrastructure problems require review.
- Incomplete static review requires review.
- Successful automated checks still require human approval.

AgentOps does not automatically authorize production approval.
"""

from typing import Optional

from agentops.models.execution import ExecutionResult
from agentops.models.qa_review import QAReview, QAFinding


class QAPolicyEngine:
    """Evaluate QA evidence and produce a structured decision."""

    SEVERE_FINDING_LEVELS = {
        "CRITICAL",
        "HIGH",
    }

    FAILURE_STATUSES = {
        "FAILED",
    }

    INFRASTRUCTURE_STATUSES = {
        "DOCKER_UNAVAILABLE",
        "IMAGE_NOT_FOUND",
        "SANDBOX_UNAVAILABLE",
        "INVALID_WORKSPACE",
        "EXECUTION_ERROR",
        "SANDBOX_REQUIRED",
    }

    def evaluate(
        self,
        execution_result: Optional[ExecutionResult],
        findings: Optional[list[QAFinding]] = None,
        static_review_completed: bool = False,
        static_review_summary: str = "",
        tests_discovered: int = 0,
        limitations: Optional[list[str]] = None,
    ) -> QAReview:
        """
        Evaluate test execution and static review evidence.

        Args:
            execution_result: Structured automated execution result.
            findings: Findings produced by the static QA review.
            static_review_completed: Whether static review completed.
            static_review_summary: Summary of static review evidence.
            tests_discovered: Number of test files discovered.
            limitations: Known limitations or warnings.

        Returns:
            Structured QAReview with a deterministic decision.
        """

        findings = findings or []
        limitations = list(limitations or [])

        tests_discovered = max(
            0,
            self._safe_count(tests_discovered),
        )

        execution_status = "NOT_RUN"
        automated_tests_executed = False

        tests_executed = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0

        collection_errors = 0
        execution_metadata = {}

        if execution_result is not None:
            execution_status = execution_result.status

            automated_tests_executed = bool(
                execution_result.tests_executed
            )

            execution_metadata = (
                execution_result.metadata or {}
            )

            tests_passed = self._safe_count(
                execution_metadata.get("tests_passed", 0)
            )

            tests_failed = self._safe_count(
                execution_metadata.get("tests_failed", 0)
            )

            tests_skipped = self._safe_count(
                execution_metadata.get("tests_skipped", 0)
            )

            collection_errors = self._safe_count(
                execution_metadata.get("collection_errors", 0)
            )

            tests_executed = (
                tests_passed
                + tests_failed
                + tests_skipped
            )

            # Preserve evidence when an older or incomplete
            # execution result does not contain count metadata.
            if execution_status == "PASSED" and tests_executed == 0:
                tests_executed = self._safe_count(
                    execution_result.test_count
                )
                tests_passed = tests_executed

            elif execution_status == "FAILED" and tests_failed == 0:
                # A failed execution without parsed test counts
                # is still a failure. Do not invent a test total.
                if (
                    execution_result.test_count > 0
                    and collection_errors == 0
                ):
                    tests_failed = 1

                    tests_executed = max(
                        tests_executed,
                        tests_passed
                        + tests_failed
                        + tests_skipped,
                    )

        severe_findings = [
            finding
            for finding in findings
            if finding.severity.upper()
            in self.SEVERE_FINDING_LEVELS
        ]

        decision = "NEEDS_REVIEW"

        if severe_findings:
            decision = "FAIL"

            limitations.append(
                "Critical or high-severity findings require correction."
            )

        elif tests_failed > 0:
            decision = "FAIL"

            limitations.append(
                "Automated tests reported one or more failures."
            )

        elif collection_errors > 0:
            decision = "FAIL"

            limitations.append(
                "Pytest encountered one or more test collection errors."
            )

        elif execution_status in self.FAILURE_STATUSES:
            decision = "FAIL"

            limitations.append(
                "Automated test execution reported a failure."
            )

        elif execution_result is None:
            limitations.append(
                "No automated execution result was available."
            )

        elif execution_status == "NO_TESTS_FOUND":
            limitations.append(
                "No automated test files or executable test cases "
                "were available."
            )

        elif execution_status in self.INFRASTRUCTURE_STATUSES:
            limitations.append(
                "Automated test execution could not be completed "
                "reliably because of an execution environment problem."
            )

        elif execution_status == "TIMEOUT":
            limitations.append(
                "Automated test execution exceeded its time limit."
            )

        elif execution_status == "CANCELLED":
            limitations.append(
                "Automated test execution was cancelled."
            )

        elif execution_status != "PASSED":
            limitations.append(
                f"Automated execution status requires review: "
                f"{execution_status}."
            )

        elif not static_review_completed:
            limitations.append(
                "The static QA review did not complete successfully."
            )

        elif tests_executed == 0:
            limitations.append(
                "No completed automated test cases were reported."
            )

        else:
            limitations.append(
                "Automated checks completed, but human approval "
                "is required before the implementation can be approved."
            )

        if not static_review_summary.strip():
            limitations.append(
                "A usable static review summary was not provided."
            )

        summary = self._build_summary(
            decision=decision,
            execution_status=execution_status,
            findings_count=len(findings),
            severe_findings_count=len(severe_findings),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            collection_errors=collection_errors,
            static_review_completed=static_review_completed,
        )

        return QAReview(
            decision=decision,
            summary=summary,
            automated_tests_executed=automated_tests_executed,
            execution_status=execution_status,
            tests_discovered=tests_discovered,
            tests_executed=tests_executed,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            static_review_completed=static_review_completed,
            findings=findings,
            limitations=self._unique_strings(limitations),
            requires_human_review=True,
            metadata={
                "static_review_summary": static_review_summary,
                "findings_count": len(findings),
                "severe_findings_count": len(severe_findings),
                "collection_errors": collection_errors,
                "execution_metadata": execution_metadata,
                "human_approval_required": True,
            },
        )

    @staticmethod
    def _safe_count(value) -> int:
        """Convert a count to a nonnegative integer safely."""

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        """Remove duplicate limitation messages while preserving order."""

        result = []
        seen = set()

        for value in values:
            if not isinstance(value, str):
                continue

            cleaned = value.strip()

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            result.append(cleaned)

        return result

    @staticmethod
    def _build_summary(
        decision: str,
        execution_status: str,
        findings_count: int,
        severe_findings_count: int,
        tests_passed: int,
        tests_failed: int,
        tests_skipped: int,
        collection_errors: int,
        static_review_completed: bool,
    ) -> str:
        """Build a concise evidence-based QA summary."""

        parts = [
            f"QA decision: {decision}.",
            f"Automated execution status: {execution_status}.",
            (
                f"Tests: {tests_passed} passed, "
                f"{tests_failed} failed, "
                f"{tests_skipped} skipped."
            ),
            f"Collection errors: {collection_errors}.",
            f"Static review completed: {static_review_completed}.",
            f"Findings identified: {findings_count}.",
            f"Critical/high findings: {severe_findings_count}.",
        ]

        if decision == "FAIL":
            parts.append(
                "The implementation requires corrective action."
            )

        elif decision == "NEEDS_REVIEW":
            parts.append(
                "The available evidence does not authorize "
                "automatic approval."
            )

        return " ".join(parts)
