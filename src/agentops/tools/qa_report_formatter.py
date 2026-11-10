
"""
AgentOps - QA Report Formatter

Converts structured QA reviews into readable reports.

The formatter presents:
- Final QA decision.
- Automated test execution results.
- Pytest collection errors.
- Static review results.
- Individual findings and recommendations.
- Known limitations.
- Human approval requirements.
"""

from agentops.models.qa_review import QAReview


class QAReportFormatter:
    """Format structured QA reviews into readable reports."""

    def format(self, review: QAReview) -> str:
        """Generate a complete human-readable QA report."""

        sections = [
            self._format_header(review),
            self._format_execution(review),
            self._format_static_review(review),
            self._format_findings(review),
            self._format_limitations(review),
            self._format_metadata(review),
        ]

        return "\n\n".join(sections)

    def _format_header(self, review: QAReview) -> str:
        """Format the decision and overall summary."""

        lines = [
            "=" * 64,
            "AGENTOPS QUALITY ASSURANCE REPORT",
            "=" * 64,
            f"Decision: {review.decision}",
            f"Summary: {review.summary}",
        ]

        if review.requires_human_review:
            lines.append("Human approval: REQUIRED")
        else:
            lines.append("Human approval: Not required")

        return "\n".join(lines)

    def _format_execution(self, review: QAReview) -> str:
        """Format automated execution evidence."""

        metadata = review.metadata or {}

        collection_errors = self._safe_count(
            metadata.get("collection_errors", 0)
        )

        lines = [
            "AUTOMATED TEST EXECUTION",
            "-" * 64,
            (
                "Automated test execution performed: "
                f"{self._yes_no(review.automated_tests_executed)}"
            ),
            f"Execution status: {review.execution_status}",
            (
                "Test files discovered: "
                f"{review.tests_discovered}"
            ),
            (
                "Test cases executed: "
                f"{review.tests_executed}"
            ),
            f"Tests passed: {review.tests_passed}",
            f"Tests failed: {review.tests_failed}",
            f"Tests skipped: {review.tests_skipped}",
            f"Test collection errors: {collection_errors}",
        ]

        if review.execution_status == "PASSED":
            lines.append(
                "Execution result: Automated test execution passed."
            )

        elif review.execution_status == "FAILED":
            lines.append(
                "Execution result: Automated execution reported "
                "one or more failures."
            )

            if collection_errors > 0:
                lines.append(
                    "Failure detail: Pytest encountered collection "
                    "errors. Some tests may not have been imported "
                    "or executed."
                )

            if review.tests_failed > 0:
                lines.append(
                    "Failure detail: One or more executed test cases "
                    "reported failures."
                )

        elif review.execution_status == "NO_TESTS_FOUND":
            lines.append(
                "Execution result: No executable automated tests "
                "were found."
            )

        elif review.execution_status == "TIMEOUT":
            lines.append(
                "Execution result: The test execution timed out."
            )

        elif review.execution_status == "CANCELLED":
            lines.append(
                "Execution result: Test execution was cancelled."
            )

        elif review.execution_status == "NOT_RUN":
            lines.append(
                "Execution result: Automated tests were not run."
            )

        else:
            lines.append(
                "Execution result: Review the execution status "
                "and limitations for additional information."
            )

        return "\n".join(lines)

    def _format_static_review(self, review: QAReview) -> str:
        """Format the LLM-assisted static review evidence."""

        lines = [
            "STATIC CODE REVIEW",
            "-" * 64,
            (
                "Static review completed: "
                f"{self._yes_no(review.static_review_completed)}"
            ),
            f"Findings identified: {len(review.findings)}",
        ]

        summary = (
            review.metadata.get("static_review_summary", "")
            if review.metadata
            else ""
        )

        if summary:
            lines.extend([
                "",
                "Review summary:",
                summary,
            ])

        return "\n".join(lines)

    def _format_findings(self, review: QAReview) -> str:
        """Format individual QA findings."""

        lines = [
            "QA FINDINGS",
            "-" * 64,
        ]

        if not review.findings:
            lines.append(
                "No findings were identified by the static review."
            )
            return "\n".join(lines)

        severity_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
            "INFO": 4,
        }

        findings = sorted(
            review.findings,
            key=lambda finding: severity_order.get(
                finding.severity.upper(),
                5,
            ),
        )

        for index, finding in enumerate(findings, start=1):
            lines.extend([
                "",
                f"{index}. [{finding.severity.upper()}] "
                f"{finding.title}",
                f"Category: {finding.category}",
                f"Description: {finding.description}",
            ])

            if finding.file_path:
                location = finding.file_path

                if finding.line_number is not None:
                    location += f":{finding.line_number}"

                lines.append(f"Location: {location}")

            if finding.recommendation:
                lines.append(
                    f"Recommendation: {finding.recommendation}"
                )

        return "\n".join(lines)

    def _format_limitations(self, review: QAReview) -> str:
        """Format limitations and required follow-up."""

        lines = [
            "LIMITATIONS AND REQUIRED FOLLOW-UP",
            "-" * 64,
        ]

        if not review.limitations:
            lines.append(
                "No additional limitations were recorded."
            )
        else:
            for limitation in review.limitations:
                lines.append(f"- {limitation}")

        return "\n".join(lines)

    def _format_metadata(self, review: QAReview) -> str:
        """Format review metadata."""

        lines = [
            "REVIEW METADATA",
            "-" * 64,
            f"Review timestamp: {review.reviewed_at}",
            (
                "Human review required: "
                f"{self._yes_no(review.requires_human_review)}"
            ),
        ]

        metadata = review.metadata or {}

        if "human_approval_required" in metadata:
            lines.append(
                "Human approval policy: "
                f"{self._yes_no(metadata['human_approval_required'])}"
            )

        lines.append("=" * 64)

        return "\n".join(lines)

    @staticmethod
    def _safe_count(value) -> int:
        """Convert a count into a nonnegative integer safely."""

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _yes_no(value: bool) -> str:
        """Convert a boolean into readable text."""

        return "Yes" if value else "No"
