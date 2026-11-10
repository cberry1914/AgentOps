
"""
AgentOps - Security Review Service

Coordinates the security review pipeline.

Responsibilities:
- Run deterministic security scanning.
- Parse AI-generated security findings.
- Combine findings into one structured review.
- Apply security policy.
- Record review history.
- Generate a readable security report.

The service coordinates existing components. It does not
execute project code or independently verify exploitability.

Important:
AI review input must be generated from sanitized project
context. Raw credentials and private key material must
never be included in the AI prompt.
"""

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Optional

from agentops.models.security_review import (
    SecurityFinding,
    SecurityReview,
)
from agentops.tools.cancellation import check_cancelled
from agentops.tools.security_finding_parser import (
    SecurityFindingParser,
    ParsedSecurityReport,
)
from agentops.tools.security_policy import SecurityPolicyEngine
from agentops.tools.security_report_formatter import (
    SecurityReportFormatter,
)
from agentops.tools.security_review_history import (
    SecurityReviewHistoryManager,
)
from agentops.tools.security_scanner import SecurityScanner


@dataclass
class SecurityReviewServiceResult:
    """Complete output from a security review operation."""

    review: SecurityReview
    report: str
    static_findings: list[SecurityFinding] = field(
        default_factory=list
    )
    ai_findings: list[SecurityFinding] = field(
        default_factory=list
    )
    parsing_warnings: list[str] = field(
        default_factory=list
    )
    history_record: Optional[dict[str, Any]] = None


class SecurityReviewService:
    """Coordinates security scanning, policy, and reporting."""

    def __init__(
        self,
        scanner: Optional[SecurityScanner] = None,
        parser: Optional[SecurityFindingParser] = None,
        policy_engine: Optional[SecurityPolicyEngine] = None,
        report_formatter: Optional[SecurityReportFormatter] = None,
        history_manager: Optional[SecurityReviewHistoryManager] = None,
    ) -> None:
        self.scanner = scanner or SecurityScanner()
        self.parser = parser or SecurityFindingParser()
        self.policy_engine = (
            policy_engine or SecurityPolicyEngine()
        )
        self.report_formatter = (
            report_formatter or SecurityReportFormatter()
        )
        self.history_manager = (
            history_manager or SecurityReviewHistoryManager()
        )

    def review(
        self,
        task_id: str,
        workspace_files: dict[str, str],
        ai_response: str,
        files_skipped: int = 0,
        limitations: Optional[list[str]] = None,
        cancel_event: Optional[Event] = None,
    ) -> SecurityReviewServiceResult:
        """
        Execute the structured security review pipeline.

        Args:
            task_id:
                AgentOps task identifier.

            workspace_files:
                In-memory mapping of relative file paths to
                file contents. Sensitive files should already
                have been excluded or safely handled.

            ai_response:
                JSON response from the Security Agent.
                The response must be based on sanitized context.

            files_skipped:
                Number of files excluded during discovery or
                collection for reasons other than scanner limits.

            limitations:
                Additional review limitations to preserve.

            cancel_event:
                Optional task cancellation event.

        Returns:
            SecurityReviewServiceResult containing the review,
            formatted report, findings, warnings, and history.
        """

        check_cancelled(cancel_event)

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(
                "task_id must be a non-empty string."
            )

        if not isinstance(workspace_files, dict):
            raise TypeError(
                "workspace_files must be a dictionary."
            )

        if not isinstance(ai_response, str):
            raise TypeError(
                "ai_response must be a string."
            )

        if isinstance(files_skipped, bool) or not isinstance(
            files_skipped,
            int,
        ):
            raise TypeError(
                "files_skipped must be an integer."
            )

        if files_skipped < 0:
            raise ValueError(
                "files_skipped cannot be negative."
            )

        for path, content in workspace_files.items():

            if not isinstance(path, str) or not isinstance(
                content,
                str,
            ):
                raise TypeError(
                    "Workspace files must map string paths "
                    "to string contents."
                )

        review_limitations = list(limitations or [])

        # --------------------------------------------------
        # 1. Determine the effective file scan boundary.
        # --------------------------------------------------

        scanner_file_limit = self.scanner.MAX_FILES

        files_scanned = min(
            len(workspace_files),
            scanner_file_limit,
        )

        scanner_limit_exceeded = (
            len(workspace_files) > scanner_file_limit
        )

        if scanner_limit_exceeded:
            review_limitations.append(
                "The deterministic scanner file limit was "
                f"reached. Only the first {scanner_file_limit} "
                "files were scanned."
            )

        effective_files_skipped = (
            files_skipped
            + max(0, len(workspace_files) - files_scanned)
        )

        check_cancelled(cancel_event)

        # --------------------------------------------------
        # 2. Run deterministic security scanning.
        # --------------------------------------------------

        static_findings = self.scanner.scan_files(
            workspace_files,
            cancel_event=cancel_event,
        )

        check_cancelled(cancel_event)

        # --------------------------------------------------
        # 3. Parse AI-generated findings.
        # --------------------------------------------------

        parsed_report: ParsedSecurityReport = self.parser.parse(
            ai_response
        )

        ai_findings = parsed_report.findings

        review_limitations.extend(
            parsed_report.limitations
        )

        review_limitations.extend(
            parsed_report.warnings
        )

        check_cancelled(cancel_event)

        # --------------------------------------------------
        # 4. Combine findings without discarding evidence.
        # --------------------------------------------------

        combined_findings = (
            static_findings + ai_findings
        )

        # --------------------------------------------------
        # 5. Determine review completion status.
        # --------------------------------------------------

        static_analysis_completed = True
        ai_review_completed = parsed_report.parsing_successful

        if not parsed_report.parsing_successful:
            status = "PARTIAL"

        elif scanner_limit_exceeded:
            status = "PARTIAL"

        else:
            status = "COMPLETED"

        # --------------------------------------------------
        # 6. Apply the security policy.
        # --------------------------------------------------

        metadata = {
            "task_id": task_id,
            "scanner_findings": len(static_findings),
            "ai_findings": len(ai_findings),
            "combined_findings": len(combined_findings),
            "scanner_file_limit": scanner_file_limit,
            "files_received": len(workspace_files),
            "files_scanned": files_scanned,
            "files_skipped": effective_files_skipped,
            "ai_response_parsed": parsed_report.parsing_successful,
            "parsing_warning_count": len(
                parsed_report.warnings
            ),
        }

        review = self.policy_engine.evaluate(
            findings=combined_findings,
            files_scanned=files_scanned,
            files_skipped=effective_files_skipped,
            static_analysis_completed=static_analysis_completed,
            ai_review_completed=ai_review_completed,
            limitations=self._unique_strings(
                review_limitations
            ),
            summary=parsed_report.summary,
            status=status,
            metadata=metadata,
        )

        check_cancelled(cancel_event)

        # --------------------------------------------------
        # 7. Format the report.
        # --------------------------------------------------

        report = self.report_formatter.format(review)

        # --------------------------------------------------
        # 8. Record structured review history.
        # --------------------------------------------------

        history_record = self.history_manager.record(
            task_id=task_id,
            review=review,
        )

        return SecurityReviewServiceResult(
            review=review,
            report=report,
            static_findings=static_findings,
            ai_findings=ai_findings,
            parsing_warnings=parsed_report.warnings,
            history_record=history_record,
        )

    def get_review_history(
        self,
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve security review history for a task."""

        return self.history_manager.get_history(task_id)

    def get_latest_review(
        self,
        task_id: str,
    ) -> Optional[dict[str, Any]]:
        """Retrieve the latest security review for a task."""

        return self.history_manager.get_latest(task_id)

    def _unique_strings(
        self,
        values: list[str],
    ) -> list[str]:
        """Remove duplicate non-empty strings while preserving order."""

        unique = []
        seen = set()

        for value in values:

            if not isinstance(value, str):
                continue

            cleaned = value.strip()

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            unique.append(cleaned)

        return unique
