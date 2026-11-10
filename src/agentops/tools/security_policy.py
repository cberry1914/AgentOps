
"""
AgentOps - Security Policy Engine

Evaluates security review findings and scan completeness.

Responsibilities:
- Combine static and AI-generated security findings.
- Apply consistent severity-based decisions.
- Identify incomplete security reviews.
- Require human review before security approval.
- Preserve review limitations and execution metadata.

The policy engine does not execute code or attempt to
prove that a reported vulnerability is exploitable.
"""

from typing import Any, Optional

from agentops.models.security_review import (
    SECURITY_DECISIONS,
    SECURITY_REVIEW_STATUSES,
    SecurityFinding,
    SecurityReview,
)


class SecurityPolicyEngine:
    """Evaluates security findings and produces a SecurityReview."""

    def evaluate(
        self,
        findings: Optional[list[SecurityFinding]] = None,
        files_scanned: int = 0,
        files_skipped: int = 0,
        static_analysis_completed: bool = False,
        ai_review_completed: bool = False,
        limitations: Optional[list[str]] = None,
        summary: str = "",
        status: str = "COMPLETED",
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecurityReview:
        """
        Evaluate a security review.

        Decision behavior:

        FAIL:
            At least one CRITICAL or HIGH finding exists.

        NEEDS_REVIEW:
            Findings exist, or the review requires human approval.

        NEEDS_INFO:
            No meaningful review could be completed.

        PASS:
            Not issued automatically. Human approval is required.
        """

        findings = list(findings or [])
        limitations = self._unique_strings(limitations or [])
        metadata = dict(metadata or {})

        self._validate_counts(files_scanned, files_skipped)

        if status not in SECURITY_REVIEW_STATUSES:
            raise ValueError(
                f"Unsupported security review status: {status}"
            )

        if any(
            not isinstance(finding, SecurityFinding)
            for finding in findings
        ):
            raise TypeError(
                "All findings must be SecurityFinding objects."
            )

        critical_count = self._count_severity(
            findings,
            "CRITICAL",
        )

        high_count = self._count_severity(
            findings,
            "HIGH",
        )

        medium_count = self._count_severity(
            findings,
            "MEDIUM",
        )

        low_count = self._count_severity(
            findings,
            "LOW",
        )

        info_count = self._count_severity(
            findings,
            "INFO",
        )

        requires_human_review = True

        decision = self._determine_decision(
            findings=findings,
            files_scanned=files_scanned,
            static_analysis_completed=static_analysis_completed,
            ai_review_completed=ai_review_completed,
            status=status,
        )

        review_summary = self._build_summary(
            decision=decision,
            status=status,
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            findings=findings,
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            static_analysis_completed=static_analysis_completed,
            ai_review_completed=ai_review_completed,
        )

        if summary.strip():
            review_summary = (
                f"{summary.strip()}\n\n{review_summary}"
            )

        review_metadata = {
            **metadata,
            "finding_count": len(findings),
            "critical_findings": critical_count,
            "high_findings": high_count,
            "medium_findings": medium_count,
            "low_findings": low_count,
            "informational_findings": info_count,
            "human_approval_required": requires_human_review,
            "automated_approval_permitted": False,
        }

        return SecurityReview(
            decision=decision,
            status=status,
            summary=review_summary,
            findings=findings,
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            static_analysis_completed=static_analysis_completed,
            ai_review_completed=ai_review_completed,
            requires_human_review=requires_human_review,
            limitations=limitations,
            metadata=review_metadata,
        )

    def _determine_decision(
        self,
        findings: list[SecurityFinding],
        files_scanned: int,
        static_analysis_completed: bool,
        ai_review_completed: bool,
        status: str,
    ) -> str:
        """Determine the security decision from available evidence."""

        if status in {"CANCELLED", "FAILED", "NOT_RUN"}:
            return "NEEDS_INFO"

        critical_or_high = any(
            finding.severity in {"CRITICAL", "HIGH"}
            for finding in findings
        )

        if critical_or_high:
            return "FAIL"

        if files_scanned == 0:
            return "NEEDS_INFO"

        if not static_analysis_completed and not ai_review_completed:
            return "NEEDS_INFO"

        if status == "PARTIAL":
            return "NEEDS_REVIEW"

        if findings:
            return "NEEDS_REVIEW"

        # A clean automated scan is not equivalent to a
        # verified secure application. Human approval remains
        # mandatory, so the engine never issues PASS.
        return "NEEDS_REVIEW"

    def _build_summary(
        self,
        decision: str,
        status: str,
        files_scanned: int,
        files_skipped: int,
        findings: list[SecurityFinding],
        critical_count: int,
        high_count: int,
        medium_count: int,
        static_analysis_completed: bool,
        ai_review_completed: bool,
    ) -> str:
        """Build a concise explanation of the policy decision."""

        lines = [
            f"Security decision: {decision}",
            f"Review status: {status}",
            f"Files scanned: {files_scanned}",
            f"Files skipped: {files_skipped}",
            f"Findings identified: {len(findings)}",
            f"Critical findings: {critical_count}",
            f"High findings: {high_count}",
            f"Medium findings: {medium_count}",
            (
                "Static analysis: "
                + (
                    "completed"
                    if static_analysis_completed
                    else "not completed"
                )
            ),
            (
                "AI review: "
                + (
                    "completed"
                    if ai_review_completed
                    else "not completed"
                )
            ),
        ]

        if decision == "FAIL":
            lines.append(
                "Critical or high-severity findings require "
                "remediation and human assessment."
            )

        elif decision == "NEEDS_INFO":
            lines.append(
                "Insufficient completed review evidence is "
                "available to make a security determination."
            )

        else:
            lines.append(
                "Human review is required before security approval."
            )

        return "\n".join(lines)

    def _count_severity(
        self,
        findings: list[SecurityFinding],
        severity: str,
    ) -> int:
        """Count findings matching a severity."""

        return sum(
            1
            for finding in findings
            if finding.severity == severity
        )

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

    def _validate_counts(
        self,
        files_scanned: int,
        files_skipped: int,
    ) -> None:
        """Validate file scan counts."""

        for name, value in (
            ("files_scanned", files_scanned),
            ("files_skipped", files_skipped),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
