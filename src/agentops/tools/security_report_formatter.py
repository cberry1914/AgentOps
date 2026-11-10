
"""
AgentOps - Security Report Formatter

Converts a structured SecurityReview into a readable report.

The formatter presents:
- Security decision and review status.
- File scanning statistics.
- Severity counts.
- Individual security findings.
- Recommendations.
- Review limitations.
- Additional metadata.

This component only formats review data. It does not
change findings or make security policy decisions.
"""

from typing import Any

from agentops.models.security_review import SecurityReview


class SecurityReportFormatter:
    """Formats SecurityReview objects into readable reports."""

    def format(self, review: SecurityReview) -> str:
        """Generate a complete human-readable security report."""

        if not isinstance(review, SecurityReview):
            raise TypeError(
                "review must be a SecurityReview instance."
            )

        sections = [
            self._format_header(review),
            self._format_scan_summary(review),
            self._format_findings(review),
            self._format_limitations(review),
            self._format_metadata(review),
        ]

        return "\n\n".join(
            section for section in sections if section.strip()
        )

    def _format_header(self, review: SecurityReview) -> str:
        """Format the decision and overall review summary."""

        lines = [
            "=" * 60,
            "AGENTOPS SECURITY REVIEW",
            "=" * 60,
            f"Decision: {review.decision}",
            f"Status: {review.status}",
            f"Reviewed at: {review.reviewed_at}",
            "",
            review.summary,
            "",
            (
                "Human approval required: "
                + (
                    "Yes"
                    if review.requires_human_review
                    else "No"
                )
            ),
        ]

        return "\n".join(lines)

    def _format_scan_summary(self, review: SecurityReview) -> str:
        """Format file scanning and review completion statistics."""

        metadata = review.metadata or {}

        critical = self._safe_count(
            metadata.get("critical_findings", 0)
        )

        high = self._safe_count(
            metadata.get("high_findings", 0)
        )

        medium = self._safe_count(
            metadata.get("medium_findings", 0)
        )

        low = self._safe_count(
            metadata.get("low_findings", 0)
        )

        informational = self._safe_count(
            metadata.get("informational_findings", 0)
        )

        lines = [
            "SCAN SUMMARY",
            "-" * 60,
            f"Files scanned: {review.files_scanned}",
            f"Files skipped: {review.files_skipped}",
            (
                "Static analysis completed: "
                + self._yes_no(review.static_analysis_completed)
            ),
            (
                "AI review completed: "
                + self._yes_no(review.ai_review_completed)
            ),
            "",
            "FINDINGS BY SEVERITY",
            f"Critical: {critical}",
            f"High: {high}",
            f"Medium: {medium}",
            f"Low: {low}",
            f"Informational: {informational}",
            f"Total: {len(review.findings)}",
        ]

        return "\n".join(lines)

    def _format_findings(self, review: SecurityReview) -> str:
        """Format each finding and its remediation recommendation."""

        lines = [
            "SECURITY FINDINGS",
            "-" * 60,
        ]

        if not review.findings:
            lines.append("No security findings were reported.")
            return "\n".join(lines)

        for index, finding in enumerate(review.findings, start=1):

            lines.extend([
                "",
                f"{index}. [{finding.severity}] {finding.title}",
                f"Category: {finding.category}",
                f"Confidence: {finding.confidence}",
                f"Source: {finding.source}",
                f"Description: {finding.description}",
            ])

            if finding.file_path:
                location = finding.file_path

                if finding.line_number is not None:
                    location += f":{finding.line_number}"

                lines.append(f"Location: {location}")

            if finding.evidence:
                lines.append(
                    f"Evidence: {finding.evidence}"
                )

            if finding.recommendation:
                lines.append(
                    f"Recommendation: {finding.recommendation}"
                )

        return "\n".join(lines)

    def _format_limitations(self, review: SecurityReview) -> str:
        """Format limitations and incomplete review information."""

        lines = [
            "REVIEW LIMITATIONS",
            "-" * 60,
        ]

        if not review.limitations:
            lines.append("No limitations were reported.")

        else:
            for limitation in review.limitations:
                lines.append(f"- {limitation}")

        return "\n".join(lines)

    def _format_metadata(self, review: SecurityReview) -> str:
        """Format additional review metadata."""

        metadata = review.metadata or {}

        if not metadata:
            return ""

        lines = [
            "REVIEW METADATA",
            "-" * 60,
        ]

        for key, value in metadata.items():

            if isinstance(value, (dict, list)):
                rendered = str(value)
            else:
                rendered = str(value)

            lines.append(f"{key}: {rendered}")

        return "\n".join(lines)

    def _safe_count(self, value: Any) -> int:
        """Return a non-negative integer for display."""

        if isinstance(value, bool):
            return 0

        try:
            count = int(value)

        except (TypeError, ValueError):
            return 0

        return max(0, count)

    def _yes_no(self, value: bool) -> str:
        """Convert a boolean into readable status text."""

        return "Yes" if value else "No"
