
"""
AgentOps - Test Report Parser

Parses pytest output into structured execution evidence.

The parser does not decide whether an application is
acceptable. It reports what pytest actually collected
and executed so the QA policy engine can make that
decision.

Collection errors are tracked separately from individual
test-case counts.
"""

import re

from dataclasses import dataclass


@dataclass
class ParsedTestReport:
    """Structured information extracted from pytest output."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    warnings: int = 0
    collection_errors: int = 0
    duration_seconds: float = 0.0
    summary_found: bool = False
    no_tests_collected: bool = False

    @property
    def has_failures(self) -> bool:
        return (
            self.failed > 0
            or self.errors > 0
            or self.collection_errors > 0
        )

    @property
    def successful(self) -> bool:
        """
        Return whether the report provides complete passing evidence.

        A successful report must contain a summary, at least one
        test case, internally consistent counts, and no failures
        or collection errors.
        """

        counted_tests = (
            self.passed
            + self.failed
            + self.skipped
            + self.errors
        )

        return (
            self.summary_found
            and self.total > 0
            and counted_tests == self.total
            and not self.has_failures
            and not self.no_tests_collected
        )


class TestReportParser:
    """Extract test counts and errors from pytest output."""

    # Supports both plain summaries and pytest-decorated summaries:
    # 5 passed in 0.42s
    # ===== 5 passed in 0.42s =====
    SUMMARY_PATTERN = re.compile(
        r"^\s*=*\s*"
        r"(?P<summary>.*?)\bin\s+"
        r"(?P<duration>\d+(?:\.\d+)?)s"
        r"\s*=*\s*$",
        re.IGNORECASE,
    )

    COUNT_PATTERN = re.compile(
        r"(?P<count>\d+)\s+"
        r"(?P<result>passed|failed|skipped|errors?|warnings?)\b",
        re.IGNORECASE,
    )

    # Supports decorated pytest collection headings.
    COLLECTION_ERROR_PATTERN = re.compile(
        r"^\s*_*\s*ERROR collecting\b",
        re.IGNORECASE,
    )

    WARNING_LINE_PATTERN = re.compile(
        r"^\s*.+:\d+:\s+\w*Warning:",
        re.IGNORECASE,
    )

    WARNING_SECTION_MARKERS = (
        "warnings summary",
        "warning summary",
    )

    def parse(
        self,
        output: str,
        exit_code: int | None = None,
    ) -> ParsedTestReport:
        """
        Parse combined pytest stdout and stderr.

        Args:
            output: Combined pytest output.
            exit_code: Process exit code, when available.

        Returns:
            ParsedTestReport containing observed test evidence.
        """

        report = ParsedTestReport()

        if not output or not output.strip():
            report.no_tests_collected = exit_code == 5
            return report

        lines = output.splitlines()

        self._count_collection_errors(lines, report)

        summary_line = self._find_summary_line(lines)

        if summary_line is not None:
            self._parse_summary(summary_line, report)

        self._count_warning_lines(lines, report)

        if not report.summary_found:
            report.no_tests_collected = (
                self._detect_no_tests_collected(output)
            )
        elif report.total == 0:
            report.no_tests_collected = True

        # Pytest exit code 5 means no tests were collected.
        # Collection errors take precedence over this condition.
        if (
            exit_code == 5
            and report.total == 0
            and report.collection_errors == 0
        ):
            report.no_tests_collected = True

        return report

    def _find_summary_line(
        self,
        lines: list[str],
    ) -> str | None:
        """Find the final pytest summary line containing duration."""

        for line in reversed(lines):
            stripped = line.strip()

            if self.SUMMARY_PATTERN.search(stripped):
                return stripped

        return None

    def _parse_summary(
        self,
        line: str,
        report: ParsedTestReport,
    ) -> None:
        """Extract counts from a pytest summary line."""

        match = self.SUMMARY_PATTERN.search(line)

        if match is None:
            return

        summary_text = match.group("summary")
        duration_text = match.group("duration")

        try:
            report.duration_seconds = float(duration_text)
        except (TypeError, ValueError):
            report.duration_seconds = 0.0

        counts = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error": 0,
            "warnings": 0,
        }

        for count_match in self.COUNT_PATTERN.finditer(summary_text):
            result_name = count_match.group("result").lower()
            count = int(count_match.group("count"))

            if result_name == "passed":
                counts["passed"] += count

            elif result_name == "failed":
                counts["failed"] += count

            elif result_name == "skipped":
                counts["skipped"] += count

            elif result_name in {"error", "errors"}:
                counts["error"] += count

            elif result_name in {"warning", "warnings"}:
                counts["warnings"] += count

        report.passed = counts["passed"]
        report.failed = counts["failed"]
        report.skipped = counts["skipped"]
        report.warnings = counts["warnings"]

        # Pytest can include collection errors in its summary
        # under "error". Keep collection errors separate from
        # individual test-case errors.
        report.errors = max(
            0,
            counts["error"] - report.collection_errors,
        )

        report.total = (
            report.passed
            + report.failed
            + report.skipped
            + report.errors
        )

        report.summary_found = True
        report.no_tests_collected = report.total == 0

    def _count_collection_errors(
        self,
        lines: list[str],
        report: ParsedTestReport,
    ) -> None:
        """
        Count pytest collection error sections.

        ImportError and SyntaxError lines are deliberately not
        counted independently because they commonly appear as
        details inside an existing collection error section.
        """

        report.collection_errors = sum(
            1
            for line in lines
            if self.COLLECTION_ERROR_PATTERN.match(line)
        )

    def _count_warning_lines(
        self,
        lines: list[str],
        report: ParsedTestReport,
    ) -> None:
        """
        Estimate warning presence when pytest provides no count.

        The fallback counts recognized warning records. If those
        are unavailable, it checks for non-decoration content
        inside the warning summary section.
        """

        if report.warnings > 0:
            return

        warning_count = sum(
            1
            for line in lines
            if self.WARNING_LINE_PATTERN.match(line)
        )

        report.warnings = warning_count

        if warning_count > 0:
            return

        in_warning_section = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            normalized = stripped.strip("= ").lower()

            if any(
                normalized.startswith(marker)
                for marker in self.WARNING_SECTION_MARKERS
            ):
                in_warning_section = True
                continue

            if in_warning_section:
                # A new decorated pytest section ends the
                # warning summary.
                if stripped.startswith("="):
                    break

                report.warnings = 1
                break

    @staticmethod
    def _detect_no_tests_collected(
        output: str,
    ) -> bool:
        """Recognize common pytest messages for empty collections."""

        normalized = output.lower()

        markers = (
            "no tests ran",
            "collected 0 items",
            "collected 0 tests",
            "no tests collected",
        )

        return any(
            marker in normalized
            for marker in markers
        )
