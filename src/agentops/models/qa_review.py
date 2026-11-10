
"""
AgentOps - QA Review Models

Defines structured findings and QA review results.

The QA review model preserves automated test evidence,
static review results, human-review requirements, and
deterministic review decisions.

The LLM does not independently assign final QA approval.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


QA_DECISIONS = {
    "PASS",
    "FAIL",
    "NEEDS_INFO",
    "NEEDS_REVIEW",
}


@dataclass
class QAFinding:
    """Represents an individual QA finding."""

    severity: str
    category: str
    title: str
    description: str

    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize finding data."""

        self.severity = str(self.severity).upper().strip()
        self.category = str(self.category).strip()
        self.title = str(self.title).strip()
        self.description = str(self.description).strip()
        self.recommendation = str(
            self.recommendation or ""
        ).strip()

        if self.file_path is not None:
            self.file_path = str(self.file_path).strip() or None

        if self.severity not in {
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
            "INFO",
        }:
            raise ValueError(
                f"Unsupported QA finding severity: {self.severity}"
            )

        if not self.category:
            raise ValueError("Finding category cannot be empty.")

        if not self.title:
            raise ValueError("Finding title cannot be empty.")

        if not self.description:
            raise ValueError(
                "Finding description cannot be empty."
            )

        if self.line_number is not None:
            if (
                isinstance(self.line_number, bool)
                or not isinstance(self.line_number, int)
                or self.line_number < 1
            ):
                raise ValueError(
                    "Finding line number must be a positive integer."
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert the finding into a serializable dictionary."""

        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAFinding":
        """Create a finding from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("Finding data must be a dictionary.")

        return cls(
            severity=data.get("severity", "INFO"),
            category=data.get("category", "General"),
            title=data.get("title", "Untitled finding"),
            description=data.get("description", ""),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            recommendation=data.get("recommendation", ""),
        )


@dataclass
class QAReview:
    """Represents the structured result of a QA review."""

    decision: str
    summary: str

    automated_tests_executed: bool = False
    execution_status: str = "NOT_RUN"

    tests_discovered: int = 0
    tests_executed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0

    static_review_completed: bool = False

    findings: list[QAFinding] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    requires_human_review: bool = True

    reviewed_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the review."""

        self.decision = str(self.decision).upper().strip()
        self.summary = str(self.summary).strip()
        self.execution_status = str(
            self.execution_status
        ).upper().strip()

        if self.decision not in QA_DECISIONS:
            raise ValueError(
                f"Unsupported QA decision: {self.decision}"
            )

        if not self.summary:
            raise ValueError(
                "QA review summary cannot be empty."
            )

        if not isinstance(self.automated_tests_executed, bool):
            raise TypeError(
                "automated_tests_executed must be a boolean."
            )

        if not isinstance(self.static_review_completed, bool):
            raise TypeError(
                "static_review_completed must be a boolean."
            )

        if not isinstance(self.requires_human_review, bool):
            raise TypeError(
                "requires_human_review must be a boolean."
            )

        count_fields = (
            "tests_discovered",
            "tests_executed",
            "tests_passed",
            "tests_failed",
            "tests_skipped",
        )

        for field_name in count_fields:
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )

        if not isinstance(self.findings, list):
            raise TypeError("Findings must be a list.")

        self.findings = [
            finding
            if isinstance(finding, QAFinding)
            else QAFinding.from_dict(finding)
            for finding in self.findings
        ]

        if not isinstance(self.limitations, list):
            raise TypeError("Limitations must be a list.")

        self.limitations = [
            str(item)
            for item in self.limitations
        ]

        if not isinstance(self.metadata, dict):
            raise TypeError("Review metadata must be a dictionary.")

        if self.decision == "PASS":
            high_severity_findings = any(
                finding.severity in {"CRITICAL", "HIGH"}
                for finding in self.findings
            )

            if high_severity_findings:
                raise ValueError(
                    "A QA review with critical or high findings "
                    "cannot receive PASS."
                )

            if not self.automated_tests_executed:
                raise ValueError(
                    "A QA review cannot PASS without automated "
                    "test execution."
                )

            if self.execution_status != "PASSED":
                raise ValueError(
                    "A QA review cannot PASS unless automated "
                    "execution status is PASSED."
                )

            if self.tests_executed < 1:
                raise ValueError(
                    "A QA review cannot PASS without executed tests."
                )

            if self.tests_failed > 0:
                raise ValueError(
                    "A QA review cannot PASS when tests have failed."
                )

            if not self.static_review_completed:
                raise ValueError(
                    "A QA review cannot PASS without a completed "
                    "static review."
                )

            if self.requires_human_review:
                raise ValueError(
                    "A QA review requiring human approval "
                    "cannot receive PASS."
                )

    @property
    def passed(self) -> bool:
        """Return whether the review satisfies all PASS conditions."""

        return self.decision == "PASS"

    @property
    def failed(self) -> bool:
        """Return whether the review received a FAIL decision."""

        return self.decision == "FAIL"

    @property
    def finding_count(self) -> int:
        """Return the total number of findings."""

        return len(self.findings)

    @property
    def high_severity_count(self) -> int:
        """Return the number of critical and high findings."""

        return sum(
            1
            for finding in self.findings
            if finding.severity in {"CRITICAL", "HIGH"}
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the review into a serializable dictionary."""

        return {
            "decision": self.decision,
            "summary": self.summary,
            "automated_tests_executed": (
                self.automated_tests_executed
            ),
            "execution_status": self.execution_status,
            "tests_discovered": self.tests_discovered,
            "tests_executed": self.tests_executed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "static_review_completed": (
                self.static_review_completed
            ),
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "limitations": list(self.limitations),
            "requires_human_review": self.requires_human_review,
            "reviewed_at": self.reviewed_at,
            "metadata": dict(self.metadata),
            "finding_count": self.finding_count,
            "high_severity_count": self.high_severity_count,
            "passed": self.passed,
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAReview":
        """Reconstruct a QA review from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("QA review data must be a dictionary.")

        return cls(
            decision=data.get("decision", "NEEDS_REVIEW"),
            summary=data.get(
                "summary",
                "No QA review summary was provided.",
            ),
            automated_tests_executed=data.get(
                "automated_tests_executed",
                False,
            ),
            execution_status=data.get(
                "execution_status",
                "NOT_RUN",
            ),
            tests_discovered=data.get("tests_discovered", 0),
            tests_executed=data.get("tests_executed", 0),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            tests_skipped=data.get("tests_skipped", 0),
            static_review_completed=data.get(
                "static_review_completed",
                False,
            ),
            findings=data.get("findings", []),
            limitations=data.get("limitations", []),
            requires_human_review=data.get(
                "requires_human_review",
                True,
            ),
            reviewed_at=data.get(
                "reviewed_at",
                datetime.now(timezone.utc).isoformat(),
            ),
            metadata=data.get("metadata") or {},
        )
