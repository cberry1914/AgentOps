
"""
AgentOps - Security Review Models

Structured models for recording security review findings
and security assessment results.

These models provide a consistent interface for:
- The Security Agent.
- Security policy evaluation.
- Workflow orchestration.
- Security review history.
- The future web dashboard.

Security findings represent potential vulnerabilities
identified through static analysis and AI-assisted review.
They do not automatically establish exploitability.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


SECURITY_SEVERITIES = {
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "INFO",
}


SECURITY_DECISIONS = {
    "PASS",
    "FAIL",
    "NEEDS_REVIEW",
    "NEEDS_INFO",
}


SECURITY_REVIEW_STATUSES = {
    "COMPLETED",
    "PARTIAL",
    "FAILED",
    "CANCELLED",
    "NOT_RUN",
}


@dataclass
class SecurityFinding:
    """Represents one potential security issue."""

    severity: str
    category: str
    title: str
    description: str

    file_path: Optional[str] = None
    line_number: Optional[int] = None

    evidence: Optional[str] = None
    recommendation: Optional[str] = None

    confidence: str = "MEDIUM"
    source: str = "ai_review"

    def __post_init__(self):
        if not isinstance(self.severity, str):
            raise TypeError("Finding severity must be a string.")

        self.severity = self.severity.upper()

        if self.severity not in SECURITY_SEVERITIES:
            raise ValueError(
                f"Unsupported security severity: {self.severity}"
            )

        required_text_fields = {
            "category": self.category,
            "title": self.title,
            "description": self.description,
        }

        for field_name, value in required_text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Finding {field_name} must be non-empty text."
                )

        if self.line_number is not None:
            if (
                not isinstance(self.line_number, int)
                or isinstance(self.line_number, bool)
            ):
                raise TypeError(
                    "Finding line number must be an integer or None."
                )

            if self.line_number < 1:
                raise ValueError(
                    "Finding line number must be positive."
                )

        if not isinstance(self.confidence, str):
            raise TypeError("Finding confidence must be a string.")

        self.confidence = self.confidence.upper()

        if self.confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError(
                "Finding confidence must be HIGH, MEDIUM, or LOW."
            )

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Finding source must be non-empty text.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the finding into a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityFinding":
        """Create a finding from serialized data."""

        if not isinstance(data, dict):
            raise TypeError("Security finding data must be a dictionary.")

        allowed_fields = cls.__dataclass_fields__.keys()

        filtered_data = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        return cls(**filtered_data)


@dataclass
class SecurityReview:
    """Structured result of a project security assessment."""

    decision: str
    status: str
    summary: str

    findings: list[SecurityFinding] = field(default_factory=list)

    files_scanned: int = 0
    files_skipped: int = 0

    static_analysis_completed: bool = False
    ai_review_completed: bool = False

    requires_human_review: bool = True

    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    reviewed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        if not isinstance(self.decision, str):
            raise TypeError("Security decision must be a string.")

        self.decision = self.decision.upper()

        if self.decision not in SECURITY_DECISIONS:
            raise ValueError(
                f"Unsupported security decision: {self.decision}"
            )

        if not isinstance(self.status, str):
            raise TypeError("Security review status must be a string.")

        self.status = self.status.upper()

        if self.status not in SECURITY_REVIEW_STATUSES:
            raise ValueError(
                f"Unsupported security review status: {self.status}"
            )

        if not isinstance(self.summary, str):
            raise TypeError("Security review summary must be a string.")

        if not isinstance(self.findings, list):
            raise TypeError("Security findings must be a list.")

        if not all(
            isinstance(finding, SecurityFinding)
            for finding in self.findings
        ):
            raise TypeError(
                "Every security finding must be a SecurityFinding."
            )

        for field_name, value in {
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
        }.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )

        if not isinstance(self.limitations, list):
            raise TypeError("Security limitations must be a list.")

        if not all(
            isinstance(item, str) for item in self.limitations
        ):
            raise TypeError(
                "Security limitations must contain strings."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "Security review metadata must be a dictionary."
            )

        # A PASS decision cannot be issued when serious findings
        # exist or the assessment is incomplete.
        if self.decision == "PASS":
            serious_findings = any(
                finding.severity in {"CRITICAL", "HIGH"}
                for finding in self.findings
            )

            if serious_findings:
                raise ValueError(
                    "A security review with critical or high findings "
                    "cannot receive a PASS decision."
                )

            if self.status != "COMPLETED":
                raise ValueError(
                    "An incomplete security review cannot receive PASS."
                )

            if not self.static_analysis_completed:
                raise ValueError(
                    "PASS requires completed static analysis."
                )

            if self.requires_human_review:
                raise ValueError(
                    "A review requiring human approval cannot receive PASS."
                )

    @property
    def passed(self) -> bool:
        return self.decision == "PASS"

    @property
    def failed(self) -> bool:
        return self.decision == "FAIL"

    @property
    def critical_count(self) -> int:
        return sum(
            finding.severity == "CRITICAL"
            for finding in self.findings
        )

    @property
    def high_count(self) -> int:
        return sum(
            finding.severity == "HIGH"
            for finding in self.findings
        )

    @property
    def medium_count(self) -> int:
        return sum(
            finding.severity == "MEDIUM"
            for finding in self.findings
        )

    @property
    def low_count(self) -> int:
        return sum(
            finding.severity == "LOW"
            for finding in self.findings
        )

    @property
    def info_count(self) -> int:
        return sum(
            finding.severity == "INFO"
            for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the review into a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityReview":
        """Create a security review from serialized data."""

        if not isinstance(data, dict):
            raise TypeError("Security review data must be a dictionary.")

        review_data = dict(data)

        review_data["findings"] = [
            finding
            if isinstance(finding, SecurityFinding)
            else SecurityFinding.from_dict(finding)
            for finding in review_data.get("findings", [])
        ]

        allowed_fields = cls.__dataclass_fields__.keys()

        filtered_data = {
            key: value
            for key, value in review_data.items()
            if key in allowed_fields
        }

        return cls(**filtered_data)
