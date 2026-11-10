"""
AgentOps - Execution Models

Standardized models for recording isolated test execution results.

These models provide a consistent interface for:
- The Docker sandbox.
- The QA Agent.
- The workflow orchestrator.
- Execution history.
- The future web dashboard.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


EXECUTION_STATUSES = {
    "PENDING",
    "RUNNING",
    "PASSED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "NO_TESTS_FOUND",
    "NO_CODE",
    "SANDBOX_REQUIRED",
    "SANDBOX_UNAVAILABLE",
    "INVALID_WORKSPACE",
    "DOCKER_UNAVAILABLE",
    "IMAGE_NOT_FOUND",
    "EXECUTION_ERROR",
    "NOT_RUN",
}


@dataclass
class ExecutionResult:
    """Normalized record of a single test execution."""

    status: str
    exit_code: Optional[int] = None
    duration_seconds: float = 0.0

    stdout: str = ""
    stderr: str = ""

    # Both names are supported for compatibility.
    # Environment is the preferred field for new integrations.
    environment: Optional[str] = None
    execution_environment: str = "docker"

    tests_executed: bool = False
    test_count: Optional[int] = None

    error_message: Optional[str] = None

    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.status, str):
            raise TypeError("Execution status must be a string.")

        self.status = self.status.upper()

        if self.status not in EXECUTION_STATUSES:
            raise ValueError(
                f"Unsupported execution status: {self.status}"
            )

        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise TypeError("Execution duration must be numeric.")

        if self.duration_seconds < 0:
            raise ValueError("Execution duration cannot be negative.")

        self.duration_seconds = float(self.duration_seconds)

        if self.test_count is not None:
            if isinstance(self.test_count, bool) or not isinstance(
                self.test_count, int
            ):
                raise TypeError("Test count must be an integer or None.")

            if self.test_count < 0:
                raise ValueError("Test count cannot be negative.")

        if self.exit_code is not None:
            if isinstance(self.exit_code, bool) or not isinstance(
                self.exit_code, int
            ):
                raise TypeError("Exit code must be an integer or None.")

        if not isinstance(self.tests_executed, bool):
            raise TypeError("tests_executed must be a boolean.")

        if self.environment is not None:
            if not isinstance(self.environment, str):
                raise TypeError("Execution environment must be a string.")

            self.execution_environment = self.environment
        else:
            if not isinstance(self.execution_environment, str):
                raise TypeError("Execution environment must be a string.")

            self.environment = self.execution_environment

        if not isinstance(self.metadata, dict):
            raise TypeError("Execution metadata must be a dictionary.")

        if not isinstance(self.stdout, str):
            raise TypeError("Execution stdout must be a string.")

        if not isinstance(self.stderr, str):
            raise TypeError("Execution stderr must be a string.")

    @property
    def succeeded(self) -> bool:
        return self.status == "PASSED"

    @property
    def failed(self) -> bool:
        return self.status == "FAILED"

    @property
    def completed(self) -> bool:
        return self.status not in {"PENDING", "RUNNING"}

    @property
    def infrastructure_error(self) -> bool:
        return self.status in {
            "SANDBOX_REQUIRED",
            "SANDBOX_UNAVAILABLE",
            "DOCKER_UNAVAILABLE",
            "IMAGE_NOT_FOUND",
            "INVALID_WORKSPACE",
            "EXECUTION_ERROR",
        }

    @property
    def requires_review(self) -> bool:
        return (
            self.infrastructure_error
            or self.status in {
                "TIMEOUT",
                "CANCELLED",
                "NO_TESTS_FOUND",
                "NO_CODE",
                "NOT_RUN",
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionResult":
        """Create a result from a serialized dictionary."""

        if not isinstance(data, dict):
            raise TypeError("Execution result data must be a dictionary.")

        allowed_fields = cls.__dataclass_fields__.keys()

        filtered_data = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        # Older or incomplete records may not contain metadata.
        if filtered_data.get("metadata") is None:
            filtered_data["metadata"] = {}

        # Support records written before the environment field existed.
        if (
            "environment" not in filtered_data
            and "execution_environment" not in filtered_data
        ):
            filtered_data["environment"] = "docker"

        return cls(**filtered_data)

    @classmethod
    def not_run(
        cls,
        reason: str = "Tests have not been executed.",
    ) -> "ExecutionResult":
        """Create a result representing an unexecuted test suite."""

        return cls(
            status="NOT_RUN",
            error_message=reason,
            environment="none",
            tests_executed=False,
        )


def create_execution_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()
