
"""
AgentOps - Test Runner

Temporary safe implementation.

Generated code must never execute directly on the
AgentOps host machine.

Automated execution will be enabled only after the
Docker-based isolation environment is implemented.
"""

import re
import time

from dataclasses import dataclass


@dataclass
class TestRunResult:
    """Structured result from an automated test execution."""

    status: str
    exit_code: int | None
    duration_seconds: float
    stdout: str
    stderr: str


class TestRunner:
    """
    Test runner interface.

    Execution is intentionally disabled until the
    Docker isolation environment is available.
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def extract_python_code(response: str) -> str:
        """Extract Python code blocks without executing them."""

        pattern = r"```(?:python|py)\s*\n(.*?)```"

        matches = re.findall(
            pattern,
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        return "\n\n".join(matches).strip()

    def run_implementation(self, response: str) -> TestRunResult:
        """
        Refuse execution until an isolated runner is configured.

        This method intentionally performs no filesystem
        writes and starts no subprocesses.
        """

        start_time = time.perf_counter()

        code = self.extract_python_code(response)

        duration = round(
            time.perf_counter() - start_time,
            3,
        )

        if not code:
            return TestRunResult(
                status="NO_CODE",
                exit_code=None,
                duration_seconds=duration,
                stdout="No Python code block was found.",
                stderr="",
            )

        return TestRunResult(
            status="SANDBOX_REQUIRED",
            exit_code=None,
            duration_seconds=duration,
            stdout=(
                "Code was identified, but automated execution "
                "is currently disabled."
            ),
            stderr=(
                "Execution is blocked because the Docker-based "
                "isolation environment has not been implemented. "
                "No code was executed on the host machine."
            ),
        )
