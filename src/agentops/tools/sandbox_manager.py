
"""
AgentOps - Sandbox Manager

Coordinates sandbox preparation and isolated test execution.

Responsibilities:
- Verify Docker availability through the sandbox components.
- Check whether the sandbox image exists.
- Build the image when necessary.
- Propagate task cancellation during image construction.
- Execute tests through DockerSandbox.
- Convert unexpected preparation and execution exceptions
  into structured results.
- Return structured preparation and execution results.

The manager does not execute project code directly.
"""

import time

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)
from agentops.tools.docker_sandbox import (
    DockerSandbox,
    DockerSandboxResult,
    SandboxConfig,
)
from agentops.tools.sandbox_image_builder import (
    ImageBuildResult,
    SandboxImageBuilder,
)


@dataclass
class SandboxPreparationResult:
    """Structured result from preparing the Docker sandbox."""

    ready: bool
    image_name: str
    image_was_built: bool
    error_message: Optional[str] = None
    build_result: Optional[ImageBuildResult] = None


class SandboxManager:
    """Coordinates sandbox image preparation and test execution."""

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        config: Optional[SandboxConfig] = None,
    ):
        self.config = config or SandboxConfig()

        self.image_builder = SandboxImageBuilder(
            project_root=project_root
        )

        self.sandbox = DockerSandbox(
            config=self.config
        )

        # The current image builder builds one fixed image tag.
        # Prevent preparation and execution from targeting different
        # images when a custom SandboxConfig is supplied.
        if self.config.image != self.image_builder.IMAGE_NAME:
            raise ValueError(
                "The configured sandbox image does not match the "
                "image currently supported by SandboxImageBuilder. "
                f"Expected '{self.image_builder.IMAGE_NAME}', "
                f"received '{self.config.image}'."
            )

    # ---------------------------------------------------------
    # IMAGE PREPARATION
    # ---------------------------------------------------------

    def prepare(
        self,
        build_if_missing: bool = True,
        cancel_event: Optional[Event] = None,
    ) -> SandboxPreparationResult:
        """
        Verify that the sandbox image is available.

        If the image is missing and building is permitted, build it.

        Unexpected preparation exceptions become structured
        preparation failures. Cancellation is always propagated
        separately so callers can preserve its meaning.
        """

        image_name = self.config.image

        try:
            check_cancelled(cancel_event)

            if self.image_builder.image_exists():
                return SandboxPreparationResult(
                    ready=True,
                    image_name=image_name,
                    image_was_built=False,
                )

            check_cancelled(cancel_event)

            if not build_if_missing:
                return SandboxPreparationResult(
                    ready=False,
                    image_name=image_name,
                    image_was_built=False,
                    error_message=(
                        "Sandbox image is not available locally and "
                        "image building is disabled."
                    ),
                )

            build_result = self.image_builder.build(
                cancel_event=cancel_event
            )

            check_cancelled(cancel_event)

            if not build_result.success:
                return SandboxPreparationResult(
                    ready=False,
                    image_name=image_name,
                    image_was_built=False,
                    error_message=(
                        build_result.error_message
                        or "Sandbox image construction failed."
                    ),
                    build_result=build_result,
                )

            if not self.image_builder.image_exists():
                return SandboxPreparationResult(
                    ready=False,
                    image_name=image_name,
                    image_was_built=True,
                    error_message=(
                        "The image build reported success, but the "
                        "sandbox image could not be found afterward."
                    ),
                    build_result=build_result,
                )

            check_cancelled(cancel_event)

            return SandboxPreparationResult(
                ready=True,
                image_name=image_name,
                image_was_built=True,
                build_result=build_result,
            )

        except TaskCancelledError:
            raise

        except Exception as exc:
            return SandboxPreparationResult(
                ready=False,
                image_name=image_name,
                image_was_built=False,
                error_message=(
                    "Unexpected sandbox preparation error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # ---------------------------------------------------------
    # TEST EXECUTION
    # ---------------------------------------------------------

    def run_tests(
        self,
        workspace_path: str | Path,
        cancel_event: Optional[Event] = None,
        build_if_missing: bool = True,
    ) -> DockerSandboxResult:
        """
        Prepare the sandbox and execute tests.

        Preparation failures are converted into structured
        SANDBOX_UNAVAILABLE results.

        Unexpected execution exceptions are converted into
        structured EXECUTION_ERROR results.

        Cancellation is preserved as a distinct CANCELLED result.

        Duration represents the complete manager operation,
        including image preparation when a build is necessary.
        """

        start_time = time.monotonic()

        try:
            check_cancelled(cancel_event)

            preparation = self.prepare(
                build_if_missing=build_if_missing,
                cancel_event=cancel_event,
            )

            if not preparation.ready:
                check_cancelled(cancel_event)

                return DockerSandboxResult(
                    status="SANDBOX_UNAVAILABLE",
                    exit_code=None,
                    duration_seconds=round(
                        time.monotonic() - start_time,
                        3,
                    ),
                    stdout="",
                    stderr="",
                    execution_environment="docker",
                    tests_executed=False,
                    error_message=preparation.error_message,
                )

            check_cancelled(cancel_event)

            execution_result = self.sandbox.run_tests(
                workspace_path=workspace_path,
                cancel_event=cancel_event,
            )

            # Preserve the sandbox's execution evidence while
            # reporting the total time spent preparing and running.
            execution_result.duration_seconds = round(
                time.monotonic() - start_time,
                3,
            )

            return execution_result

        except TaskCancelledError:
            return DockerSandboxResult(
                status="CANCELLED",
                exit_code=None,
                duration_seconds=round(
                    time.monotonic() - start_time,
                    3,
                ),
                stdout="",
                stderr="",
                execution_environment="docker",
                tests_executed=False,
                error_message=(
                    "Sandbox preparation or execution was cancelled."
                ),
            )

        except Exception as exc:
            return DockerSandboxResult(
                status="EXECUTION_ERROR",
                exit_code=None,
                duration_seconds=round(
                    time.monotonic() - start_time,
                    3,
                ),
                stdout="",
                stderr="",
                execution_environment="docker",
                tests_executed=False,
                error_message=(
                    "Unexpected sandbox manager error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
