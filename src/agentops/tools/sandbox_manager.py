
"""
AgentOps - Sandbox Manager

Coordinates Python and Node.js dependency preparation
and isolated test execution.

Responsibilities:
- Prepare the reusable Python sandbox image.
- Prepare the reusable Node.js sandbox image.
- Build project-specific dependency images.
- Discover supported Python and Node.js test projects.
- Execute pytest and npm tests in separate restricted containers.
- Preserve separate execution evidence for each environment.
- Propagate task cancellation.
- Return structured execution results.

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
from agentops.tools.node_docker_sandbox import NodeDockerSandbox
from agentops.tools.node_project_dependency_manager import (
    NodeProjectDependencyManager,
)
from agentops.tools.node_sandbox_image_builder import (
    NodeSandboxImageBuilder,
)
from agentops.tools.project_dependency_manager import (
    ProjectDependencyManager,
)
from agentops.tools.sandbox_image_builder import (
    ImageBuildResult,
    SandboxImageBuilder,
)
from agentops.tools.test_discovery import TestDiscovery


@dataclass
class SandboxPreparationResult:
    """Structured result from preparing the Python Docker sandbox."""

    ready: bool
    image_name: str
    image_was_built: bool
    error_message: Optional[str] = None
    build_result: Optional[ImageBuildResult] = None


@dataclass
class SandboxManagerResult:
    """
    Combined execution result from Python and Node.js sandboxes.

    The common fields preserve the interface previously returned
    by SandboxManager.run_tests().

    Individual results remain separate so that pytest output is
    never interpreted as npm test output, or vice versa.
    """

    status: str
    exit_code: Optional[int]
    duration_seconds: float
    stdout: str
    stderr: str
    execution_environment: str
    tests_executed: bool
    error_message: Optional[str] = None
    environment: Optional[str] = None

    python_result: Optional[DockerSandboxResult] = None
    node_result: Optional[DockerSandboxResult] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = self.execution_environment


class SandboxManager:
    """Coordinates sandbox preparation and test execution."""

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

        self.dependency_manager = ProjectDependencyManager()

        self.node_image_builder = NodeSandboxImageBuilder(
            project_root=project_root
        )

        self.node_dependency_manager = (
            NodeProjectDependencyManager()
        )

        self.test_discovery = TestDiscovery()

        # The Python base image builder currently builds one fixed image.
        if self.config.image != self.image_builder.IMAGE_NAME:
            raise ValueError(
                "The configured sandbox image does not match the "
                "image currently supported by SandboxImageBuilder. "
                f"Expected '{self.image_builder.IMAGE_NAME}', "
                f"received '{self.config.image}'."
            )

    # ---------------------------------------------------------
    # PYTHON BASE IMAGE PREPARATION
    # ---------------------------------------------------------

    def prepare(
        self,
        build_if_missing: bool = True,
        cancel_event: Optional[Event] = None,
    ) -> SandboxPreparationResult:
        """
        Verify that the Python sandbox image is available.

        If the image is missing and building is permitted, build it.
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
    # NODE.JS BASE IMAGE PREPARATION
    # ---------------------------------------------------------

    def _prepare_node_base_image(
        self,
        build_if_missing: bool,
        cancel_event: Optional[Event],
    ) -> tuple[bool, Optional[str]]:
        """Verify or build the reusable Node.js sandbox image."""

        try:
            check_cancelled(cancel_event)

            if self.node_image_builder.image_exists():
                return True, None

            if not build_if_missing:
                return False, (
                    "Node.js sandbox image is not available locally "
                    "and image building is disabled."
                )

            build_result = self.node_image_builder.build(
                cancel_event=cancel_event
            )

            check_cancelled(cancel_event)

            if not build_result.success:
                return False, (
                    build_result.error_message
                    or "Node.js sandbox image construction failed."
                )

            if not self.node_image_builder.image_exists():
                return False, (
                    "Node.js image construction reported success, "
                    "but the image could not be found afterward."
                )

            return True, None

        except TaskCancelledError:
            raise

        except Exception as exc:
            return False, (
                "Unexpected Node.js sandbox preparation error: "
                f"{type(exc).__name__}: {exc}"
            )

    # ---------------------------------------------------------
    # RESULT HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _failure_result(
        status: str,
        start_time: float,
        error_message: str,
        stdout: str = "",
        stderr: str = "",
    ) -> SandboxManagerResult:
        """Create a structured failure result."""

        return SandboxManagerResult(
            status=status,
            exit_code=None,
            duration_seconds=round(
                time.monotonic() - start_time,
                3,
            ),
            stdout=stdout,
            stderr=stderr,
            execution_environment="docker",
            tests_executed=False,
            error_message=error_message,
        )

    @staticmethod
    def _combine_results(
        python_result: Optional[DockerSandboxResult],
        node_result: Optional[DockerSandboxResult],
        start_time: float,
    ) -> SandboxManagerResult:
        """Combine execution status without merging test evidence."""

        results = [
            result
            for result in (python_result, node_result)
            if result is not None
        ]

        if not results:
            return SandboxManager._failure_result(
                status="NO_TESTS_FOUND",
                start_time=start_time,
                error_message=(
                    "No supported Python or Node.js test projects "
                    "were available for execution."
                ),
            )

        failed_statuses = [
            result.status.upper()
            for result in results
            if result.status.upper() != "PASSED"
        ]

        if failed_statuses:
            priority = (
                "CANCELLED",
                "TIMEOUT",
                "DEPENDENCY_PREPARATION_FAILED",
                "SANDBOX_UNAVAILABLE",
                "DOCKER_UNAVAILABLE",
                "IMAGE_NOT_FOUND",
                "INVALID_WORKSPACE",
                "FAILED",
                "EXECUTION_ERROR",
            )

            status = next(
                (
                    candidate
                    for candidate in priority
                    if candidate in failed_statuses
                ),
                failed_statuses[0],
            )

        else:
            status = "PASSED"

        exit_codes = [
            result.exit_code
            for result in results
            if result.exit_code is not None
        ]

        exit_code = (
            0
            if status == "PASSED"
            else next(
                (
                    code
                    for code in exit_codes
                    if code != 0
                ),
                None,
            )
        )

        environments = []

        if python_result is not None:
            environments.append("python")

        if node_result is not None:
            environments.append("node")

        errors = [
            result.error_message
            for result in results
            if result.error_message
        ]

        # Keep pytest output separate from npm output.
        # QAExecutionManager will consume node_result separately.
        python_stdout = (
            python_result.stdout
            if python_result is not None
            else ""
        )

        python_stderr = (
            python_result.stderr
            if python_result is not None
            else ""
        )

        return SandboxManagerResult(
            status=status,
            exit_code=exit_code,
            duration_seconds=round(
                time.monotonic() - start_time,
                3,
            ),
            stdout=python_stdout,
            stderr=python_stderr,
            execution_environment="docker",
            tests_executed=any(
                result.tests_executed
                for result in results
            ),
            error_message="; ".join(errors) if errors else None,
            environment="docker:" + "+".join(environments),
            python_result=python_result,
            node_result=node_result,
        )

    # ---------------------------------------------------------
    # TEST EXECUTION
    # ---------------------------------------------------------

    def run_tests(
        self,
        workspace_path: str | Path,
        cancel_event: Optional[Event] = None,
        build_if_missing: bool = True,
    ) -> SandboxManagerResult:
        """
        Prepare dependencies and execute discovered Python and
        Node.js tests in separate restricted Docker containers.

        Python and Node.js dependencies are installed in separate
        project-specific images.

        A failure in either environment makes the combined result
        unsuccessful.
        """

        start_time = time.monotonic()

        try:
            check_cancelled(cancel_event)

            workspace = Path(workspace_path).resolve(strict=True)

            if not workspace.is_dir():
                return self._failure_result(
                    status="INVALID_WORKSPACE",
                    start_time=start_time,
                    error_message="Workspace must be a directory.",
                )

            discovery = self.test_discovery.discover(
                workspace_path=workspace,
                cancel_event=cancel_event,
            )

            check_cancelled(cancel_event)

            if discovery.error_message:
                return self._failure_result(
                    status="INVALID_WORKSPACE",
                    start_time=start_time,
                    error_message=discovery.error_message,
                )

            if not discovery.discovery_complete:
                return self._failure_result(
                    status="EXECUTION_ERROR",
                    start_time=start_time,
                    error_message=(
                        "Test discovery was incomplete: "
                        + "; ".join(discovery.warnings)
                    ),
                )

            has_python_tests = (
                discovery.discovered_test_count > 0
            )

            has_node_tests = bool(
                discovery.node_project_directories
            )

            if not has_python_tests and not has_node_tests:
                return self._failure_result(
                    status="NO_TESTS_FOUND",
                    start_time=start_time,
                    error_message=(
                        "No supported Python pytest tests or "
                        "Node.js npm test projects were discovered."
                    ),
                )

            python_result = None
            node_result = None

            # -------------------------------------------------
            # PYTHON TEST EXECUTION
            # -------------------------------------------------

            if has_python_tests:
                check_cancelled(cancel_event)

                preparation = self.prepare(
                    build_if_missing=build_if_missing,
                    cancel_event=cancel_event,
                )

                if not preparation.ready:
                    return self._failure_result(
                        status="SANDBOX_UNAVAILABLE",
                        start_time=start_time,
                        error_message=preparation.error_message
                        or "Python sandbox preparation failed.",
                    )

                check_cancelled(cancel_event)

                dependency_result = self.dependency_manager.prepare(
                    project_root=workspace,
                    cancel_event=cancel_event,
                )

                check_cancelled(cancel_event)

                if not dependency_result.success:
                    return self._failure_result(
                        status="DEPENDENCY_PREPARATION_FAILED",
                        start_time=start_time,
                        error_message=(
                            dependency_result.error_message
                            or "Python dependency preparation failed."
                        ),
                        stdout=dependency_result.stdout,
                        stderr=dependency_result.stderr,
                    )

                python_sandbox = self.sandbox

                if dependency_result.image_name:
                    python_config = SandboxConfig(
                        **{
                            **self.config.__dict__,
                            "image": dependency_result.image_name,
                        }
                    )

                    python_sandbox = DockerSandbox(
                        config=python_config
                    )

                check_cancelled(cancel_event)

                python_result = python_sandbox.run_tests(
                    workspace_path=workspace,
                    cancel_event=cancel_event,
                )

            # -------------------------------------------------
            # NODE.JS TEST EXECUTION
            # -------------------------------------------------

            if has_node_tests:
                check_cancelled(cancel_event)

                node_ready, node_error = (
                    self._prepare_node_base_image(
                        build_if_missing=build_if_missing,
                        cancel_event=cancel_event,
                    )
                )

                if not node_ready:
                    return self._failure_result(
                        status="SANDBOX_UNAVAILABLE",
                        start_time=start_time,
                        error_message=node_error
                        or "Node.js sandbox preparation failed.",
                    )

                check_cancelled(cancel_event)

                node_dependency_result = (
                    self.node_dependency_manager.prepare(
                        project_root=workspace,
                        cancel_event=cancel_event,
                    )
                )

                check_cancelled(cancel_event)

                if not node_dependency_result.success:
                    return self._failure_result(
                        status="DEPENDENCY_PREPARATION_FAILED",
                        start_time=start_time,
                        error_message=(
                            node_dependency_result.error_message
                            or "Node.js dependency preparation failed."
                        ),
                        stdout=node_dependency_result.stdout,
                        stderr=node_dependency_result.stderr,
                    )

                # The dependency manager reports the manifest paths
                # it actually used. Mount that package directory.
                package_manifest = next(
                    (
                        item
                        for item in node_dependency_result.package_files
                        if item.endswith("package.json")
                    ),
                    None,
                )

                if package_manifest is None:
                    return self._failure_result(
                        status="DEPENDENCY_PREPARATION_FAILED",
                        start_time=start_time,
                        error_message=(
                            "Node.js dependency preparation did not "
                            "identify the package.json location."
                        ),
                    )

                node_workspace = (
                    workspace / Path(package_manifest).parent
                ).resolve(strict=True)

                if not node_workspace.is_dir():
                    return self._failure_result(
                        status="INVALID_WORKSPACE",
                        start_time=start_time,
                        error_message=(
                            "The discovered Node.js project directory "
                            "is not a valid directory."
                        ),
                    )

                node_image = (
                    node_dependency_result.image_name
                    or self.node_image_builder.IMAGE_NAME
                )

                node_config = SandboxConfig(
                    **{
                        **self.config.__dict__,
                        "image": node_image,
                    }
                )

                node_sandbox = NodeDockerSandbox(
                    config=node_config
                )

                check_cancelled(cancel_event)

                node_result = node_sandbox.run_tests(
                    workspace_path=node_workspace,
                    cancel_event=cancel_event,
                )

            return self._combine_results(
                python_result=python_result,
                node_result=node_result,
                start_time=start_time,
            )

        except TaskCancelledError:
            return self._failure_result(
                status="CANCELLED",
                start_time=start_time,
                error_message=(
                    "Sandbox preparation or execution was cancelled."
                ),
            )

        except Exception as exc:
            return self._failure_result(
                status="EXECUTION_ERROR",
                start_time=start_time,
                error_message=(
                    "Unexpected sandbox manager error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
