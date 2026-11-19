
"""
AgentOps - Project Dependency Manager

Builds isolated Docker images containing the Python dependencies
declared by a project's root requirements.txt file.

Project dependencies are installed inside Docker, never into the
AgentOps host Python environment.

Only the requirements manifest is copied into the temporary
Docker build context. Project-provided Dockerfiles and setup
scripts are never executed by this component.
"""

import hashlib
import subprocess
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)


@dataclass
class DependencyPreparationResult:
    """Structured result from preparing project dependencies."""

    success: bool
    image_name: Optional[str]
    image_was_built: bool
    requirements_files: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None


class ProjectDependencyManager:
    """Builds and reuses project-specific dependency images."""

    BASE_IMAGE = "agentops-python-sandbox:latest"
    BUILD_TIMEOUT_SECONDS = 900
    POLL_INTERVAL_SECONDS = 0.5
    DOCKER_INSPECT_TIMEOUT_SECONDS = 15

    def __init__(
        self,
        build_timeout_seconds: int = BUILD_TIMEOUT_SECONDS,
    ):
        if (
            isinstance(build_timeout_seconds, bool)
            or not isinstance(build_timeout_seconds, int)
            or build_timeout_seconds < 1
        ):
            raise ValueError(
                "Build timeout must be a positive integer."
            )

        self.build_timeout_seconds = build_timeout_seconds

    # ---------------------------------------------------------
    # REQUIREMENTS DISCOVERY
    # ---------------------------------------------------------

    def _discover_requirements(
        self,
        project_root: Path,
    ) -> list[Path]:
        """
        Use only the root-level requirements.txt.

        Nested requirements files are excluded until explicit
        multi-environment support is implemented.
        """

        requirements_file = project_root / "requirements.txt"

        if (
            requirements_file.is_symlink()
            or not requirements_file.is_file()
        ):
            return []

        return [requirements_file]

    # ---------------------------------------------------------
    # IMAGE IDENTIFICATION
    # ---------------------------------------------------------

    def _create_image_name(
        self,
        project_root: Path,
        requirements_files: list[Path],
    ) -> str:
        """
        Generate a stable image tag from the project and manifest.

        A changed requirements file produces a different image tag.
        """

        digest = hashlib.sha256()

        digest.update(str(project_root).encode("utf-8"))

        for requirements_file in requirements_files:
            relative_path = requirements_file.relative_to(
                project_root
            ).as_posix()

            digest.update(relative_path.encode("utf-8"))
            digest.update(requirements_file.read_bytes())

        return (
            "agentops-project-"
            f"{digest.hexdigest()[:16]}:latest"
        )

    # ---------------------------------------------------------
    # IMAGE CACHE
    # ---------------------------------------------------------

    def _image_exists(
        self,
        image_name: str,
    ) -> bool:
        """Check whether the requested project image exists locally."""

        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    image_name,
                ],
                capture_output=True,
                text=True,
                timeout=self.DOCKER_INSPECT_TIMEOUT_SECONDS,
                check=False,
            )

        except (subprocess.TimeoutExpired, OSError):
            return False

        return result.returncode == 0

    # ---------------------------------------------------------
    # DOCKER BUILD
    # ---------------------------------------------------------

    def _create_dockerfile(
        self,
        requirements_files: list[Path],
        project_root: Path,
        build_context: Path,
    ) -> Path:
        """
        Create an AgentOps-controlled Dockerfile.

        Only the root requirements manifest is copied.
        """

        requirements_directory = build_context / "requirements"
        requirements_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for requirements_file in requirements_files:
            relative_path = requirements_file.relative_to(
                project_root
            )

            destination = requirements_directory / relative_path
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination.write_bytes(
                requirements_file.read_bytes()
            )

        dockerfile = build_context / "Dockerfile"

        dockerfile.write_text(
            "\n".join(
                [
                    f"FROM {self.BASE_IMAGE}",
                    "USER root",
                    "ENV PIP_NO_CACHE_DIR=1",
                    (
                        "COPY requirements/ "
                        "/tmp/project-requirements/"
                    ),
                    (
                        "RUN python -m pip install "
                        "--no-cache-dir "
                        "-r /tmp/project-requirements/requirements.txt "
                        "&& rm -rf /tmp/project-requirements"
                    ),
                    "USER 10001:10001",
                    "WORKDIR /workspace",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return dockerfile

    def _build_image(
        self,
        image_name: str,
        dockerfile: Path,
        build_context: Path,
        cancel_event: Optional[Event],
    ) -> tuple[str, str]:
        """
        Build the project image.

        Checks the Docker exit code before reporting success.
        Supports cooperative cancellation and build timeouts.
        """

        command = [
            "docker",
            "build",
            "--pull=false",
            "--tag",
            image_name,
            "--file",
            str(dockerfile),
            str(build_context),
        ]

        process = None
        start_time = time.monotonic()

        try:
            check_cancelled(cancel_event)

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            while True:
                check_cancelled(cancel_event)

                elapsed = time.monotonic() - start_time
                remaining = self.build_timeout_seconds - elapsed

                if remaining <= 0:
                    raise TimeoutError(
                        "Project dependency image build exceeded "
                        f"{self.build_timeout_seconds} seconds."
                    )

                try:
                    stdout, stderr = process.communicate(
                        timeout=min(
                            self.POLL_INTERVAL_SECONDS,
                            remaining,
                        )
                    )
                    break

                except subprocess.TimeoutExpired:
                    continue

            if process.returncode != 0:
                raise RuntimeError(
                    "Docker dependency image build failed. "
                    f"Exit code: {process.returncode}. "
                    f"Output: {stderr[-4000:]}"
                )

            return stdout, stderr

        finally:
            if process is not None and process.poll() is None:
                process.terminate()

                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def prepare(
        self,
        project_root: str | Path,
        cancel_event: Optional[Event] = None,
    ) -> DependencyPreparationResult:
        """
        Prepare an image containing the project's dependencies.

        Projects without a root requirements.txt continue using
        the base sandbox image.

        Existing images are reused when the manifest is unchanged.
        Cancellation is propagated to the caller.
        """

        try:
            check_cancelled(cancel_event)

            root = Path(project_root)

            if root.is_symlink():
                raise ValueError(
                    "Project root cannot be a symbolic link."
                )

            if not root.exists() or not root.is_dir():
                raise ValueError(
                    "Project root must be an existing directory."
                )

            root = root.resolve(strict=True)

            requirements_files = self._discover_requirements(root)

            if not requirements_files:
                return DependencyPreparationResult(
                    success=True,
                    image_name=None,
                    image_was_built=False,
                    error_message=(
                        "No root requirements.txt was found. "
                        "The base sandbox image will be used."
                    ),
                )

            image_name = self._create_image_name(
                root,
                requirements_files,
            )

            check_cancelled(cancel_event)

            # Reuse the image if its manifest-derived tag exists.
            if self._image_exists(image_name):
                return DependencyPreparationResult(
                    success=True,
                    image_name=image_name,
                    image_was_built=False,
                    requirements_files=tuple(
                        path.relative_to(root).as_posix()
                        for path in requirements_files
                    ),
                )

            check_cancelled(cancel_event)

            with tempfile.TemporaryDirectory(
                prefix="agentops-dependencies-"
            ) as temporary_directory:

                build_context = Path(temporary_directory)

                dockerfile = self._create_dockerfile(
                    requirements_files=requirements_files,
                    project_root=root,
                    build_context=build_context,
                )

                stdout, stderr = self._build_image(
                    image_name=image_name,
                    dockerfile=dockerfile,
                    build_context=build_context,
                    cancel_event=cancel_event,
                )

            check_cancelled(cancel_event)

            # Verify that Docker actually created the image.
            if not self._image_exists(image_name):
                return DependencyPreparationResult(
                    success=False,
                    image_name=image_name,
                    image_was_built=True,
                    requirements_files=tuple(
                        path.relative_to(root).as_posix()
                        for path in requirements_files
                    ),
                    stdout=stdout,
                    stderr=stderr,
                    error_message=(
                        "Docker reported a successful build, but "
                        "the project dependency image could not "
                        "be found afterward."
                    ),
                )

            return DependencyPreparationResult(
                success=True,
                image_name=image_name,
                image_was_built=True,
                requirements_files=tuple(
                    path.relative_to(root).as_posix()
                    for path in requirements_files
                ),
                stdout=stdout,
                stderr=stderr,
            )

        except TaskCancelledError:
            raise

        except Exception as exc:
            return DependencyPreparationResult(
                success=False,
                image_name=None,
                image_was_built=False,
                error_message=(
                    "Project dependency preparation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
