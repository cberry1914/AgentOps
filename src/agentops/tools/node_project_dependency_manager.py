
"""
AgentOps - Node.js Project Dependency Manager

Builds isolated Docker images containing project-specific
Node.js dependencies declared by package.json and package-lock.json.

Dependencies are installed inside Docker, never on the AgentOps
host or inside the mounted project source directory.

Supported manifest locations:
- Project root
- Project root/frontend/

Only the package manifests are copied into the temporary Docker
build context. Project-provided Dockerfiles and setup scripts
are never executed by this component.
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
class NodeDependencyPreparationResult:
    """Structured result from preparing Node.js dependencies."""

    success: bool
    image_name: Optional[str]
    image_was_built: bool
    package_files: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None


class NodeProjectDependencyManager:
    """Builds and reuses project-specific Node.js dependency images."""

    BASE_IMAGE = "agentops-node-sandbox:latest"
    BUILD_TIMEOUT_SECONDS = 900
    POLL_INTERVAL_SECONDS = 0.5
    DOCKER_INSPECT_TIMEOUT_SECONDS = 15

    SUPPORTED_MANIFEST_DIRECTORIES = (
        Path("."),
        Path("frontend"),
    )

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
    # PACKAGE MANIFEST DISCOVERY
    # ---------------------------------------------------------

    def _discover_package_files(
        self,
        project_root: Path,
    ) -> list[Path]:
        """
        Discover one supported Node.js package environment.

        Root-level package manifests take precedence over frontend/.

        Both package.json and package-lock.json are required.
        Nested environments other than frontend/ are excluded.
        """

        for relative_directory in self.SUPPORTED_MANIFEST_DIRECTORIES:
            package_directory = project_root / relative_directory

            package_json = package_directory / "package.json"
            package_lock = package_directory / "package-lock.json"

            if not package_json.exists() and not package_lock.exists():
                continue

            if package_json.is_symlink() or package_lock.is_symlink():
                raise ValueError(
                    "Node.js package manifests cannot be symbolic links."
                )

            if not package_json.is_file():
                raise ValueError(
                    f"Missing package.json in {relative_directory}."
                )

            if not package_lock.is_file():
                raise ValueError(
                    f"Missing package-lock.json in {relative_directory}. "
                    "A lockfile is required for reproducible npm ci "
                    "installations."
                )

            return [package_json, package_lock]

        return []

    # ---------------------------------------------------------
    # IMAGE IDENTIFICATION
    # ---------------------------------------------------------

    def _create_image_name(
        self,
        project_root: Path,
        package_files: list[Path],
    ) -> str:
        """
        Generate a stable image tag from the project and manifests.

        Changes to either package manifest produce a different tag.
        """

        digest = hashlib.sha256()

        digest.update(str(project_root).encode("utf-8"))

        for package_file in package_files:
            relative_path = package_file.relative_to(
                project_root
            ).as_posix()

            digest.update(relative_path.encode("utf-8"))
            digest.update(package_file.read_bytes())

        return (
            "agentops-node-project-"
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
        package_files: list[Path],
        project_root: Path,
        build_context: Path,
    ) -> Path:
        """
        Create an AgentOps-controlled Dockerfile.

        Dependencies are installed in /workspace/node_modules.
        The project source will later be mounted at /workspace/project.
        """

        for package_file in package_files:
            destination = build_context / package_file.name

            destination.write_bytes(
                package_file.read_bytes()
            )

        dockerfile = build_context / "Dockerfile"

        dockerfile.write_text(
            "\n".join(
                [
                    f"FROM {self.BASE_IMAGE}",
                    "USER root",
                    "WORKDIR /workspace",
                    "COPY package.json package-lock.json ./",
                    (
                        "RUN npm ci "
                        "--no-audit "
                        "--no-fund"
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
                        "Node.js dependency image build exceeded "
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
                    "Docker Node.js dependency image build failed. "
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
    ) -> NodeDependencyPreparationResult:
        """
        Prepare an image containing the project's Node.js dependencies.

        Projects without supported Node.js manifests continue using
        the base Node.js sandbox image.

        Existing images are reused when their manifests are unchanged.
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

            package_files = self._discover_package_files(root)

            if not package_files:
                return NodeDependencyPreparationResult(
                    success=True,
                    image_name=None,
                    image_was_built=False,
                    error_message=(
                        "No supported Node.js package manifests "
                        "were found. The base Node.js sandbox "
                        "image will be used."
                    ),
                )

            image_name = self._create_image_name(
                root,
                package_files,
            )

            check_cancelled(cancel_event)

            # Reuse the image if its manifest-derived tag exists.
            if self._image_exists(image_name):
                return NodeDependencyPreparationResult(
                    success=True,
                    image_name=image_name,
                    image_was_built=False,
                    package_files=tuple(
                        path.relative_to(root).as_posix()
                        for path in package_files
                    ),
                )

            check_cancelled(cancel_event)

            with tempfile.TemporaryDirectory(
                prefix="agentops-node-dependencies-"
            ) as temporary_directory:

                build_context = Path(temporary_directory)

                dockerfile = self._create_dockerfile(
                    package_files=package_files,
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
                return NodeDependencyPreparationResult(
                    success=False,
                    image_name=image_name,
                    image_was_built=True,
                    package_files=tuple(
                        path.relative_to(root).as_posix()
                        for path in package_files
                    ),
                    stdout=stdout,
                    stderr=stderr,
                    error_message=(
                        "Docker reported a successful build, but "
                        "the Node.js dependency image could not "
                        "be found afterward."
                    ),
                )

            return NodeDependencyPreparationResult(
                success=True,
                image_name=image_name,
                image_was_built=True,
                package_files=tuple(
                    path.relative_to(root).as_posix()
                    for path in package_files
                ),
                stdout=stdout,
                stderr=stderr,
            )

        except TaskCancelledError:
            raise

        except Exception as exc:
            return NodeDependencyPreparationResult(
                success=False,
                image_name=None,
                image_was_built=False,
                error_message=(
                    "Node.js dependency preparation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
