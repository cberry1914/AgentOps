
"""
AgentOps - Docker Sandbox

Executes project tests inside a restricted Docker container.

Security controls:
- No network access.
- Read-only project workspace.
- Non-root container user.
- Read-only container filesystem.
- Dropped Linux capabilities.
- No-new-privileges security option.
- CPU, memory, process, and execution-time limits.
- Temporary writable filesystem limited to /tmp.
- No host directories other than the project workspace are mounted.
- No arbitrary shell commands supplied by the LLM.
- Unique container names and explicit cleanup on cancellation
  and timeout.

Output is captured continuously using bounded buffers to prevent
unlimited memory consumption from verbose test processes.

The sandbox currently supports Python projects using pytest.
Additional language runtimes and dependency strategies can be
added as the execution architecture evolves.

Important:
Docker containers share the host kernel. These controls reduce
exposure but do not provide VM-grade isolation.
"""

import re
import shutil
import subprocess
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Optional


@dataclass
class SandboxConfig:
    """Resource and runtime configuration for sandbox execution."""

    image: str = "agentops-python-sandbox:latest"
    timeout_seconds: int = 120
    memory_limit: str = "512m"
    memory_swap_limit: str = "512m"
    cpu_limit: str = "1.0"
    process_limit: int = 64
    tmpfs_size: str = "64m"

    def validate(self) -> None:
        """Validate configuration before interacting with Docker."""

        if not isinstance(self.image, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}",
            self.image,
        ):
            raise ValueError("Invalid Docker image name.")

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 900
        ):
            raise ValueError(
                "Timeout must be an integer between 1 and 900 seconds."
            )

        if (
            isinstance(self.process_limit, bool)
            or not isinstance(self.process_limit, int)
            or not 1 <= self.process_limit <= 256
        ):
            raise ValueError(
                "Process limit must be an integer between 1 and 256."
            )

        try:
            cpu_value = float(self.cpu_limit)
        except (TypeError, ValueError):
            raise ValueError("CPU limit must be a valid number.")

        if not 0.1 <= cpu_value <= 8.0:
            raise ValueError(
                "CPU limit must be between 0.1 and 8.0."
            )

        if not isinstance(self.memory_limit, str) or not re.fullmatch(
            r"[1-9][0-9]*m",
            self.memory_limit,
        ):
            raise ValueError(
                "Memory limit must use a format such as 512m."
            )

        if (
            not isinstance(self.memory_swap_limit, str)
            or not re.fullmatch(
                r"[1-9][0-9]*m",
                self.memory_swap_limit,
            )
        ):
            raise ValueError(
                "Memory swap limit must use a format such as 512m."
            )

        memory_mb = int(self.memory_limit[:-1])
        swap_mb = int(self.memory_swap_limit[:-1])

        # Docker interprets --memory-swap as the combined memory
        # and swap allowance, not as additional swap alone.
        if swap_mb < memory_mb:
            raise ValueError(
                "Memory swap limit must be greater than or equal "
                "to the memory limit."
            )

        if not isinstance(self.tmpfs_size, str) or not re.fullmatch(
            r"[1-9][0-9]*m",
            self.tmpfs_size,
        ):
            raise ValueError(
                "Temporary storage limit must use a format such as 64m."
            )


@dataclass
class DockerSandboxResult:
    """Structured result from an isolated test execution."""

    status: str
    exit_code: Optional[int]
    duration_seconds: float
    stdout: str
    stderr: str
    execution_environment: str = "docker"
    tests_executed: bool = False
    error_message: Optional[str] = None

    # Compatibility field for components that use `environment`.
    environment: Optional[str] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = self.execution_environment
        else:
            self.execution_environment = self.environment

    @property
    def succeeded(self) -> bool:
        return self.status == "PASSED"

    @property
    def failed(self) -> bool:
        return self.status == "FAILED"


class _OutputCapture:
    """
    Thread-safe bounded output capture.

    Each stream retains its most recent MAX_CHARS characters.
    Reader threads never wait for the main thread to consume output.
    """

    MAX_CHARS = 1_000_000

    def __init__(self):
        self._lock = Lock()

        self._buffers = {
            "stdout": deque(),
            "stderr": deque(),
        }

        self._character_counts = {
            "stdout": 0,
            "stderr": 0,
        }

        self._truncated = {
            "stdout": False,
            "stderr": False,
        }

    def append(self, stream_name: str, chunk: str) -> None:
        """Append output while enforcing the per-stream size limit."""

        if not chunk:
            return

        with self._lock:
            buffer = self._buffers[stream_name]

            buffer.append(chunk)
            self._character_counts[stream_name] += len(chunk)

            while (
                self._character_counts[stream_name]
                > self.MAX_CHARS
            ):
                excess = (
                    self._character_counts[stream_name]
                    - self.MAX_CHARS
                )

                oldest = buffer.popleft()

                if len(oldest) > excess:
                    buffer.appendleft(oldest[excess:])
                    self._character_counts[stream_name] -= excess
                else:
                    self._character_counts[stream_name] -= len(oldest)

                self._truncated[stream_name] = True

    def get_text(self, stream_name: str) -> str:
        """Return captured output and indicate whether it was truncated."""

        with self._lock:
            output = "".join(self._buffers[stream_name])

            if self._truncated[stream_name]:
                return (
                    "[Earlier output truncated. "
                    f"Showing the most recent {self.MAX_CHARS} characters.]\n"
                    f"{output}"
                )

            return output


class DockerSandbox:
    """
    Manages restricted Docker-based test execution.

    This class deliberately does not accept arbitrary commands,
    Docker arguments, environment variables, or mount locations
    from an agent.

    The only executable workload currently exposed is pytest.
    """

    CLEANUP_TIMEOUT_SECONDS = 10
    STOP_GRACE_SECONDS = 2
    POLL_INTERVAL_SECONDS = 0.2
    PROCESS_TERMINATION_TIMEOUT_SECONDS = 5
    OUTPUT_COLLECTION_TIMEOUT_SECONDS = 10
    OUTPUT_READ_CHUNK_SIZE = 4096

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
    ):
        self.config = config or SandboxConfig()
        self.config.validate()

    # ---------------------------------------------------------
    # DOCKER AVAILABILITY
    # ---------------------------------------------------------

    def _docker_available(self) -> bool:
        """Check whether the Docker executable is available."""

        return shutil.which("docker") is not None

    def _validate_workspace(
        self,
        workspace_path: str | Path,
    ) -> Path:
        """
        Validate and resolve the project workspace.

        The workspace must already exist and must be a real directory.
        """

        workspace = Path(workspace_path)

        if workspace.is_symlink():
            raise ValueError(
                "Workspace root cannot be a symbolic link."
            )

        if not workspace.exists():
            raise ValueError("Workspace does not exist.")

        if not workspace.is_dir():
            raise ValueError("Workspace must be a directory.")

        resolved = workspace.resolve(strict=True)

        if not resolved.is_dir():
            raise ValueError(
                "Resolved workspace is not a directory."
            )

        return resolved

    def _check_image(self) -> Optional[str]:
        """Verify that the configured sandbox image exists locally."""

        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    self.config.image,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

        except subprocess.TimeoutExpired:
            return "Docker image inspection timed out."

        except OSError as exc:
            return f"Unable to inspect Docker image: {exc}"

        if result.returncode != 0:
            return (
                f"Sandbox image '{self.config.image}' is not available. "
                "Build the AgentOps sandbox image before executing tests."
            )

        return None

    # ---------------------------------------------------------
    # CONTAINER COMMAND
    # ---------------------------------------------------------

    def _build_command(
        self,
        workspace: Path,
        container_name: str,
    ) -> list[str]:
        """
        Construct the fixed Docker execution command.

        The project is mounted read-only. The container receives
        no host credentials, Docker socket, or network access.
        """

        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit",
            str(self.config.process_limit),
            "--memory",
            self.config.memory_limit,
            "--memory-swap",
            self.config.memory_swap_limit,
            "--cpus",
            self.config.cpu_limit,
            "--ulimit",
            "nofile=1024:1024",
            "--tmpfs",
            (
                f"/tmp:rw,noexec,nosuid,nodev,"
                f"size={self.config.tmpfs_size}"
            ),
            "--mount",
            (
                f"type=bind,source={workspace},"
                "target=/workspace,readonly"
            ),
            "--workdir",
            "/workspace",
            "--user",
            "10001:10001",
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            self.config.image,
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ]

    # ---------------------------------------------------------
    # PROCESS OUTPUT
    # ---------------------------------------------------------

    @classmethod
    def _read_stream(
        cls,
        stream,
        capture: _OutputCapture,
        stream_name: str,
    ) -> None:
        """
        Continuously drain one subprocess output stream.

        Output is written directly into a bounded, thread-safe buffer.
        Reader threads never wait for the main thread to consume output.
        """

        try:
            while True:
                chunk = stream.read(cls.OUTPUT_READ_CHUNK_SIZE)

                if not chunk:
                    break

                capture.append(stream_name, chunk)

        except (OSError, ValueError):
            pass

        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _start_output_collection(
        self,
        process: subprocess.Popen,
        capture: _OutputCapture,
    ) -> list[threading.Thread]:
        """Start independent readers for stdout and stderr."""

        readers = []

        for stream_name, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            reader = threading.Thread(
                target=self._read_stream,
                args=(
                    stream,
                    capture,
                    stream_name,
                ),
                daemon=True,
                name=f"agentops-docker-{stream_name}",
            )

            reader.start()
            readers.append(reader)

        return readers

    def _join_output_readers(
        self,
        readers: list[threading.Thread],
    ) -> bool:
        """
        Wait for output readers to finish within a bounded period.

        Returns False if one or more readers remain active.
        """

        deadline = (
            time.monotonic()
            + self.OUTPUT_COLLECTION_TIMEOUT_SECONDS
        )

        for reader in readers:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            reader.join(timeout=remaining)

        return not any(reader.is_alive() for reader in readers)

    # ---------------------------------------------------------
    # PROCESS TERMINATION
    # ---------------------------------------------------------

    def _terminate_process(
        self,
        process: subprocess.Popen,
    ) -> None:
        """Terminate the local Docker CLI process and wait for exit."""

        if process.poll() is not None:
            return

        try:
            process.terminate()
        except OSError:
            try:
                process.kill()
            except OSError:
                return

        try:
            process.wait(
                timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS
            )

        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

            try:
                process.wait(
                    timeout=self.PROCESS_TERMINATION_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                pass

    # ---------------------------------------------------------
    # CONTAINER CLEANUP
    # ---------------------------------------------------------

    def _stop_container(
        self,
        container_name: str,
    ) -> Optional[str]:
        """Stop a named container if it is still running."""

        try:
            result = subprocess.run(
                [
                    "docker",
                    "stop",
                    "--time",
                    str(self.STOP_GRACE_SECONDS),
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=self.CLEANUP_TIMEOUT_SECONDS,
                check=False,
            )

            if result.returncode == 0:
                return None

            message = (
                (result.stderr or "").strip()
                or (result.stdout or "").strip()
            )

            if self._container_missing(message):
                return None

            return (
                message
                or "Docker container stop operation failed."
            )

        except subprocess.TimeoutExpired:
            return "Docker container stop operation timed out."

        except OSError as exc:
            return f"Unable to stop Docker container: {exc}"

    def _remove_container(
        self,
        container_name: str,
    ) -> Optional[str]:
        """Force-remove a named container if it still exists."""

        try:
            result = subprocess.run(
                [
                    "docker",
                    "rm",
                    "--force",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=self.CLEANUP_TIMEOUT_SECONDS,
                check=False,
            )

            if result.returncode == 0:
                return None

            message = (
                (result.stderr or "").strip()
                or (result.stdout or "").strip()
            )

            if self._container_missing(message):
                return None

            return (
                message
                or "Docker container removal failed."
            )

        except subprocess.TimeoutExpired:
            return "Docker container removal timed out."

        except OSError as exc:
            return f"Unable to remove Docker container: {exc}"

    @staticmethod
    def _container_missing(message: str) -> bool:
        """Recognize Docker responses indicating no container remains."""

        normalized = message.lower()

        return (
            "no such container" in normalized
            or "no container with name" in normalized
        )

    def _cleanup_container(
        self,
        container_name: str,
    ) -> Optional[str]:
        """
        Stop and remove a container.

        Docker's --rm option normally removes containers after they
        exit. Explicit cleanup covers cancellation and timeout paths.
        """

        stop_error = self._stop_container(container_name)
        remove_error = self._remove_container(container_name)

        # Successful removal means cleanup succeeded even if the
        # stop operation reported that the container had already exited.
        if not remove_error:
            return None

        if stop_error:
            return (
                f"Stop failed: {stop_error}; "
                f"removal failed: {remove_error}"
            )

        return remove_error

    # ---------------------------------------------------------
    # RESULT CONSTRUCTION
    # ---------------------------------------------------------

    @staticmethod
    def _build_result(
        start_time: float,
        status: str,
        exit_code: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        error_message: Optional[str] = None,
        tests_executed: bool = False,
    ) -> DockerSandboxResult:
        """Construct a consistent structured sandbox result."""

        return DockerSandboxResult(
            status=status,
            exit_code=exit_code,
            duration_seconds=round(
                time.monotonic() - start_time,
                3,
            ),
            stdout=stdout,
            stderr=stderr,
            error_message=error_message,
            tests_executed=tests_executed,
        )

    def _captured_output(
        self,
        capture: _OutputCapture,
        readers: list[threading.Thread],
    ) -> tuple[str, str, Optional[str]]:
        """Return bounded output and any reader completion warning."""

        readers_finished = self._join_output_readers(readers)

        warning = None

        if not readers_finished:
            warning = (
                "Docker output collection did not finish "
                "within the allowed time."
            )

        return (
            capture.get_text("stdout"),
            capture.get_text("stderr"),
            warning,
        )

    # ---------------------------------------------------------
    # TEST EXECUTION
    # ---------------------------------------------------------

    def run_tests(
        self,
        workspace_path: str | Path,
        cancel_event: Optional[Event] = None,
    ) -> DockerSandboxResult:
        """
        Execute the project's pytest suite inside Docker.

        Cancellation and timeout trigger explicit container cleanup.
        Ordinary execution failures are returned as structured results.
        """

        start_time = time.monotonic()

        def result(
            status: str,
            exit_code: Optional[int] = None,
            stdout: str = "",
            stderr: str = "",
            error_message: Optional[str] = None,
            tests_executed: bool = False,
        ) -> DockerSandboxResult:
            return self._build_result(
                start_time=start_time,
                status=status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                error_message=error_message,
                tests_executed=tests_executed,
            )

        if cancel_event is not None and cancel_event.is_set():
            return result(
                status="CANCELLED",
                error_message=(
                    "Execution was cancelled before starting."
                ),
            )

        try:
            workspace = self._validate_workspace(workspace_path)

        except (ValueError, OSError) as exc:
            return result(
                status="INVALID_WORKSPACE",
                error_message=str(exc),
            )

        if not self._docker_available():
            return result(
                status="DOCKER_UNAVAILABLE",
                error_message=(
                    "Docker executable was not found. "
                    "Install Docker before running sandbox tests."
                ),
            )

        if cancel_event is not None and cancel_event.is_set():
            return result(
                status="CANCELLED",
                error_message=(
                    "Execution was cancelled before container startup."
                ),
            )

        image_error = self._check_image()

        if image_error:
            return result(
                status="IMAGE_NOT_FOUND",
                error_message=image_error,
            )

        if cancel_event is not None and cancel_event.is_set():
            return result(
                status="CANCELLED",
                error_message=(
                    "Execution was cancelled before container startup."
                ),
            )

        container_name = f"agentops-test-{uuid.uuid4().hex}"

        command = self._build_command(
            workspace,
            container_name,
        )

        process = None
        readers: list[threading.Thread] = []
        capture = _OutputCapture()
        tests_started = False

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            readers = self._start_output_collection(
                process,
                capture,
            )

            tests_started = True

            deadline = (
                time.monotonic() + self.config.timeout_seconds
            )

            while process.poll() is None:

                if cancel_event is not None and cancel_event.is_set():
                    cleanup_error = self._cleanup_container(
                        container_name
                    )

                    self._terminate_process(process)

                    stdout, stderr, output_error = (
                        self._captured_output(capture, readers)
                    )

                    message = "Sandbox execution was cancelled."

                    if cleanup_error:
                        message += (
                            f" Container cleanup warning: {cleanup_error}"
                        )

                    if output_error:
                        message += (
                            f" Output collection warning: {output_error}"
                        )

                    return result(
                        status="CANCELLED",
                        exit_code=process.returncode,
                        stdout=stdout,
                        stderr=stderr,
                        error_message=message,
                        tests_executed=True,
                    )

                if time.monotonic() >= deadline:
                    cleanup_error = self._cleanup_container(
                        container_name
                    )

                    self._terminate_process(process)

                    stdout, stderr, output_error = (
                        self._captured_output(capture, readers)
                    )

                    message = (
                        "Sandbox execution exceeded the configured "
                        f"{self.config.timeout_seconds}-second limit."
                    )

                    if cleanup_error:
                        message += (
                            f" Container cleanup warning: {cleanup_error}"
                        )

                    if output_error:
                        message += (
                            f" Output collection warning: {output_error}"
                        )

                    return result(
                        status="TIMEOUT",
                        exit_code=process.returncode,
                        stdout=stdout,
                        stderr=stderr,
                        error_message=message,
                        tests_executed=True,
                    )

                time.sleep(self.POLL_INTERVAL_SECONDS)

            stdout, stderr, output_error = (
                self._captured_output(capture, readers)
            )

            if output_error:
                return result(
                    status="EXECUTION_ERROR",
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=(
                        "Docker execution ended, but output collection "
                        f"failed: {output_error}"
                    ),
                    tests_executed=tests_started,
                )

            if cancel_event is not None and cancel_event.is_set():
                cleanup_error = self._cleanup_container(
                    container_name
                )

                message = "Sandbox execution was cancelled."

                if cleanup_error:
                    message += (
                        f" Container cleanup warning: {cleanup_error}"
                    )

                return result(
                    status="CANCELLED",
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=message,
                    tests_executed=tests_started,
                )

            if process.returncode == 0:
                return result(
                    status="PASSED",
                    exit_code=0,
                    stdout=stdout,
                    stderr=stderr,
                    tests_executed=True,
                )

            return result(
                status="FAILED",
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                error_message=(
                    "The test suite returned a non-zero exit code."
                ),
                tests_executed=True,
            )

        except OSError as exc:
            cleanup_error = self._cleanup_container(
                container_name
            )

            if process is not None:
                self._terminate_process(process)

            stdout, stderr, output_error = (
                self._captured_output(capture, readers)
                if readers
                else ("", "", None)
            )

            message = f"Unable to start Docker execution: {exc}"

            if cleanup_error:
                message += (
                    f" Container cleanup warning: {cleanup_error}"
                )

            if output_error:
                message += (
                    f" Output collection warning: {output_error}"
                )

            return result(
                status="EXECUTION_ERROR",
                exit_code=(
                    process.returncode
                    if process is not None
                    else None
                ),
                stdout=stdout,
                stderr=stderr,
                error_message=message,
                tests_executed=tests_started,
            )

        except Exception as exc:
            cleanup_error = self._cleanup_container(
                container_name
            )

            if process is not None:
                self._terminate_process(process)

            stdout, stderr, output_error = (
                self._captured_output(capture, readers)
                if readers
                else ("", "", None)
            )

            message = (
                "Unexpected sandbox error: "
                f"{type(exc).__name__}: {exc}"
            )

            if cleanup_error:
                message += (
                    f" Container cleanup warning: {cleanup_error}"
                )

            if output_error:
                message += (
                    f" Output collection warning: {output_error}"
                )

            return result(
                status="EXECUTION_ERROR",
                exit_code=(
                    process.returncode
                    if process is not None
                    else None
                ),
                stdout=stdout,
                stderr=stderr,
                error_message=message,
                tests_executed=tests_started,
            )

        finally:
            # Protect against an unexpected exception leaving the
            # Docker CLI process running in the background.
            if process is not None and process.poll() is None:
                self._terminate_process(process)

            if readers:
                self._join_output_readers(readers)
