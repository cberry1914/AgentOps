
"""
AgentOps - Sandbox Image Builder

Builds and verifies the local Docker image used by DockerSandbox.

The builder:
- Locates the AgentOps project root.
- Uses the dedicated sandbox Dockerfile.
- Uses only the sandbox directory as Docker build context.
- Builds the image with a fixed image tag.
- Continuously captures build output and errors.
- Limits retained output to prevent excessive memory usage.
- Verifies the resulting image.
- Supports task cancellation during image construction.
- Detects incomplete output collection.
- Terminates the local Docker CLI process tree on cancellation
  and timeout.
- Never executes project code.

Platform support:
- Linux: POSIX process groups and signals.
- Windows: Windows process groups and taskkill process-tree
  termination.

Network access is permitted during image construction so trusted
base-image dependencies can be downloaded. Runtime network
isolation is enforced separately by DockerSandbox.

Important:
Terminating the local Docker CLI process tree does not guarantee
that Docker daemon-side image-build work is cancelled.
"""

import os
import shutil
import signal
import subprocess
import threading
import time

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Optional

from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)


@dataclass
class ImageBuildResult:
    """Structured result from a sandbox image build."""

    success: bool
    image_name: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    error_message: Optional[str] = None


class _OutputCapture:
    """Thread-safe, bounded output capture for subprocess streams."""

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

        self._reader_errors = {
            "stdout": None,
            "stderr": None,
        }

    def append(self, stream_name: str, chunk: str) -> None:
        """Append output while retaining only the most recent characters."""

        if not chunk:
            return

        with self._lock:
            buffer = self._buffers[stream_name]

            buffer.append(chunk)
            self._character_counts[stream_name] += len(chunk)

            while self._character_counts[stream_name] > self.MAX_CHARS:
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

    def record_reader_error(
        self,
        stream_name: str,
        error: Exception,
    ) -> None:
        """Preserve unexpected output-reader failures."""

        with self._lock:
            self._reader_errors[stream_name] = (
                f"{type(error).__name__}: {error}"
            )

    def get_text(self, stream_name: str) -> str:
        """Return captured output with a truncation notice if needed."""

        with self._lock:
            output = "".join(self._buffers[stream_name])

            if self._truncated[stream_name]:
                return (
                    "[Earlier output truncated. "
                    f"Showing the most recent {self.MAX_CHARS} characters.]\n"
                    f"{output}"
                )

            return output

    def get_reader_errors(self) -> list[str]:
        """Return errors reported by the output-reader threads."""

        with self._lock:
            return [
                f"{stream_name}: {error}"
                for stream_name, error in self._reader_errors.items()
                if error
            ]


class _ProcessController:
    """
    Encapsulates platform-specific subprocess management.

    The image builder uses this class without needing to know
    which operating system is running AgentOps.
    """

    TERMINATION_GRACE_SECONDS = 5
    FORCE_TERMINATION_TIMEOUT_SECONDS = 5
    PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.05

    @staticmethod
    def start(command: list[str]) -> subprocess.Popen:
        """
        Start a subprocess with platform-appropriate process isolation.

        POSIX:
            A new session creates a process group that can be
            terminated together.

        Windows:
            A new process group is created. Windows process-tree
            termination is handled separately through taskkill.
        """

        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": False,
            "bufsize": 0,
        }

        if os.name == "posix":
            kwargs["start_new_session"] = True

        elif os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )

        else:
            raise RuntimeError(
                f"Unsupported operating system: {os.name}"
            )

        return subprocess.Popen(command, **kwargs)

    @classmethod
    def terminate(cls, process: subprocess.Popen) -> None:
        """
        Terminate the subprocess and its local process tree.

        POSIX process-group cleanup is attempted even when the
        original parent process has already exited. This matters
        because descendants can outlive their parent while keeping
        inherited stdout or stderr pipes open.
        """

        if os.name == "posix":
            cls._terminate_posix(process)

        elif os.name == "nt":
            if process.poll() is not None:
                return

            cls._terminate_windows(process)

        else:
            cls._terminate_direct_process(process)

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        """
        Check whether a POSIX process group still exists.

        Permission errors are treated as evidence that the group
        exists because the caller may not have permission to signal it.
        """

        try:
            os.killpg(process_group_id, 0)
            return True

        except ProcessLookupError:
            return False

        except PermissionError:
            return True

        except OSError:
            return True

    @classmethod
    def _wait_for_process_group_exit(
        cls,
        process_group_id: int,
        timeout_seconds: float,
    ) -> bool:
        """Wait for a process group to disappear within a fixed timeout."""

        deadline = time.monotonic() + timeout_seconds

        while cls._process_group_exists(process_group_id):
            if time.monotonic() >= deadline:
                return False

            time.sleep(cls.PROCESS_GROUP_POLL_INTERVAL_SECONDS)

        return True

    @classmethod
    def _terminate_posix(
        cls,
        process: subprocess.Popen,
    ) -> None:
        """
        Terminate a POSIX process group.

        Do not return merely because the original process exited.
        Its descendants may still belong to the same process group.
        """

        process_group_id = process.pid

        try:
            os.killpg(process_group_id, signal.SIGTERM)

        except ProcessLookupError:
            cls._reap_process(process)
            return

        except OSError:
            cls._terminate_direct_process(process)
            return

        group_exited = cls._wait_for_process_group_exit(
            process_group_id,
            cls.TERMINATION_GRACE_SECONDS,
        )

        if not group_exited:
            try:
                os.killpg(process_group_id, signal.SIGKILL)

            except ProcessLookupError:
                pass

            except OSError:
                cls._terminate_direct_process(process)

            cls._wait_for_process_group_exit(
                process_group_id,
                cls.FORCE_TERMINATION_TIMEOUT_SECONDS,
            )

        cls._reap_process(process)

    @classmethod
    def _reap_process(
        cls,
        process: subprocess.Popen,
    ) -> None:
        """Reap the original subprocess without an unbounded wait."""

        try:
            process.wait(
                timeout=cls.FORCE_TERMINATION_TIMEOUT_SECONDS
            )

        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

            try:
                process.wait(
                    timeout=cls.FORCE_TERMINATION_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                pass

    @classmethod
    def _terminate_windows(
        cls,
        process: subprocess.Popen,
    ) -> None:
        """
        Terminate a Windows process tree.

        taskkill /T targets child processes as well as the specified
        process. /F requests forced termination.

        If taskkill cannot be executed or fails, fall back to direct
        process termination.

        Windows process-tree cleanup requires separate native Windows
        integration testing before production deployment.
        """

        try:
            result = subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                text=True,
                timeout=cls.FORCE_TERMINATION_TIMEOUT_SECONDS,
                check=False,
            )

            if result.returncode == 0:
                cls._reap_process(process)
                return

        except (OSError, subprocess.TimeoutExpired):
            pass

        cls._terminate_direct_process(process)

    @classmethod
    def _terminate_direct_process(
        cls,
        process: subprocess.Popen,
    ) -> None:
        """Fallback termination for the direct subprocess."""

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
                timeout=cls.TERMINATION_GRACE_SECONDS
            )

        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

            try:
                process.wait(
                    timeout=cls.FORCE_TERMINATION_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                pass


class SandboxImageBuilder:
    """Builds and verifies the AgentOps Python sandbox image."""

    IMAGE_NAME = "agentops-python-sandbox:latest"

    BUILD_TIMEOUT_SECONDS = 900
    OUTPUT_JOIN_TIMEOUT_SECONDS = 5

    READ_CHUNK_SIZE = 4096

    DOCKERFILE_RELATIVE_PATH = (
        Path("src")
        / "agentops"
        / "tools"
        / "docker"
        / "sandbox"
        / "Dockerfile"
    )

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]

        self.project_root = Path(project_root).resolve()

        self.dockerfile = (
            self.project_root / self.DOCKERFILE_RELATIVE_PATH
        )

        self.build_context = self.dockerfile.parent

    # ---------------------------------------------------------
    # DOCKER AVAILABILITY
    # ---------------------------------------------------------

    def _docker_available(self) -> bool:
        """Check whether the Docker executable is available."""

        return shutil.which("docker") is not None

    def image_exists(self) -> bool:
        """Return whether the sandbox image exists locally."""

        if not self._docker_available():
            return False

        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    self.IMAGE_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

            return result.returncode == 0

        except (OSError, subprocess.TimeoutExpired):
            return False

    # ---------------------------------------------------------
    # PROCESS MANAGEMENT
    # ---------------------------------------------------------

    @staticmethod
    def _start_process(
        command: list[str],
    ) -> subprocess.Popen:
        """Start a process using the platform-independent controller."""

        return _ProcessController.start(command)

    @staticmethod
    def _terminate_process(
        process: subprocess.Popen,
    ) -> None:
        """Terminate a process and its local process tree."""

        _ProcessController.terminate(process)

    # ---------------------------------------------------------
    # PROCESS OUTPUT
    # ---------------------------------------------------------

    @staticmethod
    def _read_stream(
        stream,
        capture: _OutputCapture,
        stream_name: str,
    ) -> None:
        """Continuously drain one unbuffered subprocess output stream."""

        try:
            while True:
                chunk = stream.read(
                    SandboxImageBuilder.READ_CHUNK_SIZE
                )

                if not chunk:
                    break

                decoded = chunk.decode(
                    "utf-8",
                    errors="replace",
                )

                capture.append(stream_name, decoded)

        except (OSError, ValueError) as exc:
            capture.record_reader_error(stream_name, exc)

        except Exception as exc:
            capture.record_reader_error(stream_name, exc)

        finally:
            try:
                stream.close()
            except OSError:
                pass

    @staticmethod
    def _start_output_readers(
        process: subprocess.Popen,
        capture: _OutputCapture,
    ) -> list[threading.Thread]:
        """Start independent readers for stdout and stderr."""

        readers = []

        for stream_name, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            if stream is None:
                raise RuntimeError(
                    f"Docker build {stream_name} pipe was not created."
                )

            reader = threading.Thread(
                target=SandboxImageBuilder._read_stream,
                args=(stream, capture, stream_name),
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
        """Wait for output readers within a bounded period."""

        deadline = (
            time.monotonic()
            + self.OUTPUT_JOIN_TIMEOUT_SECONDS
        )

        for reader in readers:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            reader.join(timeout=remaining)

        return not any(reader.is_alive() for reader in readers)

    def _stop_output_readers(
        self,
        process: Optional[subprocess.Popen],
        readers: list[threading.Thread],
    ) -> bool:
        """
        Attempt to finish output readers after process termination.

        Closing pipes can unblock readers waiting for additional
        output. Readers remain daemon threads so an operating-system
        pipe issue cannot indefinitely block the AgentOps workflow.
        """

        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

        return self._join_output_readers(readers)

    # ---------------------------------------------------------
    # RESULT CONSTRUCTION
    # ---------------------------------------------------------

    def _build_result(
        self,
        capture: _OutputCapture,
        exit_code: Optional[int],
        success: bool,
        error_message: Optional[str] = None,
    ) -> ImageBuildResult:
        """Create a structured result from captured build output."""

        reader_errors = capture.get_reader_errors()

        if reader_errors:
            reader_message = (
                "Output reader errors: "
                + "; ".join(reader_errors)
            )

            error_message = (
                f"{error_message} {reader_message}"
                if error_message
                else reader_message
            )

            success = False

        return ImageBuildResult(
            success=success,
            image_name=self.IMAGE_NAME,
            exit_code=exit_code,
            stdout=capture.get_text("stdout"),
            stderr=capture.get_text("stderr"),
            error_message=error_message,
        )

    def _output_collection_error(
        self,
        capture: _OutputCapture,
        exit_code: Optional[int],
    ) -> ImageBuildResult:
        """Create a failure result when output collection is incomplete."""

        return self._build_result(
            capture=capture,
            exit_code=exit_code,
            success=False,
            error_message=(
                "Docker build ended, but one or more output-reader "
                "threads did not finish within the allowed time."
            ),
        )

    # ---------------------------------------------------------
    # IMAGE BUILD
    # ---------------------------------------------------------

    def build(
        self,
        cancel_event: Optional[Event] = None,
    ) -> ImageBuildResult:
        """
        Build the sandbox image.

        Cancellation and timeout terminate the local Docker CLI
        process tree. Docker daemon-side cleanup is not guaranteed
        if the client is interrupted during an active image build.
        """

        check_cancelled(cancel_event)

        if not self._docker_available():
            return ImageBuildResult(
                success=False,
                image_name=self.IMAGE_NAME,
                exit_code=None,
                stdout="",
                stderr="",
                error_message="Docker executable was not found.",
            )

        if not self.project_root.is_dir():
            return ImageBuildResult(
                success=False,
                image_name=self.IMAGE_NAME,
                exit_code=None,
                stdout="",
                stderr="",
                error_message="AgentOps project root does not exist.",
            )

        if not self.dockerfile.is_file():
            return ImageBuildResult(
                success=False,
                image_name=self.IMAGE_NAME,
                exit_code=None,
                stdout="",
                stderr="",
                error_message=(
                    "Sandbox Dockerfile was not found: "
                    f"{self.dockerfile}"
                ),
            )

        if not self.build_context.is_dir():
            return ImageBuildResult(
                success=False,
                image_name=self.IMAGE_NAME,
                exit_code=None,
                stdout="",
                stderr="",
                error_message=(
                    "Sandbox Docker build context does not exist: "
                    f"{self.build_context}"
                ),
            )

        command = [
            "docker",
            "build",
            "--tag",
            self.IMAGE_NAME,
            "--file",
            str(self.dockerfile),
            str(self.build_context),
        ]

        process = None
        readers: list[threading.Thread] = []
        capture = _OutputCapture()

        start_time = time.monotonic()

        try:
            process = self._start_process(command)

            readers = self._start_output_readers(
                process,
                capture,
            )

            while process.poll() is None:
                check_cancelled(cancel_event)

                elapsed = time.monotonic() - start_time

                if elapsed >= self.BUILD_TIMEOUT_SECONDS:
                    self._terminate_process(process)

                    readers_finished = (
                        self._join_output_readers(readers)
                    )

                    if not readers_finished:
                        readers_finished = self._stop_output_readers(
                            process,
                            readers,
                        )

                    if not readers_finished:
                        return self._output_collection_error(
                            capture,
                            process.returncode,
                        )

                    return self._build_result(
                        capture=capture,
                        exit_code=process.returncode,
                        success=False,
                        error_message=(
                            "Docker image build exceeded "
                            f"{self.BUILD_TIMEOUT_SECONDS} seconds."
                        ),
                    )

                time.sleep(0.1)

            readers_finished = self._join_output_readers(readers)

            if not readers_finished:
                # The Docker CLI may have exited while a descendant
                # still holds an inherited output pipe open. On POSIX,
                # make one final process-group cleanup attempt before
                # closing the parent-side pipe handles.
                if os.name == "posix":
                    self._terminate_process(process)

                readers_finished = self._stop_output_readers(
                    process,
                    readers,
                )

            if not readers_finished:
                return self._output_collection_error(
                    capture,
                    process.returncode,
                )

            check_cancelled(cancel_event)

            if process.returncode != 0:
                return self._build_result(
                    capture=capture,
                    exit_code=process.returncode,
                    success=False,
                    error_message="Docker image build failed.",
                )

            check_cancelled(cancel_event)

            if not self.image_exists():
                return self._build_result(
                    capture=capture,
                    exit_code=process.returncode,
                    success=False,
                    error_message=(
                        "Docker reported a successful build, but the "
                        "sandbox image could not be verified."
                    ),
                )

            return self._build_result(
                capture=capture,
                exit_code=0,
                success=True,
            )

        except TaskCancelledError:
            if process is not None:
                self._terminate_process(process)

            readers_finished = self._join_output_readers(readers)

            if not readers_finished:
                self._stop_output_readers(process, readers)

            raise

        except OSError as exc:
            if process is not None:
                self._terminate_process(process)

            readers_finished = self._join_output_readers(readers)

            if not readers_finished:
                self._stop_output_readers(process, readers)

            return self._build_result(
                capture=capture,
                exit_code=(
                    process.returncode
                    if process is not None
                    else None
                ),
                success=False,
                error_message=f"Unable to execute Docker build: {exc}",
            )

        except Exception as exc:
            if process is not None:
                self._terminate_process(process)

            readers_finished = self._join_output_readers(readers)

            if not readers_finished:
                self._stop_output_readers(process, readers)

            return self._build_result(
                capture=capture,
                exit_code=(
                    process.returncode
                    if process is not None
                    else None
                ),
                success=False,
                error_message=(
                    "Unexpected Docker image build error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        finally:
            # Process termination is handled explicitly by timeout,
            # cancellation, and exception paths above. Do not signal a
            # POSIX process group unconditionally here: after a successful
            # build, the original process may already have been reaped
            # and its numeric process-group ID could eventually be reused.
            readers_finished = self._join_output_readers(readers)

            if not readers_finished:
                self._stop_output_readers(process, readers)
