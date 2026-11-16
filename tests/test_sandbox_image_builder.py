
import io
import os
import signal
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import agentops.tools.sandbox_image_builder as image_builder_module

from agentops.tools.cancellation import TaskCancelledError
from agentops.tools.sandbox_image_builder import (
    ImageBuildResult,
    SandboxImageBuilder,
    _OutputCapture,
    _ProcessController,
)


# =========================================================
# FIXTURES AND HELPERS
# =========================================================


@pytest.fixture
def project_root(tmp_path):
    docker_dir = (
        tmp_path
        / "src"
        / "agentops"
        / "tools"
        / "docker"
        / "sandbox"
    )
    docker_dir.mkdir(parents=True)

    (docker_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def builder(project_root):
    return SandboxImageBuilder(project_root=project_root)


@pytest.fixture
def available_docker(monkeypatch, builder):
    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )
    return builder


def make_process(
    returncode=0,
    stdout=b"build output",
    stderr=b"build warning",
    pid=12345,
):
    process = MagicMock()
    process.pid = pid
    process.stdout = io.BytesIO(stdout)
    process.stderr = io.BytesIO(stderr)
    process.returncode = returncode
    process.poll.return_value = returncode

    return process


def disable_builder_cleanup(monkeypatch, builder):
    """Prevent fake test processes from reaching operating-system cleanup."""

    terminate = MagicMock()

    monkeypatch.setattr(
        builder,
        "_terminate_process",
        terminate,
    )

    return terminate


# =========================================================
# IMAGE BUILD RESULT
# =========================================================


def test_image_build_result_fields():
    result = ImageBuildResult(
        success=True,
        image_name="test-image",
        exit_code=0,
        stdout="output",
        stderr="",
    )

    assert result.success is True
    assert result.image_name == "test-image"
    assert result.exit_code == 0
    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.error_message is None


# =========================================================
# BUILDER INITIALIZATION
# =========================================================


def test_builder_uses_supplied_project_root(project_root):
    builder = SandboxImageBuilder(project_root)

    assert builder.project_root == project_root.resolve()

    assert builder.dockerfile == (
        project_root
        / "src"
        / "agentops"
        / "tools"
        / "docker"
        / "sandbox"
        / "Dockerfile"
    )

    assert builder.build_context == builder.dockerfile.parent


def test_builder_resolves_string_project_root(project_root):
    builder = SandboxImageBuilder(str(project_root))

    assert builder.project_root == project_root.resolve()


def test_builder_default_image_name():
    assert (
        SandboxImageBuilder.IMAGE_NAME
        == "agentops-python-sandbox:latest"
    )


# =========================================================
# DOCKER AVAILABILITY
# =========================================================


def test_docker_available_when_executable_exists(
    monkeypatch,
    builder,
):
    monkeypatch.setattr(
        "agentops.tools.sandbox_image_builder.shutil.which",
        lambda name: "/usr/bin/docker",
    )

    assert builder._docker_available() is True


def test_docker_unavailable_when_executable_missing(
    monkeypatch,
    builder,
):
    monkeypatch.setattr(
        "agentops.tools.sandbox_image_builder.shutil.which",
        lambda name: None,
    )

    assert builder._docker_available() is False


def test_image_exists_when_docker_inspect_succeeds(
    monkeypatch,
    builder,
):
    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    inspect = MagicMock(
        return_value=MagicMock(returncode=0)
    )

    monkeypatch.setattr(
        "agentops.tools.sandbox_image_builder.subprocess.run",
        inspect,
    )

    assert builder.image_exists() is True

    inspect.assert_called_once_with(
        [
            "docker",
            "image",
            "inspect",
            builder.IMAGE_NAME,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_image_exists_returns_false_when_inspect_fails(
    monkeypatch,
    builder,
):
    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        "agentops.tools.sandbox_image_builder.subprocess.run",
        lambda *args, **kwargs: MagicMock(returncode=1),
    )

    assert builder.image_exists() is False


def test_image_exists_returns_false_when_docker_missing(
    monkeypatch,
    builder,
):
    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: False,
    )

    assert builder.image_exists() is False


@pytest.mark.parametrize(
    "exception",
    [
        OSError("Docker unavailable"),
        subprocess.TimeoutExpired("docker", 15),
    ],
)
def test_image_exists_handles_inspect_exceptions(
    monkeypatch,
    builder,
    exception,
):
    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    def raise_exception(*args, **kwargs):
        raise exception

    monkeypatch.setattr(
        "agentops.tools.sandbox_image_builder.subprocess.run",
        raise_exception,
    )

    assert builder.image_exists() is False


# =========================================================
# OUTPUT CAPTURE
# =========================================================


def test_output_capture_stores_stdout_and_stderr():
    capture = _OutputCapture()

    capture.append("stdout", "hello")
    capture.append("stderr", "warning")

    assert capture.get_text("stdout") == "hello"
    assert capture.get_text("stderr") == "warning"


def test_output_capture_ignores_empty_chunks():
    capture = _OutputCapture()

    capture.append("stdout", "")
    capture.append("stderr", None)

    assert capture.get_text("stdout") == ""
    assert capture.get_text("stderr") == ""


def test_output_capture_retains_most_recent_characters():
    capture = _OutputCapture()
    capture.MAX_CHARS = 10

    capture.append("stdout", "123456")
    capture.append("stdout", "789012")

    output = capture.get_text("stdout")

    assert output.endswith("3456789012")
    assert "Earlier output truncated" in output


def test_output_capture_truncates_streams_independently():
    capture = _OutputCapture()
    capture.MAX_CHARS = 5

    capture.append("stdout", "123456")
    capture.append("stderr", "error")

    assert capture.get_text("stdout").endswith("23456")
    assert "Earlier output truncated" in capture.get_text("stdout")
    assert capture.get_text("stderr") == "error"


def test_output_capture_does_not_mark_exact_limit_as_truncated():
    capture = _OutputCapture()
    capture.MAX_CHARS = 5

    capture.append("stdout", "12345")

    assert capture.get_text("stdout") == "12345"


def test_output_capture_handles_multiple_chunks():
    capture = _OutputCapture()

    capture.append("stdout", "first")
    capture.append("stdout", " second")
    capture.append("stdout", " third")

    assert capture.get_text("stdout") == "first second third"


def test_output_capture_records_reader_error():
    capture = _OutputCapture()

    capture.record_reader_error(
        "stdout",
        OSError("pipe failed"),
    )

    errors = capture.get_reader_errors()

    assert len(errors) == 1
    assert "stdout" in errors[0]
    assert "pipe failed" in errors[0]


def test_output_capture_returns_no_errors_initially():
    capture = _OutputCapture()

    assert capture.get_reader_errors() == []


# =========================================================
# OUTPUT READERS
# =========================================================


def test_read_stream_captures_bytes():
    capture = _OutputCapture()
    stream = io.BytesIO(b"hello docker")

    SandboxImageBuilder._read_stream(
        stream,
        capture,
        "stdout",
    )

    assert capture.get_text("stdout") == "hello docker"
    assert stream.closed


def test_read_stream_handles_invalid_utf8():
    capture = _OutputCapture()
    stream = io.BytesIO(b"hello\xffworld")

    SandboxImageBuilder._read_stream(
        stream,
        capture,
        "stdout",
    )

    assert "\ufffd" in capture.get_text("stdout")


def test_read_stream_records_reader_error():
    class BrokenStream:
        def read(self, size):
            raise OSError("read failed")

        def close(self):
            pass

    capture = _OutputCapture()

    SandboxImageBuilder._read_stream(
        BrokenStream(),
        capture,
        "stderr",
    )

    errors = capture.get_reader_errors()

    assert len(errors) == 1
    assert "read failed" in errors[0]


def test_start_output_readers_starts_two_threads():
    process = make_process()
    capture = _OutputCapture()

    readers = SandboxImageBuilder._start_output_readers(
        process,
        capture,
    )

    assert len(readers) == 2

    for reader in readers:
        reader.join(timeout=2)

    assert all(not reader.is_alive() for reader in readers)
    assert capture.get_text("stdout") == "build output"
    assert capture.get_text("stderr") == "build warning"


def test_start_output_readers_rejects_missing_pipe():
    process = MagicMock()
    process.stdout = None
    process.stderr = io.BytesIO(b"")

    with pytest.raises(
        RuntimeError,
        match="stdout pipe was not created",
    ):
        SandboxImageBuilder._start_output_readers(
            process,
            _OutputCapture(),
        )


def test_join_output_readers_returns_true_when_finished(builder):
    reader = MagicMock()
    reader.is_alive.return_value = False

    assert builder._join_output_readers([reader]) is True


def test_join_output_readers_returns_false_when_still_alive(
    builder,
    monkeypatch,
):
    reader = MagicMock()
    reader.is_alive.return_value = True

    monkeypatch.setattr(
        builder,
        "OUTPUT_JOIN_TIMEOUT_SECONDS",
        0,
    )

    assert builder._join_output_readers([reader]) is False


def test_stop_output_readers_closes_process_pipes(builder):
    process = make_process()

    reader = MagicMock()
    reader.is_alive.return_value = False

    assert builder._stop_output_readers(
        process,
        [reader],
    ) is True

    assert process.stdout.closed
    assert process.stderr.closed


# =========================================================
# PROCESS CONTROLLER: PROCESS STARTUP
# =========================================================


def test_process_controller_starts_posix_process_in_new_session(
    monkeypatch,
):
    process = make_process()
    popen = MagicMock(return_value=process)

    # Replace only the module's os reference.
    # Do not modify the shared Python os module.
    monkeypatch.setattr(
        image_builder_module,
        "os",
        SimpleNamespace(name="posix"),
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        popen,
    )

    _ProcessController.start(["docker", "build"])

    kwargs = popen.call_args.kwargs

    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["text"] is False
    assert kwargs["bufsize"] == 0


def test_process_controller_starts_windows_process_group(
    monkeypatch,
):
    process = make_process()
    popen = MagicMock(return_value=process)

    monkeypatch.setattr(
        image_builder_module,
        "os",
        SimpleNamespace(name="nt"),
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        popen,
    )

    _ProcessController.start(["docker", "build"])

    assert popen.call_args.kwargs["creationflags"] == 512


def test_process_controller_rejects_unsupported_platform(
    monkeypatch,
):
    monkeypatch.setattr(
        image_builder_module,
        "os",
        SimpleNamespace(name="unsupported"),
    )

    with pytest.raises(
        RuntimeError,
        match="Unsupported operating system",
    ):
        _ProcessController.start(["docker", "build"])


# =========================================================
# PROCESS CONTROLLER: POSIX GROUP DETECTION
# =========================================================


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_process_group_exists_when_killpg_zero_succeeds(
    monkeypatch,
):
    killpg = MagicMock()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        killpg,
    )

    assert _ProcessController._process_group_exists(12345) is True

    killpg.assert_called_once_with(12345, 0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_process_group_does_not_exist_after_lookup_error(
    monkeypatch,
):
    def missing_group(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        missing_group,
    )

    assert _ProcessController._process_group_exists(12345) is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_process_group_exists_when_permission_is_denied(
    monkeypatch,
):
    def denied_group(pid, sig):
        raise PermissionError()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        denied_group,
    )

    assert _ProcessController._process_group_exists(12345) is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_process_group_exists_when_other_os_error_occurs(
    monkeypatch,
):
    def failed_group(pid, sig):
        raise OSError("signal operation failed")

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        failed_group,
    )

    assert _ProcessController._process_group_exists(12345) is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_wait_for_process_group_exit_returns_immediately_when_absent(
    monkeypatch,
):
    monkeypatch.setattr(
        _ProcessController,
        "_process_group_exists",
        lambda pid: False,
    )

    assert _ProcessController._wait_for_process_group_exit(
        12345,
        1,
    ) is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_wait_for_process_group_exit_stops_at_deadline(
    monkeypatch,
):
    monkeypatch.setattr(
        _ProcessController,
        "_process_group_exists",
        lambda pid: True,
    )

    monkeypatch.setattr(
        image_builder_module.time,
        "monotonic",
        lambda: 100,
    )

    assert _ProcessController._wait_for_process_group_exit(
        12345,
        0,
    ) is False


# =========================================================
# PROCESS CONTROLLER: POSIX TERMINATION
# =========================================================


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_terminate_posix_group_when_parent_already_exited(
    monkeypatch,
):
    process = make_process(returncode=0)

    killpg = MagicMock()
    wait_for_group = MagicMock(return_value=True)
    reap = MagicMock()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        killpg,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_wait_for_process_group_exit",
        wait_for_group,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_reap_process",
        reap,
    )

    _ProcessController._terminate_posix(process)

    killpg.assert_called_once_with(
        process.pid,
        signal.SIGTERM,
    )

    wait_for_group.assert_called_once_with(
        process.pid,
        _ProcessController.TERMINATION_GRACE_SECONDS,
    )

    reap.assert_called_once_with(process)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_terminate_posix_uses_sigkill_after_grace_period(
    monkeypatch,
):
    process = make_process()

    killpg = MagicMock()
    wait_for_group = MagicMock(side_effect=[False, True])
    reap = MagicMock()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        killpg,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_wait_for_process_group_exit",
        wait_for_group,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_reap_process",
        reap,
    )

    _ProcessController._terminate_posix(process)

    assert killpg.call_args_list == [
        call(process.pid, signal.SIGTERM),
        call(process.pid, signal.SIGKILL),
    ]

    assert wait_for_group.call_count == 2
    reap.assert_called_once_with(process)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_terminate_posix_does_not_force_kill_group_that_exits(
    monkeypatch,
):
    process = make_process()

    killpg = MagicMock()
    wait_for_group = MagicMock(return_value=True)

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        killpg,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_wait_for_process_group_exit",
        wait_for_group,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_reap_process",
        MagicMock(),
    )

    _ProcessController._terminate_posix(process)

    assert killpg.call_args_list == [
        call(process.pid, signal.SIGTERM),
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_terminate_posix_handles_missing_group(
    monkeypatch,
):
    process = make_process()

    def missing_group(pid, sig):
        raise ProcessLookupError()

    reap = MagicMock()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        missing_group,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_reap_process",
        reap,
    )

    _ProcessController._terminate_posix(process)

    reap.assert_called_once_with(process)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-specific behavior")
def test_terminate_posix_falls_back_on_signal_error(
    monkeypatch,
):
    process = make_process()

    def failed_group(pid, sig):
        raise OSError("signal failure")

    fallback = MagicMock()

    monkeypatch.setattr(
        image_builder_module.os,
        "killpg",
        failed_group,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_terminate_direct_process",
        fallback,
    )

    _ProcessController._terminate_posix(process)

    fallback.assert_called_once_with(process)


# =========================================================
# PROCESS CONTROLLER: PROCESS REAPING
# =========================================================


def test_reap_process_waits_for_exit():
    process = make_process()

    _ProcessController._reap_process(process)

    process.wait.assert_called_once_with(
        timeout=_ProcessController.FORCE_TERMINATION_TIMEOUT_SECONDS
    )


def test_reap_process_kills_after_wait_timeout():
    process = make_process()

    process.wait.side_effect = [
        subprocess.TimeoutExpired("docker", 5),
        None,
    ]

    _ProcessController._reap_process(process)

    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_reap_process_handles_kill_error():
    process = make_process()

    process.wait.side_effect = [
        subprocess.TimeoutExpired("docker", 5),
        None,
    ]

    process.kill.side_effect = OSError("already exited")

    _ProcessController._reap_process(process)

    process.kill.assert_called_once()
    assert process.wait.call_count == 2


# =========================================================
# PROCESS CONTROLLER: DIRECT PROCESS FALLBACK
# =========================================================


def test_terminate_direct_process_does_nothing_when_finished():
    process = make_process(returncode=0)

    _ProcessController._terminate_direct_process(process)

    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_terminate_direct_process_terminates_running_process():
    process = make_process()

    process.poll.side_effect = lambda: (
        -15 if process.terminate.called else None
    )

    _ProcessController._terminate_direct_process(process)

    process.terminate.assert_called_once()

    process.wait.assert_called_once_with(
        timeout=_ProcessController.TERMINATION_GRACE_SECONDS
    )


def test_terminate_direct_process_kills_when_terminate_fails():
    process = make_process(returncode=None)

    process.poll.return_value = None
    process.terminate.side_effect = OSError("terminate failed")

    _ProcessController._terminate_direct_process(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_terminate_direct_process_kills_after_wait_timeout():
    process = make_process(returncode=None)

    process.poll.side_effect = lambda: (
        -9 if process.terminate.called else None
    )

    process.wait.side_effect = [
        subprocess.TimeoutExpired("docker", 5),
        None,
    ]

    _ProcessController._terminate_direct_process(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


# =========================================================
# PROCESS CONTROLLER: WINDOWS TERMINATION
# =========================================================


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific behavior")
def test_terminate_windows_uses_taskkill(
    monkeypatch,
):
    process = make_process()

    run = MagicMock(
        return_value=MagicMock(returncode=0)
    )

    reap = MagicMock()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "run",
        run,
    )

    monkeypatch.setattr(
        _ProcessController,
        "_reap_process",
        reap,
    )

    _ProcessController._terminate_windows(process)

    run.assert_called_once_with(
        [
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
        ],
        capture_output=True,
        text=True,
        timeout=_ProcessController.FORCE_TERMINATION_TIMEOUT_SECONDS,
        check=False,
    )

    reap.assert_called_once_with(process)


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific behavior")
def test_terminate_windows_falls_back_when_taskkill_fails(
    monkeypatch,
):
    process = make_process()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "run",
        MagicMock(
            return_value=MagicMock(returncode=1)
        ),
    )

    fallback = MagicMock()

    monkeypatch.setattr(
        _ProcessController,
        "_terminate_direct_process",
        fallback,
    )

    _ProcessController._terminate_windows(process)

    fallback.assert_called_once_with(process)


# =========================================================
# BUILDER PROCESS WRAPPERS
# =========================================================


def test_terminate_process_delegates_to_controller(monkeypatch):
    process = make_process()
    terminate = MagicMock()

    monkeypatch.setattr(
        _ProcessController,
        "terminate",
        terminate,
    )

    SandboxImageBuilder._terminate_process(process)

    terminate.assert_called_once_with(process)


# =========================================================
# RESULT CONSTRUCTION
# =========================================================


def test_build_result_preserves_captured_output(builder):
    capture = _OutputCapture()

    capture.append("stdout", "success")
    capture.append("stderr", "warning")

    result = builder._build_result(
        capture,
        exit_code=0,
        success=True,
    )

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout == "success"
    assert result.stderr == "warning"
    assert result.image_name == builder.IMAGE_NAME


def test_build_result_fails_when_reader_error_exists(builder):
    capture = _OutputCapture()

    capture.record_reader_error(
        "stderr",
        OSError("reader failed"),
    )

    result = builder._build_result(
        capture,
        exit_code=0,
        success=True,
    )

    assert result.success is False
    assert "Output reader errors" in result.error_message
    assert "reader failed" in result.error_message


def test_build_result_appends_reader_error_to_existing_error(builder):
    capture = _OutputCapture()

    capture.record_reader_error(
        "stdout",
        OSError("reader failed"),
    )

    result = builder._build_result(
        capture,
        exit_code=1,
        success=False,
        error_message="Build failed.",
    )

    assert result.success is False
    assert "Build failed." in result.error_message
    assert "reader failed" in result.error_message


def test_output_collection_error_returns_failure(builder):
    capture = _OutputCapture()

    result = builder._output_collection_error(
        capture,
        exit_code=1,
    )

    assert result.success is False
    assert result.exit_code == 1
    assert "output-reader threads" in result.error_message


# =========================================================
# BUILD VALIDATION
# =========================================================


def test_build_rejects_preexisting_cancellation(
    monkeypatch,
    builder,
):
    event = threading.Event()
    event.set()

    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    with pytest.raises(TaskCancelledError):
        builder.build(event)


def test_build_fails_when_docker_is_missing(
    monkeypatch,
    builder,
):
    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: False,
    )

    result = builder.build()

    assert result.success is False
    assert result.exit_code is None
    assert "Docker executable was not found" in result.error_message


def test_build_fails_when_project_root_is_missing(
    monkeypatch,
    tmp_path,
):
    missing_root = tmp_path / "missing"
    builder = SandboxImageBuilder(missing_root)

    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    result = builder.build()

    assert result.success is False
    assert "project root does not exist" in result.error_message


def test_build_fails_when_dockerfile_is_missing(
    monkeypatch,
    project_root,
):
    dockerfile = (
        project_root
        / "src"
        / "agentops"
        / "tools"
        / "docker"
        / "sandbox"
        / "Dockerfile"
    )

    dockerfile.unlink()

    builder = SandboxImageBuilder(project_root)

    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    result = builder.build()

    assert result.success is False
    assert "Dockerfile was not found" in result.error_message


def test_build_fails_when_build_context_is_missing(
    monkeypatch,
    project_root,
):
    dockerfile = (
        project_root
        / "src"
        / "agentops"
        / "tools"
        / "docker"
        / "sandbox"
        / "Dockerfile"
    )

    builder = SandboxImageBuilder(project_root)

    dockerfile.unlink()
    dockerfile.parent.rmdir()

    monkeypatch.setattr(
        builder,
        "_docker_available",
        lambda: True,
    )

    result = builder.build()

    assert result.success is False
    assert "Dockerfile was not found" in result.error_message


# =========================================================
# BUILD EXECUTION
# =========================================================


def test_build_constructs_expected_docker_command(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process()

    popen = MagicMock(return_value=process)

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        popen,
    )

    monkeypatch.setattr(
        builder,
        "image_exists",
        lambda: True,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is True

    command = popen.call_args.args[0]

    assert command == [
        "docker",
        "build",
        "--tag",
        builder.IMAGE_NAME,
        "--file",
        str(builder.dockerfile),
        str(builder.build_context),
    ]

    kwargs = popen.call_args.kwargs

    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["text"] is False
    assert kwargs["bufsize"] == 0


def test_build_uses_only_sandbox_directory_as_context(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        builder,
        "image_exists",
        lambda: True,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is True
    assert str(builder.build_context) == str(builder.dockerfile.parent)


def test_build_captures_stdout_and_stderr(
    monkeypatch,
    available_docker,
):
    builder = available_docker

    process = make_process(
        stdout=b"image layer created",
        stderr=b"build warning",
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        builder,
        "image_exists",
        lambda: True,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is True
    assert "image layer created" in result.stdout
    assert "build warning" in result.stderr


def test_build_fails_when_docker_returns_nonzero(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process(returncode=1)

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is False
    assert result.exit_code == 1
    assert result.error_message == "Docker image build failed."


def test_build_fails_when_image_verification_fails(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        builder,
        "image_exists",
        lambda: False,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is False
    assert "could not be verified" in result.error_message


def test_build_handles_popen_os_error(
    monkeypatch,
    available_docker,
):
    builder = available_docker

    def raise_error(*args, **kwargs):
        raise OSError("process creation failed")

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        raise_error,
    )

    result = builder.build()

    assert result.success is False
    assert result.exit_code is None
    assert "Unable to execute Docker build" in result.error_message
    assert "process creation failed" in result.error_message


def test_build_handles_unexpected_popen_error(
    monkeypatch,
    available_docker,
):
    builder = available_docker

    def raise_error(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        raise_error,
    )

    result = builder.build()

    assert result.success is False
    assert "Unexpected Docker image build error" in result.error_message
    assert "unexpected failure" in result.error_message


def test_build_cancels_running_process(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    event = threading.Event()

    process = make_process(returncode=None)

    process.poll.side_effect = lambda: (
        -15 if process.terminate.called else None
    )

    process.returncode = -15

    def cancel_during_sleep(seconds):
        event.set()

    terminate = MagicMock(
        side_effect=lambda proc: proc.terminate()
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        image_builder_module.time,
        "sleep",
        cancel_during_sleep,
    )

    monkeypatch.setattr(
        builder,
        "_terminate_process",
        terminate,
    )

    with pytest.raises(TaskCancelledError):
        builder.build(event)

    assert terminate.call_count == 1

    assert terminate.call_args_list[0] == call(process)


def test_build_enforces_timeout(
    monkeypatch,
    available_docker,
):
    builder = available_docker

    process = make_process(returncode=None)

    process.poll.side_effect = lambda: (
        -15 if process.terminate.called else None
    )

    process.returncode = -15

    terminate = MagicMock(
        side_effect=lambda proc: proc.terminate()
    )

    monkeypatch.setattr(
        builder,
        "BUILD_TIMEOUT_SECONDS",
        0,
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        builder,
        "_terminate_process",
        terminate,
    )

    result = builder.build()

    assert result.success is False
    assert "exceeded 0 seconds" in result.error_message

    assert terminate.call_count == 1
    assert terminate.call_args_list[0] == call(process)


def test_build_handles_reader_start_failure(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        builder,
        "_start_output_readers",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("reader startup failed")
        ),
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is False
    assert "reader startup failed" in result.error_message


def test_build_returns_failure_when_output_readers_do_not_finish(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        builder,
        "_join_output_readers",
        lambda readers: False,
    )

    monkeypatch.setattr(
        builder,
        "_stop_output_readers",
        lambda process, readers: False,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is False
    assert "output-reader threads" in result.error_message


def test_build_propagates_cancellation_after_process_finishes(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    event = threading.Event()
    process = make_process()

    calls = {"count": 0}

    def cancel_after_process(cancel_event):
        calls["count"] += 1

        if calls["count"] >= 2:
            event.set()

        if event.is_set():
            raise TaskCancelledError(
                "Task cancellation was requested."
            )

    monkeypatch.setattr(
        image_builder_module,
        "check_cancelled",
        cancel_after_process,
    )

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    disable_builder_cleanup(monkeypatch, builder)

    with pytest.raises(TaskCancelledError):
        builder.build(event)


def test_build_result_is_not_successful_when_reader_errors_occur(
    monkeypatch,
    available_docker,
):
    builder = available_docker
    process = make_process()

    monkeypatch.setattr(
        image_builder_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    original_start = builder._start_output_readers

    def start_readers(process, capture):
        capture.record_reader_error(
            "stdout",
            OSError("simulated reader failure"),
        )

        return original_start(process, capture)

    monkeypatch.setattr(
        builder,
        "_start_output_readers",
        start_readers,
    )

    monkeypatch.setattr(
        builder,
        "image_exists",
        lambda: True,
    )

    disable_builder_cleanup(monkeypatch, builder)

    result = builder.build()

    assert result.success is False
    assert "Output reader errors" in result.error_message
    assert "simulated reader failure" in result.error_message
