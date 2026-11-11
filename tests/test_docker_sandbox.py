
import subprocess
import threading
import time

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentops.tools.docker_sandbox import (
    DockerSandbox,
    DockerSandboxResult,
    SandboxConfig,
    _OutputCapture,
)


# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "test_example.py").write_text(
        "def test_example():\n    assert True\n",
        encoding="utf-8",
    )
    return project


@pytest.fixture
def sandbox():
    return DockerSandbox()


# ---------------------------------------------------------
# SANDBOX CONFIGURATION
# ---------------------------------------------------------


def test_default_configuration_is_valid():
    config = SandboxConfig()

    config.validate()

    assert config.image == "agentops-python-sandbox:latest"
    assert config.timeout_seconds == 120
    assert config.memory_limit == "512m"
    assert config.memory_swap_limit == "512m"
    assert config.cpu_limit == "1.0"
    assert config.process_limit == 64
    assert config.tmpfs_size == "64m"


@pytest.mark.parametrize(
    "image",
    [
        "",
        " invalid-image",
        "-invalid",
        "image with spaces",
        "image;command",
        "image\ninjection",
        "image|pipe",
        "image" * 256,
    ],
)
def test_invalid_image_names_are_rejected(image):
    config = SandboxConfig(image=image)

    with pytest.raises(ValueError, match="Invalid Docker image"):
        config.validate()


@pytest.mark.parametrize(
    "image",
    [
        "agentops-python-sandbox:latest",
        "python:3.12-slim",
        "registry.example.com/team/image:v1",
        "localhost:5000/image:latest",
        "image@sha256:abcdef123456",
    ],
)
def test_valid_image_names_are_accepted(image):
    SandboxConfig(image=image).validate()


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        901,
        1.5,
        "120",
        True,
        None,
    ],
)
def test_invalid_timeout_values_are_rejected(timeout):
    config = SandboxConfig(timeout_seconds=timeout)

    with pytest.raises(ValueError, match="Timeout"):
        config.validate()


@pytest.mark.parametrize("timeout", [1, 120, 900])
def test_valid_timeout_boundaries_are_accepted(timeout):
    SandboxConfig(timeout_seconds=timeout).validate()


@pytest.mark.parametrize(
    "process_limit",
    [
        0,
        -1,
        257,
        1.5,
        "64",
        True,
        None,
    ],
)
def test_invalid_process_limits_are_rejected(process_limit):
    config = SandboxConfig(process_limit=process_limit)

    with pytest.raises(ValueError, match="Process limit"):
        config.validate()


@pytest.mark.parametrize("limit", [1, 64, 256])
def test_valid_process_limits_are_accepted(limit):
    SandboxConfig(process_limit=limit).validate()


@pytest.mark.parametrize(
    "cpu_limit",
    [
        "0.09",
        "8.1",
        "invalid",
        "",
        None,
    ],
)
def test_invalid_cpu_limits_are_rejected(cpu_limit):
    config = SandboxConfig(cpu_limit=cpu_limit)

    with pytest.raises(ValueError, match="CPU limit"):
        config.validate()


@pytest.mark.parametrize(
    "cpu_limit",
    ["0.1", "1.0", "4", "8.0"],
)
def test_valid_cpu_limits_are_accepted(cpu_limit):
    SandboxConfig(cpu_limit=cpu_limit).validate()


@pytest.mark.parametrize(
    "memory",
    [
        "0m",
        "512",
        "512M",
        "512mb",
        "1g",
        "invalid",
        "",
        None,
    ],
)
def test_invalid_memory_limits_are_rejected(memory):
    config = SandboxConfig(memory_limit=memory)

    with pytest.raises(ValueError, match="Memory limit"):
        config.validate()


@pytest.mark.parametrize(
    "swap",
    [
        "0m",
        "512",
        "512M",
        "1g",
        "invalid",
        "",
        None,
    ],
)
def test_invalid_memory_swap_limits_are_rejected(swap):
    config = SandboxConfig(memory_swap_limit=swap)

    with pytest.raises(ValueError, match="Memory swap"):
        config.validate()


def test_memory_swap_cannot_be_less_than_memory():
    config = SandboxConfig(
        memory_limit="512m",
        memory_swap_limit="256m",
    )

    with pytest.raises(ValueError, match="greater than or equal"):
        config.validate()


def test_memory_swap_can_equal_memory():
    SandboxConfig(
        memory_limit="512m",
        memory_swap_limit="512m",
    ).validate()


def test_memory_swap_can_exceed_memory():
    SandboxConfig(
        memory_limit="512m",
        memory_swap_limit="1024m",
    ).validate()


@pytest.mark.parametrize(
    "tmpfs_size",
    [
        "0m",
        "64",
        "64M",
        "64mb",
        "1g",
        "invalid",
        "",
        None,
    ],
)
def test_invalid_tmpfs_sizes_are_rejected(tmpfs_size):
    config = SandboxConfig(tmpfs_size=tmpfs_size)

    with pytest.raises(ValueError, match="Temporary storage"):
        config.validate()


def test_custom_valid_configuration_is_accepted():
    config = SandboxConfig(
        image="python:3.12-slim",
        timeout_seconds=300,
        memory_limit="1024m",
        memory_swap_limit="2048m",
        cpu_limit="2.0",
        process_limit=128,
        tmpfs_size="128m",
    )

    config.validate()


def test_constructor_uses_default_configuration():
    sandbox = DockerSandbox()

    assert isinstance(sandbox.config, SandboxConfig)


def test_constructor_uses_supplied_configuration():
    config = SandboxConfig(timeout_seconds=45)

    sandbox = DockerSandbox(config=config)

    assert sandbox.config is config
    assert sandbox.config.timeout_seconds == 45


def test_constructor_rejects_invalid_configuration():
    config = SandboxConfig(timeout_seconds=0)

    with pytest.raises(ValueError):
        DockerSandbox(config=config)


# ---------------------------------------------------------
# RESULT MODEL
# ---------------------------------------------------------


def test_result_defaults_environment_to_execution_environment():
    result = DockerSandboxResult(
        status="PASSED",
        exit_code=0,
        duration_seconds=1.0,
        stdout="passed",
        stderr="",
    )

    assert result.environment == "docker"
    assert result.execution_environment == "docker"


def test_result_environment_overrides_execution_environment():
    result = DockerSandboxResult(
        status="PASSED",
        exit_code=0,
        duration_seconds=1.0,
        stdout="",
        stderr="",
        execution_environment="docker",
        environment="custom",
    )

    assert result.environment == "custom"
    assert result.execution_environment == "custom"


def test_result_succeeded_property():
    result = DockerSandboxResult(
        status="PASSED",
        exit_code=0,
        duration_seconds=1.0,
        stdout="",
        stderr="",
    )

    assert result.succeeded is True
    assert result.failed is False


def test_result_failed_property():
    result = DockerSandboxResult(
        status="FAILED",
        exit_code=1,
        duration_seconds=1.0,
        stdout="",
        stderr="",
    )

    assert result.failed is True
    assert result.succeeded is False


def test_nonterminal_result_is_not_successful_or_failed():
    result = DockerSandboxResult(
        status="TIMEOUT",
        exit_code=None,
        duration_seconds=1.0,
        stdout="",
        stderr="",
    )

    assert result.succeeded is False
    assert result.failed is False


# ---------------------------------------------------------
# BOUNDED OUTPUT CAPTURE
# ---------------------------------------------------------


def test_output_capture_stores_stdout():
    capture = _OutputCapture()

    capture.append("stdout", "hello")

    assert capture.get_text("stdout") == "hello"


def test_output_capture_stores_stderr():
    capture = _OutputCapture()

    capture.append("stderr", "error")

    assert capture.get_text("stderr") == "error"


def test_output_capture_keeps_streams_separate():
    capture = _OutputCapture()

    capture.append("stdout", "normal output")
    capture.append("stderr", "error output")

    assert capture.get_text("stdout") == "normal output"
    assert capture.get_text("stderr") == "error output"


def test_output_capture_ignores_empty_chunks():
    capture = _OutputCapture()

    capture.append("stdout", "")
    capture.append("stderr", "")

    assert capture.get_text("stdout") == ""
    assert capture.get_text("stderr") == ""


def test_output_capture_preserves_multiple_chunks():
    capture = _OutputCapture()

    capture.append("stdout", "first ")
    capture.append("stdout", "second ")
    capture.append("stdout", "third")

    assert capture.get_text("stdout") == "first second third"


def test_output_capture_enforces_character_limit(monkeypatch):
    monkeypatch.setattr(_OutputCapture, "MAX_CHARS", 10)

    capture = _OutputCapture()

    capture.append("stdout", "123456")
    capture.append("stdout", "789012")

    output = capture.get_text("stdout")

    assert output.endswith("3456789012")
    assert len(output.split("\n", 1)[1]) == 10
    assert "Earlier output truncated" in output


def test_output_capture_truncates_oldest_chunks_first(monkeypatch):
    monkeypatch.setattr(_OutputCapture, "MAX_CHARS", 8)

    capture = _OutputCapture()

    capture.append("stdout", "AAAA")
    capture.append("stdout", "BBBB")
    capture.append("stdout", "CCCC")

    output = capture.get_text("stdout")

    assert output.endswith("BBBBCCCC")
    assert "AAAA" not in output


def test_output_capture_truncates_part_of_oldest_chunk(monkeypatch):
    monkeypatch.setattr(_OutputCapture, "MAX_CHARS", 6)

    capture = _OutputCapture()

    capture.append("stdout", "12345678")

    output = capture.get_text("stdout")

    assert output.endswith("345678")


def test_output_capture_limits_stdout_and_stderr_independently(
    monkeypatch,
):
    monkeypatch.setattr(_OutputCapture, "MAX_CHARS", 5)

    capture = _OutputCapture()

    capture.append("stdout", "123456")
    capture.append("stderr", "abcdef")

    assert capture.get_text("stdout").endswith("23456")
    assert capture.get_text("stderr").endswith("bcdef")


def test_output_capture_is_thread_safe(monkeypatch):
    monkeypatch.setattr(_OutputCapture, "MAX_CHARS", 100_000)

    capture = _OutputCapture()

    def write_output(prefix):
        for index in range(100):
            capture.append("stdout", f"{prefix}-{index};")

    threads = [
        threading.Thread(
            target=write_output,
            args=(f"thread-{number}",),
        )
        for number in range(5)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    output = capture.get_text("stdout")

    assert len(output) > 0

    for number in range(5):
        assert f"thread-{number}-" in output


# ---------------------------------------------------------
# WORKSPACE VALIDATION
# ---------------------------------------------------------


def test_valid_workspace_is_resolved(sandbox, workspace):
    result = sandbox._validate_workspace(workspace)

    assert result == workspace.resolve()
    assert result.is_dir()


def test_workspace_string_path_is_accepted(sandbox, workspace):
    result = sandbox._validate_workspace(str(workspace))

    assert result == workspace.resolve()


def test_missing_workspace_is_rejected(sandbox, tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(ValueError, match="does not exist"):
        sandbox._validate_workspace(missing)


def test_workspace_file_is_rejected(sandbox, tmp_path):
    file_path = tmp_path / "project.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    with pytest.raises(ValueError, match="directory"):
        sandbox._validate_workspace(file_path)


def test_workspace_root_symlink_is_rejected(sandbox, tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()

    link = tmp_path / "linked-project"

    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this system.")

    with pytest.raises(ValueError, match="symbolic link"):
        sandbox._validate_workspace(link)


# ---------------------------------------------------------
# DOCKER AVAILABILITY AND IMAGE CHECKS
# ---------------------------------------------------------


def test_docker_available_when_executable_exists(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.shutil.which",
        lambda executable: "/usr/bin/docker",
    )

    assert sandbox._docker_available() is True


def test_docker_unavailable_when_executable_missing(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.shutil.which",
        lambda executable: None,
    )

    assert sandbox._docker_available() is False


def test_image_check_succeeds_when_image_exists(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="image details",
            stderr="",
        ),
    )

    assert sandbox._check_image() is None


def test_image_check_reports_missing_image(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="No such image",
        ),
    )

    error = sandbox._check_image()

    assert "not available" in error
    assert "Build the AgentOps sandbox image" in error


def test_image_check_handles_timeout(
    sandbox,
    monkeypatch,
):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="docker image inspect",
            timeout=15,
        )

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        raise_timeout,
    )

    assert sandbox._check_image() == (
        "Docker image inspection timed out."
    )


def test_image_check_handles_os_error(
    sandbox,
    monkeypatch,
):
    def raise_error(*args, **kwargs):
        raise OSError("Docker cannot be executed")

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        raise_error,
    )

    error = sandbox._check_image()

    assert "Unable to inspect Docker image" in error
    assert "Docker cannot be executed" in error


def test_image_check_uses_configured_image(
    sandbox,
    monkeypatch,
):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)

        return subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="",
        )

    sandbox.config.image = "custom-image:v2"

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        fake_run,
    )

    sandbox._check_image()

    assert commands[0] == [
        "docker",
        "image",
        "inspect",
        "custom-image:v2",
    ]


# ---------------------------------------------------------
# DOCKER COMMAND SECURITY
# ---------------------------------------------------------


def test_docker_command_uses_expected_executable(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-example",
    )

    assert command[0:2] == ["docker", "run"]


def test_docker_command_uses_unique_container_name(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-unique",
    )

    assert "agentops-test-unique" in command


def test_docker_command_disables_network(sandbox, workspace):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-network",
    )

    assert "--network=none" in command


def test_docker_command_uses_read_only_container_filesystem(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-readonly",
    )

    assert "--read-only" in command


def test_docker_command_drops_all_capabilities(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-capabilities",
    )

    assert "--cap-drop=ALL" in command


def test_docker_command_disables_privilege_escalation(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-privileges",
    )

    assert "--security-opt=no-new-privileges:true" in command


def test_docker_command_uses_non_root_user(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-user",
    )

    user_index = command.index("--user")

    assert command[user_index + 1] == "10001:10001"


def test_docker_command_mounts_workspace_read_only(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-mount",
    )

    mount_index = command.index("--mount")
    mount = command[mount_index + 1]

    assert f"source={workspace.resolve()}" in mount
    assert "target=/workspace" in mount
    assert "readonly" in mount


def test_docker_command_mounts_only_project_workspace(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-volumes",
    )

    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]

    assert len(mounts) == 1
    assert str(workspace.resolve()) in mounts[0]


def test_docker_command_uses_limited_tmpfs(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-tmpfs",
    )

    tmpfs_index = command.index("--tmpfs")
    tmpfs = command[tmpfs_index + 1]

    assert tmpfs.startswith("/tmp:")
    assert "rw" in tmpfs
    assert "noexec" in tmpfs
    assert "nosuid" in tmpfs
    assert "nodev" in tmpfs
    assert "size=64m" in tmpfs


def test_docker_command_applies_resource_limits(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-resources",
    )

    assert command[
        command.index("--pids-limit") + 1
    ] == "64"

    assert command[
        command.index("--memory") + 1
    ] == "512m"

    assert command[
        command.index("--memory-swap") + 1
    ] == "512m"

    assert command[
        command.index("--cpus") + 1
    ] == "1.0"


def test_docker_command_uses_expected_working_directory(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-workdir",
    )

    workdir_index = command.index("--workdir")

    assert command[workdir_index + 1] == "/workspace"


def test_docker_command_disables_pytest_plugin_autoload(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-plugins",
    )

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in command


def test_docker_command_uses_pytest_without_cache(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-pytest",
    )

    assert command[-6:] == [
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]


def test_docker_command_does_not_accept_custom_shell_commands(
    sandbox,
    workspace,
):
    command = sandbox._build_command(
        workspace.resolve(),
        "agentops-test-fixed",
    )

    assert command[-6:] == [
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]


# ---------------------------------------------------------
# OUTPUT READER MANAGEMENT
# ---------------------------------------------------------


def test_read_stream_captures_stream_output():
    capture = _OutputCapture()

    class FakeStream:
        def __init__(self):
            self.chunks = iter(["first", "second", ""])

        def read(self, size):
            return next(self.chunks)

        def close(self):
            pass

    DockerSandbox._read_stream(
        FakeStream(),
        capture,
        "stdout",
    )

    assert capture.get_text("stdout") == "firstsecond"


def test_read_stream_handles_stream_errors():
    capture = _OutputCapture()

    class FakeStream:
        def read(self, size):
            raise OSError("stream failed")

        def close(self):
            pass

    DockerSandbox._read_stream(
        FakeStream(),
        capture,
        "stderr",
    )

    assert capture.get_text("stderr") == ""


def test_start_output_collection_starts_two_readers(
    sandbox,
):
    process = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()

    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self)

    capture = _OutputCapture()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "agentops.tools.docker_sandbox.threading.Thread",
            FakeThread,
        )

        readers = sandbox._start_output_collection(
            process,
            capture,
        )

    assert len(readers) == 2
    assert len(started) == 2
    assert readers[0].kwargs["name"] == "agentops-docker-stdout"
    assert readers[1].kwargs["name"] == "agentops-docker-stderr"


def test_join_output_readers_returns_true_when_finished(
    sandbox,
):
    reader = MagicMock()
    reader.is_alive.return_value = False

    assert sandbox._join_output_readers([reader]) is True


def test_join_output_readers_returns_false_when_reader_is_alive(
    sandbox,
):
    reader = MagicMock()
    reader.is_alive.return_value = True

    assert sandbox._join_output_readers([reader]) is False


def test_captured_output_returns_both_streams_and_no_warning(
    sandbox,
):
    capture = _OutputCapture()
    capture.append("stdout", "test output")
    capture.append("stderr", "test warning")

    reader = MagicMock()
    reader.is_alive.return_value = False

    stdout, stderr, warning = sandbox._captured_output(
        capture,
        [reader],
    )

    assert stdout == "test output"
    assert stderr == "test warning"
    assert warning is None


def test_captured_output_reports_incomplete_readers(
    sandbox,
):
    capture = _OutputCapture()

    reader = MagicMock()
    reader.is_alive.return_value = True

    stdout, stderr, warning = sandbox._captured_output(
        capture,
        [reader],
    )

    assert stdout == ""
    assert stderr == ""
    assert "did not finish" in warning


# ---------------------------------------------------------
# PROCESS TERMINATION
# ---------------------------------------------------------


def test_terminate_process_does_nothing_if_already_exited(
    sandbox,
):
    process = MagicMock()
    process.poll.return_value = 0

    sandbox._terminate_process(process)

    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_terminate_process_sends_terminate_signal(
    sandbox,
):
    process = MagicMock()
    process.poll.side_effect = [None, 0]

    sandbox._terminate_process(process)

    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_terminate_process_kills_when_terminate_fails(
    sandbox,
):
    process = MagicMock()
    process.poll.return_value = None
    process.terminate.side_effect = OSError("terminate failed")

    sandbox._terminate_process(process)

    process.kill.assert_called_once()


def test_terminate_process_kills_after_wait_timeout(
    sandbox,
):
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired("docker", 5),
        None,
    ]

    sandbox._terminate_process(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


# ---------------------------------------------------------
# CONTAINER CLEANUP
# ---------------------------------------------------------


def test_container_missing_detection():
    assert DockerSandbox._container_missing(
        "No such container: agentops-test"
    )

    assert DockerSandbox._container_missing(
        "No container with name agentops-test"
    )


def test_container_missing_detection_is_case_insensitive():
    assert DockerSandbox._container_missing(
        "NO SUCH CONTAINER"
    )


def test_container_missing_detection_rejects_other_errors():
    assert not DockerSandbox._container_missing(
        "permission denied"
    )


def test_stop_container_succeeds(sandbox, monkeypatch):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="stopped",
            stderr="",
        ),
    )

    assert sandbox._stop_container("agentops-test") is None


def test_stop_container_ignores_missing_container(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="No such container",
        ),
    )

    assert sandbox._stop_container("agentops-test") is None


def test_stop_container_reports_docker_error(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="permission denied",
        ),
    )

    assert sandbox._stop_container("agentops-test") == (
        "permission denied"
    )


def test_stop_container_handles_timeout(
    sandbox,
    monkeypatch,
):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker stop", 10)

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        raise_timeout,
    )

    assert sandbox._stop_container("agentops-test") == (
        "Docker container stop operation timed out."
    )


def test_stop_container_handles_os_error(
    sandbox,
    monkeypatch,
):
    def raise_error(*args, **kwargs):
        raise OSError("Docker unavailable")

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        raise_error,
    )

    error = sandbox._stop_container("agentops-test")

    assert "Unable to stop Docker container" in error


def test_remove_container_succeeds(sandbox, monkeypatch):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="removed",
            stderr="",
        ),
    )

    assert sandbox._remove_container("agentops-test") is None


def test_remove_container_ignores_missing_container(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="No such container",
        ),
    )

    assert sandbox._remove_container("agentops-test") is None


def test_remove_container_reports_docker_error(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="permission denied",
        ),
    )

    assert sandbox._remove_container("agentops-test") == (
        "permission denied"
    )


def test_remove_container_handles_timeout(
    sandbox,
    monkeypatch,
):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker rm", 10)

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        raise_timeout,
    )

    assert sandbox._remove_container("agentops-test") == (
        "Docker container removal timed out."
    )


def test_remove_container_handles_os_error(
    sandbox,
    monkeypatch,
):
    def raise_error(*args, **kwargs):
        raise OSError("Docker unavailable")

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.run",
        raise_error,
    )

    error = sandbox._remove_container("agentops-test")

    assert "Unable to remove Docker container" in error


def test_cleanup_succeeds_when_stop_and_remove_succeed(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_stop_container",
        lambda name: None,
    )

    monkeypatch.setattr(
        sandbox,
        "_remove_container",
        lambda name: None,
    )

    assert sandbox._cleanup_container("agentops-test") is None


def test_cleanup_succeeds_when_container_already_exited(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_stop_container",
        lambda name: "No such container",
    )

    monkeypatch.setattr(
        sandbox,
        "_remove_container",
        lambda name: None,
    )

    assert sandbox._cleanup_container("agentops-test") is None


def test_cleanup_reports_stop_and_remove_failures(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_stop_container",
        lambda name: "stop failed",
    )

    monkeypatch.setattr(
        sandbox,
        "_remove_container",
        lambda name: "remove failed",
    )

    error = sandbox._cleanup_container("agentops-test")

    assert "Stop failed: stop failed" in error
    assert "removal failed: remove failed" in error


def test_cleanup_reports_remove_failure(
    sandbox,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_stop_container",
        lambda name: None,
    )

    monkeypatch.setattr(
        sandbox,
        "_remove_container",
        lambda name: "remove failed",
    )

    assert sandbox._cleanup_container("agentops-test") == (
        "remove failed"
    )


# ---------------------------------------------------------
# RESULT CONSTRUCTION
# ---------------------------------------------------------


def test_build_result_creates_structured_result():
    start_time = time.monotonic()

    result = DockerSandbox._build_result(
        start_time=start_time,
        status="PASSED",
        exit_code=0,
        stdout="success",
        stderr="",
        tests_executed=True,
    )

    assert isinstance(result, DockerSandboxResult)
    assert result.status == "PASSED"
    assert result.exit_code == 0
    assert result.stdout == "success"
    assert result.stderr == ""
    assert result.tests_executed is True
    assert result.duration_seconds >= 0


def test_build_result_preserves_error_message():
    result = DockerSandbox._build_result(
        start_time=time.monotonic(),
        status="FAILED",
        error_message="Test failure",
    )

    assert result.error_message == "Test failure"


# ---------------------------------------------------------
# TEST EXECUTION: EARLY EXIT CONDITIONS
# ---------------------------------------------------------


def test_run_tests_returns_cancelled_before_starting(
    sandbox,
    workspace,
):
    cancel_event = threading.Event()
    cancel_event.set()

    result = sandbox.run_tests(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.tests_executed is False
    assert "before starting" in result.error_message


def test_run_tests_rejects_invalid_workspace(
    sandbox,
    tmp_path,
):
    missing = tmp_path / "missing"

    result = sandbox.run_tests(missing)

    assert result.status == "INVALID_WORKSPACE"
    assert result.tests_executed is False


def test_run_tests_returns_docker_unavailable(
    sandbox,
    workspace,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: False,
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "DOCKER_UNAVAILABLE"
    assert result.tests_executed is False
    assert "Docker executable was not found" in result.error_message


def test_run_tests_returns_image_not_found(
    sandbox,
    workspace,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: "Sandbox image missing.",
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "IMAGE_NOT_FOUND"
    assert result.tests_executed is False
    assert result.error_message == "Sandbox image missing."


def test_run_tests_cancels_after_workspace_validation(
    sandbox,
    workspace,
    monkeypatch,
):
    cancel_event = threading.Event()

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: (
            cancel_event.set() is None
            and True
        ),
    )

    result = sandbox.run_tests(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.tests_executed is False
    assert "before container startup" in result.error_message


def test_run_tests_cancels_after_image_check(
    sandbox,
    workspace,
    monkeypatch,
):
    cancel_event = threading.Event()

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    def image_check():
        cancel_event.set()
        return None

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        image_check,
    )

    result = sandbox.run_tests(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.tests_executed is False


# ---------------------------------------------------------
# TEST EXECUTION: MOCKED DOCKER PROCESS
# ---------------------------------------------------------


def _configure_mocked_docker(
    sandbox,
    monkeypatch,
    return_code=0,
    stdout="2 passed",
    stderr="",
):
    """Configure the Docker preflight checks and process."""

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = return_code
    process.poll.return_value = return_code

    capture = _OutputCapture()
    capture.append("stdout", stdout)
    capture.append("stderr", stderr)

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        lambda process, output_capture: [],
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda output_capture, readers: (
            capture.get_text("stdout"),
            capture.get_text("stderr"),
            None,
        ),
    )

    return process


def test_run_tests_returns_passed_for_zero_exit_code(
    sandbox,
    workspace,
    monkeypatch,
):
    _configure_mocked_docker(
        sandbox,
        monkeypatch,
        return_code=0,
        stdout="2 passed",
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "PASSED"
    assert result.exit_code == 0
    assert result.stdout == "2 passed"
    assert result.tests_executed is True
    assert result.succeeded is True


def test_run_tests_returns_failed_for_nonzero_exit_code(
    sandbox,
    workspace,
    monkeypatch,
):
    _configure_mocked_docker(
        sandbox,
        monkeypatch,
        return_code=1,
        stdout="1 failed",
        stderr="AssertionError",
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "FAILED"
    assert result.exit_code == 1
    assert result.stdout == "1 failed"
    assert result.stderr == "AssertionError"
    assert result.tests_executed is True
    assert result.failed is True
    assert "non-zero exit code" in result.error_message


def test_run_tests_reports_execution_error_when_output_readers_fail(
    sandbox,
    workspace,
    monkeypatch,
):
    _configure_mocked_docker(
        sandbox,
        monkeypatch,
        return_code=0,
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda capture, readers: (
            "",
            "",
            "Docker output collection did not finish.",
        ),
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "EXECUTION_ERROR"
    assert result.exit_code == 0
    assert result.tests_executed is True
    assert "output collection failed" in result.error_message


def test_run_tests_reports_execution_error_when_popen_fails(
    sandbox,
    workspace,
    monkeypatch,
):
    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    def raise_error(*args, **kwargs):
        raise OSError("Docker process could not start")

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        raise_error,
    )

    monkeypatch.setattr(
        sandbox,
        "_cleanup_container",
        lambda name: None,
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "EXECUTION_ERROR"
    assert result.tests_executed is False
    assert "Unable to start Docker execution" in result.error_message
    assert "Docker process could not start" in result.error_message


def test_run_tests_passes_read_only_workspace_to_docker(
    sandbox,
    workspace,
    monkeypatch,
):
    commands = []

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = 0
    process.poll.return_value = 0

    def fake_popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        fake_popen,
    )

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        lambda process, capture: [],
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda capture, readers: ("", "", None),
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "PASSED"
    assert len(commands) == 1

    command = commands[0]

    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command

    mount_index = command.index("--mount")

    assert "readonly" in command[mount_index + 1]


def test_run_tests_uses_unique_container_names(
    sandbox,
    workspace,
    monkeypatch,
):
    names = []

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = 0
    process.poll.return_value = 0

    def fake_popen(command, **kwargs):
        name_index = command.index("--name")
        names.append(command[name_index + 1])
        return process

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        fake_popen,
    )

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        lambda process, capture: [],
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda capture, readers: ("", "", None),
    )

    first = sandbox.run_tests(workspace)
    second = sandbox.run_tests(workspace)

    assert first.status == "PASSED"
    assert second.status == "PASSED"
    assert len(names) == 2
    assert names[0] != names[1]
    assert names[0].startswith("agentops-test-")
    assert names[1].startswith("agentops-test-")


def test_run_tests_cleans_up_container_on_cancellation(
    sandbox,
    workspace,
    monkeypatch,
):
    cancel_event = threading.Event()
    cleanup_names = []

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = None

    # The process remains active until terminate() is called.
    process.poll.side_effect = (
        lambda: -15 if process.terminate.called else None
    )

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        lambda process, capture: [],
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda capture, readers: (
            "partial output",
            "",
            None,
        ),
    )

    def fake_cleanup(name):
        cleanup_names.append(name)
        return None

    monkeypatch.setattr(
        sandbox,
        "_cleanup_container",
        fake_cleanup,
    )

    def cancel_during_sleep(seconds):
        cancel_event.set()

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.time.sleep",
        cancel_during_sleep,
    )

    result = sandbox.run_tests(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert result.tests_executed is True
    assert result.stdout == "partial output"
    assert len(cleanup_names) == 1
    process.terminate.assert_called_once()


def test_run_tests_cleans_up_container_on_timeout(
    sandbox,
    workspace,
    monkeypatch,
):
    cleanup_names = []

    sandbox.config.timeout_seconds = 1

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = None

    # The process remains active until terminate() is called.
    process.poll.side_effect = (
        lambda: -15 if process.terminate.called else None
    )

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        lambda process, capture: [],
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda capture, readers: (
            "partial output",
            "",
            None,
        ),
    )

    def fake_cleanup(name):
        cleanup_names.append(name)
        return None

    monkeypatch.setattr(
        sandbox,
        "_cleanup_container",
        fake_cleanup,
    )

    # Use the real monotonic clock and sleep.
    # The configured one-second timeout will expire naturally.

    result = sandbox.run_tests(workspace)

    assert result.status == "TIMEOUT"
    assert result.tests_executed is True
    assert "exceeded the configured" in result.error_message
    assert len(cleanup_names) == 1
    process.terminate.assert_called_once()


def test_run_tests_cleans_up_after_unexpected_exception(
    sandbox,
    workspace,
    monkeypatch,
):
    cleanup_names = []

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = None

    # Once terminated, poll() reports that the process exited.
    process.poll.side_effect = lambda: process.returncode

    def mark_process_terminated():
        process.returncode = -15

    process.terminate.side_effect = mark_process_terminated

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    def raise_error(*args, **kwargs):
        raise RuntimeError("reader startup failed")

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        raise_error,
    )

    monkeypatch.setattr(
        sandbox,
        "_cleanup_container",
        lambda name: cleanup_names.append(name),
    )

    result = sandbox.run_tests(workspace)

    assert result.status == "EXECUTION_ERROR"
    assert "RuntimeError" in result.error_message
    assert "reader startup failed" in result.error_message
    assert len(cleanup_names) == 1
    process.terminate.assert_called_once()


def test_run_tests_preserves_cleanup_warning_on_cancellation(
    sandbox,
    workspace,
    monkeypatch,
):
    cancel_event = threading.Event()

    monkeypatch.setattr(
        sandbox,
        "_docker_available",
        lambda: True,
    )

    monkeypatch.setattr(
        sandbox,
        "_check_image",
        lambda: None,
    )

    process = MagicMock()
    process.stdout = None
    process.stderr = None
    process.returncode = None

    # The process exits only after the termination request.
    process.poll.side_effect = (
        lambda: -15 if process.terminate.called else None
    )

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    monkeypatch.setattr(
        sandbox,
        "_start_output_collection",
        lambda process, capture: [],
    )

    monkeypatch.setattr(
        sandbox,
        "_captured_output",
        lambda capture, readers: ("", "", None),
    )

    monkeypatch.setattr(
        sandbox,
        "_cleanup_container",
        lambda name: "remove failed",
    )

    monkeypatch.setattr(
        "agentops.tools.docker_sandbox.time.sleep",
        lambda seconds: cancel_event.set(),
    )

    result = sandbox.run_tests(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.status == "CANCELLED"
    assert "cleanup warning" in result.error_message
    assert "remove failed" in result.error_message
