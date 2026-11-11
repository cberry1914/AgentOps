
import threading
from unittest.mock import MagicMock

import pytest

from agentops.tools.cancellation import TaskCancelledError
from agentops.tools.docker_sandbox import (
    DockerSandboxResult,
    SandboxConfig,
)
from agentops.tools.sandbox_image_builder import ImageBuildResult
from agentops.tools.sandbox_manager import (
    SandboxManager,
    SandboxPreparationResult,
)


@pytest.fixture
def manager(tmp_path):
    return SandboxManager(project_root=tmp_path)


@pytest.fixture
def successful_build():
    return ImageBuildResult(
        success=True,
        image_name=SandboxManager().image_builder.IMAGE_NAME,
        exit_code=0,
        stdout="Image built",
        stderr="",
    )


@pytest.fixture
def successful_execution():
    return DockerSandboxResult(
        status="PASSED",
        exit_code=0,
        duration_seconds=1.0,
        stdout="2 passed",
        stderr="",
        execution_environment="docker",
        tests_executed=True,
    )


def test_preparation_result_fields():
    result = SandboxPreparationResult(
        ready=True,
        image_name="test-image",
        image_was_built=True,
    )

    assert result.ready is True
    assert result.image_name == "test-image"
    assert result.image_was_built is True
    assert result.error_message is None
    assert result.build_result is None


def test_manager_initializes_components(tmp_path):
    manager = SandboxManager(project_root=tmp_path)

    assert manager.config is not None
    assert manager.image_builder.project_root == tmp_path.resolve()
    assert manager.sandbox.config is manager.config


def test_manager_uses_default_config(tmp_path):
    manager = SandboxManager(project_root=tmp_path)

    assert isinstance(manager.config, SandboxConfig)
    assert manager.config.image == manager.image_builder.IMAGE_NAME


def test_manager_accepts_matching_custom_config(tmp_path):
    config = SandboxConfig(
        image=SandboxManager().image_builder.IMAGE_NAME
    )

    manager = SandboxManager(
        project_root=tmp_path,
        config=config,
    )

    assert manager.config is config


def test_manager_rejects_unsupported_custom_image(tmp_path):
    config = SandboxConfig(
        image="custom-sandbox:latest"
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        SandboxManager(
            project_root=tmp_path,
            config=config,
        )


def test_prepare_returns_ready_when_image_exists(manager, monkeypatch):
    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: True,
    )

    result = manager.prepare()

    assert result.ready is True
    assert result.image_name == manager.config.image
    assert result.image_was_built is False
    assert result.error_message is None
    assert result.build_result is None


def test_prepare_does_not_build_when_image_exists(manager, monkeypatch):
    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: True,
    )

    manager.image_builder.build = MagicMock()

    result = manager.prepare()

    assert result.ready is True
    manager.image_builder.build.assert_not_called()


def test_prepare_returns_unavailable_when_building_disabled(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: False,
    )

    manager.image_builder.build = MagicMock()

    result = manager.prepare(build_if_missing=False)

    assert result.ready is False
    assert result.image_was_built is False
    assert "building is disabled" in result.error_message
    manager.image_builder.build.assert_not_called()


def test_prepare_builds_missing_image(
    manager,
    successful_build,
    monkeypatch,
):
    image_checks = iter([False, True])

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: next(image_checks),
    )
    manager.image_builder.build = MagicMock(
        return_value=successful_build
    )

    result = manager.prepare()

    assert result.ready is True
    assert result.image_was_built is True
    assert result.build_result is successful_build
    assert result.error_message is None

    manager.image_builder.build.assert_called_once_with(
        cancel_event=None
    )


def test_prepare_passes_cancellation_event_to_builder(
    manager,
    successful_build,
    monkeypatch,
):
    event = threading.Event()
    image_checks = iter([False, True])

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: next(image_checks),
    )
    manager.image_builder.build = MagicMock(
        return_value=successful_build
    )

    result = manager.prepare(cancel_event=event)

    assert result.ready is True
    manager.image_builder.build.assert_called_once_with(
        cancel_event=event
    )


def test_prepare_returns_failure_when_build_fails(
    manager,
    monkeypatch,
):
    failed_build = ImageBuildResult(
        success=False,
        image_name=manager.config.image,
        exit_code=1,
        stdout="",
        stderr="Build error",
        error_message="Docker build failed.",
    )

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: False,
    )
    manager.image_builder.build = MagicMock(
        return_value=failed_build
    )

    result = manager.prepare()

    assert result.ready is False
    assert result.image_was_built is False
    assert result.error_message == "Docker build failed."
    assert result.build_result is failed_build


def test_prepare_uses_default_build_error_message(
    manager,
    monkeypatch,
):
    failed_build = ImageBuildResult(
        success=False,
        image_name=manager.config.image,
        exit_code=1,
        stdout="",
        stderr="",
        error_message=None,
    )

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: False,
    )
    manager.image_builder.build = MagicMock(
        return_value=failed_build
    )

    result = manager.prepare()

    assert result.ready is False
    assert result.error_message == (
        "Sandbox image construction failed."
    )


def test_prepare_fails_when_built_image_cannot_be_verified(
    manager,
    successful_build,
    monkeypatch,
):
    image_checks = iter([False, False])

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: next(image_checks),
    )
    manager.image_builder.build = MagicMock(
        return_value=successful_build
    )

    result = manager.prepare()

    assert result.ready is False
    assert result.image_was_built is True
    assert "could not be found afterward" in result.error_message
    assert result.build_result is successful_build


def test_prepare_propagates_task_cancellation(manager, monkeypatch):
    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: False,
    )

    manager.image_builder.build = MagicMock(
        side_effect=TaskCancelledError("Cancelled")
    )

    with pytest.raises(TaskCancelledError):
        manager.prepare()


def test_prepare_propagates_preexisting_cancellation(manager):
    event = threading.Event()
    event.set()

    manager.image_builder.image_exists = MagicMock()

    with pytest.raises(TaskCancelledError):
        manager.prepare(cancel_event=event)

    manager.image_builder.image_exists.assert_not_called()


def test_prepare_converts_image_check_exception_to_failure(
    manager,
    monkeypatch,
):
    def raise_error():
        raise RuntimeError("Docker inspection failed")

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        raise_error,
    )

    result = manager.prepare()

    assert result.ready is False
    assert result.image_was_built is False
    assert "Unexpected sandbox preparation error" in result.error_message
    assert "Docker inspection failed" in result.error_message


def test_prepare_converts_builder_exception_to_failure(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: False,
    )
    manager.image_builder.build = MagicMock(
        side_effect=RuntimeError("Build crashed")
    )

    result = manager.prepare()

    assert result.ready is False
    assert "Unexpected sandbox preparation error" in result.error_message
    assert "Build crashed" in result.error_message


def test_prepare_checks_cancellation_before_build(
    manager,
    monkeypatch,
):
    event = threading.Event()
    checks = {"count": 0}

    def cancel_during_image_check():
        checks["count"] += 1
        event.set()
        return False

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        cancel_during_image_check,
    )
    manager.image_builder.build = MagicMock()

    with pytest.raises(TaskCancelledError):
        manager.prepare(cancel_event=event)

    manager.image_builder.build.assert_not_called()


def test_run_tests_returns_sandbox_unavailable_when_preparation_fails(
    manager,
    monkeypatch,
):
    preparation = SandboxPreparationResult(
        ready=False,
        image_name=manager.config.image,
        image_was_built=False,
        error_message="Image unavailable.",
    )

    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: preparation,
    )

    result = manager.run_tests("/workspace")

    assert result.status == "SANDBOX_UNAVAILABLE"
    assert result.exit_code is None
    assert result.tests_executed is False
    assert result.execution_environment == "docker"
    assert result.error_message == "Image unavailable."


def test_run_tests_does_not_execute_when_preparation_fails(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=False,
            image_name=manager.config.image,
            image_was_built=False,
            error_message="Not ready.",
        ),
    )

    manager.sandbox.run_tests = MagicMock()

    manager.run_tests("/workspace")

    manager.sandbox.run_tests.assert_not_called()


def test_run_tests_executes_after_successful_preparation(
    manager,
    successful_execution,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=False,
        ),
    )

    manager.sandbox.run_tests = MagicMock(
        return_value=successful_execution
    )

    result = manager.run_tests("/workspace")

    assert result is successful_execution
    assert result.status == "PASSED"

    manager.sandbox.run_tests.assert_called_once_with(
        workspace_path="/workspace",
        cancel_event=None,
    )


def test_run_tests_passes_cancellation_event_to_preparation_and_sandbox(
    manager,
    successful_execution,
    monkeypatch,
):
    event = threading.Event()

    original_prepare = manager.prepare

    prepare_mock = MagicMock(
        side_effect=original_prepare
    )

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        lambda: True,
    )
    monkeypatch.setattr(
        manager,
        "prepare",
        prepare_mock,
    )

    manager.sandbox.run_tests = MagicMock(
        return_value=successful_execution
    )

    result = manager.run_tests(
        "/workspace",
        cancel_event=event,
    )

    assert result is successful_execution

    prepare_mock.assert_called_once_with(
        build_if_missing=True,
        cancel_event=event,
    )

    manager.sandbox.run_tests.assert_called_once_with(
        workspace_path="/workspace",
        cancel_event=event,
    )


def test_run_tests_passes_build_if_missing_option(
    manager,
    successful_execution,
    monkeypatch,
):
    prepare_mock = MagicMock(
        return_value=SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=False,
        )
    )

    monkeypatch.setattr(
        manager,
        "prepare",
        prepare_mock,
    )
    manager.sandbox.run_tests = MagicMock(
        return_value=successful_execution
    )

    manager.run_tests(
        "/workspace",
        build_if_missing=False,
    )

    prepare_mock.assert_called_once_with(
        build_if_missing=False,
        cancel_event=None,
    )


def test_run_tests_includes_preparation_in_duration(
    manager,
    successful_execution,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=True,
        ),
    )

    manager.sandbox.run_tests = MagicMock(
        return_value=successful_execution
    )

    result = manager.run_tests("/workspace")

    assert result.duration_seconds >= 0
    assert isinstance(result.duration_seconds, float)


def test_run_tests_returns_cancelled_when_preparation_raises(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        MagicMock(
            side_effect=TaskCancelledError("Cancelled")
        ),
    )

    result = manager.run_tests("/workspace")

    assert result.status == "CANCELLED"
    assert result.exit_code is None
    assert result.tests_executed is False
    assert result.execution_environment == "docker"
    assert "was cancelled" in result.error_message


def test_run_tests_returns_cancelled_for_preexisting_event(manager):
    event = threading.Event()
    event.set()

    manager.prepare = MagicMock()

    result = manager.run_tests(
        "/workspace",
        cancel_event=event,
    )

    assert result.status == "CANCELLED"
    manager.prepare.assert_not_called()


def test_run_tests_returns_execution_error_when_sandbox_raises(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=False,
        ),
    )

    manager.sandbox.run_tests = MagicMock(
        side_effect=RuntimeError("Execution crashed")
    )

    result = manager.run_tests("/workspace")

    assert result.status == "EXECUTION_ERROR"
    assert result.exit_code is None
    assert result.tests_executed is False
    assert "Execution crashed" in result.error_message


def test_run_tests_returns_execution_error_when_prepare_raises_unexpected(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        MagicMock(
            side_effect=RuntimeError("Preparation crashed")
        ),
    )

    result = manager.run_tests("/workspace")

    assert result.status == "EXECUTION_ERROR"
    assert "Preparation crashed" in result.error_message


def test_run_tests_preserves_sandbox_failure_result(
    manager,
    monkeypatch,
):
    failed_execution = DockerSandboxResult(
        status="FAILED",
        exit_code=1,
        duration_seconds=2.0,
        stdout="1 failed",
        stderr="AssertionError",
        execution_environment="docker",
        tests_executed=True,
        error_message="Tests failed.",
    )

    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=False,
        ),
    )

    manager.sandbox.run_tests = MagicMock(
        return_value=failed_execution
    )

    result = manager.run_tests("/workspace")

    assert result.status == "FAILED"
    assert result.exit_code == 1
    assert result.stdout == "1 failed"
    assert result.stderr == "AssertionError"
    assert result.tests_executed is True
    assert result.error_message == "Tests failed."


def test_run_tests_returns_docker_unavailable_from_sandbox(
    manager,
    monkeypatch,
):
    docker_failure = DockerSandboxResult(
        status="DOCKER_UNAVAILABLE",
        exit_code=None,
        duration_seconds=0.1,
        stdout="",
        stderr="",
        execution_environment="docker",
        tests_executed=False,
        error_message="Docker daemon unavailable.",
    )

    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=False,
        ),
    )

    manager.sandbox.run_tests = MagicMock(
        return_value=docker_failure
    )

    result = manager.run_tests("/workspace")

    assert result.status == "DOCKER_UNAVAILABLE"
    assert result.error_message == "Docker daemon unavailable."


def test_run_tests_preserves_sandbox_cancelled_result(
    manager,
    monkeypatch,
):
    cancelled_execution = DockerSandboxResult(
        status="CANCELLED",
        exit_code=None,
        duration_seconds=0.5,
        stdout="",
        stderr="",
        execution_environment="docker",
        tests_executed=False,
        error_message="Execution cancelled.",
    )

    monkeypatch.setattr(
        manager,
        "prepare",
        lambda **kwargs: SandboxPreparationResult(
            ready=True,
            image_name=manager.config.image,
            image_was_built=False,
        ),
    )

    manager.sandbox.run_tests = MagicMock(
        return_value=cancelled_execution
    )

    result = manager.run_tests("/workspace")

    assert result.status == "CANCELLED"
    assert result.error_message == "Execution cancelled."


def test_run_tests_uses_docker_environment_for_manager_errors(
    manager,
    monkeypatch,
):
    monkeypatch.setattr(
        manager,
        "prepare",
        MagicMock(
            side_effect=RuntimeError("Unexpected")
        ),
    )

    result = manager.run_tests("/workspace")

    assert result.execution_environment == "docker"


def test_prepare_build_failure_does_not_verify_image_again(
    manager,
    monkeypatch,
):
    failed_build = ImageBuildResult(
        success=False,
        image_name=manager.config.image,
        exit_code=1,
        stdout="",
        stderr="",
        error_message="Build failed.",
    )

    image_check = MagicMock(return_value=False)

    monkeypatch.setattr(
        manager.image_builder,
        "image_exists",
        image_check,
    )
    manager.image_builder.build = MagicMock(
        return_value=failed_build
    )

    result = manager.prepare()

    assert result.ready is False
    image_check.assert_called_once()
