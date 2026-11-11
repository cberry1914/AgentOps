
"""
AgentOps - Test Discovery Tests

Tests pytest discovery, configuration detection,
filesystem boundaries, cancellation, limits, and
structured result serialization.
"""

from pathlib import Path
from threading import Event

import pytest

from agentops.tools.test_discovery import (
    TestDiscovery,
    TestDiscoveryResult as DiscoveryResult,
)


@pytest.fixture
def discovery():
    return TestDiscovery()


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


def create_file(root, relative_path, content=""):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# Result model


def test_result_defaults():
    result = DiscoveryResult()

    assert result.supported_frameworks == []
    assert result.test_files == []
    assert result.configuration_files == []
    assert result.test_directories == []
    assert result.discovered_test_count == 0
    assert result.discovery_complete is True
    assert result.warnings == []
    assert result.error_message is None
    assert result.tests_found is False


def test_result_tests_found_property():
    result = DiscoveryResult(discovered_test_count=2)

    assert result.tests_found is True


def test_result_to_dict():
    result = DiscoveryResult(
        supported_frameworks=["pytest"],
        test_files=["tests/test_example.py"],
        configuration_files=["pytest.ini"],
        test_directories=["tests"],
        discovered_test_count=1,
        discovery_complete=True,
        warnings=["Example warning"],
        error_message=None,
    )

    assert result.to_dict() == {
        "supported_frameworks": ["pytest"],
        "test_files": ["tests/test_example.py"],
        "configuration_files": ["pytest.ini"],
        "test_directories": ["tests"],
        "discovered_test_count": 1,
        "discovery_complete": True,
        "warnings": ["Example warning"],
        "error_message": None,
    }


def test_result_to_dict_returns_serializable_values():
    result = DiscoveryResult(
        supported_frameworks=["pytest"],
        test_files=["tests/test_example.py"],
        discovered_test_count=1,
    )

    data = result.to_dict()

    assert isinstance(data, dict)

    for value in data.values():
        assert isinstance(
            value,
            (list, int, bool, type(None), str),
        )


# Workspace validation


def test_validate_workspace_accepts_valid_directory(
    discovery,
    workspace,
):
    resolved = discovery._validate_workspace(workspace)

    assert resolved == workspace.resolve()
    assert resolved.is_dir()


def test_discover_rejects_missing_workspace(discovery, tmp_path):
    result = discovery.discover(tmp_path / "missing")

    assert result.discovery_complete is False
    assert result.error_message == "Workspace does not exist."
    assert result.test_files == []


def test_discover_rejects_file_as_workspace(
    discovery,
    tmp_path,
):
    file_path = tmp_path / "project.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    result = discovery.discover(file_path)

    assert result.discovery_complete is False
    assert result.error_message == "Workspace must be a directory."


def test_discover_rejects_none_workspace(discovery):
    result = discovery.discover(None)

    assert result.discovery_complete is False
    assert result.error_message == "Workspace path is required."


def test_validate_workspace_rejects_symlink(
    discovery,
    tmp_path,
):
    actual_directory = tmp_path / "actual"
    actual_directory.mkdir()

    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(
        actual_directory,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        discovery._validate_workspace(linked_directory)


# Test filename identification


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("test_example.py", True),
        ("test_api.py", True),
        ("example_test.py", True),
        ("API_TEST.PY", False),
        ("Test_example.py", False),
        ("example.py", False),
        ("tests.py", False),
        ("test_example.txt", False),
        ("example_test.txt", False),
        ("conftest.py", False),
        ("pytest.ini", False),
    ],
)
def test_is_test_file(
    discovery,
    tmp_path,
    filename,
    expected,
):
    path = tmp_path / filename

    assert discovery._is_test_file(path) is expected


# Basic discovery


def test_discover_finds_test_files(discovery, workspace):
    create_file(workspace, "tests/test_users.py")
    create_file(workspace, "tests/test_orders.py")
    create_file(workspace, "src/api_test.py")

    result = discovery.discover(workspace)

    assert result.test_files == [
        "src/api_test.py",
        "tests/test_orders.py",
        "tests/test_users.py",
    ]
    assert result.discovered_test_count == 3
    assert result.tests_found is True
    assert result.supported_frameworks == ["pytest"]
    assert result.discovery_complete is True


def test_discover_finds_test_configuration_files(
    discovery,
    workspace,
):
    for filename in (
        "pytest.ini",
        "tox.ini",
        "pyproject.toml",
        "setup.cfg",
    ):
        create_file(workspace, filename)

    result = discovery.discover(workspace)

    assert result.configuration_files == [
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
    ]


def test_discover_finds_nested_configuration_files(
    discovery,
    workspace,
):
    create_file(workspace, "project/pytest.ini")
    create_file(workspace, "project/tests/test_example.py")

    result = discovery.discover(workspace)

    assert result.configuration_files == ["project/pytest.ini"]


def test_discover_finds_conventional_test_directories(
    discovery,
    workspace,
):
    create_file(workspace, "tests/unit/test_unit.py")
    create_file(workspace, "application/tests/test_app.py")
    create_file(workspace, "testing/test_other.py")

    result = discovery.discover(workspace)

    assert result.test_directories == [
        "application/tests",
        "tests",
    ]


def test_discover_returns_sorted_results(
    discovery,
    workspace,
):
    create_file(workspace, "z_tests/test_z.py")
    create_file(workspace, "a_tests/test_a.py")
    create_file(workspace, "pytest.ini")
    create_file(workspace, "a/pytest.ini")

    result = discovery.discover(workspace)

    assert result.test_files == sorted(result.test_files)
    assert result.configuration_files == sorted(
        result.configuration_files
    )
    assert result.test_directories == sorted(
        result.test_directories
    )


def test_discover_returns_no_tests_warning(
    discovery,
    workspace,
):
    create_file(workspace, "src/application.py")

    result = discovery.discover(workspace)

    assert result.test_files == []
    assert result.discovered_test_count == 0
    assert result.supported_frameworks == []
    assert result.discovery_complete is True
    assert any(
        "No conventional Python pytest test files" in warning
        for warning in result.warnings
    )


def test_discover_empty_workspace(discovery, workspace):
    result = discovery.discover(workspace)

    assert result.test_files == []
    assert result.discovered_test_count == 0
    assert result.discovery_complete is True


# Excluded directories


@pytest.mark.parametrize(
    "directory_name",
    sorted(TestDiscovery.EXCLUDED_DIRECTORIES),
)
def test_discover_skips_excluded_directories(
    discovery,
    workspace,
    directory_name,
):
    create_file(
        workspace,
        f"{directory_name}/tests/test_hidden.py",
    )
    create_file(workspace, "tests/test_visible.py")

    result = discovery.discover(workspace)

    assert result.test_files == ["tests/test_visible.py"]
    assert result.discovery_complete is True


# Symbolic links


def test_discover_does_not_follow_directory_symlinks(
    discovery,
    workspace,
    tmp_path,
):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()

    try:
        create_file(outside, "tests/test_external.py")

        linked_directory = workspace / "external"
        linked_directory.symlink_to(
            outside,
            target_is_directory=True,
        )

        create_file(workspace, "tests/test_local.py")

        result = discovery.discover(workspace)

        assert result.test_files == ["tests/test_local.py"]

    finally:
        import shutil
        shutil.rmtree(outside, ignore_errors=True)


def test_discover_ignores_file_symlinks(
    discovery,
    workspace,
    tmp_path,
):
    outside_file = tmp_path.parent / f"{tmp_path.name}_external.py"

    try:
        outside_file.write_text(
            "def test_external(): pass",
            encoding="utf-8",
        )

        linked_file = workspace / "test_link.py"
        linked_file.symlink_to(outside_file)

        result = discovery.discover(workspace)

        assert result.test_files == []

    finally:
        outside_file.unlink(missing_ok=True)


# Cancellation


def test_discover_handles_cancellation_before_scan(
    discovery,
    workspace,
):
    cancel_event = Event()
    cancel_event.set()

    result = discovery.discover(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.discovery_complete is False
    assert result.test_files == []
    assert any(
        "cancelled" in warning.lower()
        for warning in result.warnings
    )


def test_discover_handles_cancellation_during_scan(
    discovery,
    workspace,
    monkeypatch,
):
    create_file(workspace, "tests/test_example.py")

    cancel_event = Event()
    original_iterdir = Path.iterdir

    def cancel_when_tests_directory_is_scanned(path):
        if path == workspace / "tests":
            cancel_event.set()

        return original_iterdir(path)

    monkeypatch.setattr(
        Path,
        "iterdir",
        cancel_when_tests_directory_is_scanned,
    )

    result = discovery.discover(
        workspace,
        cancel_event=cancel_event,
    )

    assert result.discovery_complete is False
    assert any(
        "cancelled" in warning.lower()
        for warning in result.warnings
    )


def test_discover_does_not_require_cancellation_event(
    discovery,
    workspace,
):
    create_file(workspace, "tests/test_example.py")

    result = discovery.discover(workspace)

    assert result.discovery_complete is True
    assert result.discovered_test_count == 1


# Directory depth


def test_discover_stops_at_maximum_directory_depth(
    discovery,
    workspace,
):
    allowed_path = workspace

    for index in range(TestDiscovery.MAX_DEPTH):
        allowed_path = allowed_path / f"level_{index}"

    create_file(allowed_path, "test_allowed.py")

    result = discovery.discover(workspace)

    expected_path = str(
        Path(
            *[
                f"level_{index}"
                for index in range(TestDiscovery.MAX_DEPTH)
            ],
            "test_allowed.py",
        )
    )

    assert expected_path in result.test_files
    assert result.discovery_complete is True


def test_discover_marks_incomplete_beyond_maximum_depth(
    discovery,
    workspace,
):
    deep_path = workspace

    for index in range(TestDiscovery.MAX_DEPTH + 1):
        deep_path = deep_path / f"level_{index}"

    create_file(deep_path, "test_too_deep.py")

    result = discovery.discover(workspace)

    assert result.discovery_complete is False
    assert any(
        "Maximum directory depth reached" in warning
        for warning in result.warnings
    )


# File count limit


def test_discover_stops_after_maximum_file_count(
    discovery,
    workspace,
):
    for index in range(TestDiscovery.MAX_FILES + 1):
        create_file(workspace, f"file_{index:04d}.txt")

    result = discovery.discover(workspace)

    assert result.discovery_complete is False
    assert any(
        "Discovery stopped after" in warning
        for warning in result.warnings
    )


def test_discover_file_limit_marks_partial_test_results(
    discovery,
    workspace,
):
    create_file(workspace, "test_0000.py")

    for index in range(TestDiscovery.MAX_FILES):
        create_file(workspace, f"file_{index:04d}.txt")

    result = discovery.discover(workspace)

    assert result.discovery_complete is False
    assert result.discovered_test_count <= 1


# Filesystem errors


def test_discover_handles_directory_iteration_error(
    discovery,
    workspace,
    monkeypatch,
):
    original_iterdir = Path.iterdir

    def failing_iterdir(path):
        if path == workspace:
            raise PermissionError("Permission denied")

        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)

    result = discovery.discover(workspace)

    assert result.discovery_complete is False
    assert any(
        "Unable to inspect directory" in warning
        for warning in result.warnings
    )


def test_discover_continues_after_nested_directory_error(
    discovery,
    workspace,
    monkeypatch,
):
    create_file(workspace, "tests/test_visible.py")

    inaccessible = workspace / "restricted"
    inaccessible.mkdir()

    original_iterdir = Path.iterdir

    def failing_iterdir(path):
        if path == inaccessible:
            raise PermissionError("Permission denied")

        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)

    result = discovery.discover(workspace)

    assert result.discovery_complete is False
    assert result.test_files == ["tests/test_visible.py"]
    assert any(
        "Unable to inspect directory" in warning
        for warning in result.warnings
    )


def test_discover_handles_symlink_creation_failure(
    discovery,
    tmp_path,
):
    actual = tmp_path / "actual"
    actual.mkdir()

    linked = tmp_path / "linked"

    try:
        linked.symlink_to(
            actual,
            target_is_directory=True,
        )
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links are unavailable.")

    result = discovery.discover(linked)

    assert result.discovery_complete is False
    assert "symbolic link" in result.error_message.lower()
