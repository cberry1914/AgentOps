
from unittest.mock import Mock

import pytest

from agentops.agents.developer import DeveloperAgent
from agentops.tools.file_manager import WorkspaceError


@pytest.fixture
def agent():
    developer = DeveloperAgent.__new__(DeveloperAgent)
    developer.workspace_manager = Mock()
    return developer


def test_apply_file_changes_writes_file(agent):
    result = agent._apply_file_changes(
        task_id="test-task",
        proposed_files=[
            {
                "path": "app.py",
                "content": "print('Hello AgentOps')",
            }
        ],
        existing_directories=set(),
    )

    assert result == ["app.py"]

    agent.workspace_manager.write_file.assert_called_once_with(
        task_id="test-task",
        relative_path="app.py",
        content="print('Hello AgentOps')",
        agent="developer",
    )


def test_apply_file_changes_creates_parent_directories(agent):
    directories = set()

    result = agent._apply_file_changes(
        task_id="test-task",
        proposed_files=[
            {
                "path": "src/app/main.py",
                "content": "print('Hello')",
            }
        ],
        existing_directories=directories,
    )

    assert result == ["src/app/main.py"]
    assert directories == {"src", "src/app"}
    assert agent.workspace_manager.create_directory.call_count == 2

    agent.workspace_manager.write_file.assert_called_once_with(
        task_id="test-task",
        relative_path="src/app/main.py",
        content="print('Hello')",
        agent="developer",
    )


def test_apply_file_changes_does_not_recreate_existing_directories(agent):
    directories = {"src", "src/app"}

    agent._apply_file_changes(
        task_id="test-task",
        proposed_files=[
            {
                "path": "src/app/main.py",
                "content": "print('Hello')",
            }
        ],
        existing_directories=directories,
    )

    agent.workspace_manager.create_directory.assert_not_called()


def test_apply_file_changes_rejects_duplicate_paths_before_writing(agent):
    proposed_files = [
        {
            "path": "app.py",
            "content": "print('First')",
        },
        {
            "path": "app.py",
            "content": "print('Second')",
        },
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate file operation",
    ):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=proposed_files,
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()


def test_apply_file_changes_rejects_unsafe_path_before_writing(agent):
    proposed_files = [
        {
            "path": "safe.py",
            "content": "print('Safe')",
        },
        {
            "path": "../outside.py",
            "content": "print('Unsafe')",
        },
    ]

    with pytest.raises(WorkspaceError):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=proposed_files,
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()


def test_apply_file_changes_rejects_non_string_content(agent):
    with pytest.raises(
        ValueError,
        match="File content must be a string",
    ):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=[
                {
                    "path": "app.py",
                    "content": None,
                }
            ],
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()


def test_apply_file_changes_rejects_too_many_files(agent):
    proposed_files = [
        {
            "path": f"file_{index}.py",
            "content": "print('test')",
        }
        for index in range(agent.MAX_FILE_CHANGES + 1)
    ]

    with pytest.raises(
        ValueError,
        match="maximum allowed",
    ):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=proposed_files,
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()


def test_apply_file_changes_rejects_oversized_file(agent):
    oversized_content = "x" * (agent.MAX_FILE_BYTES + 1)

    with pytest.raises(
        ValueError,
        match="maximum allowed size",
    ):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=[
                {
                    "path": "large.py",
                    "content": oversized_content,
                }
            ],
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()


def test_apply_file_changes_rejects_non_object_file_operation(agent):
    with pytest.raises(
        ValueError,
        match="Each file operation must be an object",
    ):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=["app.py"],
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()


def test_apply_file_changes_rejects_missing_path(agent):
    with pytest.raises(
        ValueError,
        match="valid relative path",
    ):
        agent._apply_file_changes(
            task_id="test-task",
            proposed_files=[
                {
                    "content": "print('Hello')",
                }
            ],
            existing_directories=set(),
        )

    agent.workspace_manager.write_file.assert_not_called()
