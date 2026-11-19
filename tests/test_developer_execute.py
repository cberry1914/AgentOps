
import json
from unittest.mock import Mock

import pytest

from agentops.agents.developer import DeveloperAgent
from agentops.tools.file_manager import WorkspaceError


def make_agent(workspace_manager=None):
    """Create a DeveloperAgent without initializing a real model."""
    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.workspace_manager = (
        workspace_manager
        if workspace_manager is not None
        else object()
    )
    agent.system_prompt = "Test developer system prompt."
    return agent


def configure_successful_execution(agent, monkeypatch):
    workspace_files = [
        {
            "path": "existing.py",
            "content": "def existing_function(): pass",
        }
    ]
    existing_directories = {"src"}
    limitations = ["Some workspace files could not be inspected."]

    collect_mock = Mock(
        return_value=(
            workspace_files,
            existing_directories,
            limitations,
        )
    )

    response = json.dumps(
        {
            "summary": "Created a new Python module.",
            "files": [
                {
                    "path": "src/new_module.py",
                    "content": "def new_function():\n    return True\n",
                }
            ],
        }
    )

    stream_mock = Mock(return_value=response)
    apply_mock = Mock(return_value=["src/new_module.py"])

    monkeypatch.setattr(
        agent,
        "_collect_workspace_files",
        collect_mock,
    )
    monkeypatch.setattr(
        agent,
        "_stream_response",
        stream_mock,
    )
    monkeypatch.setattr(
        agent,
        "_apply_file_changes",
        apply_mock,
    )

    return collect_mock, stream_mock, apply_mock


def test_execute_rejects_empty_task():
    agent = make_agent()

    with pytest.raises(
        ValueError,
        match="development task cannot be empty",
    ):
        agent.execute("   ", task_id="task-123")


def test_execute_requires_workspace_manager():
    agent = make_agent(workspace_manager=None)
    agent.workspace_manager = None

    with pytest.raises(
        WorkspaceError,
        match="requires a WorkspaceManager",
    ):
        agent.execute(
            "Create a Python module.",
            task_id="task-123",
        )


def test_execute_requires_task_id():
    agent = make_agent()

    with pytest.raises(
        ValueError,
        match="A task ID is required",
    ):
        agent.execute("Create a Python module.")


def test_execute_completes_full_execution_path(
    monkeypatch,
):
    agent = make_agent()

    collect_mock, stream_mock, apply_mock = (
        configure_successful_execution(
            agent,
            monkeypatch,
        )
    )

    report = agent.execute(
        "Create a Python module.",
        task_id="task-123",
    )

    collect_mock.assert_called_once_with(
        "task-123",
        None,
    )

    stream_mock.assert_called_once()

    messages = stream_mock.call_args.args[0]

    assert len(messages) == 2
    assert messages[0].content == agent.system_prompt
    assert "Create a Python module." in messages[1].content
    assert "existing.py" in messages[1].content

    apply_mock.assert_called_once_with(
        task_id="task-123",
        proposed_files=[
            {
                "path": "src/new_module.py",
                "content": "def new_function():\n    return True\n",
            }
        ],
        existing_directories={"src"},
        cancel_event=None,
    )

    assert "TASK UNDERSTANDING:" in report
    assert "Create a Python module." in report
    assert "Created a new Python module." in report
    assert "- src/new_module.py" in report
    assert "WORKSPACE LIMITATIONS:" in report
    assert "Some workspace files could not be inspected." in report
    assert "TECHNICAL DECISIONS:" in report
    assert "No code execution or automated testing was performed." in report


def test_execute_does_not_apply_files_when_json_is_invalid(
    monkeypatch,
):
    agent = make_agent()

    collect_mock = Mock(
        return_value=([], set(), [])
    )
    stream_mock = Mock(
        return_value="This is not valid JSON."
    )
    apply_mock = Mock()

    monkeypatch.setattr(
        agent,
        "_collect_workspace_files",
        collect_mock,
    )
    monkeypatch.setattr(
        agent,
        "_stream_response",
        stream_mock,
    )
    monkeypatch.setattr(
        agent,
        "_apply_file_changes",
        apply_mock,
    )

    with pytest.raises(
        ValueError,
        match="did not return a JSON object",
    ):
        agent.execute(
            "Create a Python module.",
            task_id="task-123",
        )

    collect_mock.assert_called_once()
    stream_mock.assert_called_once()
    apply_mock.assert_not_called()


def test_execute_reports_when_no_files_are_changed(
    monkeypatch,
):
    agent = make_agent()

    collect_mock = Mock(
        return_value=([], set(), [])
    )
    stream_mock = Mock(
        return_value=json.dumps(
            {
                "summary": "No changes were necessary.",
                "files": [],
            }
        )
    )
    apply_mock = Mock(return_value=[])

    monkeypatch.setattr(
        agent,
        "_collect_workspace_files",
        collect_mock,
    )
    monkeypatch.setattr(
        agent,
        "_stream_response",
        stream_mock,
    )
    monkeypatch.setattr(
        agent,
        "_apply_file_changes",
        apply_mock,
    )

    report = agent.execute(
        "Review the existing implementation.",
        task_id="task-123",
    )

    assert "No changes were necessary." in report
    assert "No files were created or modified." in report
    apply_mock.assert_called_once()
