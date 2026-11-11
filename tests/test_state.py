
from typing import Any, Dict, List, Optional, get_type_hints

from agentops.models.state import AgentState


EXPECTED_FIELDS = {
    "task_id",
    "original_task",
    "workspace_path",
    "implementation",
    "developer_feedback",
    "retry_source",
    "qa_review",
    "qa_decision",
    "security_review",
    "security_decision",
    "security_status",
    "security_review_completed",
    "security_findings",
    "architecture_report",
    "architecture_status",
    "current_agent",
    "workflow_status",
    "retry_count",
    "max_retries",
    "progress_percentage",
    "workflow_stage",
    "completed_stages",
    "cancellation_requested",
    "execution_history",
    "final_result",
    "error_message",
}


def make_agent_state() -> AgentState:
    return {
        "task_id": "12345678-1234-1234-1234-123456789012",
        "original_task": "Build a REST API",
        "workspace_path": "/tmp/agentops/workspaces/task",
        "implementation": "print('Hello World')",
        "developer_feedback": "",
        "retry_source": "",
        "qa_review": "",
        "qa_decision": "PENDING",
        "security_review": "",
        "security_decision": "PENDING",
        "security_status": "PENDING",
        "security_review_completed": False,
        "security_findings": [],
        "architecture_report": "",
        "architecture_status": "PENDING",
        "current_agent": "developer",
        "workflow_status": "RUNNING",
        "retry_count": 0,
        "max_retries": 2,
        "progress_percentage": 0,
        "workflow_stage": "DEVELOPMENT",
        "completed_stages": [],
        "cancellation_requested": False,
        "execution_history": [],
        "final_result": None,
        "error_message": None,
    }


def test_agent_state_is_a_typed_dict():
    assert AgentState.__class__.__name__ == "_TypedDictMeta"


def test_agent_state_is_total():
    assert AgentState.__total__ is True


def test_agent_state_contains_all_expected_fields():
    assert set(AgentState.__annotations__) == EXPECTED_FIELDS


def test_agent_state_has_no_unexpected_fields():
    assert set(AgentState.__annotations__) - EXPECTED_FIELDS == set()


def test_all_agent_state_fields_are_required():
    assert AgentState.__required_keys__ == EXPECTED_FIELDS


def test_agent_state_has_no_optional_keys():
    assert AgentState.__optional_keys__ == set()


def test_agent_state_can_be_created_as_a_dictionary():
    state = make_agent_state()

    assert isinstance(state, dict)
    assert state["task_id"] == "12345678-1234-1234-1234-123456789012"
    assert state["workflow_status"] == "RUNNING"


def test_agent_state_contains_task_identification_fields():
    state = make_agent_state()

    assert state["task_id"]
    assert state["original_task"]
    assert state["workspace_path"]


def test_agent_state_contains_developer_fields():
    state = make_agent_state()

    assert state["implementation"] == "print('Hello World')"
    assert state["developer_feedback"] == ""
    assert state["retry_source"] == ""


def test_agent_state_contains_qa_fields():
    state = make_agent_state()

    assert state["qa_review"] == ""
    assert state["qa_decision"] == "PENDING"


def test_agent_state_contains_security_fields():
    state = make_agent_state()

    assert state["security_review"] == ""
    assert state["security_decision"] == "PENDING"
    assert state["security_status"] == "PENDING"
    assert state["security_review_completed"] is False
    assert state["security_findings"] == []


def test_agent_state_contains_architect_fields():
    state = make_agent_state()

    assert state["architecture_report"] == ""
    assert state["architecture_status"] == "PENDING"


def test_agent_state_contains_workflow_tracking_fields():
    state = make_agent_state()

    assert state["current_agent"] == "developer"
    assert state["workflow_status"] == "RUNNING"
    assert state["retry_count"] == 0
    assert state["max_retries"] == 2


def test_agent_state_contains_progress_fields():
    state = make_agent_state()

    assert state["progress_percentage"] == 0
    assert state["workflow_stage"] == "DEVELOPMENT"
    assert state["completed_stages"] == []


def test_agent_state_contains_cancellation_field():
    state = make_agent_state()

    assert state["cancellation_requested"] is False


def test_agent_state_contains_execution_history():
    state = make_agent_state()

    assert state["execution_history"] == []


def test_final_result_can_be_none():
    state = make_agent_state()

    assert state["final_result"] is None


def test_error_message_can_be_none():
    state = make_agent_state()

    assert state["error_message"] is None


def test_final_result_can_hold_a_string():
    state = make_agent_state()
    state["final_result"] = "Task completed successfully"

    assert state["final_result"] == "Task completed successfully"


def test_error_message_can_hold_a_string():
    state = make_agent_state()
    state["error_message"] = "Developer execution failed"

    assert state["error_message"] == "Developer execution failed"


def test_security_findings_can_hold_structured_records():
    state = make_agent_state()

    state["security_findings"] = [
        {
            "severity": "HIGH",
            "title": "Hardcoded credential",
            "file": "config.py",
            "line": 12,
        }
    ]

    finding = state["security_findings"][0]

    assert finding["severity"] == "HIGH"
    assert finding["title"] == "Hardcoded credential"
    assert finding["file"] == "config.py"
    assert finding["line"] == 12


def test_execution_history_can_hold_structured_records():
    state = make_agent_state()

    state["execution_history"] = [
        {
            "agent": "qa",
            "status": "FAILED",
            "retry_count": 0,
        },
        {
            "agent": "qa",
            "status": "PASSED",
            "retry_count": 1,
        },
    ]

    assert len(state["execution_history"]) == 2
    assert state["execution_history"][0]["status"] == "FAILED"
    assert state["execution_history"][1]["status"] == "PASSED"


def test_completed_stages_can_track_multiple_stages():
    state = make_agent_state()

    state["completed_stages"] = [
        "DEVELOPMENT",
        "QA",
        "SECURITY",
    ]

    assert state["completed_stages"] == [
        "DEVELOPMENT",
        "QA",
        "SECURITY",
    ]


def test_workflow_progress_can_be_updated():
    state = make_agent_state()

    state["progress_percentage"] = 75
    state["workflow_stage"] = "ARCHITECTURE"

    assert state["progress_percentage"] == 75
    assert state["workflow_stage"] == "ARCHITECTURE"


def test_retry_tracking_can_be_updated():
    state = make_agent_state()

    state["retry_count"] = 1
    state["retry_source"] = "QA"

    assert state["retry_count"] == 1
    assert state["retry_source"] == "QA"


def test_cancellation_can_be_recorded():
    state = make_agent_state()

    state["cancellation_requested"] = True

    assert state["cancellation_requested"] is True


def test_agent_state_type_annotations_are_correct():
    annotations = get_type_hints(AgentState)

    assert annotations["task_id"] is str
    assert annotations["original_task"] is str
    assert annotations["workspace_path"] is str

    assert annotations["implementation"] is str
    assert annotations["developer_feedback"] is str
    assert annotations["retry_source"] is str

    assert annotations["qa_review"] is str
    assert annotations["qa_decision"] is str

    assert annotations["security_review"] is str
    assert annotations["security_decision"] is str
    assert annotations["security_status"] is str
    assert annotations["security_review_completed"] is bool
    assert annotations["security_findings"] == List[Dict[str, Any]]

    assert annotations["architecture_report"] is str
    assert annotations["architecture_status"] is str

    assert annotations["current_agent"] is str
    assert annotations["workflow_status"] is str
    assert annotations["retry_count"] is int
    assert annotations["max_retries"] is int

    assert annotations["progress_percentage"] is int
    assert annotations["workflow_stage"] is str
    assert annotations["completed_stages"] == List[str]

    assert annotations["cancellation_requested"] is bool
    assert annotations["execution_history"] == List[Dict[str, Any]]

    assert annotations["final_result"] == Optional[str]
    assert annotations["error_message"] == Optional[str]


def test_agent_state_can_represent_completed_workflow():
    state = make_agent_state()

    state.update(
        {
            "current_agent": "architect",
            "workflow_status": "COMPLETED",
            "progress_percentage": 100,
            "workflow_stage": "COMPLETED",
            "completed_stages": [
                "DEVELOPMENT",
                "QA",
                "SECURITY",
                "ARCHITECTURE",
            ],
            "final_result": "Implementation completed successfully",
        }
    )

    assert state["workflow_status"] == "COMPLETED"
    assert state["progress_percentage"] == 100
    assert state["current_agent"] == "architect"
    assert state["final_result"] is not None


def test_agent_state_can_represent_cancelled_workflow():
    state = make_agent_state()

    state.update(
        {
            "workflow_status": "CANCELLED",
            "cancellation_requested": True,
            "error_message": "Task cancellation was requested.",
        }
    )

    assert state["workflow_status"] == "CANCELLED"
    assert state["cancellation_requested"] is True
    assert state["error_message"] == "Task cancellation was requested."


def test_agent_state_can_represent_failed_workflow():
    state = make_agent_state()

    state.update(
        {
            "workflow_status": "FAILED",
            "error_message": "Unexpected workflow error",
        }
    )

    assert state["workflow_status"] == "FAILED"
    assert state["error_message"] == "Unexpected workflow error"
