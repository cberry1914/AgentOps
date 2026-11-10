
"""
AgentOps - Workflow State

Defines the shared state passed between agents
and workflow nodes.

The state tracks:
- Original development request.
- Workspace information.
- Developer implementation.
- QA review results.
- Security review results.
- Architecture documentation.
- Workflow progress and execution history.
- Correction attempt source.
- Cancellation and error information.

Agents communicate through this shared state.
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    """Shared state for the AgentOps multi-agent workflow."""

    # Task identification
    task_id: str
    original_task: str
    workspace_path: str

    # Developer Agent
    implementation: str
    developer_feedback: str
    retry_source: str

    # QA Agent
    qa_review: str
    qa_decision: str

    # Security Agent
    security_review: str
    security_decision: str
    security_status: str
    security_review_completed: bool
    security_findings: List[Dict[str, Any]]

    # Architect Agent
    architecture_report: str
    architecture_status: str

    # Workflow tracking
    current_agent: str
    workflow_status: str
    retry_count: int
    max_retries: int

    # Observable progress
    progress_percentage: int
    workflow_stage: str
    completed_stages: List[str]

    # Cancellation
    cancellation_requested: bool

    # Activity and execution history
    execution_history: List[Dict[str, Any]]

    # Final outcome
    final_result: Optional[str]
    error_message: Optional[str]
