
"""
AgentOps - Project Manager / Orchestrator

Coordinates Developer, QA, Security, and Architect Agents
using LangGraph.

Responsibilities:
- Delegate development tasks.
- Preserve previous implementations during corrections.
- Coordinate QA reviews and automated testing.
- Coordinate security reviews.
- Manage bounded correction attempts.
- Track which review caused each correction.
- Distinguish agent failures from review rejections.
- Run architecture analysis after QA and security reviews.
- Track observable workflow checkpoints.
- Publish live agent activity and progress.
- Preserve the latest workflow state during failures.
- Support task cancellation.
- Produce execution history and a final result.
"""

import json
import re
import uuid

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from agentops.agents.developer import DeveloperAgent
from agentops.agents.qa import QAAgent
from agentops.agents.security import SecurityAgent
from agentops.agents.architect import ArchitectAgent

from agentops.models.state import AgentState
from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)
from agentops.tools.file_manager import WorkspaceManager


class ProjectManager:
    """Coordinates the AgentOps multi-agent development workflow."""

    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
        max_retries: int = 3,
    ):
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.max_retries = max(0, int(max_retries))

        # All agents must use the same workspace manager.
        self.developer = DeveloperAgent(
            workspace_manager=self.workspace_manager
        )

        self.qa = QAAgent(
            workspace_manager=self.workspace_manager
        )

        self.security = SecurityAgent(
            workspace_manager=self.workspace_manager
        )

        self.architect = ArchitectAgent(
            workspace_manager=self.workspace_manager
        )

        self._progress_callback = None

        # Retain the most recently published workflow checkpoint.
        # This allows exception and cancellation handlers to preserve
        # completed work instead of reverting to the initial state.
        self._latest_state: Optional[AgentState] = None

        # Protect state snapshots if progress publication and
        # workflow handling occur from different threads.
        self._state_lock = RLock()

    # ---------------------------------------------------------
    # State retention
    # ---------------------------------------------------------

    def _save_latest_state(self, state: AgentState) -> None:
        """Store an independent snapshot of the latest workflow state."""

        snapshot = deepcopy(dict(state))

        with self._state_lock:
            self._latest_state = snapshot

    def _get_latest_state(
        self,
        fallback: Optional[AgentState] = None,
    ) -> AgentState:
        """Return the latest saved state or a supplied fallback."""

        with self._state_lock:
            if self._latest_state is not None:
                return deepcopy(self._latest_state)

        if fallback is not None:
            return deepcopy(dict(fallback))

        raise RuntimeError(
            "No workflow state is available."
        )

    # ---------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------

    def _create_initial_state(
        self,
        task: str,
        task_id: str,
        workspace_path: str = "",
    ) -> AgentState:
        """Create a complete initial workflow state."""

        return {
            "task_id": task_id,
            "original_task": task,
            "workspace_path": str(Path(workspace_path))
            if workspace_path
            else "",
            "implementation": "",
            "developer_feedback": "",
            "retry_source": "",
            "qa_review": "",
            "qa_decision": "NOT_RUN",
            "security_review": "",
            "security_decision": "NOT_RUN",
            "security_status": "NOT_RUN",
            "security_review_completed": False,
            "security_findings": [],
            "architecture_report": "",
            "architecture_status": "NOT_RUN",
            "current_agent": "orchestrator",
            "workflow_status": "RUNNING",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "progress_percentage": 0,
            "workflow_stage": "initialization",
            "completed_stages": [],
            "cancellation_requested": False,
            "execution_history": [],
            "final_result": None,
            "error_message": None,
        }

    # ---------------------------------------------------------
    # Workflow activity and progress
    # ---------------------------------------------------------

    def _record_activity(
        self,
        state: AgentState,
        agent: str,
        activity: str,
        details: str = "",
    ) -> list:
        """Append a timestamped activity to execution history."""

        history = list(state.get("execution_history", []))

        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": agent,
                "activity": activity,
                "details": details,
                "status": state.get("workflow_status", "RUNNING"),
                "workflow_stage": state.get("workflow_stage", ""),
                "progress_percentage": state.get(
                    "progress_percentage", 0
                ),
            }
        )

        return history

    def _calculate_progress(self, completed_stages: list) -> int:
        """Calculate progress from completed workflow milestones."""

        stage_weights = {
            "developer": 25,
            "qa": 20,
            "security": 20,
            "architect": 25,
            "finalize": 10,
        }

        completed = set(completed_stages)

        return min(
            100,
            sum(
                weight
                for stage, weight in stage_weights.items()
                if stage in completed
            ),
        )

    def _publish_progress(self, state: AgentState) -> None:
        """
        Save the latest state and publish it to the optional callback.

        State retention happens even when no callback is configured.
        Callback errors must never interrupt the workflow.
        """

        snapshot = deepcopy(dict(state))

        self._save_latest_state(snapshot)

        if self._progress_callback is None:
            return

        try:
            # Give the callback its own copy so it cannot mutate
            # the orchestrator's retained workflow state.
            self._progress_callback(deepcopy(snapshot))

        except Exception:
            # A progress callback must not interrupt the workflow.
            pass

    def _record_agent_start(
        self,
        state: AgentState,
        agent: str,
        stage: str,
        details: str = "",
    ) -> Dict[str, Any]:
        """Record and publish the start of an agent's work."""

        started_state = dict(state)

        started_state.update(
            {
                "current_agent": agent,
                "workflow_stage": stage,
                "workflow_status": "RUNNING",
            }
        )

        started_state["execution_history"] = self._record_activity(
            started_state,
            agent,
            "Agent started",
            details,
        )

        self._publish_progress(started_state)

        return started_state

    def _record_stage(
        self,
        state: AgentState,
        stage: str,
        agent: str,
        activity: str,
        details: str = "",
    ) -> Dict[str, Any]:
        """Record a completed stage and publish updated progress."""

        completed_stages = list(state.get("completed_stages", []))

        if stage not in completed_stages:
            completed_stages.append(stage)

        updated_state = dict(state)

        updated_state.update(
            {
                "workflow_stage": stage,
                "current_agent": agent,
                "completed_stages": completed_stages,
                "progress_percentage": self._calculate_progress(
                    completed_stages
                ),
            }
        )

        updated_state["execution_history"] = self._record_activity(
            updated_state,
            agent,
            activity,
            details,
        )

        self._publish_progress(updated_state)

        return {
            "workflow_stage": updated_state["workflow_stage"],
            "current_agent": updated_state["current_agent"],
            "completed_stages": updated_state["completed_stages"],
            "progress_percentage": updated_state["progress_percentage"],
            "execution_history": updated_state["execution_history"],
        }

    # ---------------------------------------------------------
    # Cancellation
    # ---------------------------------------------------------

    def _cancelled_result(
        self,
        state: AgentState,
        message: str = "Task cancellation was requested.",
    ) -> AgentState:
        """Build a consistent cancelled result from the latest state."""

        cancelled_state = deepcopy(dict(state))

        cancelled_state.update(
            {
                "workflow_status": "CANCELLED",
                "workflow_stage": "cancelled",
                "current_agent": "orchestrator",
                "cancellation_requested": True,
                "error_message": message,
                "final_result": (
                    "The task was cancelled before the workflow "
                    "could complete."
                ),
            }
        )

        cancelled_state["execution_history"] = self._record_activity(
            cancelled_state,
            "orchestrator",
            "Task cancelled",
            message,
        )

        self._publish_progress(cancelled_state)

        return cancelled_state

    # ---------------------------------------------------------
    # Developer
    # ---------------------------------------------------------

    def _developer_node(
        self,
        state: AgentState,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Run the Developer Agent."""

        check_cancelled(cancel_event)

        retry_count = state.get("retry_count", 0)

        working_state = self._record_agent_start(
            state,
            "developer",
            "developer",
            (
                f"Starting implementation attempt "
                f"{retry_count + 1}."
            ),
        )

        task_id = state["task_id"]
        original_task = state["original_task"]

        if retry_count > 0:
            previous_implementation = state.get(
                "implementation", ""
            )

            feedback = state.get("developer_feedback", "")

            task = f"""
You are correcting an existing implementation.

Original development request:
{original_task}

Previous implementation:
{previous_implementation}

Review feedback:
{feedback}

Correction requirements:
1. Preserve existing working functionality.
2. Address the specific review findings.
3. Do not unnecessarily rewrite unrelated files.
4. Return the complete updated file contents for every file
   that needs to be created or modified.
5. Do not claim that tests passed unless they were actually run.
"""
        else:
            task = original_task

        result = self.developer.execute(
            task_id=task_id,
            task=task,
            cancel_event=cancel_event,
        )

        if isinstance(result, dict):
            implementation = result.get(
                "implementation",
                result.get("report", ""),
            )
        else:
            implementation = str(result)

        updated_state = {
            "implementation": implementation,
            "developer_feedback": "",
            "error_message": None,
            "workflow_status": "RUNNING",
            "retry_source": "",
        }

        # Include the new implementation in the checkpoint itself.
        # This ensures cancellation immediately after Developer
        # completion does not lose the generated implementation.
        stage_state = {
            **working_state,
            **updated_state,
        }

        updated_state.update(
            self._record_stage(
                stage_state,
                "developer",
                "developer",
                "Implementation completed",
                (
                    f"Developer completed implementation attempt "
                    f"{retry_count + 1}."
                ),
            )
        )

        return updated_state

    # ---------------------------------------------------------
    # QA
    # ---------------------------------------------------------

    def _extract_qa_decision(self, result: Any) -> str:
        """Extract the QA decision from the agent result."""

        if isinstance(result, dict):
            decision = result.get("decision")

            if decision:
                return str(decision).upper()

            report = result.get(
                "report",
                result.get("summary", ""),
            )
        else:
            report = str(result)

        match = re.search(
            r"Decision\s*:\s*(PASS|FAIL|NEEDS_INFO|NEEDS_REVIEW)",
            str(report),
            re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

        return "NEEDS_REVIEW"

    def _qa_node(
        self,
        state: AgentState,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Run the QA Agent and record its review decision."""

        check_cancelled(cancel_event)

        working_state = self._record_agent_start(
            state,
            "qa",
            "qa",
            "Starting quality assurance review.",
        )

        result = self.qa.execute(
            state["original_task"],
            state["task_id"],
            state.get("implementation", ""),
            cancel_event,
        )

        if isinstance(result, dict):
            qa_report = result.get(
                "report",
                result.get("summary", ""),
            )
        else:
            qa_report = str(result)

        decision = self._extract_qa_decision(result)

        retry_source = "QA" if decision == "FAIL" else ""

        updated_state = {
            "qa_review": qa_report,
            "qa_decision": decision,
            "retry_source": retry_source,
        }

        updated_state.update(
            self._record_stage(
                {
                    **working_state,
                    **updated_state,
                },
                "qa",
                "qa",
                "QA review completed",
                f"QA decision: {decision}",
            )
        )

        return updated_state

    # ---------------------------------------------------------
    # Security
    # ---------------------------------------------------------

    def _security_node(
        self,
        state: AgentState,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Run the Security Agent and record its review."""

        check_cancelled(cancel_event)

        working_state = self._record_agent_start(
            state,
            "security",
            "security",
            "Starting security review.",
        )

        result = self.security.execute(
            task_id=state["task_id"],
            task=state["original_task"],
            cancel_event=cancel_event,
        )

        if isinstance(result, str):
            try:
                security_data = json.loads(result)
            except json.JSONDecodeError:
                security_data = {
                    "summary": result,
                    "decision": "NEEDS_REVIEW",
                    "status": "PARTIAL",
                    "findings": [],
                    "report": result,
                }

        elif isinstance(result, dict):
            security_data = result

        else:
            security_data = {
                "summary": str(result),
                "decision": "NEEDS_REVIEW",
                "status": "PARTIAL",
                "findings": [],
                "report": str(result),
            }

        decision = str(
            security_data.get(
                "decision",
                "NEEDS_REVIEW",
            )
        ).upper()

        security_status = str(
            security_data.get(
                "status",
                "PARTIAL",
            )
        ).upper()

        security_report = security_data.get(
            "report",
            security_data.get("summary", ""),
        )

        findings = security_data.get("findings", [])

        retry_source = (
            "Security" if decision == "FAIL" else ""
        )

        updated_state = {
            "security_review": str(security_report),
            "security_decision": decision,
            "security_status": security_status,
            "security_review_completed": (
                security_status == "COMPLETED"
            ),
            "security_findings": findings,
            "retry_source": retry_source,
        }

        updated_state.update(
            self._record_stage(
                {
                    **working_state,
                    **updated_state,
                },
                "security",
                "security",
                "Security review completed",
                (
                    f"Security decision: {decision}; "
                    f"status: {security_status}."
                ),
            )
        )

        return updated_state

    # ---------------------------------------------------------
    # Retry coordination
    # ---------------------------------------------------------

    def _should_retry(self, state: AgentState) -> str:
        """Determine whether the latest review requires correction."""

        retry_source = state.get("retry_source", "")

        latest_review_failed = retry_source in {
            "QA",
            "Security",
        }

        retries_remaining = (
            state.get("retry_count", 0)
            < state.get("max_retries", self.max_retries)
        )

        if latest_review_failed and retries_remaining:
            return "retry"

        return "continue"

    def _retry_node(self, state: AgentState) -> Dict[str, Any]:
        """Prepare feedback from the review that caused the retry."""

        retry_source = state.get("retry_source", "")

        if retry_source == "QA":
            feedback = state.get("qa_review", "")

        elif retry_source == "Security":
            feedback = state.get("security_review", "")

        else:
            raise RuntimeError(
                "Cannot determine which review caused the retry. "
                "Expected retry_source to be 'QA' or 'Security'."
            )

        retry_count = state.get("retry_count", 0) + 1

        updated_state = {
            "developer_feedback": feedback,
            "retry_count": retry_count,
            "retry_source": "",
            "current_agent": "developer",
            "workflow_stage": "retry",
            "workflow_status": "RUNNING",
        }

        merged_state = {
            **state,
            **updated_state,
        }

        updated_state["execution_history"] = (
            self._record_activity(
                merged_state,
                "orchestrator",
                "Correction requested",
                (
                    f"Retry {retry_count} of "
                    f"{state.get('max_retries', self.max_retries)}. "
                    f"Feedback source: {retry_source}."
                ),
            )
        )

        self._publish_progress(
            {
                **state,
                **updated_state,
            }
        )

        return updated_state

    # ---------------------------------------------------------
    # Architect
    # ---------------------------------------------------------

    def _architect_node(
        self,
        state: AgentState,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Run architecture analysis after the review stages."""

        check_cancelled(cancel_event)

        working_state = self._record_agent_start(
            state,
            "architect",
            "architect",
            "Starting architecture analysis.",
        )

        result = self.architect.execute(
            task_id=state["task_id"],
            task=state["original_task"],
            cancel_event=cancel_event,
        )

        if isinstance(result, dict):
            architecture_report = result.get(
                "report",
                result.get("architecture_report", ""),
            )
        else:
            architecture_report = str(result)

        updated_state = {
            "architecture_report": architecture_report,
            "architecture_status": "COMPLETED",
        }

        updated_state.update(
            self._record_stage(
                {
                    **working_state,
                    **updated_state,
                },
                "architect",
                "architect",
                "Architecture analysis completed",
                "Architecture report generated.",
            )
        )

        return updated_state

    # ---------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------

    def _finalize_node(
        self,
        state: AgentState,
        cancel_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Determine the final workflow status and result."""

        # Cancellation may arrive after Architect finishes but
        # before LangGraph begins the finalization node.
        check_cancelled(cancel_event)

        qa_decision = state.get(
            "qa_decision",
            "NEEDS_REVIEW",
        )

        security_decision = state.get(
            "security_decision",
            "NEEDS_REVIEW",
        )

        security_status = state.get(
            "security_status",
            "NOT_RUN",
        )

        if state.get("error_message"):
            workflow_status = "FAILED"

        elif state.get("cancellation_requested"):
            workflow_status = "CANCELLED"

        elif (
            qa_decision == "FAIL"
            or security_decision == "FAIL"
        ):
            workflow_status = "REJECTED"

        else:
            workflow_status = "NEEDS_REVIEW"

        if workflow_status == "REJECTED":
            final_result = (
                "The implementation was rejected because QA or "
                "Security identified blocking findings."
            )

        elif workflow_status == "CANCELLED":
            final_result = "The workflow was cancelled."

        elif workflow_status == "FAILED":
            final_result = (
                "The workflow encountered an execution error: "
                f"{state.get('error_message', 'Unknown error')}"
            )

        else:
            review_note = ""

            if (
                security_status != "COMPLETED"
                or not state.get(
                    "security_review_completed",
                    False,
                )
            ):
                review_note = (
                    " The security review was incomplete or partial."
                )

            final_result = (
                "The workflow completed its automated stages. "
                "Human review is required before approval."
                f"{review_note}"
            )

        completed_stages = list(
            state.get("completed_stages", [])
        )

        if "finalize" not in completed_stages:
            completed_stages.append("finalize")

        finalized_state = dict(state)

        finalized_state.update(
            {
                "workflow_status": workflow_status,
                "workflow_stage": "finalize",
                "current_agent": "orchestrator",
                "completed_stages": completed_stages,
                "progress_percentage": 100,
                "final_result": final_result,
            }
        )

        finalized_state["execution_history"] = (
            self._record_activity(
                finalized_state,
                "orchestrator",
                "Workflow finalized",
                f"Final status: {workflow_status}.",
            )
        )

        # Check again immediately before publishing the final state.
        # If cancellation arrived while finalization was being
        # assembled, preserve the previous checkpoint instead.
        check_cancelled(cancel_event)

        self._publish_progress(finalized_state)

        return {
            "workflow_status": finalized_state["workflow_status"],
            "workflow_stage": finalized_state["workflow_stage"],
            "current_agent": finalized_state["current_agent"],
            "completed_stages": finalized_state["completed_stages"],
            "progress_percentage": finalized_state["progress_percentage"],
            "final_result": finalized_state["final_result"],
            "execution_history": finalized_state["execution_history"],
        }

    # ---------------------------------------------------------
    # LangGraph construction
    # ---------------------------------------------------------

    def _build_graph(
        self,
        cancel_event: Optional[Event] = None,
    ):
        """Build and compile the LangGraph workflow."""

        graph = StateGraph(AgentState)

        graph.add_node(
            "developer",
            lambda state: self._developer_node(
                state,
                cancel_event,
            ),
        )

        graph.add_node(
            "qa",
            lambda state: self._qa_node(
                state,
                cancel_event,
            ),
        )

        graph.add_node(
            "security",
            lambda state: self._security_node(
                state,
                cancel_event,
            ),
        )

        graph.add_node("retry", self._retry_node)

        graph.add_node(
            "architect",
            lambda state: self._architect_node(
                state,
                cancel_event,
            ),
        )

        graph.add_node(
            "finalize",
            lambda state: self._finalize_node(
                state,
                cancel_event,
            ),
        )

        graph.add_edge(START, "developer")
        graph.add_edge("developer", "qa")

        graph.add_conditional_edges(
            "qa",
            self._should_retry,
            {
                "retry": "retry",
                "continue": "security",
            },
        )

        graph.add_conditional_edges(
            "security",
            self._should_retry,
            {
                "retry": "retry",
                "continue": "architect",
            },
        )

        graph.add_edge("retry", "developer")
        graph.add_edge("architect", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    # ---------------------------------------------------------
    # Public workflow interface
    # ---------------------------------------------------------

    def run(
        self,
        task: str,
        task_id: Optional[str] = None,
        cancel_event: Optional[Event] = None,
        progress_callback=None,
    ) -> AgentState:
        """Execute the complete development workflow."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "A non-empty development task is required."
            )

        task_id = task_id or str(uuid.uuid4())

        try:
            task_id = str(uuid.UUID(task_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "task_id must be a valid UUID."
            ) from exc

        self._progress_callback = progress_callback

        initial_state = self._create_initial_state(
            task=task,
            task_id=task_id,
        )

        # Start each run with a clean retained state.
        self._save_latest_state(initial_state)

        try:
            if cancel_event is not None and cancel_event.is_set():
                return self._cancelled_result(
                    self._get_latest_state(initial_state)
                )

            workspace_path = self.workspace_manager.create_workspace(
                task_id
            )

            initial_state = self._create_initial_state(
                task=task,
                task_id=task_id,
                workspace_path=str(workspace_path),
            )

            initial_state["execution_history"] = (
                self._record_activity(
                    initial_state,
                    "orchestrator",
                    "Workflow started",
                    "Development task accepted.",
                )
            )

            self._publish_progress(initial_state)

            workflow = self._build_graph(
                cancel_event=cancel_event
            )

            final_state = workflow.invoke(initial_state)

            # Preserve the final graph state as the authoritative
            # snapshot, including any fields not sent by callbacks.
            self._save_latest_state(final_state)

            return final_state

        except TaskCancelledError as exc:
            latest_state = self._get_latest_state(initial_state)

            return self._cancelled_result(
                latest_state,
                str(exc),
            )

        except Exception as exc:
            # Do not rebuild from the original initialization state.
            # Preserve all checkpoints completed before the failure.
            error_state = self._get_latest_state(initial_state)

            error_state.update(
                {
                    "current_agent": "orchestrator",
                    "workflow_status": "FAILED",
                    "workflow_stage": "failed",
                    "cancellation_requested": False,
                    "error_message": str(exc),
                    "final_result": (
                        "The workflow failed after preserving its "
                        f"latest completed state: {exc}"
                    ),
                }
            )

            error_state["execution_history"] = (
                self._record_activity(
                    error_state,
                    "orchestrator",
                    "Workflow failed",
                    str(exc),
                )
            )

            self._publish_progress(error_state)

            return error_state

        finally:
            self._progress_callback = None
