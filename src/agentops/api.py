
"""
AgentOps - FastAPI Application

Provides HTTP endpoints for submitting, monitoring,
and cancelling multi-agent development workflows.

Serves the browser dashboard and executes tasks
in background threads.

Current limitations:
- Task records are stored in memory.
- Task records do not survive application restarts.
- API authentication has not yet been implemented.
- This initial server is intended for local development.
"""

import threading
import uuid

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentops.orchestrator.workflow import ProjectManager


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
DASHBOARD_FILE = STATIC_DIR / "index.html"


if not STATIC_DIR.is_dir():
    raise RuntimeError(
        f"AgentOps static directory was not found: {STATIC_DIR}"
    )

if not DASHBOARD_FILE.is_file():
    raise RuntimeError(
        f"AgentOps dashboard file was not found: {DASHBOARD_FILE}"
    )


# ---------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="AgentOps API",
    description=(
        "Multi-agent AI development platform. "
        "Submit tasks, monitor agent progress, and "
        "retrieve development, QA, security, and "
        "architecture results."
    ),
    version="0.1.0",
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class CreateTaskRequest(BaseModel):
    """Request body for creating a development task."""

    task: str = Field(
        min_length=1,
        max_length=20000,
        description="Development task to be executed by AgentOps.",
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of correction attempts.",
    )


# ---------------------------------------------------------
# TASK MANAGEMENT
# ---------------------------------------------------------

class TaskManager:
    """
    Thread-safe in-memory task registry.

    Each API task has its own ProjectManager instance,
    cancellation event, and background execution thread.

    Task records remain available after execution.
    Active workflow and thread references are released
    when execution finishes.
    """

    TERMINAL_STATUSES = {
        "COMPLETED",
        "NEEDS_REVIEW",
        "REJECTED",
        "FAILED",
        "CANCELLED",
    }

    RESULT_FIELDS = {
        "task_id": "workflow_task_id",
        "workflow_status": "workflow_status",
        "progress_percentage": "progress_percentage",
        "workflow_stage": "workflow_stage",
        "current_agent": "current_agent",
        "retry_count": "retry_count",
        "max_retries": "max_retries",
        "completed_stages": "completed_stages",
        "qa_decision": "qa_decision",
        "qa_review": "qa_review",
        "security_decision": "security_decision",
        "security_status": "security_status",
        "security_review_completed": "security_review_completed",
        "security_review": "security_review",
        "security_findings": "security_findings",
        "architecture_status": "architecture_status",
        "architecture_report": "architecture_report",
        "execution_history": "execution_history",
        "final_result": "final_result",
        "error_message": "error_message",
        "cancellation_requested": "cancellation_requested",
    }

    def __init__(self):
        self._tasks: dict[str, dict[str, Any]] = {}
        self._workflows: dict[str, ProjectManager] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _timestamp() -> str:
        """Return a UTC ISO 8601 timestamp."""

        return datetime.now(timezone.utc).isoformat()

    # -----------------------------------------------------
    # Task creation
    # -----------------------------------------------------

    def create_task(
        self,
        task: str,
        max_retries: int,
    ) -> str:
        """Create and register a task with its workflow."""

        api_task_id = str(uuid.uuid4())

        workflow = ProjectManager(
            max_retries=max_retries,
        )

        cancel_event = threading.Event()

        task_record = {
            "task_id": api_task_id,
            "workflow_task_id": None,
            "task": task,
            "workflow_status": "PENDING",
            "progress_percentage": 0,
            "workflow_stage": "queued",
            "current_agent": "NONE",
            "retry_count": 0,
            "max_retries": max_retries,
            "completed_stages": [],
            "qa_decision": "NOT_RUN",
            "qa_review": "",
            "security_decision": "NOT_RUN",
            "security_status": "NOT_RUN",
            "security_review_completed": False,
            "security_review": "",
            "security_findings": [],
            "architecture_status": "PENDING",
            "architecture_report": "",
            "execution_history": [],
            "final_result": None,
            "error_message": None,
            "created_at": self._timestamp(),
            "started_at": None,
            "completed_at": None,
            "cancellation_requested": False,
        }

        with self._lock:
            self._tasks[api_task_id] = task_record
            self._workflows[api_task_id] = workflow
            self._cancel_events[api_task_id] = cancel_event

        return api_task_id

    # -----------------------------------------------------
    # Progress updates
    # -----------------------------------------------------

    def update_progress(
        self,
        api_task_id: str,
        update: dict[str, Any],
    ) -> None:
        """
        Apply an observable workflow progress update.

        Updates are ignored after a task reaches a terminal
        status so delayed callbacks cannot overwrite results.
        """

        if not isinstance(update, dict):
            return

        with self._lock:
            task_record = self._tasks.get(api_task_id)

            if task_record is None:
                return

            if (
                task_record["workflow_status"]
                in self.TERMINAL_STATUSES
            ):
                return

            field_mapping = {
                "task_id": "workflow_task_id",
                "progress_percentage": "progress_percentage",
                "workflow_stage": "workflow_stage",
                "current_agent": "current_agent",
                "workflow_status": "workflow_status",
                "retry_count": "retry_count",
                "max_retries": "max_retries",
                "completed_stages": "completed_stages",
                "cancellation_requested": "cancellation_requested",
                "qa_decision": "qa_decision",
                "qa_review": "qa_review",
                "security_decision": "security_decision",
                "security_status": "security_status",
                "security_review_completed": "security_review_completed",
                "security_review": "security_review",
                "security_findings": "security_findings",
                "architecture_status": "architecture_status",
                "architecture_report": "architecture_report",
                "execution_history": "execution_history",
                "final_result": "final_result",
                "error_message": "error_message",
            }

            for source, destination in field_mapping.items():
                if source not in update:
                    continue

                value = deepcopy(update[source])

                if source == "progress_percentage":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue

                    value = max(0, min(100, value))

                    # Observable progress must not move backwards.
                    value = max(
                        task_record["progress_percentage"],
                        value,
                    )

                task_record[destination] = value

            if task_record["started_at"] is None:
                task_record["started_at"] = self._timestamp()

    # -----------------------------------------------------
    # Final result handling
    # -----------------------------------------------------

    def _apply_final_result(
        self,
        task_record: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Copy the final workflow state into the API task record."""

        for source, destination in self.RESULT_FIELDS.items():
            if source in result:
                task_record[destination] = deepcopy(result[source])

        try:
            final_progress = int(
                result.get(
                    "progress_percentage",
                    task_record["progress_percentage"],
                )
            )
        except (TypeError, ValueError):
            final_progress = task_record["progress_percentage"]

        task_record["progress_percentage"] = max(
            task_record["progress_percentage"],
            min(100, max(0, final_progress)),
        )

        if (
            task_record["workflow_status"]
            not in self.TERMINAL_STATUSES
        ):
            task_record["workflow_status"] = "FAILED"
            task_record["workflow_stage"] = "invalid_workflow_result"
            task_record["current_agent"] = "NONE"
            task_record["error_message"] = (
                "The workflow returned without a terminal status."
            )

        if task_record["started_at"] is None:
            task_record["started_at"] = self._timestamp()

        task_record["completed_at"] = self._timestamp()

    # -----------------------------------------------------
    # Background execution
    # -----------------------------------------------------

    def execute_task(
        self,
        api_task_id: str,
        task: str,
    ) -> None:
        """Execute a workflow and capture its final result."""

        with self._lock:
            workflow = self._workflows.get(api_task_id)
            task_record = self._tasks.get(api_task_id)
            cancel_event = self._cancel_events.get(api_task_id)

            if (
                workflow is None
                or task_record is None
                or cancel_event is None
            ):
                return

            if task_record["workflow_status"] != "RUNNING":
                return

        try:
            result = workflow.run(
                task=task,
                task_id=api_task_id,
                cancel_event=cancel_event,
                progress_callback=lambda update: (
                    self.update_progress(api_task_id, update)
                ),
            )

            if not isinstance(result, dict):
                raise TypeError(
                    "The workflow returned an invalid result."
                )

            with self._lock:
                task_record = self._tasks.get(api_task_id)

                if task_record is None:
                    return

                self._apply_final_result(
                    task_record,
                    result,
                )

        except Exception as error:
            with self._lock:
                task_record = self._tasks.get(api_task_id)

                if task_record is not None:
                    task_record["workflow_status"] = "FAILED"
                    task_record["workflow_stage"] = "execution_error"
                    task_record["current_agent"] = "NONE"
                    task_record["error_message"] = (
                        f"{type(error).__name__}: {error}"
                    )
                    task_record["completed_at"] = self._timestamp()

                    task_record["execution_history"].append(
                        {
                            "timestamp": self._timestamp(),
                            "agent": "orchestrator",
                            "activity": "Workflow execution failed",
                            "details": str(error),
                            "status": "FAILED",
                            "workflow_stage": "execution_error",
                            "progress_percentage": (
                                task_record["progress_percentage"]
                            ),
                        }
                    )

        finally:
            with self._lock:
                self._workflows.pop(api_task_id, None)
                self._threads.pop(api_task_id, None)
                self._cancel_events.pop(api_task_id, None)

    # -----------------------------------------------------
    # Thread management
    # -----------------------------------------------------

    def start_task(
        self,
        api_task_id: str,
        task: str,
    ) -> None:
        """Start workflow execution in a background thread."""

        worker = threading.Thread(
            target=self.execute_task,
            args=(api_task_id, task),
            name=f"AgentOps-{api_task_id}",
            daemon=True,
        )

        with self._lock:
            task_record = self._tasks.get(api_task_id)

            if task_record is None:
                raise ValueError("Task does not exist.")

            if task_record["workflow_status"] != "PENDING":
                raise ValueError(
                    "Task is not in a pending state."
                )

            if api_task_id in self._threads:
                raise ValueError(
                    "A worker thread already exists for this task."
                )

            self._threads[api_task_id] = worker
            task_record["workflow_status"] = "RUNNING"
            task_record["workflow_stage"] = "starting"
            task_record["started_at"] = self._timestamp()

            try:
                # The worker waits for this lock before beginning
                # workflow execution.
                worker.start()

            except Exception:
                self._threads.pop(api_task_id, None)
                self._workflows.pop(api_task_id, None)
                self._cancel_events.pop(api_task_id, None)
                self._tasks.pop(api_task_id, None)
                raise

    # -----------------------------------------------------
    # Task retrieval
    # -----------------------------------------------------

    def get_task(
        self,
        api_task_id: str,
    ) -> Optional[dict[str, Any]]:
        """Return an independent copy of a task record."""

        with self._lock:
            task_record = self._tasks.get(api_task_id)

            if task_record is None:
                return None

            return deepcopy(task_record)

    def get_all_tasks(self) -> list[dict[str, Any]]:
        """Return independent copies of all registered tasks."""

        with self._lock:
            return deepcopy(list(self._tasks.values()))

    # -----------------------------------------------------
    # Cancellation
    # -----------------------------------------------------

    def cancel_task(
        self,
        api_task_id: str,
    ) -> bool:
        """Request cancellation of a pending or running workflow."""

        with self._lock:
            task_record = self._tasks.get(api_task_id)
            cancel_event = self._cancel_events.get(api_task_id)

            if task_record is None or cancel_event is None:
                return False

            if task_record["workflow_status"] not in {
                "PENDING",
                "RUNNING",
            }:
                return False

            if task_record["cancellation_requested"]:
                return True

            cancel_event.set()
            task_record["cancellation_requested"] = True

            return True


# ---------------------------------------------------------
# GLOBAL TASK MANAGER
# ---------------------------------------------------------

task_manager = TaskManager()


# ---------------------------------------------------------
# DASHBOARD ROUTE
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the AgentOps browser dashboard."""

    return FileResponse(
        path=str(DASHBOARD_FILE),
        media_type="text/html",
    )


# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """Return API health information."""

    return {
        "status": "healthy",
        "application": "AgentOps",
        "timestamp": task_manager._timestamp(),
    }


@app.post(
    "/tasks",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_task(request: CreateTaskRequest):
    """Submit a development task for background execution."""

    task = request.task.strip()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A development task cannot be empty.",
        )

    api_task_id = None

    try:
        api_task_id = task_manager.create_task(
            task=task,
            max_retries=request.max_retries,
        )

        task_manager.start_task(
            api_task_id=api_task_id,
            task=task,
        )

        return {
            "task_id": api_task_id,
            "status": "RUNNING",
            "message": (
                "Development task accepted for background execution."
            ),
            "status_url": f"/tasks/{api_task_id}",
            "cancel_url": f"/tasks/{api_task_id}/cancel",
        }

    except Exception as error:
        if api_task_id is not None:
            with task_manager._lock:
                task_manager._tasks.pop(api_task_id, None)
                task_manager._workflows.pop(api_task_id, None)
                task_manager._threads.pop(api_task_id, None)
                task_manager._cancel_events.pop(api_task_id, None)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to submit task: "
                f"{type(error).__name__}: {error}"
            ),
        )


@app.get("/tasks")
def list_tasks():
    """List registered development tasks."""

    tasks = task_manager.get_all_tasks()

    return {
        "count": len(tasks),
        "tasks": tasks,
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Retrieve workflow status and agent results."""

    task_record = task_manager.get_task(task_id)

    if task_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    return task_record


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """Request cancellation of a running development task."""

    task_record = task_manager.get_task(task_id)

    if task_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    cancelled = task_manager.cancel_task(task_id)

    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Task cannot be cancelled because it is no longer "
                "pending or running."
            ),
        )

    return {
        "task_id": task_id,
        "status": "CANCELLATION_REQUESTED",
        "message": (
            "Cancellation was requested. "
            "The workflow will stop at its next cancellation checkpoint."
        ),
    }
