
"""
AgentOps - Command Line Interface

Provides access to individual agents and the
complete multi-agent development workflow.
"""

import queue
import select
import sys
import threading
import uuid

from pathlib import Path

from agentops.agents.developer import DeveloperAgent
from agentops.agents.qa import QAAgent
from agentops.agents.architect import ArchitectAgent
from agentops.agents.instructor import InstructorAgent

from agentops.orchestrator.workflow import ProjectManager

from agentops.tools.file_manager import WorkspaceManager
from agentops.tools.project_importer import ProjectImporter


# ---------------------------------------------------------
# WORKSPACE MANAGEMENT
# ---------------------------------------------------------

def get_workspace_manager():
    """Create a workspace manager using the project workspaces directory."""

    project_root = Path(__file__).resolve().parents[2]

    return WorkspaceManager(
        root_dir=project_root / "workspaces"
    )


def create_task_workspace():
    """Create a task ID and its isolated workspace."""

    task_id = str(uuid.uuid4())
    workspace_manager = get_workspace_manager()

    workspace_manager.create_workspace(task_id)

    return task_id, workspace_manager


def prepare_analysis_workspace(agent_name: str):
    """Select an existing workspace or import a project for analysis."""

    print(f"\n========== {agent_name} Workspace ==========")
    print("1. Inspect an existing task workspace")
    print("2. Import a project directory")
    print("============================================")

    choice = input("Select an option: ").strip()

    workspace_manager = get_workspace_manager()

    if choice == "1":
        task_id = input(
            "\nEnter the existing workspace task ID: "
        ).strip()

        if not task_id:
            print("A task ID is required.")
            return None

        try:
            workspace_manager.list_files(
                task_id=task_id,
                relative_path=".",
                agent="orchestrator",
            )

            return task_id, workspace_manager

        except Exception as error:
            print(f"\nUnable to access workspace: {error}")
            return None

    if choice == "2":
        source_directory = input(
            "\nEnter the full path to the project directory:\n"
        ).strip()

        if not source_directory:
            print("A project directory is required.")
            return None

        print("\nSelected project directory:")
        print(source_directory)

        confirmation = input(
            "\nType IMPORT to confirm importing this project: "
        ).strip()

        if confirmation != "IMPORT":
            print("Import cancelled.")
            return None

        try:
            task_id, workspace_manager = create_task_workspace()

            importer = ProjectImporter(
                workspace_manager=workspace_manager
            )

            print("\nImporting approved project files...")

            result = importer.import_project(
                source_directory=source_directory,
                task_id=task_id,
            )

            print("\nProject import completed.")
            print(f"Task ID: {result.task_id}")
            print(f"Files imported: {result.files_imported}")
            print(f"Total bytes: {result.total_bytes}")
            print(
                f"Analysis workspace: "
                f"{result.workspace_directory}"
            )

            return task_id, workspace_manager

        except Exception as error:
            print(f"\nProject import failed: {error}")
            return None

    print("Returning to main menu.")
    return None


def get_multiline_input(prompt: str) -> str:
    """Collect multiline input, with back/exit commands."""

    print(prompt)
    print("Enter END on a separate line when finished.")
    print("Enter BACK on the first line to return to the main menu.")

    lines = []

    while True:
        line = input()

        if not lines and line.strip().lower() in {
            "back", "exit", "quit"
        }:
            return "BACK"

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()


# ---------------------------------------------------------
# INDIVIDUAL AGENTS
# ---------------------------------------------------------

def run_developer():
    """Run the Developer Agent independently."""

    task = get_multiline_input(
        "\nDeveloper Agent - Enter your task:\n"
    )

    if task == "BACK":
        return

    if not task:
        print("Please enter a task.")
        return

    try:
        task_id, workspace_manager = create_task_workspace()

        agent = DeveloperAgent(
            workspace_manager=workspace_manager
        )

        print("\nDeveloper Agent is working...")
        print(f"Task ID: {task_id}\n")

        result = agent.execute(
            task=task,
            task_id=task_id,
        )

        print("\n" + str(result))

    except Exception as error:
        print(f"\nDeveloper Agent error: {error}")


def run_qa():
    """
    Run the QA Agent independently.

    The submitted implementation is first written into the
    task workspace. QA then reviews the workspace as its
    source of truth.
    """

    task = get_multiline_input(
        "\nQA Agent - Enter the original development task:\n"
    )

    if task == "BACK":
        return

    if not task:
        print("Please enter the original task.")
        return

    implementation = get_multiline_input(
        "\nPaste the implementation for QA review:\n"
    )

    if implementation == "BACK":
        return

    if not implementation:
        print("Please provide an implementation.")
        return

    try:
        task_id, workspace_manager = create_task_workspace()

        # Store the submitted implementation in the workspace
        # before invoking QA. This makes the workspace the
        # authoritative source for the review.
        workspace_manager.write_file(
            task_id=task_id,
            relative_path="submitted_implementation.py",
            content=implementation,
            agent="orchestrator",
        )

        agent = QAAgent(
            workspace_manager=workspace_manager
        )

        print("\nQA Agent is reviewing the implementation...")
        print(f"Task ID: {task_id}")
        print("Submitted file: submitted_implementation.py\n")

        result = agent.execute(
            task=task,
            task_id=task_id,
            implementation=implementation,
        )

        print("\n" + str(result))

    except Exception as error:
        print(f"\nQA Agent error: {error}")


def run_architect():
    """Run the Architect Agent independently."""

    print("\n========== Architect Agent ==========")
    print("Analyzes project architecture, components,")
    print("integrations, security, and deployment.")
    print("=====================================")

    workspace = prepare_analysis_workspace("Architect")

    if workspace is None:
        return

    task_id, workspace_manager = workspace

    task = get_multiline_input(
        "\nDescribe the architecture analysis you want:\n"
    )

    if task == "BACK":
        return

    if not task:
        print("Please describe the architecture analysis.")
        return

    try:
        agent = ArchitectAgent(
            workspace_manager=workspace_manager
        )

        print("\nArchitect Agent is analyzing the workspace...")
        print(f"Task ID: {task_id}")
        print("This may take a few minutes.\n")

        report = agent.execute(
            task_id=task_id,
            task=task,
        )

        print("\n========== ARCHITECTURE REPORT ==========\n")
        print(report)
        print("\n=========================================")

    except Exception as error:
        print(f"\nArchitect Agent error: {error}")


def run_instructor():
    """Run the Instructor Agent independently."""

    print("\n========== Instructor Agent ==========")
    print("Answers questions using project source code")
    print("and documentation.")
    print("======================================")

    workspace = prepare_analysis_workspace("Instructor")

    if workspace is None:
        return

    task_id, workspace_manager = workspace

    task = get_multiline_input(
        "\nWhat would you like the Instructor to explain?\n"
    )

    if task == "BACK":
        return

    if not task:
        print("Please enter a question.")
        return

    try:
        agent = InstructorAgent(
            workspace_manager=workspace_manager
        )

        print("\nInstructor Agent is inspecting the workspace...")
        print(f"Task ID: {task_id}")
        print("Generating an explanation...\n")

        response = agent.execute(
            task=task,
            task_id=task_id,
        )

        print("\n========== INSTRUCTOR RESPONSE ==========\n")
        print(response)
        print("\n=========================================")

    except Exception as error:
        print(f"\nInstructor Agent error: {error}")


# ---------------------------------------------------------
# WORKFLOW PROGRESS DISPLAY
# ---------------------------------------------------------

def format_progress_bar(
    percentage: int,
    width: int = 30,
) -> str:
    """Create a terminal progress bar."""

    percentage = max(0, min(percentage, 100))
    filled = round(width * percentage / 100)

    return (
        "["
        + "#" * filled
        + "-"
        * (width - filled)
        + f"] {percentage:3d}%"
    )


def display_progress(update: dict) -> None:
    """Display a workflow progress update."""

    percentage = update.get("progress_percentage", 0)
    agent = update.get("current_agent", "UNKNOWN")
    stage = update.get("workflow_stage", "unknown")
    status = update.get("workflow_status", "UNKNOWN")
    retry_count = update.get("retry_count", 0)
    max_retries = update.get("max_retries", 0)

    print("\n----------------------------------------")
    print(format_progress_bar(percentage))
    print(f"Agent: {agent}")
    print(f"Stage: {stage}")
    print(f"Status: {status}")
    print(f"Corrections used: {retry_count}/{max_retries}")

    qa_decision = update.get("qa_decision", "")

    if qa_decision and qa_decision != "NOT_RUN":
        print(f"Latest QA decision: {qa_decision}")

    security_decision = update.get(
        "security_decision",
        "",
    )

    if security_decision and security_decision != "NOT_RUN":
        print(f"Security decision: {security_decision}")

    security_status = update.get(
        "security_status",
        "NOT_RUN",
    )

    if security_status != "NOT_RUN":
        print(f"Security review status: {security_status}")

    architecture_status = update.get(
        "architecture_status",
        "NOT_RUN",
    )

    if architecture_status != "NOT_RUN":
        print(f"Architecture: {architecture_status}")

    if status == "CANCELLED":
        print("Cancellation acknowledged.")

    print("----------------------------------------", flush=True)


# ---------------------------------------------------------
# EXECUTION HISTORY DISPLAY
# ---------------------------------------------------------

def display_execution_history(history: list) -> None:
    """Display execution history using the current state schema."""

    print("\n========== EXECUTION HISTORY ==========")

    if not history:
        print("No execution history was recorded.")
        return

    for entry in history:
        timestamp = entry.get("timestamp", "Unknown time")
        agent = entry.get("agent", "Unknown agent")
        activity = entry.get("activity", "No activity description")
        details = entry.get("details", "")

        print(f"\n[{timestamp}] {agent}")
        print(activity)

        if details:
            print(details)

        stage = entry.get("workflow_stage")

        if stage:
            print(f"Stage: {stage}")


# ---------------------------------------------------------
# COMPLETE MULTI-AGENT WORKFLOW
# ---------------------------------------------------------

def run_full_workflow():
    """Run the complete workflow with live progress and cancellation."""

    task = get_multiline_input(
        "\nAgentOps Workflow - Enter your development task:\n"
    )

    if task == "BACK":
        return

    if not task:
        print("Please enter a development task.")
        return

    print("\n========== AgentOps Multi-Agent Workflow ==========")
    print("Agents:")
    print("  1. Developer - Implements the requested changes")
    print("  2. QA - Reviews implementation and runs tests")
    print("  3. Security - Scans for security risks")
    print("  4. Architect - Analyzes architecture and dependencies")
    print("  5. Orchestrator - Coordinates the agents")
    print("====================================================")
    print("Maximum correction attempts: 2")
    print("Starting workflow...\n")

    try:
        progress_queue = queue.Queue()
        result_container = {}
        cancel_event = threading.Event()

        task_id = str(uuid.uuid4())
        workspace_manager = get_workspace_manager()

        workflow = ProjectManager(
            workspace_manager=workspace_manager,
            max_retries=2,
        )

        def execute_workflow():
            """Run the workflow in a background thread."""

            try:
                result_container["result"] = workflow.run(
                    task=task,
                    task_id=task_id,
                    cancel_event=cancel_event,
                    progress_callback=progress_queue.put,
                )

            except Exception as error:
                result_container["error"] = error

        worker = threading.Thread(
            target=execute_workflow,
            name="AgentOpsWorkflow",
            daemon=False,
        )

        worker.start()

        print("Workflow is running.")
        print("Type CANCEL and press Enter to request cancellation.")
        print("Other input will be ignored.\n")

        cancellation_sent = False

        while worker.is_alive() or not progress_queue.empty():

            while True:
                try:
                    update = progress_queue.get_nowait()
                    display_progress(update)

                except queue.Empty:
                    break

            if worker.is_alive() and not cancellation_sent:
                readable, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.25,
                )

                if readable:
                    command = sys.stdin.readline().strip().upper()

                    if command == "CANCEL":
                        cancel_event.set()
                        cancellation_sent = True

                        print("\nCancellation requested.")
                        print(
                            "Waiting for the agents to stop...\n"
                        )

            elif worker.is_alive():
                worker.join(timeout=0.25)

        worker.join()

        while True:
            try:
                update = progress_queue.get_nowait()
                display_progress(update)

            except queue.Empty:
                break

        if "error" in result_container:
            raise result_container["error"]

        result = result_container.get("result")

        if result is None:
            print("Workflow ended without returning a result.")
            return

        # -------------------------------------------------
        # FINAL WORKFLOW REPORT
        # -------------------------------------------------

        print("\n========== WORKFLOW REPORT ==========")

        print(f"\nTask ID: {result['task_id']}")
        print(f"Status: {result['workflow_status']}")
        print(f"Progress: {result['progress_percentage']}%")
        print(f"Final Stage: {result['workflow_stage']}")
        print(f"Correction Attempts: {result['retry_count']}")

        print("\n========== FINAL RESULT ==========")
        print(result.get("final_result") or "No final result.")

        if result.get("error_message"):
            print(f"\nWorkflow message: {result['error_message']}")

        # QA results
        print("\n========== QA REVIEW ==========")

        print(
            f"QA Decision: "
            f"{result.get('qa_decision') or 'NOT AVAILABLE'}"
        )

        print(
            result.get("qa_review")
            or "No QA review available."
        )

        # Security results
        print("\n========== SECURITY REVIEW ==========")

        print(
            f"Security Decision: "
            f"{result.get('security_decision') or 'NOT AVAILABLE'}"
        )

        print(
            f"Security Status: "
            f"{result.get('security_status', 'NOT_RUN')}"
        )

        print(
            f"Security Review Completed: "
            f"{result.get('security_review_completed', False)}"
        )

        findings = result.get("security_findings", [])

        print(f"Security Findings: {len(findings)}")

        if findings:
            for index, finding in enumerate(findings, start=1):
                print(f"\nFinding {index}:")

                if isinstance(finding, dict):
                    print(
                        f"  Severity: "
                        f"{finding.get('severity', 'UNKNOWN')}"
                    )
                    print(
                        f"  Category: "
                        f"{finding.get('category', 'UNKNOWN')}"
                    )
                    print(
                        f"  Title: "
                        f"{finding.get('title', 'Untitled finding')}"
                    )
                    print(
                        f"  File: "
                        f"{finding.get('file_path') or 'Not specified'}"
                    )
                    print(
                        f"  Recommendation: "
                        f"{finding.get('recommendation') or 'None provided'}"
                    )

                else:
                    print(f"  {finding}")

        security_report = result.get("security_review")

        if security_report:
            print("\n========== SECURITY REPORT DETAILS ==========")
            print(security_report)

        # Architecture results
        print("\n========== ARCHITECTURE ANALYSIS ==========")

        print(
            f"Architecture Status: "
            f"{result.get('architecture_status', 'NOT_RUN')}"
        )

        architecture_report = result.get("architecture_report")

        if architecture_report:
            print("\n" + architecture_report)

        else:
            print("No architecture report available.")

        # Activity history
        display_execution_history(
            result.get("execution_history", [])
        )

        print("\n=======================================")

    except Exception as error:
        print(f"\nAgentOps Workflow error: {error}")


# ---------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------

def main():
    """Display the AgentOps main menu."""

    while True:

        print("\n========== AgentOps ==========")
        print("1. Developer Agent")
        print("2. QA Agent")
        print("3. Run Full AgentOps Workflow")
        print("4. Architect Agent")
        print("5. Instructor / Helper Agent")
        print("6. Exit")
        print("==============================")

        choice = input("Select an option: ").strip()

        if choice == "1":
            run_developer()

        elif choice == "2":
            run_qa()

        elif choice == "3":
            run_full_workflow()

        elif choice == "4":
            run_architect()

        elif choice == "5":
            run_instructor()

        elif choice == "6":
            print("Exiting AgentOps.")
            break

        else:
            print(
                "Invalid selection. "
                "Please choose 1, 2, 3, 4, 5, or 6."
            )


if __name__ == "__main__":
    main()
