
"""
AgentOps - QA Agent

Performs static application review and coordinates
evidence-based automated QA execution.

The QA Agent:
- Inspects project files using read-only WorkspaceManager access.
- Selects relevant source files for review.
- Uses the LLM to identify potential defects.
- Parses structured findings.
- Delegates test execution and final policy decisions
  to QAReviewService.
- Produces a human-readable QA report.
- Reports workspace discovery limitations.

The LLM does not independently approve an implementation.
"""

import json

from pathlib import Path, PurePosixPath
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agentops.tools.cancellation import check_cancelled
from agentops.tools.context_manager import ProjectContextManager
from agentops.tools.file_manager import (
    WorkspaceManager,
    WorkspaceError,
)
from agentops.tools.qa_review_service import QAReviewService


class QAAgent:
    """Read-only application reviewer and QA coordinator."""

    MAX_DISCOVERED_FILES = 150
    MAX_WORKSPACE_CHARACTERS = 250_000
    MAX_FILE_SIZE_BYTES = 1_000_000
    MAX_DIRECTORY_DEPTH = 8

    SKIP_DIRECTORIES = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        ".terraform",
        "workspaces",
    }

    SENSITIVE_FILENAMES = {
        ".env",
        ".env.local",
        ".env.production",
        "id_rsa",
        "id_ed25519",
        "credentials",
        "secrets",
        "secrets.json",
        "authorized_keys",
        "known_hosts",
        ".netrc",
        ".npmrc",
        ".pypirc",
    }

    SENSITIVE_EXTENSIONS = {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".ppk",
        ".tfstate",
    }

    def __init__(
        self,
        model_name: str = "qwen3:4b-instruct",
        workspace_manager: Optional[WorkspaceManager] = None,
        context_manager: Optional[ProjectContextManager] = None,
        review_service: Optional[QAReviewService] = None,
    ) -> None:
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
        )

        self.workspace_manager = (
            workspace_manager or WorkspaceManager()
        )

        self.context_manager = (
            context_manager or ProjectContextManager()
        )

        self.review_service = (
            review_service or QAReviewService()
        )

    # -----------------------------------------------------
    # MAIN EXECUTION
    # -----------------------------------------------------

    def execute(
        self,
        task: str,
        task_id: str,
        implementation: str = "",
        cancel_event=None,
    ) -> str:
        """
        Review an implementation and return a QA report.

        The implementation argument is retained for compatibility
        with the existing orchestrator interface. The workspace
        remains the source of truth for file inspection.
        """

        check_cancelled(cancel_event)

        if not task_id:
            raise ValueError("A task ID is required.")

        if not task or not task.strip():
            raise ValueError("A QA review task cannot be empty.")

        workspace_path = self._get_workspace_path(task_id)

        if not workspace_path.is_dir():
            raise FileNotFoundError(
                f"QA workspace does not exist or is not a directory: "
                f"{workspace_path}"
            )

        (
            files,
            workspace_limitations,
            files_skipped,
        ) = self._collect_workspace_files(
            task_id=task_id,
            cancel_event=cancel_event,
        )

        check_cancelled(cancel_event)

        selected_files = self._select_context(
            task=task,
            files=files,
        )

        check_cancelled(cancel_event)

        static_review_response = ""
        static_review_summary = ""
        static_review_completed = False
        static_review_error = None

        if not selected_files:
            static_review_summary = (
                "Static review could not be completed because "
                "no readable project files were available "
                "for LLM review."
            )

        else:
            try:
                static_review_response = self._run_static_review(
                    task=task,
                    selected_files=selected_files,
                    workspace_limitations=workspace_limitations,
                    cancel_event=cancel_event,
                )

                static_review_summary = self._extract_summary(
                    static_review_response
                )

                static_review_completed = bool(
                    static_review_response.strip()
                    and static_review_summary
                    and not static_review_summary.startswith(
                        "Unable to extract"
                    )
                )

            except Exception as exc:
                check_cancelled(cancel_event)

                static_review_error = (
                    f"Static review could not be completed: "
                    f"{type(exc).__name__}: {exc}"
                )

        check_cancelled(cancel_event)

        if static_review_error:
            static_review_response = (
                '{"summary": "Static review failed.", '
                '"findings": []}'
            )

            static_review_summary = static_review_error

        # Automated test execution and policy evaluation remain
        # the responsibility of QAReviewService.
        result = self.review_service.review(
            task_id=task_id,
            workspace_path=workspace_path,
            static_review_response=static_review_response,
            static_review_completed=static_review_completed,
            cancel_event=cancel_event,
            static_review_summary=static_review_summary,
        )

        check_cancelled(cancel_event)

        report = result.report

        report += (
            "\n\nWORKSPACE REVIEW CONTEXT\n"
            f"Files discovered: {len(files)}\n"
            f"Files selected for LLM review: {len(selected_files)}\n"
            f"Files skipped or excluded: {files_skipped}\n"
            f"Workspace: {workspace_path}\n"
        )

        report += "\nWORKSPACE DISCOVERY LIMITATIONS\n"

        report += "\n".join(
            f"- {item}"
            for item in workspace_limitations
        )

        if static_review_error:
            report += (
                "\n\nSTATIC REVIEW ERROR\n"
                f"{static_review_error}\n"
            )

        return report

    # -----------------------------------------------------
    # WORKSPACE PATH
    # -----------------------------------------------------

    def _get_workspace_path(
        self,
        task_id: str,
    ) -> Path:
        """Resolve the task workspace path."""

        root_dir = Path(self.workspace_manager.root_dir)

        return root_dir / task_id

    # -----------------------------------------------------
    # WORKSPACE DISCOVERY
    # -----------------------------------------------------

    def _collect_workspace_files(
        self,
        task_id: str,
        cancel_event=None,
    ) -> tuple[dict[str, str], list[str], int]:
        """
        Discover and read workspace files through WorkspaceManager.

        Sensitive files and excluded directories are skipped.

        Returns:
            - Dictionary of relative paths and file contents.
            - Workspace discovery limitations.
            - Number of skipped or excluded files/directories.
        """

        collected_files: dict[str, str] = {}
        limitations: list[str] = []

        total_characters = 0
        files_skipped = 0

        file_limit_reached = False
        character_limit_reached = False
        depth_limit_reached = False

        def scan(
            relative_directory: str = ".",
            depth: int = 0,
        ) -> None:
            nonlocal total_characters
            nonlocal files_skipped
            nonlocal file_limit_reached
            nonlocal character_limit_reached
            nonlocal depth_limit_reached

            check_cancelled(cancel_event)

            if depth > self.MAX_DIRECTORY_DEPTH:
                depth_limit_reached = True

                limitations.append(
                    f"Directory depth limit reached at "
                    f"{relative_directory}."
                )
                return

            if len(collected_files) >= self.MAX_DISCOVERED_FILES:
                file_limit_reached = True
                return

            # WorkspaceManager controls directory access.
            # The QA role has read-only permissions.
            entries = self.workspace_manager.list_files(
                task_id=task_id,
                relative_path=relative_directory,
                agent="qa",
            )

            for entry in entries:
                check_cancelled(cancel_event)

                if len(collected_files) >= self.MAX_DISCOVERED_FILES:
                    file_limit_reached = True
                    return

                if not isinstance(entry, dict):
                    files_skipped += 1

                    limitations.append(
                        "An invalid workspace directory entry was skipped."
                    )
                    continue

                entry_path = entry.get("path")
                entry_type = entry.get("type")

                if not isinstance(entry_path, str):
                    files_skipped += 1

                    limitations.append(
                        "A workspace entry with an invalid path was skipped."
                    )
                    continue

                try:
                    normalized_path = self._validate_relative_path(
                        entry_path
                    )

                except WorkspaceError:
                    files_skipped += 1
                    continue

                parts = PurePosixPath(normalized_path).parts

                if not parts:
                    files_skipped += 1
                    continue

                filename = parts[-1].lower()

                if entry_type == "directory":
                    if filename in self.SKIP_DIRECTORIES:
                        files_skipped += 1
                        continue

                    scan(
                        normalized_path,
                        depth + 1,
                    )
                    continue

                if entry_type != "file":
                    continue

                if filename in self.SENSITIVE_FILENAMES:
                    files_skipped += 1
                    continue

                suffix = PurePosixPath(filename).suffix

                if suffix in self.SENSITIVE_EXTENSIONS:
                    files_skipped += 1
                    continue

                check_cancelled(cancel_event)

                try:
                    content = self.workspace_manager.read_file(
                        task_id=task_id,
                        relative_path=normalized_path,
                        agent="qa",
                    )

                except UnicodeDecodeError:
                    files_skipped += 1

                    limitations.append(
                        f"{normalized_path} was skipped because "
                        "it could not be decoded as text."
                    )
                    continue

                except (OSError, ValueError, PermissionError) as exc:
                    check_cancelled(cancel_event)

                    files_skipped += 1

                    limitations.append(
                        f"Could not read {normalized_path}: "
                        f"{type(exc).__name__}."
                    )
                    continue

                except WorkspaceError as exc:
                    check_cancelled(cancel_event)

                    files_skipped += 1

                    limitations.append(
                        f"Workspace access prevented reading "
                        f"{normalized_path}: {exc}"
                    )
                    continue

                check_cancelled(cancel_event)

                if not isinstance(content, str):
                    files_skipped += 1

                    limitations.append(
                        f"{normalized_path} returned non-text content."
                    )
                    continue

                content_size = len(content.encode("utf-8"))

                if content_size > self.MAX_FILE_SIZE_BYTES:
                    files_skipped += 1

                    limitations.append(
                        f"{normalized_path} exceeded the maximum "
                        f"file size of {self.MAX_FILE_SIZE_BYTES:,} bytes."
                    )
                    continue

                remaining = (
                    self.MAX_WORKSPACE_CHARACTERS
                    - total_characters
                )

                if remaining <= 0:
                    character_limit_reached = True

                    limitations.append(
                        "The workspace character limit was reached. "
                        "Additional file contents were not reviewed."
                    )
                    return

                if len(content) > remaining:
                    content = content[:remaining]

                    collected_files[normalized_path] = content
                    total_characters += len(content)

                    character_limit_reached = True

                    limitations.append(
                        f"{normalized_path} was truncated because "
                        "the workspace character limit was reached."
                    )
                    return

                collected_files[normalized_path] = content
                total_characters += len(content)

        scan()

        if file_limit_reached:
            limitations.append(
                f"The discovery limit of {self.MAX_DISCOVERED_FILES} "
                "files was reached. Additional files were not reviewed."
            )

        if character_limit_reached:
            limitations.append(
                f"The workspace context limit of "
                f"{self.MAX_WORKSPACE_CHARACTERS:,} characters "
                "was reached."
            )

        if depth_limit_reached:
            limitations.append(
                f"The maximum directory depth of "
                f"{self.MAX_DIRECTORY_DEPTH} was reached."
            )

        if not limitations:
            limitations.append(
                "No workspace discovery limitations were detected."
            )

        return (
            collected_files,
            self._unique_strings(limitations),
            files_skipped,
        )

    @classmethod
    def _validate_relative_path(
        cls,
        relative_path: str,
    ) -> str:
        """Validate workspace-relative paths and exclude sensitive locations."""

        if not isinstance(relative_path, str) or not relative_path:
            raise WorkspaceError("A valid relative path is required.")

        if relative_path != relative_path.strip():
            raise WorkspaceError(
                "Paths cannot contain surrounding whitespace."
            )

        if "\x00" in relative_path or "\\" in relative_path:
            raise WorkspaceError("Invalid path characters.")

        path = PurePosixPath(relative_path)

        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise WorkspaceError("Invalid workspace-relative path.")

        parts = [part.lower() for part in path.parts]

        if any(part in cls.SKIP_DIRECTORIES for part in parts):
            raise WorkspaceError("Excluded directory.")

        filename = parts[-1]

        if filename in cls.SENSITIVE_FILENAMES:
            raise WorkspaceError("Sensitive filename.")

        if filename == ".env" or filename.startswith(".env."):
            raise WorkspaceError("Environment files are protected.")

        if PurePosixPath(filename).suffix in cls.SENSITIVE_EXTENSIONS:
            raise WorkspaceError("Sensitive file extension.")

        return path.as_posix()

    # -----------------------------------------------------
    # CONTEXT SELECTION
    # -----------------------------------------------------

    def _select_context(
        self,
        task: str,
        files: dict[str, str],
    ) -> list:
        """Select relevant project files for static review."""

        if not files:
            return []

        context_input = [
            {
                "path": path,
                "content": content,
            }
            for path, content in files.items()
        ]

        context = self.context_manager.select_context(
            task=task,
            files=context_input,
        )

        return context.selected_files

    # -----------------------------------------------------
    # STATIC REVIEW
    # -----------------------------------------------------

    def _run_static_review(
        self,
        task: str,
        selected_files: list,
        workspace_limitations: list[str] | None = None,
        cancel_event=None,
    ) -> str:
        """Ask the LLM to inspect selected files for defects."""

        file_sections = []

        for selected_file in selected_files:
            check_cancelled(cancel_event)

            file_sections.append(
                f"FILE: {selected_file.path}\n"
                "```text\n"
                f"{selected_file.content}\n"
                "```"
            )

        project_context = "\n\n".join(file_sections)

        if not project_context:
            raise ValueError(
                "Static review requires at least one selected file."
            )

        limitations_text = "\n".join(
            f"- {item}"
            for item in (workspace_limitations or [])
        )

        system_prompt = """
You are the AgentOps Quality Assurance Agent.

Your job is to identify concrete software defects and
implementation risks using only the supplied project evidence.

Review for:
- Logic errors and incorrect behavior.
- Missing error handling.
- Unsafe input handling.
- Security weaknesses.
- Incorrect file or resource handling.
- Integration and configuration problems.
- Missing validation.
- Reliability and maintainability concerns.

Rules:
1. Do not invent files, functions, or implementation details.
2. Do not claim that tests were executed.
3. Do not claim that an application was run.
4. Distinguish confirmed defects from potential risks.
5. Reference a file path when identifying a finding.
6. Include a line number only when it can be determined
   from the supplied file content.
7. Do not report stylistic preferences as defects.
8. Do not automatically approve the implementation.
9. Account for workspace discovery limitations.
10. Do not assume omitted files or directories do not exist.
11. Do not claim the entire project was reviewed if context
    was incomplete.

Return exactly one JSON object.

Required format:
{
  "summary": "Concise summary of the static review.",
  "findings": [
    {
      "severity": "CRITICAL | HIGH | MEDIUM | LOW | INFO",
      "category": "security | correctness | reliability | performance | maintainability | configuration",
      "title": "Short finding title",
      "description": "Evidence-based explanation",
      "file_path": "relative/path/to/file",
      "line_number": null,
      "recommendation": "Specific corrective recommendation"
    }
  ]
}

If no findings are identified, return an empty findings array.
"""

        human_prompt = f"""
DEVELOPMENT TASK:
{task}

WORKSPACE DISCOVERY LIMITATIONS:
{limitations_text or "No discovery limitations were reported."}

PROJECT FILES:
{project_context}

Perform the static review using only the supplied evidence.

If workspace limitations prevent a reliable conclusion,
state that uncertainty in the summary.

Return the required JSON object.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response_parts = []

        for chunk in self.llm.stream(messages):
            check_cancelled(cancel_event)

            content = getattr(chunk, "content", "")

            if isinstance(content, str):
                response_parts.append(content)

        check_cancelled(cancel_event)

        return "".join(response_parts).strip()

    # -----------------------------------------------------
    # RESPONSE PARSING
    # -----------------------------------------------------

    @staticmethod
    def _extract_summary(response: str) -> str:
        """Extract the summary field from the LLM JSON response."""

        if not response or not response.strip():
            return "Unable to extract static review summary."

        cleaned = response.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict):
                summary = parsed.get("summary")

                if isinstance(summary, str) and summary.strip():
                    return summary.strip()

        except json.JSONDecodeError:
            pass

        return "Unable to extract static review summary."

    # -----------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------

    @staticmethod
    def _unique_strings(
        values: list[str],
    ) -> list[str]:
        """Remove duplicate non-empty strings while preserving order."""

        unique = []
        seen = set()

        for value in values:
            if not isinstance(value, str):
                continue

            cleaned = value.strip()

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            unique.append(cleaned)

        return unique
