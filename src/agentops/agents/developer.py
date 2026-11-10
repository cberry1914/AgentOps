
"""
AgentOps - Developer Agent

The Developer Agent uses a local Ollama model to analyze
development tasks, inspect its assigned workspace, and
propose file changes.

All filesystem operations are performed through WorkspaceManager.
The LLM never receives unrestricted filesystem or shell access.

Workspace discovery limitations are explicitly reported to the
model and included in the final implementation report.
"""

import json

from pathlib import PurePosixPath
from threading import Event
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agentops.tools.file_manager import (
    WorkspaceManager,
    WorkspaceError,
)

from agentops.tools.cancellation import (
    TaskCancelledError,
    check_cancelled,
)


class DeveloperAgent:
    """AI agent responsible for controlled software development."""

    MAX_WORKSPACE_FILES = 30
    MAX_WORKSPACE_CONTEXT_CHARS = 80_000
    MAX_DIRECTORY_DEPTH = 8
    MAX_FILE_CHANGES = 20
    MAX_FILE_BYTES = 1_000_000

    PROTECTED_DIRECTORIES = {
        ".git",
        ".ssh",
        ".aws",
        ".azure",
        ".gnupg",
    }

    PROTECTED_FILENAMES = {
        "credentials",
        "secrets",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
        "authorized_keys",
        "known_hosts",
        ".netrc",
        ".npmrc",
        ".pypirc",
    }

    PROTECTED_EXTENSIONS = {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".ppk",
    }

    def __init__(
        self,
        model_name: str = "qwen3:4b-instruct",
        temperature: float = 0.2,
        workspace_manager: WorkspaceManager | None = None,
    ):
        self.model_name = model_name
        self.workspace_manager = workspace_manager

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

        self.system_prompt = """
You are the Developer Agent in an autonomous software engineering
platform called AgentOps.

YOUR RESPONSIBILITIES:
1. Understand the development task.
2. Inspect the files provided from your assigned workspace.
3. Identify requirements and expected behavior.
4. Design a practical implementation.
5. Generate complete file contents when changes are needed.
6. Preserve existing functionality unless a change is required.
7. Identify assumptions, risks, and edge cases.

ENGINEERING RULES:
- Use the simplest reliable solution.
- Follow Python best practices when writing Python.
- Do not invent libraries, APIs, or project requirements.
- Do not claim code has been executed or tested unless it has.
- Clearly identify assumptions when requirements are incomplete.
- Prefer readable, maintainable code over unnecessary complexity.
- Consider security and error handling.
- Do not request access to files outside the assigned workspace.
- Do not generate shell commands as file operations.
- Do not delete files.
- Do not modify files merely to make unnecessary changes.
- Return complete file contents, not partial patches.

WORKSPACE RULES:
- You can see the existing workspace files supplied in the task.
- You can propose new or updated files using the required JSON format.
- Only propose files necessary to complete the task.
- Use workspace-relative paths.
- Never use absolute paths or directory traversal.
- Do not propose credentials, secrets, or environment files.
- Do not modify Git metadata or configuration.
- Do not create private keys, certificates, or credential files.
- Workspace discovery may be incomplete.
- Review any reported workspace limitations before proposing changes.
- Do not assume that files omitted from the supplied context do not exist.
- State assumptions when missing workspace context affects implementation.

RESPONSE FORMAT:

Return ONLY one valid JSON object.

The JSON object must have this structure:

{
    "summary": "Brief explanation of the implementation",
    "files": [
        {
            "path": "relative/path/to/file.py",
            "content": "Complete file contents as a string"
        }
    ]
}

If no file changes are necessary, return an empty files array.

Do not wrap the JSON object in Markdown code fences.
Do not include explanations outside the JSON object.
"""

    # --------------------------------------------------
    # Path validation
    # --------------------------------------------------

    @classmethod
    def _validate_relative_path(cls, relative_path: str) -> str:
        """Validate a workspace-relative path and reject sensitive locations."""

        if not isinstance(relative_path, str) or not relative_path:
            raise WorkspaceError("A valid file path is required.")

        if relative_path != relative_path.strip():
            raise WorkspaceError(
                "File paths cannot contain surrounding whitespace."
            )

        if "\x00" in relative_path or "\\" in relative_path:
            raise WorkspaceError("Invalid characters in file path.")

        path = PurePosixPath(relative_path)

        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise WorkspaceError(
                "Invalid workspace-relative file path."
            )

        if len(path.parts[0]) >= 2 and path.parts[0][1] == ":":
            raise WorkspaceError(
                "Drive-style paths are not allowed."
            )

        parts = [part.lower() for part in path.parts]

        if any(
            part in cls.PROTECTED_DIRECTORIES
            for part in parts
        ):
            raise WorkspaceError(
                "Access to protected directories is prohibited."
            )

        filename = parts[-1]

        if filename == ".env" or filename.startswith(".env."):
            raise WorkspaceError(
                "Environment files are protected."
            )

        if filename in cls.PROTECTED_FILENAMES:
            raise WorkspaceError(
                "This filename is protected."
            )

        if any(
            filename.endswith(extension)
            for extension in cls.PROTECTED_EXTENSIONS
        ):
            raise WorkspaceError(
                "Credential and private-key files are protected."
            )

        if filename.startswith("terraform.tfstate"):
            raise WorkspaceError(
                "Terraform state files are protected."
            )

        return path.as_posix()

    # --------------------------------------------------
    # Workspace discovery
    # --------------------------------------------------

    def _collect_workspace_files(
        self,
        task_id: str,
        cancel_event: Event | None = None,
    ) -> tuple[list[dict[str, Any]], set[str], list[str]]:
        """
        Collect workspace files and report discovery limitations.

        Returns:
            - Collected workspace files.
            - Discovered directory paths.
            - Human-readable discovery limitations.
        """

        if self.workspace_manager is None:
            raise WorkspaceError(
                "Workspace Manager is not configured."
            )

        files: list[dict[str, Any]] = []
        directories: set[str] = set()
        limitations: list[str] = []

        total_chars = 0
        file_limit_reached = False
        character_limit_reached = False
        depth_limit_reached = False

        def scan(
            relative_directory: str = ".",
            depth: int = 0,
        ) -> None:
            nonlocal total_chars
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

            if len(files) >= self.MAX_WORKSPACE_FILES:
                file_limit_reached = True
                return

            try:
                entries = self.workspace_manager.list_files(
                    task_id=task_id,
                    relative_path=relative_directory,
                    agent="developer",
                )

            except Exception as exc:
                check_cancelled(cancel_event)

                limitations.append(
                    f"Could not inspect directory "
                    f"{relative_directory}: {type(exc).__name__}."
                )
                return

            for entry in entries:
                check_cancelled(cancel_event)

                if len(files) >= self.MAX_WORKSPACE_FILES:
                    file_limit_reached = True
                    return

                if not isinstance(entry, dict):
                    limitations.append(
                        "An invalid workspace directory entry was skipped."
                    )
                    continue

                relative_path = entry.get("path")
                entry_type = entry.get("type")

                if not isinstance(relative_path, str):
                    limitations.append(
                        "A workspace entry with an invalid path was skipped."
                    )
                    continue

                try:
                    relative_path = self._validate_relative_path(
                        relative_path
                    )
                except WorkspaceError:
                    limitations.append(
                        "A protected or invalid workspace path was excluded."
                    )
                    continue

                if entry_type == "directory":
                    directories.add(relative_path)

                    scan(
                        relative_path,
                        depth + 1,
                    )
                    continue

                if entry_type != "file":
                    continue

                check_cancelled(cancel_event)

                try:
                    content = self.workspace_manager.read_file(
                        task_id=task_id,
                        relative_path=relative_path,
                        agent="developer",
                    )

                except UnicodeDecodeError:
                    limitations.append(
                        f"{relative_path} was skipped because it "
                        "could not be decoded as text."
                    )
                    continue

                except Exception as exc:
                    check_cancelled(cancel_event)

                    limitations.append(
                        f"Could not read {relative_path}: "
                        f"{type(exc).__name__}."
                    )
                    continue

                check_cancelled(cancel_event)

                if not isinstance(content, str):
                    limitations.append(
                        f"{relative_path} returned non-text content."
                    )
                    continue

                remaining = (
                    self.MAX_WORKSPACE_CONTEXT_CHARS - total_chars
                )

                if remaining <= 0:
                    character_limit_reached = True
                    limitations.append(
                        "The workspace character limit was reached. "
                        "Additional files were omitted."
                    )
                    return

                if len(content) > remaining:
                    content = content[:remaining]

                    files.append({
                        "path": relative_path,
                        "content": content,
                        "truncated": True,
                    })

                    total_chars += len(content)
                    character_limit_reached = True

                    limitations.append(
                        f"{relative_path} was truncated because "
                        "the workspace character limit was reached."
                    )
                    return

                files.append({
                    "path": relative_path,
                    "content": content,
                    "truncated": False,
                })

                total_chars += len(content)

        scan()

        if file_limit_reached:
            limitations.append(
                f"The workspace discovery limit of "
                f"{self.MAX_WORKSPACE_FILES} files was reached. "
                "Additional files were not supplied to the model."
            )

        if character_limit_reached:
            limitations.append(
                f"The workspace context limit of "
                f"{self.MAX_WORKSPACE_CONTEXT_CHARS:,} characters "
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
            files,
            directories,
            self._unique_strings(limitations),
        )

    # --------------------------------------------------
    # LLM streaming
    # --------------------------------------------------

    def _stream_response(
        self,
        messages: list,
        cancel_event: Event | None = None,
    ) -> str:
        """
        Stream the model response and monitor cancellation.

        Attempts to close the underlying stream when cancellation
        is detected. Actual Ollama generation termination must be
        verified against the installed runtime.
        """

        check_cancelled(cancel_event)

        stream = self.llm.stream(messages)
        response_parts: list[str] = []

        try:
            for chunk in stream:
                check_cancelled(cancel_event)

                content = chunk.content

                if isinstance(content, str):
                    response_parts.append(content)
                elif content is not None:
                    raise ValueError(
                        "The model returned an unexpected response format."
                    )

            check_cancelled(cancel_event)

        except TaskCancelledError:
            close = getattr(stream, "close", None)

            if callable(close):
                close()

            raise

        finally:
            close = getattr(stream, "close", None)

            if callable(close):
                close()

        return "".join(response_parts)

    # --------------------------------------------------
    # Model response parsing
    # --------------------------------------------------

    def _extract_json(self, response: str) -> dict[str, Any]:
        """Extract and validate the model's JSON response."""

        if not isinstance(response, str):
            raise ValueError(
                "The model returned an invalid response."
            )

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end <= start:
            raise ValueError(
                "The model did not return a JSON object."
            )

        try:
            result = json.loads(response[start:end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"The model returned invalid JSON: {error}"
            ) from error

        if not isinstance(result, dict):
            raise ValueError(
                "The model response must be a JSON object."
            )

        if not isinstance(result.get("summary"), str):
            raise ValueError(
                "The response must contain a summary string."
            )

        if not isinstance(result.get("files"), list):
            raise ValueError(
                "The response must contain a files array."
            )

        return result

    # --------------------------------------------------
    # File operation helpers
    # --------------------------------------------------

    def _ensure_parent_directories(
        self,
        task_id: str,
        relative_path: str,
        existing_directories: set[str],
        cancel_event: Event | None = None,
    ) -> None:
        """Create missing parent directories through WorkspaceManager."""

        check_cancelled(cancel_event)

        relative_path = self._validate_relative_path(
            relative_path
        )

        path = PurePosixPath(relative_path)
        parent_parts = path.parts[:-1]

        if not parent_parts:
            return

        for index in range(1, len(parent_parts) + 1):
            check_cancelled(cancel_event)

            directory = PurePosixPath(
                *parent_parts[:index]
            ).as_posix()

            if directory in existing_directories:
                continue

            self.workspace_manager.create_directory(
                task_id=task_id,
                relative_path=directory,
                agent="developer",
            )

            existing_directories.add(directory)

    def _apply_file_changes(
        self,
        task_id: str,
        proposed_files: list[dict[str, Any]],
        existing_directories: set[str],
        cancel_event: Event | None = None,
    ) -> list[str]:
        """
        Validate and apply proposed file changes.

        All proposed operations are validated before writing begins.
        Each write is delegated to WorkspaceManager.
        """

        if self.workspace_manager is None:
            raise WorkspaceError(
                "Workspace Manager is not configured."
            )

        check_cancelled(cancel_event)

        if len(proposed_files) > self.MAX_FILE_CHANGES:
            raise ValueError(
                f"The Developer Agent proposed {len(proposed_files)} "
                f"files. The maximum allowed is {self.MAX_FILE_CHANGES}."
            )

        validated_files: list[dict[str, str]] = []
        seen_paths: set[str] = set()

        for item in proposed_files:
            check_cancelled(cancel_event)

            if not isinstance(item, dict):
                raise ValueError(
                    "Each file operation must be an object."
                )

            relative_path = item.get("path")
            content = item.get("content")

            if (
                not isinstance(relative_path, str)
                or not relative_path.strip()
            ):
                raise ValueError(
                    "Each file needs a valid relative path."
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"File content must be a string: {relative_path}"
                )

            normalized = self._validate_relative_path(
                relative_path
            )

            if len(content.encode("utf-8")) > self.MAX_FILE_BYTES:
                raise ValueError(
                    f"File exceeds the maximum allowed size: {normalized}"
                )

            if normalized in seen_paths:
                raise ValueError(
                    f"Duplicate file operation: {normalized}"
                )

            seen_paths.add(normalized)

            validated_files.append({
                "path": normalized,
                "content": content,
            })

        check_cancelled(cancel_event)

        changed_files: list[str] = []

        for item in validated_files:
            check_cancelled(cancel_event)

            relative_path = item["path"]

            self._ensure_parent_directories(
                task_id,
                relative_path,
                existing_directories,
                cancel_event,
            )

            check_cancelled(cancel_event)

            self.workspace_manager.write_file(
                task_id=task_id,
                relative_path=relative_path,
                content=item["content"],
                agent="developer",
            )

            changed_files.append(relative_path)

        return changed_files

    # --------------------------------------------------
    # Main execution
    # --------------------------------------------------

    def execute(
        self,
        task: str,
        task_id: str | None = None,
        cancel_event: Event | None = None,
    ) -> str:
        """
        Process a development task and apply file operations
        to the assigned workspace.

        The model proposes changes. WorkspaceManager enforces
        the actual filesystem permissions and path boundaries.
        """

        check_cancelled(cancel_event)

        if not task or not task.strip():
            raise ValueError(
                "A development task cannot be empty."
            )

        if self.workspace_manager is None:
            raise WorkspaceError(
                "Developer Agent requires a WorkspaceManager."
            )

        if not task_id:
            raise ValueError(
                "A task ID is required."
            )

        (
            workspace_files,
            existing_directories,
            workspace_limitations,
        ) = self._collect_workspace_files(
            task_id,
            cancel_event,
        )

        check_cancelled(cancel_event)

        workspace_context = json.dumps(
            {
                "existing_files": workspace_files,
                "existing_directories": sorted(
                    existing_directories
                ),
                "workspace_limitations": workspace_limitations,
            },
            indent=2,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"""
Complete the following development task.

TASK:
{task}

ASSIGNED WORKSPACE CONTENTS:
{workspace_context}

WORKSPACE DISCOVERY LIMITATIONS:
{chr(10).join(f"- {item}" for item in workspace_limitations)}

IMPORTANT:
The workspace context may be incomplete.

Do not assume that omitted files or directories do not exist.
Do not overwrite or replace existing functionality based on
an assumption that an undiscovered file is absent.

If the limitations prevent a reliable implementation,
explain the uncertainty in your implementation summary.

Analyze the supplied files before proposing changes.

Return the required JSON object containing your implementation
summary and complete contents for each new or modified file.
"""
            ),
        ]

        response_content = self._stream_response(
            messages,
            cancel_event,
        )

        check_cancelled(cancel_event)

        result = self._extract_json(response_content)

        check_cancelled(cancel_event)

        changed_files = self._apply_file_changes(
            task_id=task_id,
            proposed_files=result["files"],
            existing_directories=existing_directories,
            cancel_event=cancel_event,
        )

        check_cancelled(cancel_event)

        report = [
            "TASK UNDERSTANDING:",
            task,
            "",
            "IMPLEMENTATION:",
            result["summary"],
            "",
            "WORKSPACE FILE CHANGES:",
        ]

        if changed_files:
            report.extend(
                f"- {path}" for path in changed_files
            )
        else:
            report.append(
                "No files were created or modified."
            )

        report.extend([
            "",
            "WORKSPACE LIMITATIONS:",
        ])

        report.extend(
            f"- {item}"
            for item in workspace_limitations
        )

        report.extend([
            "",
            "TECHNICAL DECISIONS:",
            "File operations were applied through WorkspaceManager.",
            "",
            "EXECUTION STATUS:",
            "Files were written to the assigned workspace.",
            "No code execution or automated testing was performed.",
        ])

        return "\n".join(report)

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

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
