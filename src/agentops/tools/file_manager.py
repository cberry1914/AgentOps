
"""
AgentOps - Shared Workspace and File Management

Provides controlled file access within isolated task workspaces.

Security boundaries:
- Agents can only access their assigned task workspace.
- Paths must be relative to that workspace.
- Directory traversal and symlink paths are rejected.
- Protected files and directories cannot be accessed or modified.
- Developer can read and modify files.
- QA can only read and list files.
- No shell execution or file deletion is provided.

This is an application-level security layer, not a hardened
sandbox against malicious concurrent filesystem changes.
"""

import os
import re
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceError(Exception):
    """Base exception for workspace operations."""


class WorkspacePermissionError(WorkspaceError):
    """Raised when an agent attempts an unauthorized operation."""


class WorkspaceManager:
    """
    Manages isolated workspaces for AgentOps tasks.

    Default workspace location:
        <project_root>/workspaces/<task_id>/
    """

    ALLOWED_ROLES = {
        "developer": {"list", "read", "write", "mkdir"},
        "qa": {"list", "read"},
        "orchestrator": {"list", "read", "write", "mkdir"},
    }

    # Directories that agents cannot access or create.
    PROTECTED_DIRECTORIES = {
        ".git",
        ".ssh",
        ".aws",
        ".azure",
        ".gnupg",
    }

    # Filenames that may contain credentials or secrets.
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
        root_dir: Path | str | None = None,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError(
                "Maximum file size must be greater than zero."
            )

        if root_dir is None:
            root_dir = Path.cwd() / "workspaces"

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_file_size_bytes = max_file_size_bytes

        self.root_dir.mkdir(parents=True, exist_ok=True)

        # In-memory audit history for this application session.
        self.audit_events: list[dict[str, Any]] = []

    def _validate_task_id(self, task_id: str) -> str:
        """Accept only UUID-formatted task IDs."""

        if not isinstance(task_id, str):
            raise WorkspaceError("Task ID must be a string.")

        if not re.fullmatch(
            r"[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}",
            task_id,
        ):
            raise WorkspaceError("Invalid task ID format.")

        return task_id.lower()

    def _authorize(self, agent: str, operation: str) -> None:
        """Enforce the permissions assigned to each agent."""

        permissions = self.ALLOWED_ROLES.get(agent)

        if permissions is None:
            raise WorkspacePermissionError(
                f"Unknown agent role: {agent}"
            )

        if operation not in permissions:
            raise WorkspacePermissionError(
                f"Agent '{agent}' cannot perform '{operation}'."
            )

    def _validate_relative_path(
        self,
        relative_path: str,
    ) -> str:
        """
        Validate a workspace-relative path.

        Reject absolute paths, traversal, Windows-style paths,
        protected directories, and sensitive filenames.
        """

        if not isinstance(relative_path, str) or not relative_path:
            raise WorkspaceError("A valid file path is required.")

        if relative_path != relative_path.strip():
            raise WorkspaceError(
                "File paths cannot contain surrounding whitespace."
            )

        if "\x00" in relative_path or "\\" in relative_path:
            raise WorkspaceError("Invalid characters in file path.")

        if relative_path.startswith("/"):
            raise WorkspaceError("Absolute paths are not allowed.")

        parts = relative_path.split("/")

        if any(part in ("", ".", "..") for part in parts):
            if relative_path != ".":
                raise WorkspaceError(
                    "Invalid or traversing path components."
                )

        # Reject Windows drive-style paths.
        if len(parts[0]) >= 2 and parts[0][1] == ":":
            raise WorkspaceError(
                "Drive-style paths are not allowed."
            )

        normalized_parts = [part.lower() for part in parts]

        if any(
            part in self.PROTECTED_DIRECTORIES
            for part in normalized_parts
        ):
            raise WorkspaceError(
                "Access to protected directories is prohibited."
            )

        filename = normalized_parts[-1]

        # Protect all environment files, including .env.local.
        if filename == ".env" or filename.startswith(".env."):
            raise WorkspaceError(
                "Environment files are protected."
            )

        if filename in self.PROTECTED_FILENAMES:
            raise WorkspaceError(
                "This filename is protected."
            )

        if any(
            filename.endswith(extension)
            for extension in self.PROTECTED_EXTENSIONS
        ):
            raise WorkspaceError(
                "Credential and private-key files are protected."
            )

        if filename.startswith("terraform.tfstate"):
            raise WorkspaceError(
                "Terraform state files are protected."
            )

        return "/".join(parts)

    def _get_workspace(self, task_id: str) -> Path:
        """Return an existing task workspace."""

        validated_id = self._validate_task_id(task_id)
        workspace = self.root_dir / validated_id

        if workspace.is_symlink():
            raise WorkspaceError(
                "Workspace cannot be a symbolic link."
            )

        if not workspace.is_dir():
            raise WorkspaceError(
                f"Workspace does not exist for task {validated_id}."
            )

        resolved = workspace.resolve()

        if not resolved.is_relative_to(self.root_dir):
            raise WorkspaceError(
                "Workspace path escaped its root."
            )

        return resolved

    def _resolve_path(
        self,
        task_id: str,
        relative_path: str,
    ) -> Path:
        """
        Resolve a path inside a task workspace.

        Rejects absolute paths, traversal, null bytes,
        protected paths, and existing symbolic links.
        """

        workspace = self._get_workspace(task_id)

        validated_path = self._validate_relative_path(
            relative_path
        )

        requested = Path(validated_path)
        current = workspace

        for part in requested.parts:
            if part in ("", "."):
                continue

            current = current / part

            if current.is_symlink():
                raise WorkspaceError(
                    "Symbolic links are not allowed in workspaces."
                )

        resolved = current.resolve(strict=False)

        if not resolved.is_relative_to(workspace):
            raise WorkspaceError(
                "Path escaped the task workspace."
            )

        return resolved

    def _record_event(
        self,
        task_id: str,
        agent: str,
        operation: str,
        relative_path: str,
        status: str,
        details: str = "",
    ) -> None:
        """Record an operation without storing file contents."""

        self.audit_events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "agent": agent,
                "operation": operation,
                "path": relative_path,
                "status": status,
                "details": details,
            }
        )

    def create_workspace(self, task_id: str) -> Path:
        """Create a new isolated workspace for a task."""

        validated_id = self._validate_task_id(task_id)
        workspace = self.root_dir / validated_id

        if workspace.exists() or workspace.is_symlink():
            raise WorkspaceError(
                f"Workspace already exists for task {validated_id}."
            )

        workspace.mkdir(parents=False, exist_ok=False)

        self._record_event(
            task_id=validated_id,
            agent="orchestrator",
            operation="create_workspace",
            relative_path=".",
            status="SUCCESS",
        )

        return workspace

    def list_files(
        self,
        task_id: str,
        relative_path: str = ".",
        agent: str = "developer",
    ) -> list[dict[str, Any]]:
        """List files and directories inside a workspace directory."""

        self._authorize(agent, "list")
        target = self._resolve_path(task_id, relative_path)

        if not target.is_dir():
            raise WorkspaceError(
                "Requested path is not a directory."
            )

        results = []

        for item in sorted(
            target.iterdir(),
            key=lambda entry: entry.name,
        ):
            if item.is_symlink():
                raise WorkspaceError(
                    "Workspace contains a symbolic link."
                )

            item_relative_path = item.relative_to(
                self._get_workspace(task_id)
            ).as_posix()

            # Never expose protected entries in directory listings.
            try:
                self._validate_relative_path(item_relative_path)
            except WorkspaceError:
                continue

            results.append(
                {
                    "name": item.name,
                    "path": item_relative_path,
                    "type": (
                        "directory" if item.is_dir() else "file"
                    ),
                    "size_bytes": (
                        item.stat().st_size if item.is_file() else None
                    ),
                }
            )

        self._record_event(
            task_id,
            agent,
            "list",
            relative_path,
            "SUCCESS",
        )

        return results

    def read_file(
        self,
        task_id: str,
        relative_path: str,
        agent: str = "developer",
    ) -> str:
        """Read a text file from the task workspace."""

        self._authorize(agent, "read")
        target = self._resolve_path(task_id, relative_path)

        if not target.is_file():
            raise WorkspaceError(
                "Requested file does not exist."
            )

        if target.stat().st_size > self.max_file_size_bytes:
            raise WorkspaceError(
                "File exceeds the configured size limit."
            )

        content = target.read_text(encoding="utf-8")

        self._record_event(
            task_id,
            agent,
            "read",
            relative_path,
            "SUCCESS",
        )

        return content

    def create_directory(
        self,
        task_id: str,
        relative_path: str,
        agent: str = "developer",
    ) -> None:
        """Create a directory inside the task workspace."""

        self._authorize(agent, "mkdir")
        target = self._resolve_path(task_id, relative_path)

        if target.exists():
            raise WorkspaceError(
                "Directory or file already exists."
            )

        target.mkdir(parents=True, exist_ok=False)

        self._record_event(
            task_id,
            agent,
            "mkdir",
            relative_path,
            "SUCCESS",
        )

    def write_file(
        self,
        task_id: str,
        relative_path: str,
        content: str,
        agent: str = "developer",
    ) -> None:
        """
        Create or replace a text file inside the task workspace.

        Writes are atomic: a temporary file is written first,
        then moved into place.
        """

        self._authorize(agent, "write")

        if not isinstance(content, str):
            raise WorkspaceError(
                "File content must be a string."
            )

        encoded = content.encode("utf-8")

        if len(encoded) > self.max_file_size_bytes:
            raise WorkspaceError(
                "Content exceeds the configured size limit."
            )

        target = self._resolve_path(task_id, relative_path)

        if not target.parent.is_dir():
            raise WorkspaceError(
                "Parent directory does not exist. Create it first."
            )

        if target.exists() and not target.is_file():
            raise WorkspaceError(
                "Target path is not a regular file."
            )

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=".agentops-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            # Recheck immediately before replacing the destination.
            if target.is_symlink():
                raise WorkspaceError(
                    "Symbolic links are not allowed in workspaces."
                )

            os.replace(temporary_path, target)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

        self._record_event(
            task_id,
            agent,
            "write",
            relative_path,
            "SUCCESS",
        )

    def get_audit_events(
        self,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return audit events, optionally filtered by task ID."""

        if task_id is None:
            return list(self.audit_events)

        validated_id = self._validate_task_id(task_id)

        return [
            event
            for event in self.audit_events
            if event["task_id"] == validated_id
        ]
