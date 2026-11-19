
"""
AgentOps - Controlled Project Importer

Creates a safe, read-only analysis snapshot of a user-selected
project inside an isolated AgentOps task workspace.

The importer:
- Reads only regular text/source files.
- Does not follow symbolic links.
- Excludes credentials, environment files, and sensitive directories.
- Enforces file-count, file-size, total-size, and depth limits.
- Preflights the complete import before writing files.
- Uses WorkspaceManager for all destination writes.
- Sets permissions required by the non-root Docker QA sandbox.

This is an application-level safety boundary, not a hardened
sandbox against malicious concurrent filesystem changes.
"""

from dataclasses import dataclass
from pathlib import Path

from agentops.tools.file_manager import (
    WorkspaceError,
    WorkspaceManager,
)


class ProjectImportError(Exception):
    """Raised when a project cannot be safely imported."""


@dataclass
class ProjectImportResult:
    """Summary of a completed project import."""

    source_directory: str
    task_id: str
    files_imported: int
    total_bytes: int
    workspace_directory: str


class ProjectImporter:
    """Safely imports an approved project into an AgentOps workspace."""

    MAX_FILES = 100
    MAX_FILE_BYTES = 250_000
    MAX_TOTAL_BYTES = 5_000_000
    MAX_DIRECTORY_DEPTH = 10

    DIRECTORY_MODE = 0o755
    FILE_MODE = 0o644

    SKIPPED_DIRECTORIES = {
        ".git", ".ssh", ".aws", ".azure", ".gnupg",
        ".venv", "venv", "env", "__pycache__",
        "node_modules", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", ".tox", ".terraform", ".idea",
        ".vscode", "dist", "build", "workspaces",
        "site-packages", ".next", ".nuxt", "coverage",
    }

    PROTECTED_FILENAMES = {
        "credentials", "secrets", "secrets.json",
        "id_rsa", "id_ed25519", "authorized_keys",
        "known_hosts", ".netrc", ".npmrc", ".pypirc",
        "terraform.tfstate", "terraform.tfstate.backup",
    }

    PROTECTED_EXTENSIONS = {
        ".pem", ".key", ".p12", ".pfx", ".ppk",
        ".sqlite", ".db", ".dump", ".bak", ".bin",
        ".exe", ".dll", ".so", ".dylib", ".zip",
        ".tar", ".gz", ".7z", ".rar", ".pdf",
        ".docx", ".xlsx", ".pptx", ".png", ".jpg",
        ".jpeg", ".gif", ".webp", ".ico", ".mp3",
        ".mp4", ".mov", ".wav",
    }

    ALLOWED_EXTENSIONS = {
        ".py", ".pyi", ".md", ".rst", ".txt",
        ".toml", ".yaml", ".yml", ".json", ".jsonc",
        ".ini", ".cfg", ".conf", ".sql", ".sh",
        ".bash", ".html", ".css", ".scss", ".js",
        ".jsx", ".ts", ".tsx", ".xml", ".csv",
        ".tf", ".hcl", ".dockerfile", ".properties",
        ".gradle", ".java", ".cs", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".ps1",
        ".bat", ".psm1", ".ipynb",
    }

    ALLOWED_EXTENSIONLESS_FILENAMES = {
        "readme", "license", "notice", "dockerfile",
        "makefile", "procfile", "justfile", "taskfile",
        "gemfile", "rakefile", "jenkinsfile",
    }

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self.workspace_manager = workspace_manager

    def _is_protected_file(self, filename: str) -> bool:
        normalized = filename.lower()
        path = Path(normalized)

        if normalized in self.PROTECTED_FILENAMES:
            return True

        if normalized == ".env" or normalized.startswith(".env."):
            return True

        if normalized.startswith("terraform.tfstate"):
            return True

        return path.suffix in self.PROTECTED_EXTENSIONS

    def _is_allowed_file(self, filename: str) -> bool:
        normalized = filename.lower()

        if self._is_protected_file(normalized):
            return False

        suffix = Path(normalized).suffix

        if suffix:
            return suffix in self.ALLOWED_EXTENSIONS

        return normalized in self.ALLOWED_EXTENSIONLESS_FILENAMES

    def _collect_project_files(
        self,
        source_directory: Path,
    ) -> list[tuple[str, str, int]]:
        collected = []
        total_bytes = 0

        def inspect_directory(
            directory: Path,
            relative_directory: Path,
        ) -> None:
            nonlocal total_bytes

            depth = len(relative_directory.parts)

            if depth > self.MAX_DIRECTORY_DEPTH:
                raise ProjectImportError(
                    "Project exceeds the maximum directory depth."
                )

            try:
                entries = sorted(
                    directory.iterdir(),
                    key=lambda entry: entry.name.lower(),
                )
            except OSError as error:
                raise ProjectImportError(
                    f"Unable to inspect directory: {relative_directory}"
                ) from error

            for entry in entries:
                if entry.is_symlink():
                    raise ProjectImportError(
                        "Symbolic links are not allowed in imported "
                        f"projects: {entry.name}"
                    )

                relative_path = relative_directory / entry.name

                if entry.is_dir():
                    if entry.name.lower() in self.SKIPPED_DIRECTORIES:
                        continue

                    inspect_directory(entry, relative_path)
                    continue

                if not entry.is_file():
                    continue

                if not self._is_allowed_file(entry.name):
                    continue

                if len(collected) >= self.MAX_FILES:
                    raise ProjectImportError(
                        f"Project exceeds the {self.MAX_FILES}-file limit."
                    )

                try:
                    file_size = entry.stat().st_size
                except OSError as error:
                    raise ProjectImportError(
                        f"Unable to inspect file: {relative_path}"
                    ) from error

                if file_size > self.MAX_FILE_BYTES:
                    raise ProjectImportError(
                        f"File exceeds the size limit: {relative_path}"
                    )

                total_bytes += file_size

                if total_bytes > self.MAX_TOTAL_BYTES:
                    raise ProjectImportError(
                        "Project exceeds the total import size limit."
                    )

                try:
                    content = entry.read_text(encoding="utf-8")
                except UnicodeDecodeError as error:
                    raise ProjectImportError(
                        f"File is not valid UTF-8 text: {relative_path}"
                    ) from error
                except OSError as error:
                    raise ProjectImportError(
                        f"Unable to read file: {relative_path}"
                    ) from error

                workspace_relative_path = relative_path.as_posix()

                try:
                    self.workspace_manager._validate_relative_path(
                        workspace_relative_path
                    )
                except WorkspaceError as error:
                    raise ProjectImportError(
                        f"Project contains a prohibited path: "
                        f"{workspace_relative_path}"
                    ) from error

                collected.append(
                    (
                        workspace_relative_path,
                        content,
                        file_size,
                    )
                )

        inspect_directory(source_directory, Path("."))

        if not collected:
            raise ProjectImportError(
                "No eligible source or documentation files were found."
            )

        return collected

    def _set_sandbox_read_permissions(
        self,
        workspace_root: Path,
        project_files: list[tuple[str, str, int]],
    ) -> None:
        """
        Allow the non-root Docker sandbox user to read imported files.

        The sandbox mounts this workspace read-only. These permissions
        make directories traversable and files readable without
        granting write access to other users.
        """

        try:
            workspace_root.chmod(self.DIRECTORY_MODE)

            directories = set()

            for relative_path, _, _ in project_files:
                parent = Path(relative_path).parent

                while parent != Path("."):
                    directories.add(parent)
                    parent = parent.parent

            for directory in sorted(
                directories,
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                (workspace_root / directory).chmod(
                    self.DIRECTORY_MODE
                )

            for relative_path, _, _ in project_files:
                (workspace_root / relative_path).chmod(
                    self.FILE_MODE
                )

        except OSError as error:
            raise ProjectImportError(
                "Unable to configure imported project permissions "
                "for the Docker QA sandbox."
            ) from error

    def import_project(
        self,
        source_directory: str | Path,
        task_id: str,
    ) -> ProjectImportResult:
        """Import a selected project into an existing empty workspace."""

        try:
            source = Path(source_directory).expanduser()

            if source.is_symlink():
                raise ProjectImportError(
                    "The selected project directory cannot be a "
                    "symbolic link."
                )

            if not source.exists() or not source.is_dir():
                raise ProjectImportError(
                    "The selected project directory does not exist."
                )

            source = source.resolve(strict=True)

            if source.is_relative_to(self.workspace_manager.root_dir):
                raise ProjectImportError(
                    "An AgentOps task workspace cannot be used "
                    "as an import source."
                )

            existing_entries = self.workspace_manager.list_files(
                task_id=task_id,
                relative_path=".",
                agent="orchestrator",
            )

            if existing_entries:
                raise ProjectImportError(
                    "The destination task workspace must be empty. "
                    "Create a new task workspace before importing."
                )

            project_files = self._collect_project_files(source)

            workspace_root = (
                self.workspace_manager.root_dir / task_id.lower()
            )

            directories = set()

            for relative_path, _, _ in project_files:
                parent = Path(relative_path)

                while parent.parent != Path("."):
                    parent = parent.parent
                    directories.add(parent.as_posix())

            for directory in sorted(
                directories,
                key=lambda value: (
                    len(Path(value).parts),
                    value,
                ),
            ):
                destination = workspace_root / directory

                if not destination.exists():
                    self.workspace_manager.create_directory(
                        task_id=task_id,
                        relative_path=directory,
                        agent="orchestrator",
                    )

            for relative_path, content, _ in project_files:
                self.workspace_manager.write_file(
                    task_id=task_id,
                    relative_path=relative_path,
                    content=content,
                    agent="orchestrator",
                )

            self._set_sandbox_read_permissions(
                workspace_root,
                project_files,
            )

            total_bytes = sum(
                file_size
                for _, _, file_size in project_files
            )

            return ProjectImportResult(
                source_directory=str(source),
                task_id=task_id.lower(),
                files_imported=len(project_files),
                total_bytes=total_bytes,
                workspace_directory=str(workspace_root),
            )

        except WorkspaceError as error:
            raise ProjectImportError(
                f"Workspace operation failed: {error}"
            ) from error
