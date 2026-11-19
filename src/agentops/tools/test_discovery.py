
"""
AgentOps - Test Discovery

Discovers supported automated tests in an imported project.

Current support:
- Python pytest tests.
- test_*.py files.
- *_test.py files.
- Conventional tests/ directories.
- Conventional pytest configuration files.
- Node.js projects with a package.json test script.
- Node.js package manifests at the project root or frontend/.

Security:
- Read-only filesystem discovery.
- No test execution.
- No imports of project modules.
- No symlink traversal.
- Excludes virtual environments, dependency folders,
  Git metadata, and AgentOps workspaces.

This component identifies candidate tests. It does not
guarantee that discovered tests are valid or executable.
"""

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TestDiscoveryResult:
    """Structured result of project test discovery."""

    supported_frameworks: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    configuration_files: list[str] = field(default_factory=list)
    test_directories: list[str] = field(default_factory=list)
    node_project_directories: list[str] = field(default_factory=list)
    discovered_test_count: int = 0
    discovery_complete: bool = True
    warnings: list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def tests_found(self) -> bool:
        return (
            self.discovered_test_count > 0
            or bool(self.node_project_directories)
        )

    def to_dict(self) -> dict:
        """Convert discovery results into a serializable dictionary."""

        return {
            "supported_frameworks": self.supported_frameworks,
            "test_files": self.test_files,
            "configuration_files": self.configuration_files,
            "test_directories": self.test_directories,
            "node_project_directories": self.node_project_directories,
            "discovered_test_count": self.discovered_test_count,
            "discovery_complete": self.discovery_complete,
            "warnings": self.warnings,
            "error_message": self.error_message,
        }


class TestDiscovery:
    """Discovers supported automated tests without executing them."""

    MAX_FILES = 1000
    MAX_DEPTH = 12

    EXCLUDED_DIRECTORIES = {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "build",
        "dist",
        "site-packages",
        "workspaces",
        ".terraform",
    }

    TEST_CONFIG_FILES = {
        "pytest.ini",
        "tox.ini",
        "pyproject.toml",
        "setup.cfg",
    }

    NODE_PACKAGE_LOCATIONS = (
        Path("package.json"),
        Path("frontend") / "package.json",
    )

    def _validate_workspace(
        self,
        workspace_path: str | Path,
    ) -> Path:
        """Validate and resolve the project directory."""

        if workspace_path is None:
            raise ValueError("Workspace path is required.")

        workspace = Path(workspace_path)

        if workspace.is_symlink():
            raise ValueError(
                "Workspace root cannot be a symbolic link."
            )

        if not workspace.exists():
            raise ValueError("Workspace does not exist.")

        if not workspace.is_dir():
            raise ValueError("Workspace must be a directory.")

        resolved = workspace.resolve(strict=True)

        if not resolved.is_dir():
            raise ValueError("Invalid workspace directory.")

        return resolved

    def _is_test_file(self, path: Path) -> bool:
        """Identify conventional pytest test filenames."""

        if path.suffix.lower() != ".py":
            return False

        return (
            path.name.startswith("test_")
            or path.name.endswith("_test.py")
        )

    @staticmethod
    def _is_cancelled(cancel_event) -> bool:
        """Check whether discovery cancellation was requested."""

        return (
            cancel_event is not None
            and cancel_event.is_set()
        )

    def _discover_node_projects(
        self,
        workspace: Path,
        result: TestDiscoveryResult,
        cancel_event=None,
    ) -> None:
        """
        Identify supported Node.js test projects.

        Only the project root and frontend/ are supported.
        A project is recognized when package.json contains
        a nonempty scripts.test command.

        This method reads manifests only. It does not execute
        package scripts or install dependencies.
        """

        discovered_directories = []

        for relative_manifest in self.NODE_PACKAGE_LOCATIONS:

            if self._is_cancelled(cancel_event):
                result.discovery_complete = False
                result.warnings.append(
                    "Test discovery was cancelled."
                )
                break

            manifest = workspace / relative_manifest

            try:
                # Do not follow symbolic links.
                if manifest.is_symlink():
                    result.warnings.append(
                        f"Skipped symbolic-link Node.js manifest: "
                        f"{relative_manifest}"
                    )
                    continue

                if not manifest.exists():
                    continue

                if not manifest.is_file():
                    continue

                with manifest.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    package_data = json.load(file)

                if not isinstance(package_data, dict):
                    result.discovery_complete = False
                    result.warnings.append(
                        f"Invalid Node.js package manifest: "
                        f"{relative_manifest} must contain a JSON object."
                    )
                    continue

                scripts = package_data.get("scripts", {})

                if not isinstance(scripts, dict):
                    result.discovery_complete = False
                    result.warnings.append(
                        f"Invalid scripts section in "
                        f"{relative_manifest}."
                    )
                    continue

                test_script = scripts.get("test")

                if not isinstance(test_script, str) or not test_script.strip():
                    result.warnings.append(
                        f"No usable scripts.test command found in "
                        f"{relative_manifest}."
                    )
                    continue

                project_directory = str(
                    relative_manifest.parent
                )

                if project_directory == ".":
                    project_directory = "."

                discovered_directories.append(project_directory)

            except json.JSONDecodeError as exc:
                result.discovery_complete = False
                result.warnings.append(
                    f"Unable to parse Node.js manifest "
                    f"{relative_manifest}: {exc}"
                )

            except (OSError, UnicodeError) as exc:
                result.discovery_complete = False
                result.warnings.append(
                    f"Unable to read Node.js manifest "
                    f"{relative_manifest}: {exc}"
                )

        result.node_project_directories = sorted(
            set(discovered_directories)
        )

        if result.node_project_directories:
            result.supported_frameworks.append("npm")

    def discover(
        self,
        workspace_path: str | Path,
        cancel_event=None,
    ) -> TestDiscoveryResult:
        """
        Discover candidate Python and Node.js tests.

        The filesystem scan is bounded by file count and
        directory depth.

        Filesystem errors are recorded as warnings so that one
        inaccessible entry does not terminate the entire scan.
        """

        result = TestDiscoveryResult()

        if self._is_cancelled(cancel_event):
            result.discovery_complete = False
            result.warnings.append(
                "Test discovery was cancelled."
            )
            return result

        try:
            workspace = self._validate_workspace(workspace_path)

        except (ValueError, OSError, TypeError) as exc:
            result.discovery_complete = False
            result.error_message = str(exc)
            return result

        visited_files = 0

        discovered_tests = set()
        discovered_configs = set()
        discovered_directories = set()

        directories = [(workspace, 0)]
        cancelled = False
        file_limit_reached = False

        while directories:

            if self._is_cancelled(cancel_event):
                cancelled = True
                result.discovery_complete = False
                result.warnings.append(
                    "Test discovery was cancelled."
                )
                break

            current_directory, depth = directories.pop()

            if depth > self.MAX_DEPTH:
                result.discovery_complete = False

                try:
                    relative_directory = str(
                        current_directory.relative_to(workspace)
                    )
                except ValueError:
                    relative_directory = current_directory.name

                result.warnings.append(
                    f"Maximum directory depth reached: "
                    f"{relative_directory}"
                )
                continue

            try:
                entries = sorted(
                    current_directory.iterdir(),
                    key=lambda item: item.name.lower(),
                )

            except OSError as exc:
                result.discovery_complete = False
                result.warnings.append(
                    f"Unable to inspect directory "
                    f"{current_directory}: {exc}"
                )
                continue

            for entry in entries:

                if self._is_cancelled(cancel_event):
                    cancelled = True
                    result.discovery_complete = False
                    result.warnings.append(
                        "Test discovery was cancelled."
                    )
                    break

                try:
                    # Never traverse symbolic links.
                    if entry.is_symlink():
                        continue

                    if entry.is_dir():

                        if entry.name in self.EXCLUDED_DIRECTORIES:
                            continue

                        if entry.name == "tests":
                            discovered_directories.add(
                                str(entry.relative_to(workspace))
                            )

                        directories.append(
                            (entry, depth + 1)
                        )
                        continue

                    if not entry.is_file():
                        continue

                    visited_files += 1

                    if visited_files > self.MAX_FILES:
                        result.discovery_complete = False
                        result.warnings.append(
                            f"Discovery stopped after "
                            f"{self.MAX_FILES} files."
                        )

                        directories.clear()
                        file_limit_reached = True
                        break

                    relative_path = str(
                        entry.relative_to(workspace)
                    )

                    if entry.name in self.TEST_CONFIG_FILES:
                        discovered_configs.add(relative_path)

                    if self._is_test_file(entry):
                        discovered_tests.add(relative_path)

                except OSError as exc:
                    result.discovery_complete = False
                    result.warnings.append(
                        f"Unable to inspect entry "
                        f"{entry.name}: {exc}"
                    )
                    continue

            if cancelled or file_limit_reached:
                break

        result.test_files = sorted(discovered_tests)
        result.configuration_files = sorted(discovered_configs)
        result.test_directories = sorted(discovered_directories)

        result.discovered_test_count = len(result.test_files)

        if result.discovered_test_count:
            result.supported_frameworks.append("pytest")

        self._discover_node_projects(
            workspace=workspace,
            result=result,
            cancel_event=cancel_event,
        )

        if not result.tests_found and result.discovery_complete:
            result.warnings.append(
                "No supported Python pytest tests or Node.js "
                "npm test scripts were discovered."
            )

        return result
