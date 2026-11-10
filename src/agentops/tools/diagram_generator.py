
"""
AgentOps - Architecture Diagram Generator

Generates Mermaid diagrams from observable project information.

Diagram types:
1. Project structure
2. Python module dependencies
3. Component architecture

Design principles:
- Diagrams are derived from actual workspace files.
- Python dependencies are identified through AST import analysis.
- No LLM-generated or assumed relationships.
- Workspace access is read-only.
- Sensitive files and excluded directories are ignored.
- Cancellation is checked during workspace inspection.
- Mermaid labels are sanitized before rendering.
"""

import ast
import hashlib
import re

from pathlib import PurePosixPath
from threading import Event
from typing import Optional, Any

from agentops.tools.file_manager import WorkspaceManager
from agentops.tools.cancellation import check_cancelled


class ProjectDiagramGenerator:
    """Generate deterministic architecture diagrams from source code."""

    MAX_FILES = 150
    MAX_FILE_SIZE_BYTES = 1_000_000
    MAX_DIRECTORY_DEPTH = 8
    MAX_STRUCTURE_NODES = 80
    MAX_DEPENDENCY_NODES = 60
    MAX_DEPENDENCY_EDGES = 150

    SKIPPED_DIRECTORIES = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".terraform",
        "terraform",
        "coverage",
        ".next",
        ".idea",
    }

    SENSITIVE_FILENAMES = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "id_rsa",
        "id_ed25519",
        "credentials",
        "credentials.json",
        "secrets.json",
        "secret.json",
        "terraform.tfstate",
        "terraform.tfstate.backup",
    }

    SENSITIVE_EXTENSIONS = {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".tfstate",
    }

    COMPONENT_DIRECTORIES = {
        "agents": "Agents",
        "agent": "Agents",
        "orchestrator": "Orchestration",
        "orchestrators": "Orchestration",
        "models": "Models",
        "model": "Models",
        "tools": "Tools",
        "tool": "Tools",
        "api": "API",
        "routes": "API",
        "endpoints": "API",
        "services": "Services",
        "controllers": "Controllers",
        "database": "Database",
        "db": "Database",
        "middleware": "Middleware",
        "config": "Configuration",
        "configuration": "Configuration",
        "tests": "Tests",
        "test": "Tests",
        "frontend": "Frontend",
        "backend": "Backend",
        "ui": "Frontend",
    }

    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
    ) -> None:
        self.workspace_manager = (
            workspace_manager or WorkspaceManager()
        )

    # ------------------------------------------------------
    # GENERAL HELPERS
    # ------------------------------------------------------

    @staticmethod
    def _node_id(value: str) -> str:
        """Create a stable Mermaid-safe identifier."""

        digest = hashlib.sha1(
            value.encode("utf-8")
        ).hexdigest()[:10]

        return f"node_{digest}"

    @staticmethod
    def _safe_label(value: str) -> str:
        """Sanitize text used inside Mermaid labels."""

        value = value.replace("\\", "/")
        value = value.replace('"', "'")
        value = value.replace("&", "&amp;")
        value = value.replace("[", "&#91;")
        value = value.replace("]", "&#93;")
        value = value.replace("{", "&#123;")
        value = value.replace("}", "&#125;")
        value = value.replace("<", "&lt;")
        value = value.replace(">", "&gt;")

        value = re.sub(r"[\r\n\t]+", " ", value)

        return value.strip()[:100]

    @staticmethod
    def _is_sensitive_file(relative_path: str) -> bool:
        path = PurePosixPath(relative_path)

        filename = path.name.lower()
        extension = path.suffix.lower()

        if filename in ProjectDiagramGenerator.SENSITIVE_FILENAMES:
            return True

        if extension in ProjectDiagramGenerator.SENSITIVE_EXTENSIONS:
            return True

        if filename.startswith(".env"):
            return True

        if filename.endswith(".tfstate"):
            return True

        return False

    @staticmethod
    def _should_skip_directory(directory_name: str) -> bool:
        return (
            directory_name.lower()
            in ProjectDiagramGenerator.SKIPPED_DIRECTORIES
        )

    # ------------------------------------------------------
    # WORKSPACE DISCOVERY
    # ------------------------------------------------------

    def _discover_workspace(
        self,
        task_id: str,
        cancel_event: Optional[Event] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Discover readable workspace files.

        Uses WorkspaceManager with the QA role to ensure
        inspection remains read-only.
        """

        collected_files = []
        skipped_files = []

        pending_directories = [(".", 0)]

        while pending_directories:
            check_cancelled(cancel_event)

            current_directory, current_depth = (
                pending_directories.pop()
            )

            if current_depth > self.MAX_DIRECTORY_DEPTH:
                skipped_files.append(
                    f"{current_directory}: maximum directory depth reached"
                )
                continue

            entries = self.workspace_manager.list_files(
                task_id,
                relative_path=current_directory,
                agent="qa",
            )

            for entry in entries:
                check_cancelled(cancel_event)

                relative_path = entry.get("path", "")
                filename = entry.get("name", "")
                entry_type = entry.get("type", "")

                if not relative_path or not filename:
                    continue

                if entry_type == "directory":
                    if self._should_skip_directory(filename):
                        continue

                    if current_depth >= self.MAX_DIRECTORY_DEPTH:
                        skipped_files.append(
                            f"{relative_path}: maximum directory depth reached"
                        )
                        continue

                    pending_directories.append(
                        (relative_path, current_depth + 1)
                    )

                    continue

                if entry_type != "file":
                    continue

                if self._is_sensitive_file(relative_path):
                    skipped_files.append(
                        f"{relative_path}: sensitive file excluded"
                    )
                    continue

                if len(collected_files) >= self.MAX_FILES:
                    skipped_files.append(
                        f"{relative_path}: maximum file count reached"
                    )
                    continue

                file_size = entry.get("size_bytes", 0)

                if file_size > self.MAX_FILE_SIZE_BYTES:
                    skipped_files.append(
                        f"{relative_path}: file exceeds size limit"
                    )
                    continue

                collected_files.append({
                    "path": relative_path,
                    "size_bytes": file_size,
                    "content": None,
                })

        return collected_files, skipped_files

    def _read_python_sources(
        self,
        task_id: str,
        discovered_files: list[dict[str, Any]],
        cancel_event: Optional[Event] = None,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Read Python source files for dependency analysis."""

        python_files = []
        skipped_files = []

        for file_info in discovered_files:
            check_cancelled(cancel_event)

            relative_path = file_info["path"]

            if not relative_path.lower().endswith(".py"):
                continue

            try:
                content = self.workspace_manager.read_file(
                    task_id,
                    relative_path,
                    agent="qa",
                )

            except Exception as exc:
                skipped_files.append(
                    f"{relative_path}: read failed "
                    f"({type(exc).__name__})"
                )
                continue

            check_cancelled(cancel_event)

            if not isinstance(content, str):
                skipped_files.append(
                    f"{relative_path}: unsupported source content"
                )
                continue

            python_files.append({
                "path": relative_path,
                "content": content,
            })

        return python_files, skipped_files

    # ------------------------------------------------------
    # PROJECT STRUCTURE DIAGRAM
    # ------------------------------------------------------

    def generate_project_structure(
        self,
        discovered_files: list[dict[str, Any]],
    ) -> str:
        """
        Generate a diagram showing actual directories and files.

        Directory and file relationships are based on their
        actual workspace paths.
        """

        lines = [
            "flowchart TD",
            '    root["Project Workspace"]',
        ]

        directory_nodes = {}
        node_count = 1
        truncated = False

        sorted_files = sorted(
            discovered_files,
            key=lambda item: item["path"].lower(),
        )

        for file_info in sorted_files:
            relative_path = PurePosixPath(file_info["path"])

            parts = relative_path.parts

            if not parts:
                continue

            parent_id = "root"
            current_path = ""

            for directory in parts[:-1]:
                current_path = (
                    f"{current_path}/{directory}"
                    if current_path
                    else directory
                )

                if current_path not in directory_nodes:
                    if node_count >= self.MAX_STRUCTURE_NODES:
                        truncated = True
                        break

                    directory_id = self._node_id(
                        f"directory:{current_path}"
                    )

                    directory_nodes[current_path] = directory_id

                    label = self._safe_label(directory)

                    lines.append(
                        f'    {directory_id}(["{label}/"])'
                    )

                    node_count += 1

                child_id = directory_nodes.get(current_path)

                if child_id is None:
                    break

                lines.append(
                    f"    {parent_id} --> {child_id}"
                )

                parent_id = child_id

            if truncated:
                break

            if node_count >= self.MAX_STRUCTURE_NODES:
                truncated = True
                break

            filename = parts[-1]

            file_id = self._node_id(
                f"file:{file_info['path']}"
            )

            label = self._safe_label(filename)

            lines.append(
                f'    {file_id}["{label}"]'
            )

            lines.append(
                f"    {parent_id} --> {file_id}"
            )

            node_count += 1

        if truncated:
            lines.append(
                '    truncated["Additional files omitted from diagram"]'
            )
            lines.append(
                "    root -.-> truncated"
            )

        return "\n".join(lines)

    # ------------------------------------------------------
    # PYTHON MODULE IDENTIFICATION
    # ------------------------------------------------------

    @staticmethod
    def _module_name(relative_path: str) -> Optional[str]:
        """Convert a Python file path into a module name."""

        path = PurePosixPath(relative_path)

        if path.suffix.lower() != ".py":
            return None

        parts = list(path.with_suffix("").parts)

        if parts and parts[-1] == "__init__":
            parts.pop()

        if not parts:
            return None

        if not all(
            part.isidentifier()
            for part in parts
        ):
            return None

        return ".".join(parts)

    @staticmethod
    def _resolve_relative_import(
        current_module: str,
        is_package: bool,
        module: Optional[str],
        level: int,
    ) -> str:
        """Resolve a Python relative import to a dotted module."""

        current_parts = current_module.split(".")

        if not is_package and current_parts:
            current_parts = current_parts[:-1]

        if level > 0:
            remove_count = max(level - 1, 0)

            if remove_count:
                current_parts = current_parts[:(
                    len(current_parts) - remove_count
                )]

        if module:
            current_parts.extend(module.split("."))

        return ".".join(
            part for part in current_parts if part
        )

    # ------------------------------------------------------
    # PYTHON DEPENDENCY DIAGRAM
    # ------------------------------------------------------

    def _extract_python_imports(
        self,
        python_files: list[dict[str, str]],
        cancel_event: Optional[Event] = None,
    ) -> tuple[dict[str, set[str]], list[str]]:
        """
        Extract local Python module relationships using AST.

        Only relationships that resolve to discovered local
        Python modules are included.
        """

        module_paths = {}

        for file_info in python_files:
            module_name = self._module_name(
                file_info["path"]
            )

            if module_name:
                module_paths[module_name] = file_info["path"]

        dependencies = {}
        parse_warnings = []

        for file_info in python_files:
            check_cancelled(cancel_event)

            relative_path = file_info["path"]
            content = file_info["content"]

            current_module = self._module_name(relative_path)

            if not current_module:
                continue

            is_package = (
                PurePosixPath(relative_path).name == "__init__.py"
            )

            try:
                syntax_tree = ast.parse(
                    content,
                    filename=relative_path,
                )

            except SyntaxError as exc:
                parse_warnings.append(
                    f"{relative_path}: Python syntax could not be parsed "
                    f"(line {exc.lineno})"
                )
                continue

            except Exception as exc:
                parse_warnings.append(
                    f"{relative_path}: AST parsing failed "
                    f"({type(exc).__name__})"
                )
                continue

            local_dependencies = set()

            for node in ast.walk(syntax_tree):
                check_cancelled(cancel_event)

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name

                        if imported_module in module_paths:
                            local_dependencies.add(imported_module)
                            continue

                        top_level = imported_module.split(".")[0]

                        if top_level in module_paths:
                            local_dependencies.add(top_level)

                elif isinstance(node, ast.ImportFrom):
                    base_module = node.module or ""

                    if node.level:
                        base_module = self._resolve_relative_import(
                            current_module,
                            is_package,
                            node.module,
                            node.level,
                        )

                    if base_module in module_paths:
                        local_dependencies.add(base_module)

                    for alias in node.names:
                        if alias.name == "*":
                            continue

                        candidate = (
                            f"{base_module}.{alias.name}"
                            if base_module
                            else alias.name
                        )

                        if candidate in module_paths:
                            local_dependencies.add(candidate)

            dependencies[current_module] = local_dependencies

        return dependencies, parse_warnings

    def generate_python_dependencies(
        self,
        dependencies: dict[str, set[str]],
    ) -> str:
        """
        Generate a directed graph of discovered local Python imports.

        An arrow A --> B means module A imports module B.
        """

        lines = [
            "flowchart LR",
        ]

        module_names = sorted(dependencies.keys())

        visible_modules = module_names[
            :self.MAX_DEPENDENCY_NODES
        ]

        visible_set = set(visible_modules)

        for module_name in visible_modules:
            node_id = self._node_id(
                f"module:{module_name}"
            )

            label = self._safe_label(module_name)

            lines.append(
                f'    {node_id}["{label}"]'
            )

        edge_count = 0
        truncated_edges = False

        for source_module in visible_modules:
            targets = sorted(
                dependencies.get(source_module, set())
            )

            for target_module in targets:
                if target_module not in visible_set:
                    continue

                if edge_count >= self.MAX_DEPENDENCY_EDGES:
                    truncated_edges = True
                    break

                source_id = self._node_id(
                    f"module:{source_module}"
                )

                target_id = self._node_id(
                    f"module:{target_module}"
                )

                lines.append(
                    f"    {source_id} --> {target_id}"
                )

                edge_count += 1

            if truncated_edges:
                break

        if not visible_modules:
            lines.append(
                '    empty["No Python modules discovered"]'
            )

        if len(module_names) > len(visible_modules):
            lines.append(
                '    omitted["Additional modules omitted"]'
            )

        if truncated_edges:
            lines.append(
                '    edge_limit["Additional dependency edges omitted"]'
            )

        return "\n".join(lines)

    # ------------------------------------------------------
    # COMPONENT ARCHITECTURE DIAGRAM
    # ------------------------------------------------------

    def _component_for_path(self, relative_path: str) -> str:
        """Determine a component from recognized directory names."""

        parts = PurePosixPath(relative_path).parts[:-1]

        for part in parts:
            component = self.COMPONENT_DIRECTORIES.get(
                part.lower()
            )

            if component:
                return component

        return "Other"

    def generate_component_architecture(
        self,
        python_files: list[dict[str, str]],
        dependencies: dict[str, set[str]],
    ) -> str:
        """
        Group Python modules by recognizable project components.

        Component relationships are based on actual Python imports.
        """

        module_components = {}

        for file_info in python_files:
            module_name = self._module_name(
                file_info["path"]
            )

            if not module_name:
                continue

            module_components[module_name] = (
                self._component_for_path(
                    file_info["path"]
                )
            )

        component_modules = {}

        for module_name, component in module_components.items():
            component_modules.setdefault(
                component, []
            ).append(module_name)

        lines = [
            "flowchart LR",
        ]

        component_ids = {}

        for component in sorted(component_modules):
            component_id = self._node_id(
                f"component:{component}"
            )

            component_ids[component] = component_id

            label = self._safe_label(component)

            lines.append(
                f'    subgraph {component_id}["{label}"]'
            )

            for module_name in sorted(
                component_modules[component]
            ):
                module_id = self._node_id(
                    f"module:{module_name}"
                )

                module_label = self._safe_label(
                    module_name
                )

                lines.append(
                    f'        {module_id}["{module_label}"]'
                )

            lines.append(
                "    end"
            )

        component_edges = set()

        for source_module, targets in dependencies.items():
            source_component = module_components.get(
                source_module
            )

            if not source_component:
                continue

            for target_module in targets:
                target_component = module_components.get(
                    target_module
                )

                if not target_component:
                    continue

                if source_component == target_component:
                    continue

                component_edges.add(
                    (source_component, target_component)
                )

        for source_component, target_component in sorted(
            component_edges
        ):
            source_id = component_ids[source_component]
            target_id = component_ids[target_component]

            lines.append(
                f"    {source_id} --> {target_id}"
            )

        if not component_modules:
            lines.append(
                '    empty["No recognizable Python components found"]'
            )

        return "\n".join(lines)

    # ------------------------------------------------------
    # MAIN GENERATION INTERFACE
    # ------------------------------------------------------

    def generate_all(
        self,
        task_id: str,
        cancel_event: Optional[Event] = None,
    ) -> dict[str, Any]:
        """
        Generate all supported architecture diagrams.

        Returns a dictionary containing Mermaid source,
        inspection metadata, and warnings.

        This method does not modify the workspace.
        """

        check_cancelled(cancel_event)

        discovered_files, discovery_warnings = (
            self._discover_workspace(
                task_id,
                cancel_event,
            )
        )

        check_cancelled(cancel_event)

        python_files, read_warnings = (
            self._read_python_sources(
                task_id,
                discovered_files,
                cancel_event,
            )
        )

        check_cancelled(cancel_event)

        dependencies, parse_warnings = (
            self._extract_python_imports(
                python_files,
                cancel_event,
            )
        )

        check_cancelled(cancel_event)

        project_structure = self.generate_project_structure(
            discovered_files
        )

        python_dependencies = self.generate_python_dependencies(
            dependencies
        )

        component_architecture = (
            self.generate_component_architecture(
                python_files,
                dependencies,
            )
        )

        all_warnings = (
            discovery_warnings
            + read_warnings
            + parse_warnings
        )

        dependency_count = sum(
            len(targets)
            for targets in dependencies.values()
        )

        return {
            "diagrams": {
                "project_structure": project_structure,
                "python_dependencies": python_dependencies,
                "component_architecture": component_architecture,
            },
            "metadata": {
                "files_discovered": len(discovered_files),
                "python_files_read": len(python_files),
                "python_modules_identified": len(dependencies),
                "local_import_relationships": dependency_count,
                "warnings_count": len(all_warnings),
            },
            "warnings": all_warnings,
            "workspace_modified": False,
        }
