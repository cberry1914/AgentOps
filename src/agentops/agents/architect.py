
"""
AgentOps - Architect Agent

Inspects project workspaces and produces architecture reports.

Responsibilities:
- Discover project files through WorkspaceManager.
- Select relevant files through ProjectContextManager.
- Generate deterministic Mermaid architecture diagrams.
- Analyze application structure and design using the local LLM.
- Include project structure, Python dependencies, and component diagrams.
- Report inspection limitations and diagram-generation warnings.
- Preserve read-only workspace access.
- Support cooperative cancellation.

Architecture diagrams are generated from observable project
files and Python import relationships. The LLM explains the
evidence but does not invent diagram relationships.
"""

from threading import Event
from typing import Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agentops.tools.file_manager import WorkspaceManager
from agentops.tools.context_manager import ProjectContextManager
from agentops.tools.diagram_generator import ProjectDiagramGenerator
from agentops.tools.cancellation import check_cancelled


class ArchitectAgent:
    """Read-only architecture inspection and diagram generation agent."""

    MAX_WORKSPACE_FILES = 150
    MAX_WORKSPACE_CONTEXT_CHARS = 250_000
    MAX_DIRECTORY_DEPTH = 8
    MAX_FILE_SIZE_BYTES = 1_000_000

    SYSTEM_PROMPT = """
You are the AgentOps Architect Agent.

Your responsibility is to inspect an existing software project
and produce an accurate, technically useful architecture report.

You are an architecture analyst.

You do not modify source files.

You receive:
- Project source files selected for relevance.
- A project file inventory.
- Deterministically generated Mermaid diagrams.
- Project inspection metadata.
- Any file discovery or parsing warnings.

ARCHITECTURE ANALYSIS RESPONSIBILITIES

1. Identify the application's entry points.
2. Explain the major application components.
3. Explain the responsibilities of agents, tools, models,
   orchestrators, APIs, and services when supported by evidence.
4. Describe observable dependencies between Python modules.
5. Explain the application execution flow when supported by code.
6. Identify important configuration and dependency files.
7. Identify architectural strengths.
8. Identify architectural weaknesses and technical debt.
9. Identify security and reliability concerns.
10. Recommend practical architectural improvements.

EVIDENCE RULES

- Base conclusions on the supplied source material and diagrams.
- Do not invent files, modules, services, or relationships.
- Do not assume an import proves runtime communication.
- Do not assume a directory name proves a component's behavior.
- Distinguish confirmed relationships from architectural inferences.
- Identify missing evidence explicitly.
- Do not claim to have executed the application or its tests.
- Do not claim to have inspected files that were not supplied.

DIAGRAM RULES

The supplied Mermaid diagrams are generated from actual
workspace paths and Python import statements.

Explain what they show.

Do not silently correct, fabricate, or add relationships.

A Python import relationship indicates a source-level import,
not necessarily a runtime call or execution dependency.

REQUIRED REPORT FORMAT

1. EXECUTIVE SUMMARY
   Summarize the application architecture.

2. PROJECT STRUCTURE
   Explain the directory organization and entry points.

3. COMPONENT RESPONSIBILITIES
   Explain the major components and their responsibilities.

4. APPLICATION FLOW
   Describe the observable execution flow.

5. MODULE DEPENDENCIES
   Explain confirmed source-level dependencies.

6. ARCHITECTURAL STRENGTHS
   Identify good architectural decisions.

7. ARCHITECTURAL RISKS
   Identify design, reliability, security, and maintainability risks.

8. RECOMMENDATIONS
   Provide prioritized improvements.

9. INSPECTION LIMITATIONS
   Identify missing files, parsing failures, and unverified assumptions.

Be specific, technically accurate, and concise.
"""

    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
    ) -> None:
        self.workspace_manager = (
            workspace_manager or WorkspaceManager()
        )

        self.context_manager = ProjectContextManager()

        self.diagram_generator = ProjectDiagramGenerator(
            workspace_manager=self.workspace_manager
        )

        self.llm = ChatOllama(
            model="qwen3:4b-instruct",
            temperature=0.1,
        )

    # --------------------------------------------------
    # WORKSPACE DISCOVERY
    # --------------------------------------------------

    def _collect_workspace_files(
        self,
        task_id: str,
        cancel_event: Optional[Event] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Collect project files through read-only workspace access."""

        collected_files = []
        skipped_files = []

        pending_directories = [(".", 0)]
        total_characters = 0

        skipped_directories = {
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

        sensitive_filenames = {
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

        sensitive_extensions = {
            ".pem",
            ".key",
            ".p12",
            ".pfx",
            ".tfstate",
        }

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
                    if filename.lower() in skipped_directories:
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

                filename_lower = filename.lower()

                if filename_lower in sensitive_filenames:
                    skipped_files.append(
                        f"{relative_path}: sensitive file excluded"
                    )
                    continue

                if filename_lower.startswith(".env"):
                    skipped_files.append(
                        f"{relative_path}: environment file excluded"
                    )
                    continue

                if any(
                    filename_lower.endswith(extension)
                    for extension in sensitive_extensions
                ):
                    skipped_files.append(
                        f"{relative_path}: sensitive file extension"
                    )
                    continue

                if len(collected_files) >= self.MAX_WORKSPACE_FILES:
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

                if total_characters >= self.MAX_WORKSPACE_CONTEXT_CHARS:
                    skipped_files.append(
                        f"{relative_path}: discovery context limit reached"
                    )
                    continue

                try:
                    content = self.workspace_manager.read_file(
                        task_id,
                        relative_path,
                        agent="qa",
                    )

                except Exception as exc:
                    check_cancelled(cancel_event)

                    skipped_files.append(
                        f"{relative_path}: read failed "
                        f"({type(exc).__name__})"
                    )
                    continue

                check_cancelled(cancel_event)

                if not isinstance(content, str):
                    skipped_files.append(
                        f"{relative_path}: unsupported content"
                    )
                    continue

                if not content.strip():
                    skipped_files.append(
                        f"{relative_path}: empty file"
                    )
                    continue

                remaining_chars = (
                    self.MAX_WORKSPACE_CONTEXT_CHARS
                    - total_characters
                )

                if len(content) > remaining_chars:
                    content = content[:remaining_chars]

                    skipped_files.append(
                        f"{relative_path}: content truncated"
                    )

                collected_files.append({
                    "path": relative_path,
                    "content": content,
                })

                total_characters += len(content)

        return collected_files, skipped_files

    # --------------------------------------------------
    # CONTEXT SELECTION AND FORMATTING
    # --------------------------------------------------

    @staticmethod
    def _format_workspace_context(
        files: list[Any],
    ) -> str:
        """Format selected project files for the LLM."""

        sections = []

        for file_info in files:
            sections.append(
                f"FILE: {file_info.path}\n"
                f"{file_info.content}"
            )

        return "\n\n".join(sections)

    def _select_project_context(
        self,
        task: str,
        workspace_files: list[dict[str, str]],
    ):
        """Select relevant project files using ProjectContextManager."""

        return self.context_manager.select_context(
            task=task or (
                "Analyze the complete application architecture."
            ),
            files=workspace_files,
        )

    # --------------------------------------------------
    # DIAGRAM FORMATTING
    # --------------------------------------------------

    @staticmethod
    def _format_diagrams(
        diagram_result: dict[str, Any],
    ) -> str:
        """Format generated Mermaid diagrams for the report."""

        diagrams = diagram_result.get("diagrams", {})

        sections = [
            "ARCHITECTURE DIAGRAMS",
            "=====================",
            "",
            "PROJECT STRUCTURE DIAGRAM",
            "-------------------------",
            "```mermaid",
            diagrams.get(
                "project_structure",
                "flowchart TD\n    A[Diagram unavailable]",
            ),
            "```",
            "",
            "PYTHON MODULE DEPENDENCY DIAGRAM",
            "--------------------------------",
            "```mermaid",
            diagrams.get(
                "python_dependencies",
                "flowchart LR\n    A[Diagram unavailable]",
            ),
            "```",
            "",
            "COMPONENT ARCHITECTURE DIAGRAM",
            "------------------------------",
            "```mermaid",
            diagrams.get(
                "component_architecture",
                "flowchart LR\n    A[Diagram unavailable]",
            ),
            "```",
        ]

        return "\n".join(sections)

    # --------------------------------------------------
    # LLM STREAMING
    # --------------------------------------------------

    def _stream_response(
        self,
        messages: list,
        cancel_event: Optional[Event] = None,
    ) -> str:
        """Stream the architecture analysis with cancellation checks."""

        response_parts = []
        stream = None

        try:
            stream = self.llm.stream(messages)

            for chunk in stream:
                check_cancelled(cancel_event)

                content = chunk.content

                if isinstance(content, str):
                    response_parts.append(content)

                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            text_content = item.get("text")

                            if isinstance(text_content, str):
                                response_parts.append(text_content)

            check_cancelled(cancel_event)

            return "".join(response_parts).strip()

        finally:
            if stream is not None:
                close_method = getattr(stream, "close", None)

                if callable(close_method):
                    close_method()

    # --------------------------------------------------
    # MAIN EXECUTION
    # --------------------------------------------------

    def execute(
        self,
        task_id: str,
        task: str = "",
        cancel_event: Optional[Event] = None,
    ) -> str:
        """
        Inspect a project and generate an architecture report.

        The workspace remains read-only.
        """

        check_cancelled(cancel_event)

        # STEP 1: Discover workspace files.

        workspace_files, discovery_warnings = (
            self._collect_workspace_files(
                task_id,
                cancel_event,
            )
        )

        check_cancelled(cancel_event)

        # STEP 2: Select relevant LLM context.

        project_context = self._select_project_context(
            task=task,
            workspace_files=workspace_files,
        )

        selected_files = project_context.selected_files

        workspace_context = self._format_workspace_context(
            selected_files
        )

        selected_inventory = (
            "\n".join(
                f"- {item.path}"
                for item in selected_files
            )
            if selected_files
            else "No readable files selected."
        )

        context_omissions = (
            discovery_warnings
            + project_context.omitted_files
        )

        omission_report = (
            "\n".join(
                f"- {item}"
                for item in context_omissions
            )
            if context_omissions
            else "No files were skipped or omitted."
        )

        # STEP 3: Generate deterministic diagrams.

        check_cancelled(cancel_event)

        try:
            diagram_result = self.diagram_generator.generate_all(
                task_id,
                cancel_event,
            )

        except Exception as exc:
            check_cancelled(cancel_event)

            diagram_result = {
                "diagrams": {
                    "project_structure": (
                        "flowchart TD\n"
                        '    A["Project structure diagram unavailable"]'
                    ),
                    "python_dependencies": (
                        "flowchart LR\n"
                        '    A["Dependency diagram unavailable"]'
                    ),
                    "component_architecture": (
                        "flowchart LR\n"
                        '    A["Component diagram unavailable"]'
                    ),
                },
                "metadata": {},
                "warnings": [
                    "Diagram generation failed: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                ],
                "workspace_modified": False,
            }

        check_cancelled(cancel_event)

        diagram_text = self._format_diagrams(
            diagram_result
        )

        diagram_metadata = diagram_result.get(
            "metadata",
            {},
        )

        diagram_warnings = diagram_result.get(
            "warnings",
            [],
        )

        # STEP 4: Build architecture analysis request.

        user_message = f"""
ARCHITECTURE ANALYSIS REQUEST
=============================

{task or "Analyze the complete application architecture."}

PROJECT DISCOVERY
=================

Files discovered: {len(workspace_files)}
Files selected for LLM context: {len(selected_files)}
Context characters used: {project_context.total_characters}

SELECTED FILE INVENTORY
=======================

{selected_inventory}

FILES SKIPPED OR OMITTED
========================

{omission_report}

PROJECT SOURCE MATERIAL
=======================

{workspace_context}

{diagram_text}

DIAGRAM GENERATION METADATA
===========================

{diagram_metadata}

DIAGRAM WARNINGS
================

{chr(10).join(diagram_warnings) if diagram_warnings else "No diagram-generation warnings."}

ANALYSIS REQUIREMENTS
=====================

Use the supplied source material and diagrams to produce
a technically accurate architecture report.

Explain the diagrams and their observable relationships.

Do not treat source-level imports as proof of runtime
communication.

Do not invent missing components or dependencies.

Clearly identify inspection limitations.

The workspace has not been modified.
Automated tests have not been executed.
"""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # STEP 5: Generate architecture report.

        check_cancelled(cancel_event)

        architecture_report = self._stream_response(
            messages,
            cancel_event,
        )

        check_cancelled(cancel_event)

        if not architecture_report:
            architecture_report = (
                "Architecture analysis could not be generated. "
                "The local model returned an empty response."
            )

        # STEP 6: Append inspection metadata and diagrams.

        metadata_report = f"""

ARCHITECT WORKSPACE INSPECTION
==============================

Files discovered: {len(workspace_files)}
Files selected for LLM context: {len(selected_files)}
Files skipped or omitted: {len(context_omissions)}
Context characters used: {project_context.total_characters}

PYTHON DIAGRAM STATISTICS
=========================

Python files read: {diagram_metadata.get('python_files_read', 0)}
Python modules identified: {diagram_metadata.get('python_modules_identified', 0)}
Local import relationships: {diagram_metadata.get('local_import_relationships', 0)}
Diagram warnings: {len(diagram_warnings)}

WORKSPACE MODIFICATION STATUS
=============================

No project source files were modified by the Architect Agent.

TEST EXECUTION STATUS
=====================

Automated tests were not executed.
Architecture conclusions are based on static inspection.
"""

        warning_report = ""

        if diagram_warnings:
            warning_report = (
                "\n\nDIAGRAM GENERATION WARNINGS\n"
                "===========================\n"
                + "\n".join(
                    f"- {warning}"
                    for warning in diagram_warnings
                )
            )

        omission_section = (
            "\n\nFILES SKIPPED OR OMITTED\n"
            "========================\n"
            + omission_report
        )

        return (
            architecture_report
            + "\n\n"
            + diagram_text
            + metadata_report
            + warning_report
            + omission_section
        )
