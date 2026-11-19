
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
- Separately reports Python and Node.js execution evidence.

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
            num_ctx=16384,
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

        # Report execution evidence separately from the static
        # review and deterministic QA decision. Python test-case
        # counts and Node.js command execution are not interchangeable.
        report += self._format_execution_evidence(
            result.execution_result
        )

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
    # EXECUTION EVIDENCE REPORTING
    # -----------------------------------------------------

    @classmethod
    def _format_execution_evidence(
        cls,
        execution_result,
    ) -> str:
        """
        Format execution evidence without combining unlike test counts.

        Python test-case counts come from the pytest parser.
        Node.js execution status comes from the Node sandbox.

        A successful npm command does not establish how many
        individual assertions or test cases were executed.
        """

        if execution_result is None:
            return (
                "\n\nAUTOMATED EXECUTION EVIDENCE\n"
                "Execution result: NOT_AVAILABLE\n"
                "No structured execution result was returned.\n"
            )

        metadata = getattr(
            execution_result,
            "metadata",
            None,
        ) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        environment_results = metadata.get(
            "environment_results",
            {},
        )

        if not isinstance(environment_results, dict):
            environment_results = {}

        supported_environments = metadata.get(
            "supported_test_environments",
            [],
        )

        if not isinstance(supported_environments, list):
            supported_environments = []

        lines = [
            "\n\nAUTOMATED EXECUTION EVIDENCE",
            (
                "Overall execution status: "
                f"{getattr(execution_result, 'status', 'UNKNOWN')}"
            ),
        ]

        # Python / pytest evidence.
        python_result = environment_results.get("python")

        if isinstance(python_result, dict):
            lines.append("")
            lines.append("PYTHON / PYTEST")

            discovered = python_result.get("discovered")

            if discovered is not None:
                lines.append(
                    f"Environment discovered: {discovered}"
                )

            lines.append(
                f"Execution status: "
                f"{python_result.get('status', 'NOT_AVAILABLE')}"
            )

            exit_code = python_result.get("exit_code")

            if exit_code is not None:
                lines.append(f"Exit code: {exit_code}")

            tests_executed = python_result.get(
                "tests_executed"
            )

            if tests_executed is not None:
                lines.append(
                    f"Tests executed: {tests_executed}"
                )

            duration = python_result.get(
                "duration_seconds"
            )

            if duration is not None:
                lines.append(
                    f"Duration: {cls._format_duration(duration)}"
                )

            lines.append(
                "Test files discovered: "
                f"{cls._safe_count(metadata.get('tests_discovered', 0))}"
            )

            lines.append(
                "Test cases reported: "
                f"{cls._safe_count(metadata.get('test_cases_reported', 0))}"
            )

            lines.append(
                "Passed: "
                f"{cls._safe_count(metadata.get('tests_passed', 0))}"
            )

            lines.append(
                "Failed: "
                f"{cls._safe_count(metadata.get('tests_failed', 0))}"
            )

            lines.append(
                "Skipped: "
                f"{cls._safe_count(metadata.get('tests_skipped', 0))}"
            )

            lines.append(
                "Collection errors: "
                f"{cls._safe_count(metadata.get('collection_errors', 0))}"
            )

        elif "python" in supported_environments:
            lines.extend([
                "",
                "PYTHON / PYTEST",
                "Environment was identified, but separate execution "
                "details were not available.",
            ])

        # Node.js / npm evidence.
        node_result = environment_results.get("node")

        if isinstance(node_result, dict):
            lines.append("")
            lines.append("NODE.JS / NPM")

            discovered = node_result.get("discovered")

            if discovered is not None:
                lines.append(
                    f"Environment discovered: {discovered}"
                )

            lines.append(
                f"Execution status: "
                f"{node_result.get('status', 'NOT_AVAILABLE')}"
            )

            exit_code = node_result.get("exit_code")

            if exit_code is not None:
                lines.append(f"Exit code: {exit_code}")

            tests_executed = node_result.get(
                "tests_executed"
            )

            if tests_executed is not None:
                lines.append(
                    f"npm test command executed: {tests_executed}"
                )

            duration = node_result.get(
                "duration_seconds"
            )

            if duration is not None:
                lines.append(
                    f"Duration: {cls._format_duration(duration)}"
                )

            lines.append(
                "Individual Node.js test-case count: "
                "NOT_REPORTED"
            )

            lines.append(
                "A successful npm test command indicates that "
                "the command completed successfully. It does not "
                "establish an individual test-case or assertion count."
            )

        elif "node" in supported_environments:
            lines.extend([
                "",
                "NODE.JS / NPM",
                "Environment was identified, but separate execution "
                "details were not available.",
            ])

        if not environment_results:
            lines.extend([
                "",
                "Environment-specific execution details "
                "were not available in the execution metadata.",
            ])

        return "\n".join(lines) + "\n"

    @staticmethod
    def _safe_count(value) -> int:
        """Convert a count into a nonnegative integer safely."""

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _format_duration(value) -> str:
        """Format a nonnegative execution duration."""

        try:
            duration = float(value)

            if duration < 0:
                return "NOT_AVAILABLE"

            return f"{duration:.2f} seconds"

        except (TypeError, ValueError):
            return "NOT_AVAILABLE"

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

Your responsibility is to identify concrete software defects
and meaningful implementation risks using only the supplied
project evidence.

The quality of a review is determined by the accuracy and
actionability of its findings, not by the number of findings.

A short report with no findings is preferable to a report
containing speculative, duplicated, or unsupported findings.

REVIEW SCOPE:

Review for issues that are relevant to the actual implementation:

- Incorrect logic or behavior.
- Defects in input handling and validation.
- Demonstrable security weaknesses.
- Missing error handling where a failure can realistically occur
  and the application does not handle it appropriately.
- Incorrect file, resource, or dependency handling.
- Integration and configuration defects.
- Reliability problems.
- Maintainability problems that create a concrete operational risk.

Do not treat every possible improvement as a defect.

EVIDENCE REQUIREMENTS:

1. Use only code and configuration actually supplied in this review.

2. Before reporting a finding, identify the exact code expression,
   behavior, or configuration that supports it.

3. Explain the actual consequence of the identified behavior.
   Do not rely on vague statements such as "could cause problems"
   or "may introduce vulnerabilities."

4. Do not assume that omitted code, files, dependencies, or
   application layers do not exist.

5. Do not claim that a function, validation rule, test, assertion,
   exception handler, or feature exists unless it is visible
   in the supplied evidence.

6. Do not describe recommended behavior as existing behavior.

7. Do not invent execution results, test results, runtime behavior,
   user input sources, API endpoints, or application features.

8. If a conclusion depends on an assumption about omitted code
   or an unknown application boundary, do not report it as
   a confirmed defect.

9. If evidence is insufficient to establish a defect, omit
   the finding or explicitly identify the uncertainty in
   the review summary.

10. Never invent files, functions, dependencies, line numbers,
    or implementation details.

INPUT VALIDATION RULES:

1. Do not automatically classify missing input validation
   as a defect.

2. First establish the expected input contract from the
   supplied code, development task, or visible application context.

3. Do not assume that zero, negative numbers, empty strings,
   or other particular values are invalid unless the expected
   behavior establishes that restriction.

4. Do not assume that function parameters originate from
   untrusted external input unless the supplied evidence
   establishes that source or boundary.

5. A function accepting parameters without explicit validation
   is not automatically defective. Consider the function's
   responsibility, callers, documented contract, and context.

6. When an actual invalid-input defect is established,
   identify the specific input and the resulting incorrect
   behavior.

SECURITY REVIEW RULES:

1. Security findings require a concrete security-relevant
   behavior supported by the supplied code.

2. Do not report injection, command execution, path traversal,
   authentication bypass, authorization failure, or data
   exposure merely because a function accepts parameters.

3. For an input-related security finding, establish as much
   of the following as the supplied evidence allows:
   - The source of untrusted input.
   - How that input reaches the affected operation.
   - The missing or ineffective security control.
   - The realistic security consequence.

4. Do not assume that an ordinary calculation or parameter
   assignment creates an injection vulnerability.

5. Do not duplicate a correctness finding as a security finding
   unless there is a distinct and demonstrable security impact.

ERROR HANDLING AND TESTING RULES:

1. Do not require try/except blocks around ordinary test assertions
   or normal function calls simply because an exception is possible.

2. Test frameworks such as pytest already report uncaught
   exceptions as test failures. Do not describe that behavior
   as a missing test error-handling defect.

3. Report missing exception handling in application code only
   when the supplied evidence establishes a meaningful failure
   scenario and the absence of handling creates a concrete risk.

4. Do not treat a test's lack of exception handling as a defect
   when normal test-framework behavior already provides
   appropriate failure reporting.

5. Distinguish between:
   - A test file that exists.
   - A test function that contains assertions.
   - A script that merely prints a success message.
   - A test command that completed successfully.
   - Individual test cases that were actually reported.

6. A script that prints a message without checking behavior
   should not be described as a meaningful functional test.

7. Do not claim that a test suite provides coverage beyond
   what is visible in its actual test code.

DUPLICATE FINDING RULES:

1. Report each underlying defect only once.

2. Do not report the same missing validation behavior under
   multiple categories unless there are separate, independently
   supported consequences.

3. Do not split one defect into multiple findings merely
   to discuss different possible impacts.

4. Combine closely related symptoms when they share the
   same root cause and corrective action.

5. Findings should represent distinct problems that can
   be independently understood and addressed.

SEVERITY GUIDELINES:

Use severity proportionally to the evidence and demonstrated impact.

CRITICAL:
A demonstrable issue that can cause catastrophic impact,
such as major unauthorized access, widespread compromise,
or severe data loss.

HIGH:
A confirmed defect with significant impact on a core
application function, important data, or a meaningful
security boundary.

MEDIUM:
A concrete defect that affects functionality, reliability,
or expected behavior but has a limited or recoverable impact.

LOW:
A minor defect or limited risk with a relatively small
impact on users or operations.

INFO:
An observation, limitation, or improvement opportunity
that is not itself a confirmed defect.

Additional severity requirements:

1. Missing validation is not automatically HIGH.

2. A hypothetical security concern is not automatically
   a security finding.

3. Do not increase severity because an issue can be imagined
   to affect a production environment that is not established
   by the supplied evidence.

4. Do not use HIGH or CRITICAL merely to encourage a developer
   to implement a recommendation.

5. If a concern is only an optional improvement, use LOW or INFO
   as appropriate.

REVIEW RULES:

1. Distinguish confirmed defects from potential risks
   and informational observations.

2. Reference a file path when identifying a finding.

3. Include a line number only when it can be determined
   from the supplied file content.

4. Do not report stylistic preferences as defects.

5. Do not automatically approve the implementation.

6. Account for workspace discovery limitations.

7. Do not claim the entire project was reviewed if context
   was incomplete.

8. Do not claim that tests were executed.

9. Do not claim that an application was run.

10. Keep findings specific, factual, and proportional
    to the evidence.

11. Recommendations must be clearly described as recommendations,
    not as existing implementation behavior.

12. Do not invent additional findings to make the report
    appear more comprehensive.

OUTPUT REQUIREMENTS:

Return exactly one JSON object.

Required format:
{
  "summary": "Concise summary of the static review.",
  "findings": [
    {
      "severity": "CRITICAL | HIGH | MEDIUM | LOW | INFO",
      "category": "security | correctness | reliability | performance | maintainability | configuration",
      "title": "Short finding title",
      "description": "Evidence-based explanation identifying the relevant code and its actual consequence",
      "file_path": "relative/path/to/file",
      "line_number": null,
      "recommendation": "Specific corrective recommendation"
    }
  ]
}

For every finding:

- Use one category that best represents its primary impact.
- Keep the title specific to the underlying defect.
- Identify the relevant code behavior in the description.
- Explain why that behavior is a defect or meaningful risk.
- Recommend a proportionate corrective action.
- Do not include unsupported claims.

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

Before including each finding, verify all of the following:

1. The referenced file and behavior are present in the supplied code.
2. The finding describes an actual defect or meaningful risk,
   not merely an optional best practice.
3. The expected behavior or input contract supports the conclusion.
4. The described consequence follows from the visible implementation.
5. The severity is proportional to the demonstrated impact.
6. The finding is not a duplicate of another finding.
7. The recommendation addresses the identified problem.

For missing input validation, do not assume particular values
are invalid without evidence.

For security findings, do not invent an external input source
or attack path.

For test code, do not recommend ordinary try/except wrappers
merely to catch exceptions that pytest already reports.

If a concern is speculative, omit it or describe the uncertainty
in the summary rather than presenting it as a confirmed defect.

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
