"""
AgentOps - Project Context Manager

Selects the most relevant project files for local LLM analysis.

The context manager:
- Prioritizes application entry points and core modules.
- Prioritizes dependency manifests and documentation.
- Uses the user's question to rank relevant files.
- Uses source-content keyword matching as an additional signal.
- Enforces a conservative context character budget.
- Reports files that were excluded or truncated.

File selection is deterministic. The LLM does not decide
which files it is allowed to inspect.
"""

import re

from dataclasses import dataclass
from typing import Any


@dataclass
class ProjectContext:
    """Selected project context and its inspection metadata."""

    selected_files: list[dict[str, str]]
    omitted_files: list[str]
    total_characters: int


class ProjectContextManager:
    """Select relevant source files for an LLM request."""

    MAX_CONTEXT_CHARS = 5_500
    MAX_SELECTED_FILES = 18

    # Files that frequently describe the overall application.
    DOCUMENTATION_FILES = {
        "readme.md",
        "readme.rst",
        "readme.txt",
        "architecture.md",
        "design.md",
        "contributing.md",
    }

    DEPENDENCY_FILES = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "poetry.lock",
        "pipfile",
        "pipfile.lock",
        "package.json",
        "package-lock.json",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "composer.json",
    }

    CONFIGURATION_FILES = {
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
        "makefile",
        "justfile",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "ruff.toml",
        ".gitignore",
    }

    ENTRY_POINT_FILES = {
        "main.py",
        "app.py",
        "application.py",
        "server.py",
        "manage.py",
        "cli.py",
        "__main__.py",
        "main.js",
        "index.js",
        "server.js",
        "index.ts",
        "main.go",
        "program.cs",
    }

    IMPORTANT_DIRECTORY_NAMES = {
        "agents",
        "orchestrator",
        "api",
        "routes",
        "models",
        "services",
        "controllers",
        "database",
        "db",
        "middleware",
        "config",
        "tools",
        "tests",
    }

    STOP_WORDS = {
        "about",
        "after",
        "again",
        "also",
        "analyze",
        "and",
        "application",
        "are",
        "architecture",
        "can",
        "code",
        "describe",
        "does",
        "explain",
        "for",
        "from",
        "how",
        "identify",
        "into",
        "its",
        "make",
        "me",
        "of",
        "project",
        "show",
        "that",
        "the",
        "their",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }

    def __init__(
        self,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_selected_files: int = MAX_SELECTED_FILES,
    ) -> None:

        if max_context_chars <= 0:
            raise ValueError(
                "Context character limit must be greater than zero."
            )

        if max_selected_files <= 0:
            raise ValueError(
                "Maximum selected files must be greater than zero."
            )

        self.max_context_chars = max_context_chars
        self.max_selected_files = max_selected_files

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        """Extract normalized words from text."""

        return {
            token.lower()
            for token in re.findall(
                r"[a-zA-Z][a-zA-Z0-9_]{1,}",
                value,
            )
        }

    def _question_keywords(self, query: str) -> set[str]:
        """Extract meaningful keywords from the user's question."""

        return (
            self._tokenize(query)
            - self.STOP_WORDS
        )

    def _base_priority(self, relative_path: str) -> int:
        """Assign priority based on the file's architectural role."""

        normalized_path = relative_path.lower()
        filename = normalized_path.rsplit("/", 1)[-1]

        score = 0

        if filename in self.DOCUMENTATION_FILES:
            score += 100

        if filename in self.DEPENDENCY_FILES:
            score += 90

        if filename in self.CONFIGURATION_FILES:
            score += 80

        if filename in self.ENTRY_POINT_FILES:
            score += 110

        path_parts = normalized_path.split("/")

        if any(
            part in self.IMPORTANT_DIRECTORY_NAMES
            for part in path_parts[:-1]
        ):
            score += 20

        if filename.startswith("test_") or filename.endswith("_test.py"):
            score += 5

        return score

    def _relevance_score(
        self,
        relative_path: str,
        content: str,
        question_keywords: set[str],
    ) -> int:
        """
        Calculate a deterministic relevance score.

        File path matches receive more weight than content matches.
        """

        score = self._base_priority(relative_path)

        if not question_keywords:
            return score

        path_keywords = self._tokenize(relative_path)
        content_keywords = self._tokenize(content)

        path_matches = path_keywords & question_keywords
        content_matches = content_keywords & question_keywords

        score += len(path_matches) * 15
        score += min(len(content_matches), 20) * 2

        return score

    def select_files(
        self,
        files: list[dict[str, Any]],
        query: str,
    ) -> ProjectContext:
        """
        Rank files and select those that fit the context budget.

        Args:
            files: Dictionaries containing path and content.
            query: User's analysis request or technical question.

        Returns:
            ProjectContext containing selected and omitted files.
        """

        question_keywords = self._question_keywords(query)

        ranked_files = []

        for file_info in files:
            relative_path = file_info.get("path")
            content = file_info.get("content")

            if not isinstance(relative_path, str):
                continue

            if not isinstance(content, str):
                continue

            if not content.strip():
                continue

            score = self._relevance_score(
                relative_path,
                content,
                question_keywords,
            )

            ranked_files.append(
                {
                    "path": relative_path,
                    "content": content,
                    "score": score,
                }
            )

        # Higher relevance first.
        # Alphabetical path ordering makes ties deterministic.
        ranked_files.sort(
            key=lambda item: (
                -item["score"],
                item["path"].lower(),
            )
        )

        selected_files = []
        omitted_files = []
        total_characters = 0

        for file_info in ranked_files:

            relative_path = file_info["path"]
            content = file_info["content"]

            if len(selected_files) >= self.max_selected_files:
                omitted_files.append(
                    f"{relative_path}: maximum selected-file limit reached"
                )
                continue

            remaining_chars = (
                self.max_context_chars - total_characters
            )

            if remaining_chars <= 0:
                omitted_files.append(
                    f"{relative_path}: context character budget exhausted"
                )
                continue

            # Account for file labels and separators in the budget.
            header = f"\n\nFILE: {relative_path}\n"
            available_content_chars = remaining_chars - len(header)

            if available_content_chars <= 0:
                omitted_files.append(
                    f"{relative_path}: insufficient context space"
                )
                continue

            selected_content = content

            if len(content) > available_content_chars:
                selected_content = content[:available_content_chars]

                omitted_files.append(
                    f"{relative_path}: content truncated"
                )

            selected_files.append(
                {
                    "path": relative_path,
                    "content": selected_content,
                }
            )

            total_characters += (
                len(header) + len(selected_content)
            )

        return ProjectContext(
            selected_files=selected_files,
            omitted_files=omitted_files,
            total_characters=total_characters,
        )