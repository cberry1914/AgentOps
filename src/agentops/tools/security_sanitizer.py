
"""
AgentOps - Security Content Sanitizer

Removes sensitive credential material from project content
before that content is included in an AI model prompt.

Design:
- Original content remains available to local security scanning.
- Sanitized content is intended for AI review.
- Credential values are replaced with redaction markers.
- Private key blocks are removed as a whole.
- File paths are preserved.
- No network access or code execution.

This is a defensive preprocessing layer, not a guarantee
that every possible secret format will be detected.
"""

import re

from dataclasses import dataclass, field
from typing import Optional

from agentops.tools.cancellation import check_cancelled


@dataclass
class SanitizationResult:
    """Contains sanitized project content and redaction metadata."""

    files: dict[str, str] = field(default_factory=dict)
    files_modified: int = 0
    redactions_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return serializable sanitization metadata."""

        return {
            "files": dict(self.files),
            "files_modified": self.files_modified,
            "redactions_count": self.redactions_count,
            "warnings": list(self.warnings),
        }


class SecurityContentSanitizer:
    """Redacts common secret formats before AI processing."""

    MAX_FILES = 1000
    MAX_FILE_SIZE_CHARS = 1_000_000
    MAX_REDACTIONS_PER_FILE = 1000

    REDACTION_MARKER = "[REDACTED_SECRET]"

    # Supports common PEM private key types, including encrypted keys.
    PRIVATE_KEY_PATTERN = re.compile(
        r"-----BEGIN "
        r"(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?"
        r"PRIVATE KEY-----"
        r".*?"
        r"-----END "
        r"(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?"
        r"PRIVATE KEY-----",
        re.DOTALL,
    )

    # AWS secret access key assignments.
    AWS_SECRET_ASSIGNMENT_PATTERN = re.compile(
        r"""
        (
            ["']?
            (?:aws_secret_access_key|secret_access_key)
            ["']?
            \s*[:=]\s*
            ["']
        )
        ([^"'\r\n]+)
        (["'])
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # Common named credential assignments in source and configuration files.
    CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
        r"""
        (
            \b
            (?:
                password
                |passwd
                |pwd
                |secret
                |api[_-]?key
                |access[_-]?token
                |auth[_-]?token
                |client[_-]?secret
                |db[_-]?password
                |private[_-]?key
                |connection[_-]?string
            )
            \b
            \s*[:=]\s*
            ["']
        )
        ([^"'\r\n]{4,})
        (["'])
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    BEARER_TOKEN_PATTERN = re.compile(
        r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}"
    )

    AWS_ACCESS_KEY_PATTERN = re.compile(
        r"\bAKIA[0-9A-Z]{16}\b"
    )

    GITHUB_TOKEN_PATTERN = re.compile(
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    )

    SLACK_TOKEN_PATTERN = re.compile(
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
    )

    GENERIC_JWT_PATTERN = re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}\b"
    )

    def sanitize_files(
        self,
        files: dict[str, str],
        cancel_event: Optional[object] = None,
    ) -> SanitizationResult:
        """
        Sanitize a mapping of project paths to file contents.

        The input dictionary is not modified.

        Returns:
            A new mapping suitable for AI review.
        """

        if not isinstance(files, dict):
            raise TypeError("files must be a dictionary.")

        result = SanitizationResult()

        if len(files) > self.MAX_FILES:
            result.warnings.append(
                f"File count exceeded the sanitizer limit of "
                f"{self.MAX_FILES}. Additional files were omitted."
            )

        for index, (path, content) in enumerate(files.items()):
            check_cancelled(cancel_event)

            if index >= self.MAX_FILES:
                break

            if not isinstance(path, str) or not isinstance(
                content,
                str,
            ):
                result.warnings.append(
                    "A file with an invalid path or content type "
                    "was omitted."
                )
                continue

            if len(content) > self.MAX_FILE_SIZE_CHARS:
                result.warnings.append(
                    f"{path}: file exceeded the sanitizer size "
                    "limit and was omitted from AI context."
                )
                continue

            sanitized, redactions = self._sanitize_content(
                content,
                cancel_event=cancel_event,
            )

            result.files[path] = sanitized
            result.redactions_count += redactions

            if redactions:
                result.files_modified += 1

        return result

    def _sanitize_content(
        self,
        content: str,
        cancel_event: Optional[object] = None,
    ) -> tuple[str, int]:
        """Redact sensitive material from one file."""

        check_cancelled(cancel_event)

        redactions = 0
        sanitized = content

        # Remove complete private key blocks before other processing.
        sanitized, count = self._substitute(
            self.PRIVATE_KEY_PATTERN,
            sanitized,
            self.REDACTION_MARKER,
            cancel_event,
        )
        redactions += count

        check_cancelled(cancel_event)

        # Preserve assignment names and quotation marks.
        for pattern in (
            self.AWS_SECRET_ASSIGNMENT_PATTERN,
            self.CREDENTIAL_ASSIGNMENT_PATTERN,
        ):
            sanitized, count = self._substitute(
                pattern,
                sanitized,
                lambda match: (
                    match.group(1)
                    + self.REDACTION_MARKER
                    + match.group(3)
                ),
                cancel_event,
            )
            redactions += count

        check_cancelled(cancel_event)

        # Redact common standalone token formats.
        token_patterns = (
            self.BEARER_TOKEN_PATTERN,
            self.AWS_ACCESS_KEY_PATTERN,
            self.GITHUB_TOKEN_PATTERN,
            self.SLACK_TOKEN_PATTERN,
            self.GENERIC_JWT_PATTERN,
        )

        for pattern in token_patterns:
            sanitized, count = self._substitute(
                pattern,
                sanitized,
                self._token_replacement,
                cancel_event,
            )
            redactions += count

            check_cancelled(cancel_event)

        return sanitized, redactions

    def _token_replacement(
        self,
        match: re.Match,
    ) -> str:
        """
        Preserve a useful prefix when the pattern provides one.

        For example, a Bearer authorization scheme remains visible,
        but the associated token is replaced.
        """

        if match.lastindex and match.lastindex >= 1:
            prefix = match.group(1)

            if prefix:
                return prefix + self.REDACTION_MARKER

        return self.REDACTION_MARKER

    def _substitute(
        self,
        pattern: re.Pattern,
        content: str,
        replacement,
        cancel_event: Optional[object] = None,
    ) -> tuple[str, int]:
        """Apply a bounded substitution and return its match count."""

        count = 0

        def replace_match(match: re.Match) -> str:
            nonlocal count

            if count >= self.MAX_REDACTIONS_PER_FILE:
                return match.group(0)

            check_cancelled(cancel_event)

            count += 1

            if callable(replacement):
                return replacement(match)

            return str(replacement)

        sanitized = pattern.sub(
            replace_match,
            content,
        )

        return sanitized, count
