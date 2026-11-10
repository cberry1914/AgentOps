
"""
AgentOps - Security Scanner

Performs deterministic, local security checks against
project source files.

Initial capabilities:
- Detects common hardcoded credentials.
- Detects AWS access key identifiers.
- Detects common GitHub and Slack tokens.
- Detects inline private key markers.
- Redacts detected secret values from evidence.
- Produces structured SecurityFinding objects.

Security boundaries:
- Does not execute project code.
- Does not modify project files.
- Does not transmit source code or secrets.
- Does not guarantee detection of every secret.
- Findings require contextual validation.
"""

import re

from typing import Optional

from agentops.models.security_review import SecurityFinding


class SecurityScanner:
    """Performs deterministic security checks on source files."""

    MAX_FILES = 1000
    MAX_FINDINGS = 500
    MAX_FILE_SIZE_CHARS = 1_000_000

    PLACEHOLDER_VALUES = {
        "",
        "password",
        "changeme",
        "change_me",
        "your_password",
        "your_api_key",
        "your_token",
        "example",
        "placeholder",
        "none",
        "null",
        "undefined",
        "xxxxx",
        "xxxxxx",
        "replace_me",
        "replace_with_secret",
    }

    SECRET_PATTERNS = [
        (
            "AWS Access Key",
            "HIGH",
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        ),
        (
            "GitHub Token",
            "HIGH",
            re.compile(
                r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"
            ),
        ),
        (
            "Slack Token",
            "HIGH",
            re.compile(
                r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
            ),
        ),
        (
            "Private Key Material",
            "CRITICAL",
            re.compile(
                r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?"
                r"PRIVATE KEY-----"
            ),
        ),
    ]

    ASSIGNMENT_PATTERN = re.compile(
        r"""
        (?ix)
        \b
        (?P<name>
            password
            |passwd
            |pwd
            |secret
            |api[_-]?key
            |access[_-]?token
            |auth[_-]?token
            |client[_-]?secret
            |private[_-]?key
            |db[_-]?password
        )
        \b
        \s*
        (?:
            =
            | :
        )
        \s*
        (?P<quote>["'])
        (?P<value>[^"'\r\n]{4,})
        (?P=quote)
        """
    )

    def __init__(self):
        pass

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        """Identify common example and placeholder values."""

        normalized = value.strip().lower()

        if normalized in SecurityScanner.PLACEHOLDER_VALUES:
            return True

        if normalized.startswith("${") and normalized.endswith("}"):
            return True

        if normalized.startswith("os.environ"):
            return True

        if normalized.startswith("getenv("):
            return True

        if normalized.startswith("env("):
            return True

        if normalized.startswith("<") and normalized.endswith(">"):
            return True

        return False

    @staticmethod
    def _redact_line(
        line: str,
        secret: str,
    ) -> str:
        """Replace a detected secret with a redaction marker."""

        if not secret:
            return line

        return line.replace(secret, "[REDACTED]")

    def _create_finding(
        self,
        *,
        relative_path: str,
        line_number: int,
        severity: str,
        category: str,
        title: str,
        description: str,
        line: str,
        secret: str,
        recommendation: str,
        confidence: str = "HIGH",
    ) -> SecurityFinding:
        """Create a finding without exposing the detected value."""

        redacted_evidence = self._redact_line(
            line.strip(),
            secret,
        )

        return SecurityFinding(
            severity=severity,
            category=category,
            title=title,
            description=description,
            file_path=relative_path,
            line_number=line_number,
            evidence=redacted_evidence[:500],
            recommendation=recommendation,
            confidence=confidence,
            source="static_analysis",
        )

    def _scan_known_patterns(
        self,
        relative_path: str,
        line_number: int,
        line: str,
    ) -> list[SecurityFinding]:
        """Detect known credential and private-key patterns."""

        findings = []

        for category, severity, pattern in self.SECRET_PATTERNS:
            for match in pattern.finditer(line):

                secret = match.group(0)

                findings.append(
                    self._create_finding(
                        relative_path=relative_path,
                        line_number=line_number,
                        severity=severity,
                        category=category,
                        title=f"Potential {category} exposure",
                        description=(
                            f"The source contains text matching a known "
                            f"{category.lower()} pattern. If this is a "
                            f"real credential, it may expose access to "
                            f"an external service or protected system."
                        ),
                        line=line,
                        secret=secret,
                        recommendation=(
                            "Remove the credential from source code. "
                            "Store secrets in an approved secret manager "
                            "or protected environment configuration. "
                            "Rotate the credential if it was exposed."
                        ),
                    )
                )

        return findings

    def _scan_assignments(
        self,
        relative_path: str,
        line_number: int,
        line: str,
    ) -> list[SecurityFinding]:
        """Detect potentially hardcoded sensitive assignments."""

        findings = []

        for match in self.ASSIGNMENT_PATTERN.finditer(line):

            variable_name = match.group("name")
            secret = match.group("value")

            if self._is_placeholder(secret):
                continue

            # Avoid reporting obvious environment-variable references.
            if secret.strip().startswith("$"):
                continue

            findings.append(
                self._create_finding(
                    relative_path=relative_path,
                    line_number=line_number,
                    severity="HIGH",
                    category="Hardcoded Credential",
                    title="Potential hardcoded credential",
                    description=(
                        f"A value is assigned directly to a variable "
                        f"named '{variable_name}'. This may expose a "
                        f"password, token, key, or other sensitive value "
                        f"if the source is shared or committed."
                    ),
                    line=line,
                    secret=secret,
                    recommendation=(
                        "Replace hardcoded credentials with references "
                        "to a protected secret manager or environment "
                        "configuration. Rotate the credential if it "
                        "has already been exposed."
                    ),
                    confidence="MEDIUM",
                )
            )

        return findings

    def scan_files(
        self,
        files: dict[str, str],
        cancel_event: Optional[object] = None,
    ) -> list[SecurityFinding]:
        """
        Scan a collection of project files.

        Args:
            files:
                Dictionary mapping relative file paths to text content.
            cancel_event:
                Optional cancellation event.

        Returns:
            A list of structured security findings.

        The scanner never prints or returns the original detected
        credential values.
        """

        if not isinstance(files, dict):
            raise TypeError("Files must be provided as a dictionary.")

        findings = []
        seen_findings = set()

        for file_index, (relative_path, content) in enumerate(
            files.items()
        ):

            if file_index >= self.MAX_FILES:
                break

            if cancel_event is not None and cancel_event.is_set():
                break

            if not isinstance(relative_path, str):
                continue

            if not isinstance(content, str):
                continue

            if len(content) > self.MAX_FILE_SIZE_CHARS:
                content = content[:self.MAX_FILE_SIZE_CHARS]

            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            ):

                if cancel_event is not None and cancel_event.is_set():
                    return findings

                line_findings = self._scan_known_patterns(
                    relative_path,
                    line_number,
                    line,
                )

                line_findings.extend(
                    self._scan_assignments(
                        relative_path,
                        line_number,
                        line,
                    )
                )

                for finding in line_findings:

                    # Deduplicate overlapping detections.
                    identity = (
                        finding.file_path,
                        finding.line_number,
                        finding.category,
                        finding.title,
                    )

                    if identity in seen_findings:
                        continue

                    seen_findings.add(identity)
                    findings.append(finding)

                    if len(findings) >= self.MAX_FINDINGS:
                        return findings

        return findings
