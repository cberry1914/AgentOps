
"""
AgentOps - Security Finding Parser

Parses the Security Agent's JSON response into validated
SecurityFinding objects.

Responsibilities:
- Extract JSON from model responses.
- Validate finding structures.
- Normalize supported finding formats.
- Reject malformed or invalid findings.
- Preserve review limitations.
- Report parsing warnings.

The parser does not determine whether a finding is
a verified vulnerability. Security policy handles
review decisions separately.
"""

import json
import re

from dataclasses import dataclass, field
from typing import Any

from agentops.models.security_review import SecurityFinding


@dataclass
class ParsedSecurityReport:
    """Structured result produced by the security report parser."""

    findings: list[SecurityFinding] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    parsing_successful: bool = False

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "summary": self.summary,
            "parsing_successful": self.parsing_successful,
            "finding_count": self.finding_count,
        }


class SecurityFindingParser:
    """Converts Security Agent output into structured findings."""

    MAX_FINDINGS = 100
    MAX_LIMITATIONS = 50
    MAX_TEXT_LENGTH = 5000

    VALID_SEVERITIES = {
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "INFO",
    }

    VALID_CONFIDENCE = {
        "HIGH",
        "MEDIUM",
        "LOW",
    }

    def parse(self, response: str) -> ParsedSecurityReport:
        """
        Parse a Security Agent response.

        Expected JSON format:

        {
            "summary": "...",
            "findings": [
                {
                    "severity": "HIGH",
                    "category": "Authentication",
                    "title": "...",
                    "description": "...",
                    "file_path": "src/example.py",
                    "line_number": 12,
                    "evidence": "...",
                    "recommendation": "...",
                    "confidence": "MEDIUM"
                }
            ],
            "limitations": []
        }
        """

        result = ParsedSecurityReport()

        if not isinstance(response, str) or not response.strip():
            result.warnings.append(
                "Security Agent returned an empty response."
            )
            return result

        json_text = self._extract_json(response)

        if json_text is None:
            result.warnings.append(
                "No valid JSON object or array could be extracted "
                "from the Security Agent response."
            )
            return result

        try:
            payload = json.loads(json_text)

        except json.JSONDecodeError as exc:
            result.warnings.append(
                f"Security Agent returned invalid JSON: {exc.msg}."
            )
            return result

        if isinstance(payload, list):
            payload = {"findings": payload}

        if not isinstance(payload, dict):
            result.warnings.append(
                "Security Agent response must be a JSON object."
            )
            return result

        summary = payload.get("summary", "")

        if isinstance(summary, str):
            result.summary = summary[:self.MAX_TEXT_LENGTH]
        else:
            result.warnings.append(
                "Security report summary was not a string."
            )

        raw_limitations = payload.get("limitations", [])

        if isinstance(raw_limitations, list):
            for limitation in raw_limitations[:self.MAX_LIMITATIONS]:

                if isinstance(limitation, str) and limitation.strip():
                    result.limitations.append(
                        limitation.strip()[:self.MAX_TEXT_LENGTH]
                    )

                else:
                    result.warnings.append(
                        "An invalid security limitation was ignored."
                    )

        elif raw_limitations is not None:
            result.warnings.append(
                "Security report limitations must be a list."
            )

        raw_findings = payload.get("findings")

        if not isinstance(raw_findings, list):
            result.warnings.append(
                "Security report did not contain a valid findings list."
            )
            return result

        if len(raw_findings) > self.MAX_FINDINGS:
            result.warnings.append(
                f"Security findings exceeded the limit of "
                f"{self.MAX_FINDINGS}. Additional findings were ignored."
            )

        for index, raw_finding in enumerate(
            raw_findings[:self.MAX_FINDINGS]
        ):

            if not isinstance(raw_finding, dict):
                result.warnings.append(
                    f"Finding {index + 1} was not a JSON object."
                )
                continue

            try:
                finding_data = self._normalize_finding(raw_finding)

                finding = SecurityFinding.from_dict(finding_data)

                result.findings.append(finding)

            except (TypeError, ValueError, KeyError) as exc:
                result.warnings.append(
                    f"Finding {index + 1} was rejected: {exc}"
                )

        result.parsing_successful = True

        return result

    def _extract_json(self, response: str) -> str | None:
        """Extract a JSON object or array from model output."""

        cleaned = response.strip()

        # Remove common Markdown JSON fences.
        cleaned = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```\s*$",
            "",
            cleaned,
        )

        try:
            json.loads(cleaned)
            return cleaned

        except json.JSONDecodeError:
            pass

        # Locate the first JSON object or array.
        start_positions = [
            position
            for position in (
                cleaned.find("{"),
                cleaned.find("["),
            )
            if position >= 0
        ]

        if not start_positions:
            return None

        start = min(start_positions)

        opening_character = cleaned[start]

        closing_character = (
            "}" if opening_character == "{" else "]"
        )

        depth = 0
        inside_string = False
        escaped = False

        for position in range(start, len(cleaned)):
            character = cleaned[position]

            if inside_string:

                if escaped:
                    escaped = False

                elif character == "\\":
                    escaped = True

                elif character == '"':
                    inside_string = False

                continue

            if character == '"':
                inside_string = True

            elif character == opening_character:
                depth += 1

            elif character == closing_character:
                depth -= 1

                if depth == 0:
                    candidate = cleaned[start:position + 1]

                    try:
                        json.loads(candidate)
                        return candidate

                    except json.JSONDecodeError:
                        return None

        return None

    def _normalize_finding(
        self,
        raw_finding: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize supported fields before model validation."""

        finding = dict(raw_finding)

        severity = finding.get("severity")

        if isinstance(severity, str):
            finding["severity"] = severity.strip().upper()

        confidence = finding.get("confidence", "MEDIUM")

        if isinstance(confidence, str):
            confidence = confidence.strip().upper()

        if confidence not in self.VALID_CONFIDENCE:
            confidence = "LOW"

        finding["confidence"] = confidence

        # AI findings must retain their source identity.
        # Static scanner findings use static_analysis instead.
        finding["source"] = "ai_review"

        # Normalize optional text fields.
        for field_name in (
            "file_path",
            "evidence",
            "recommendation",
        ):
            value = finding.get(field_name)

            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{field_name} must be a string or null."
                )

            if isinstance(value, str):
                finding[field_name] = (
                    value[:self.MAX_TEXT_LENGTH]
                )

        # Required text fields must be strings.
        for field_name in (
            "category",
            "title",
            "description",
        ):
            value = finding.get(field_name)

            if not isinstance(value, str):
                raise ValueError(
                    f"{field_name} must be a string."
                )

            finding[field_name] = (
                value.strip()[:self.MAX_TEXT_LENGTH]
            )

        if finding.get("severity") not in self.VALID_SEVERITIES:
            raise ValueError(
                "Finding contains an unsupported severity."
            )

        return finding
