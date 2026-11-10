
"""
AgentOps - QA Finding Parser

Converts structured QA agent output into validated
QAFinding objects.

The LLM may identify findings, but it cannot determine
the final QA decision. QAPolicyEngine handles that.
"""

import json
import re
from typing import Any

from agentops.models.qa_review import QAFinding


class QAFindingParser:
    """Parse and validate findings returned by the QA agent."""

    ALLOWED_SEVERITIES = {
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "INFO",
    }

    MAX_FINDINGS = 100
    MAX_TEXT_LENGTH = 5000
    MAX_PATH_LENGTH = 500

    def parse(
        self,
        response: str,
    ) -> tuple[list[QAFinding], list[str]]:
        """
        Parse QA findings from a JSON response.

        Returns:
            A tuple containing:
            - Validated findings.
            - Parsing warnings.
        """

        warnings: list[str] = []

        if not response or not response.strip():
            return [], ["The QA agent returned an empty response."]

        payload = self._extract_json(response)

        if payload is None:
            return [], [
                "The QA agent response did not contain valid JSON."
            ]

        if isinstance(payload, dict):
            raw_findings = payload.get("findings", [])
        elif isinstance(payload, list):
            raw_findings = payload
        else:
            return [], [
                "The QA findings must be a JSON object or array."
            ]

        if not isinstance(raw_findings, list):
            return [], [
                "The findings property must contain a JSON array."
            ]

        if len(raw_findings) > self.MAX_FINDINGS:
            warnings.append(
                f"Finding list exceeded the limit of "
                f"{self.MAX_FINDINGS}; additional entries were ignored."
            )
            raw_findings = raw_findings[:self.MAX_FINDINGS]

        findings: list[QAFinding] = []

        for index, item in enumerate(raw_findings):
            finding, warning = self._parse_finding(
                item,
                index,
            )

            if warning:
                warnings.append(warning)

            if finding is not None:
                findings.append(finding)

        return findings, warnings

    def _extract_json(
        self,
        response: str,
    ) -> Any:
        """Extract JSON from plain or Markdown-fenced output."""

        cleaned = response.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()

        for index, character in enumerate(cleaned):
            if character not in "[{":
                continue

            try:
                payload, _ = decoder.raw_decode(
                    cleaned[index:]
                )
                return payload
            except json.JSONDecodeError:
                continue

        return None

    def _parse_finding(
        self,
        item: Any,
        index: int,
    ) -> tuple[QAFinding | None, str | None]:
        """Validate and construct one finding."""

        prefix = f"Finding {index + 1}"

        if not isinstance(item, dict):
            return None, (
                f"{prefix} was ignored because it was not an object."
            )

        severity = self._clean_text(
            item.get("severity", "INFO"),
            50,
        ).upper()

        if severity not in self.ALLOWED_SEVERITIES:
            return None, (
                f"{prefix} was ignored because its severity "
                f"was invalid: {severity!r}."
            )

        category = self._clean_text(
            item.get("category", "General"),
            100,
        )

        title = self._clean_text(
            item.get("title", ""),
            250,
        )

        description = self._clean_text(
            item.get("description", ""),
            self.MAX_TEXT_LENGTH,
        )

        if not title:
            return None, (
                f"{prefix} was ignored because it had no title."
            )

        if not description:
            return None, (
                f"{prefix} was ignored because it had no description."
            )

        file_path = self._clean_text(
            item.get("file_path", ""),
            self.MAX_PATH_LENGTH,
        )

        line_number = item.get("line_number")

        if isinstance(line_number, bool):
            line_number = None

        if not isinstance(line_number, int) or line_number < 1:
            line_number = None

        recommendation = self._clean_text(
            item.get("recommendation", ""),
            self.MAX_TEXT_LENGTH,
        )

        try:
            finding = QAFinding(
                severity=severity,
                category=category or "General",
                title=title,
                description=description,
                file_path=file_path or None,
                line_number=line_number,
                recommendation=recommendation or None,
            )
        except (TypeError, ValueError) as exc:
            return None, (
                f"{prefix} could not be constructed: {exc}"
            )

        return finding, None

    @staticmethod
    def _clean_text(
        value: Any,
        max_length: int,
    ) -> str:
        """Normalize untrusted text fields."""

        if not isinstance(value, str):
            return ""

        cleaned = value.strip()

        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]

        return cleaned
