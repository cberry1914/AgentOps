
"""
AgentOps - Security Finding Parser Tests

Tests JSON extraction, finding validation, normalization,
warnings, limitations, and parser boundaries.
"""

import json

import pytest

from agentops.models.security_review import SecurityFinding
from agentops.tools.security_finding_parser import (
    ParsedSecurityReport,
    SecurityFindingParser,
)


@pytest.fixture
def parser():
    return SecurityFindingParser()


def valid_finding(**overrides):
    finding = {
        "severity": "HIGH",
        "category": "Authentication",
        "title": "Weak authentication configuration",
        "description": "Authentication configuration may be weak.",
        "file_path": "src/auth.py",
        "line_number": 12,
        "evidence": "authentication_enabled = False",
        "recommendation": "Review authentication configuration.",
        "confidence": "MEDIUM",
    }
    finding.update(overrides)
    return finding


def valid_payload(**overrides):
    payload = {
        "summary": "Security review completed.",
        "findings": [valid_finding()],
        "limitations": ["Dynamic analysis was not performed."],
    }
    payload.update(overrides)
    return payload


# Successful parsing and JSON extraction


def test_parse_valid_json_object(parser):
    report = parser.parse(json.dumps(valid_payload()))

    assert report.parsing_successful is True
    assert report.finding_count == 1
    assert isinstance(report.findings[0], SecurityFinding)


def test_parse_json_array_as_findings(parser):
    report = parser.parse(json.dumps([valid_finding()]))

    assert report.parsing_successful is True
    assert report.finding_count == 1


def test_parse_json_inside_markdown_fence(parser):
    response = (
        "```json\n"
        + json.dumps(valid_payload())
        + "\n```"
    )

    report = parser.parse(response)

    assert report.parsing_successful is True
    assert report.finding_count == 1


def test_parse_json_with_surrounding_explanation(parser):
    response = (
        "Here is the security review:\n"
        + json.dumps(valid_payload())
        + "\nReview complete."
    )

    report = parser.parse(response)

    assert report.parsing_successful is True
    assert report.finding_count == 1


def test_parse_json_with_braces_inside_string(parser):
    payload = valid_payload(
        summary='Review found a configuration value: {"enabled": false}'
    )

    report = parser.parse(json.dumps(payload))

    assert report.parsing_successful is True
    assert report.summary == payload["summary"]


def test_parse_json_with_escaped_quotes(parser):
    finding = valid_finding(
        description='The application returned "access denied".'
    )

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert report.parsing_successful is True
    assert report.findings[0].description == finding["description"]


# Invalid and incomplete responses


@pytest.mark.parametrize(
    "response",
    ["", "   ", None, 123],
)
def test_empty_or_non_string_response_returns_warning(parser, response):
    report = parser.parse(response)

    assert report.parsing_successful is False
    assert report.finding_count == 0
    assert report.warnings


def test_non_json_response_returns_warning(parser):
    report = parser.parse("The security review is complete.")

    assert report.parsing_successful is False
    assert any("No valid JSON" in warning for warning in report.warnings)


def test_malformed_json_returns_warning(parser):
    report = parser.parse('{"findings": [}')

    assert report.parsing_successful is False
    assert report.warnings
    assert any(
        "No valid JSON" in warning
        for warning in report.warnings
    )


def test_json_scalar_is_rejected(parser):
    report = parser.parse(json.dumps(123))

    assert report.parsing_successful is False
    assert any("must be a JSON object" in warning for warning in report.warnings)


def test_missing_findings_list_returns_unsuccessful_report(parser):
    report = parser.parse(json.dumps({"summary": "No findings field."}))

    assert report.parsing_successful is False
    assert report.finding_count == 0
    assert any("findings list" in warning for warning in report.warnings)


def test_invalid_finding_items_are_skipped(parser):
    payload = valid_payload(
        findings=[
            valid_finding(),
            "not an object",
            valid_finding(severity="EXTREME"),
        ]
    )

    report = parser.parse(json.dumps(payload))

    assert report.parsing_successful is True
    assert report.finding_count == 1
    assert len(report.warnings) == 2


def test_malformed_finding_does_not_discard_valid_findings(parser):
    payload = valid_payload(
        findings=[
            {"severity": "HIGH"},
            valid_finding(),
        ]
    )

    report = parser.parse(json.dumps(payload))

    assert report.parsing_successful is True
    assert report.finding_count == 1
    assert report.warnings


# Field normalization and validation


def test_severity_is_normalized(parser):
    report = parser.parse(
        json.dumps(
            valid_payload(
                findings=[valid_finding(severity=" high ")]
            )
        )
    )

    assert report.findings[0].severity == "HIGH"


def test_confidence_is_normalized(parser):
    report = parser.parse(
        json.dumps(
            valid_payload(
                findings=[valid_finding(confidence=" low ")]
            )
        )
    )

    assert report.findings[0].confidence == "LOW"


def test_invalid_confidence_defaults_to_low(parser):
    report = parser.parse(
        json.dumps(
            valid_payload(
                findings=[valid_finding(confidence="CERTAIN")]
            )
        )
    )

    assert report.findings[0].confidence == "LOW"


def test_missing_confidence_defaults_to_medium(parser):
    finding = valid_finding()
    finding.pop("confidence")

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert report.findings[0].confidence == "MEDIUM"


def test_ai_source_is_assigned(parser):
    report = parser.parse(
        json.dumps(
            valid_payload(
                findings=[valid_finding(source="static_analysis")]
            )
        )
    )

    assert report.findings[0].source == "ai_review"


@pytest.mark.parametrize(
    "field_name",
    ["category", "title", "description"],
)
def test_missing_required_finding_fields_are_rejected(parser, field_name):
    finding = valid_finding()
    finding.pop(field_name)

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert report.finding_count == 0
    assert report.warnings


@pytest.mark.parametrize(
    "field_name",
    ["file_path", "evidence", "recommendation"],
)
def test_non_string_optional_fields_are_rejected(parser, field_name):
    finding = valid_finding()
    finding[field_name] = 123

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert report.finding_count == 0
    assert report.warnings


def test_non_integer_line_number_is_rejected(parser):
    finding = valid_finding(line_number="12")

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert report.finding_count == 0
    assert report.warnings


def test_blank_required_text_is_rejected(parser):
    finding = valid_finding(title="   ")

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert report.finding_count == 0
    assert report.warnings


# Limitations and summaries


def test_limitations_are_trimmed(parser):
    report = parser.parse(
        json.dumps(
            valid_payload(
                limitations=["  Dynamic testing was not performed.  "]
            )
        )
    )

    assert report.limitations == [
        "Dynamic testing was not performed."
    ]


def test_invalid_limitations_are_skipped_with_warning(parser):
    report = parser.parse(
        json.dumps(
            valid_payload(
                limitations=[
                    "Valid limitation",
                    123,
                    "",
                    None,
                ]
            )
        )
    )

    assert report.limitations == ["Valid limitation"]
    assert len(report.warnings) == 3


def test_non_list_limitations_returns_warning(parser):
    report = parser.parse(
        json.dumps(valid_payload(limitations="not a list"))
    )

    assert report.parsing_successful is True
    assert any("must be a list" in warning for warning in report.warnings)


def test_non_string_summary_returns_warning(parser):
    report = parser.parse(
        json.dumps(valid_payload(summary=123))
    )

    assert report.parsing_successful is True
    assert report.summary == ""
    assert any("summary was not a string" in warning for warning in report.warnings)


# Size limits


def test_finding_limit_is_enforced(parser):
    findings = [
        valid_finding(title=f"Finding {index}")
        for index in range(parser.MAX_FINDINGS + 5)
    ]

    report = parser.parse(
        json.dumps(valid_payload(findings=findings))
    )

    assert report.finding_count == parser.MAX_FINDINGS
    assert any("exceeded the limit" in warning for warning in report.warnings)


def test_limitation_limit_is_enforced(parser):
    limitations = [
        f"Limitation {index}"
        for index in range(parser.MAX_LIMITATIONS + 5)
    ]

    report = parser.parse(
        json.dumps(valid_payload(limitations=limitations))
    )

    assert len(report.limitations) == parser.MAX_LIMITATIONS


def test_summary_length_is_limited(parser):
    summary = "S" * (parser.MAX_TEXT_LENGTH + 100)

    report = parser.parse(
        json.dumps(valid_payload(summary=summary))
    )

    assert len(report.summary) == parser.MAX_TEXT_LENGTH


def test_finding_text_length_is_limited(parser):
    finding = valid_finding(
        description="D" * (parser.MAX_TEXT_LENGTH + 100)
    )

    report = parser.parse(
        json.dumps(valid_payload(findings=[finding]))
    )

    assert len(report.findings[0].description) == parser.MAX_TEXT_LENGTH


# Result serialization


def test_parsed_report_serialization(parser):
    report = parser.parse(json.dumps(valid_payload()))

    serialized = report.to_dict()

    assert serialized["finding_count"] == 1
    assert serialized["parsing_successful"] is True
    assert isinstance(serialized["findings"][0], dict)
    assert serialized["limitations"] == report.limitations
    assert serialized["warnings"] == report.warnings


def test_parsed_report_default_values():
    report = ParsedSecurityReport()

    assert report.findings == []
    assert report.limitations == []
    assert report.warnings == []
    assert report.summary == ""
    assert report.parsing_successful is False
    assert report.finding_count == 0
