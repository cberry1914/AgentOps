
"""
AgentOps - Security Report Formatter Tests

Tests report structure, finding details, optional fields,
limitations, metadata, and safe formatting.
"""

import pytest

from agentops.models.security_review import (
    SecurityFinding,
    SecurityReview,
)
from agentops.tools.security_report_formatter import (
    SecurityReportFormatter,
)


@pytest.fixture
def formatter():
    return SecurityReportFormatter()


def make_finding(**overrides):
    values = {
        "severity": "HIGH",
        "category": "Authentication",
        "title": "Weak authentication configuration",
        "description": "Authentication settings may be insecure.",
        "file_path": "src/auth.py",
        "line_number": 25,
        "evidence": "authentication_enabled = False",
        "recommendation": "Review authentication configuration.",
        "confidence": "HIGH",
        "source": "static_analysis",
    }
    values.update(overrides)
    return SecurityFinding(**values)


def make_review(**overrides):
    values = {
        "decision": "NEEDS_REVIEW",
        "status": "COMPLETED",
        "summary": "Security review completed.",
        "findings": [make_finding()],
        "files_scanned": 10,
        "files_skipped": 2,
        "static_analysis_completed": True,
        "ai_review_completed": True,
        "requires_human_review": True,
        "limitations": ["Dynamic analysis was not performed."],
        "metadata": {
            "critical_findings": 0,
            "high_findings": 1,
            "medium_findings": 0,
            "low_findings": 0,
            "informational_findings": 0,
            "scanner_version": "1.0",
        },
    }
    values.update(overrides)
    return SecurityReview(**values)


# Input validation


def test_formatter_rejects_non_security_review(formatter):
    with pytest.raises(TypeError):
        formatter.format({"decision": "PASS"})


# Header tests


def test_report_contains_header_and_decision(formatter):
    report = formatter.format(make_review())

    assert "AGENTOPS SECURITY REVIEW" in report
    assert "Decision: NEEDS_REVIEW" in report
    assert "Status: COMPLETED" in report


def test_report_contains_review_timestamp(formatter):
    review = make_review(reviewed_at="2026-11-10T12:00:00+00:00")

    report = formatter.format(review)

    assert "Reviewed at: 2026-11-10T12:00:00+00:00" in report


def test_report_contains_summary(formatter):
    report = formatter.format(make_review())

    assert "Security review completed." in report


def test_report_formats_human_approval_required(formatter):
    report = formatter.format(
        make_review(requires_human_review=True)
    )

    assert "Human approval required: Yes" in report


def test_report_formats_human_approval_not_required(formatter):
    report = formatter.format(
        make_review(
            decision="PASS",
            findings=[],
            static_analysis_completed=True,
            requires_human_review=False,
        )
    )

    assert "Human approval required: No" in report


# Scan summary tests


def test_report_contains_scan_statistics(formatter):
    report = formatter.format(make_review())

    assert "Files scanned: 10" in report
    assert "Files skipped: 2" in report


def test_report_contains_analysis_completion_status(formatter):
    report = formatter.format(make_review())

    assert "Static analysis completed: Yes" in report
    assert "AI review completed: Yes" in report


def test_report_formats_incomplete_analysis(formatter):
    report = formatter.format(
        make_review(
            static_analysis_completed=False,
            ai_review_completed=False,
        )
    )

    assert "Static analysis completed: No" in report
    assert "AI review completed: No" in report


def test_report_contains_severity_counts(formatter):
    report = formatter.format(make_review())

    assert "Critical: 0" in report
    assert "High: 1" in report
    assert "Medium: 0" in report
    assert "Low: 0" in report
    assert "Informational: 0" in report
    assert "Total: 1" in report


def test_report_handles_missing_severity_metadata(formatter):
    review = make_review(metadata={})

    report = formatter.format(review)

    assert "Critical: 0" in report
    assert "High: 0" in report
    assert "Total: 1" in report


# Finding formatting tests


def test_report_contains_finding_details(formatter):
    report = formatter.format(make_review())

    assert "[HIGH] Weak authentication configuration" in report
    assert "Category: Authentication" in report
    assert "Confidence: HIGH" in report
    assert "Source: static_analysis" in report
    assert "Authentication settings may be insecure." in report


def test_report_contains_file_location_and_line(formatter):
    report = formatter.format(make_review())

    assert "Location: src/auth.py:25" in report


def test_report_contains_file_location_without_line(formatter):
    finding = make_finding(line_number=None)

    report = formatter.format(
        make_review(findings=[finding])
    )

    assert "Location: src/auth.py" in report


def test_report_omits_location_when_file_path_missing(formatter):
    finding = make_finding(
        file_path=None,
        line_number=None,
    )

    report = formatter.format(
        make_review(findings=[finding])
    )

    assert "Location:" not in report


def test_report_contains_evidence_and_recommendation(formatter):
    report = formatter.format(make_review())

    assert "Evidence: authentication_enabled = False" in report
    assert "Recommendation: Review authentication configuration." in report


def test_report_omits_optional_finding_details(formatter):
    finding = make_finding(
        file_path=None,
        line_number=None,
        evidence=None,
        recommendation=None,
    )

    report = formatter.format(
        make_review(findings=[finding])
    )

    assert "Evidence:" not in report
    assert "Recommendation:" not in report
    assert "Location:" not in report


def test_report_numbers_multiple_findings(formatter):
    findings = [
        make_finding(title="First finding"),
        make_finding(
            severity="MEDIUM",
            title="Second finding",
        ),
    ]

    report = formatter.format(
        make_review(findings=findings)
    )

    assert "1. [HIGH] First finding" in report
    assert "2. [MEDIUM] Second finding" in report


def test_report_handles_no_findings(formatter):
    report = formatter.format(
        make_review(findings=[])
    )

    assert "No security findings were reported." in report


# Limitations tests


def test_report_contains_limitations(formatter):
    report = formatter.format(make_review())

    assert "REVIEW LIMITATIONS" in report
    assert "- Dynamic analysis was not performed." in report


def test_report_handles_no_limitations(formatter):
    report = formatter.format(
        make_review(limitations=[])
    )

    assert "No limitations were reported." in report


# Metadata tests


def test_report_contains_metadata(formatter):
    report = formatter.format(make_review())

    assert "REVIEW METADATA" in report
    assert "scanner_version: 1.0" in report


def test_report_formats_nested_metadata(formatter):
    review = make_review(
        metadata={
            "scanner": {
                "name": "local-scanner",
                "version": "1.0",
            },
            "warnings": ["warning one", "warning two"],
        }
    )

    report = formatter.format(review)

    assert "scanner: {'name': 'local-scanner', 'version': '1.0'}" in report
    assert "warnings: ['warning one', 'warning two']" in report


def test_report_omits_metadata_section_when_empty(formatter):
    report = formatter.format(
        make_review(metadata={})
    )

    assert "REVIEW METADATA" not in report


# Safe count formatting


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, 5),
        ("7", 7),
        (0, 0),
        (-3, 0),
        ("invalid", 0),
        (None, 0),
        (True, 0),
        (False, 0),
        (3.9, 3),
    ],
)
def test_safe_count_formats_values(formatter, value, expected):
    assert formatter._safe_count(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "Yes"),
        (False, "No"),
    ],
)
def test_yes_no_formats_boolean_values(formatter, value, expected):
    assert formatter._yes_no(value) == expected
