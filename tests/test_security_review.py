
"""
AgentOps - Security Review Model Tests

Tests finding validation, review validation, serialization,
decision constraints, and severity counters.
"""

import pytest

from agentops.models.security_review import (
    SecurityFinding,
    SecurityReview,
)


@pytest.fixture
def finding():
    return SecurityFinding(
        severity="HIGH",
        category="Hardcoded Credential",
        title="Potential hardcoded credential",
        description="A credential may be exposed.",
        file_path="config.py",
        line_number=10,
        evidence='password = "[REDACTED]"',
        recommendation="Use a protected secret manager.",
        confidence="MEDIUM",
        source="static_analysis",
    )


def make_review(**overrides):
    values = {
        "decision": "NEEDS_REVIEW",
        "status": "COMPLETED",
        "summary": "Security review completed.",
    }
    values.update(overrides)
    return SecurityReview(**values)


# SecurityFinding tests


def test_finding_normalizes_severity_and_confidence(finding):
    assert finding.severity == "HIGH"
    assert finding.confidence == "MEDIUM"


def test_finding_accepts_lowercase_severity():
    item = SecurityFinding(
        severity="high",
        category="Credential",
        title="Potential credential",
        description="Potential exposure.",
        confidence="low",
    )

    assert item.severity == "HIGH"
    assert item.confidence == "LOW"


def test_finding_rejects_invalid_severity():
    with pytest.raises(ValueError):
        SecurityFinding(
            severity="EXTREME",
            category="Credential",
            title="Potential credential",
            description="Potential exposure.",
        )


@pytest.mark.parametrize(
    "field_name",
    ["category", "title", "description"],
)
def test_finding_rejects_empty_required_text(field_name):
    values = {
        "severity": "HIGH",
        "category": "Credential",
        "title": "Potential credential",
        "description": "Potential exposure.",
    }
    values[field_name] = " "

    with pytest.raises(ValueError):
        SecurityFinding(**values)


def test_finding_rejects_invalid_line_number():
    with pytest.raises(ValueError):
        SecurityFinding(
            severity="HIGH",
            category="Credential",
            title="Potential credential",
            description="Potential exposure.",
            line_number=0,
        )


def test_finding_rejects_boolean_line_number():
    with pytest.raises(TypeError):
        SecurityFinding(
            severity="HIGH",
            category="Credential",
            title="Potential credential",
            description="Potential exposure.",
            line_number=True,
        )


def test_finding_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        SecurityFinding(
            severity="HIGH",
            category="Credential",
            title="Potential credential",
            description="Potential exposure.",
            confidence="CERTAIN",
        )


def test_finding_rejects_empty_source():
    with pytest.raises(ValueError):
        SecurityFinding(
            severity="HIGH",
            category="Credential",
            title="Potential credential",
            description="Potential exposure.",
            source=" ",
        )


def test_finding_serialization_round_trip(finding):
    serialized = finding.to_dict()
    restored = SecurityFinding.from_dict(serialized)

    assert isinstance(restored, SecurityFinding)
    assert restored.to_dict() == serialized


def test_finding_from_dict_ignores_unknown_fields(finding):
    serialized = finding.to_dict()
    serialized["unexpected_field"] = "ignored"

    restored = SecurityFinding.from_dict(serialized)

    assert not hasattr(restored, "unexpected_field")
    assert restored.title == finding.title


def test_finding_from_dict_rejects_non_dictionary():
    with pytest.raises(TypeError):
        SecurityFinding.from_dict("invalid")


# SecurityReview tests


def test_review_normalizes_decision_and_status():
    review = make_review(
        decision="needs_review",
        status="completed",
    )

    assert review.decision == "NEEDS_REVIEW"
    assert review.status == "COMPLETED"


def test_review_rejects_invalid_decision():
    with pytest.raises(ValueError):
        make_review(decision="APPROVED")


def test_review_rejects_invalid_status():
    with pytest.raises(ValueError):
        make_review(status="IN_PROGRESS")


def test_review_rejects_negative_file_counts():
    with pytest.raises(ValueError):
        make_review(files_scanned=-1)


def test_review_rejects_boolean_file_count():
    with pytest.raises(ValueError):
        make_review(files_scanned=True)


def test_review_rejects_invalid_findings_collection():
    with pytest.raises(TypeError):
        make_review(findings="not a list")


def test_review_rejects_non_finding_items():
    with pytest.raises(TypeError):
        make_review(findings=["invalid finding"])


def test_review_counts_findings_by_severity():
    findings = [
        SecurityFinding(
            severity=severity,
            category="Test",
            title=f"{severity} finding",
            description="Test finding.",
        )
        for severity in [
            "CRITICAL",
            "HIGH",
            "HIGH",
            "MEDIUM",
            "LOW",
            "INFO",
        ]
    ]

    review = make_review(findings=findings)

    assert review.critical_count == 1
    assert review.high_count == 2
    assert review.medium_count == 1
    assert review.low_count == 1
    assert review.info_count == 1


def test_review_passed_and_failed_properties():
    passed = make_review(
        decision="PASS",
        static_analysis_completed=True,
        requires_human_review=False,
    )

    failed = make_review(decision="FAIL")

    assert passed.passed is True
    assert passed.failed is False
    assert failed.failed is True
    assert failed.passed is False


def test_review_rejects_pass_with_critical_finding():
    finding = SecurityFinding(
        severity="CRITICAL",
        category="Private Key Material",
        title="Private key detected",
        description="Private key material was detected.",
    )

    with pytest.raises(ValueError):
        make_review(
            decision="PASS",
            findings=[finding],
            static_analysis_completed=True,
            requires_human_review=False,
        )


def test_review_rejects_pass_with_high_finding():
    finding = SecurityFinding(
        severity="HIGH",
        category="Credential",
        title="Credential detected",
        description="A credential was detected.",
    )

    with pytest.raises(ValueError):
        make_review(
            decision="PASS",
            findings=[finding],
            static_analysis_completed=True,
            requires_human_review=False,
        )


def test_review_rejects_pass_with_incomplete_status():
    with pytest.raises(ValueError):
        make_review(
            decision="PASS",
            status="PARTIAL",
            static_analysis_completed=True,
            requires_human_review=False,
        )


def test_review_rejects_pass_without_static_analysis():
    with pytest.raises(ValueError):
        make_review(
            decision="PASS",
            static_analysis_completed=False,
            requires_human_review=False,
        )


def test_review_rejects_pass_when_human_review_required():
    with pytest.raises(ValueError):
        make_review(
            decision="PASS",
            static_analysis_completed=True,
            requires_human_review=True,
        )


def test_review_serialization_round_trip(finding):
    review = make_review(findings=[finding])

    serialized = review.to_dict()
    restored = SecurityReview.from_dict(serialized)

    assert isinstance(restored, SecurityReview)
    assert isinstance(restored.findings[0], SecurityFinding)
    assert restored.to_dict() == serialized


def test_review_from_dict_rejects_non_dictionary():
    with pytest.raises(TypeError):
        SecurityReview.from_dict("invalid")
