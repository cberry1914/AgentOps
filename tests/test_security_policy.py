
"""
AgentOps - Security Policy Engine Tests

Tests severity-based decisions, incomplete reviews,
human approval requirements, metadata, and validation.
"""

import pytest

from agentops.models.security_review import SecurityFinding
from agentops.tools.security_policy import SecurityPolicyEngine


@pytest.fixture
def policy():
    return SecurityPolicyEngine()


def make_finding(severity="MEDIUM"):
    return SecurityFinding(
        severity=severity,
        category="Test Finding",
        title=f"{severity} security finding",
        description="A test security finding.",
    )


def evaluate_clean(policy, **overrides):
    values = {
        "files_scanned": 5,
        "static_analysis_completed": True,
        "ai_review_completed": True,
    }
    values.update(overrides)
    return policy.evaluate(**values)


# Decision tests


@pytest.mark.parametrize("severity", ["CRITICAL", "HIGH"])
def test_critical_and_high_findings_produce_fail(policy, severity):
    review = evaluate_clean(
        policy,
        findings=[make_finding(severity)],
    )

    assert review.decision == "FAIL"


@pytest.mark.parametrize("severity", ["MEDIUM", "LOW", "INFO"])
def test_lower_severity_findings_require_review(policy, severity):
    review = evaluate_clean(
        policy,
        findings=[make_finding(severity)],
    )

    assert review.decision == "NEEDS_REVIEW"


def test_clean_completed_scan_requires_human_review(policy):
    review = evaluate_clean(policy)

    assert review.decision == "NEEDS_REVIEW"
    assert review.requires_human_review is True


def test_policy_never_automatically_issues_pass(policy):
    review = evaluate_clean(policy)

    assert review.decision != "PASS"
    assert review.metadata["automated_approval_permitted"] is False


@pytest.mark.parametrize(
    "status",
    ["CANCELLED", "FAILED", "NOT_RUN"],
)
def test_uncompleted_review_status_produces_needs_info(policy, status):
    review = evaluate_clean(
        policy,
        status=status,
        findings=[make_finding("CRITICAL")],
    )

    assert review.decision == "NEEDS_INFO"


def test_zero_scanned_files_produces_needs_info(policy):
    review = policy.evaluate(
        files_scanned=0,
        static_analysis_completed=True,
        ai_review_completed=True,
    )

    assert review.decision == "NEEDS_INFO"


def test_no_completed_analysis_produces_needs_info(policy):
    review = policy.evaluate(files_scanned=5)

    assert review.decision == "NEEDS_INFO"


def test_partial_review_produces_needs_review(policy):
    review = evaluate_clean(
        policy,
        status="PARTIAL",
    )

    assert review.decision == "NEEDS_REVIEW"


def test_high_finding_overrides_partial_status(policy):
    review = evaluate_clean(
        policy,
        status="PARTIAL",
        findings=[make_finding("HIGH")],
    )

    assert review.decision == "FAIL"


def test_ai_review_alone_is_sufficient_for_needs_review(policy):
    review = policy.evaluate(
        files_scanned=3,
        ai_review_completed=True,
    )

    assert review.decision == "NEEDS_REVIEW"


def test_static_analysis_alone_is_sufficient_for_needs_review(policy):
    review = policy.evaluate(
        files_scanned=3,
        static_analysis_completed=True,
    )

    assert review.decision == "NEEDS_REVIEW"


# Human approval and metadata tests


def test_human_approval_is_always_required(policy):
    review = evaluate_clean(policy)

    assert review.requires_human_review is True
    assert review.metadata["human_approval_required"] is True


def test_metadata_contains_severity_counts(policy):
    findings = [
        make_finding("CRITICAL"),
        make_finding("HIGH"),
        make_finding("HIGH"),
        make_finding("MEDIUM"),
        make_finding("LOW"),
        make_finding("INFO"),
    ]

    review = evaluate_clean(
        policy,
        findings=findings,
    )

    assert review.metadata["finding_count"] == 6
    assert review.metadata["critical_findings"] == 1
    assert review.metadata["high_findings"] == 2
    assert review.metadata["medium_findings"] == 1
    assert review.metadata["low_findings"] == 1
    assert review.metadata["informational_findings"] == 1


def test_custom_metadata_is_preserved(policy):
    review = evaluate_clean(
        policy,
        metadata={"scanner_version": "1.0"},
    )

    assert review.metadata["scanner_version"] == "1.0"


def test_policy_metadata_overrides_conflicting_custom_values(policy):
    review = evaluate_clean(
        policy,
        metadata={
            "finding_count": 999,
            "automated_approval_permitted": True,
        },
    )

    assert review.metadata["finding_count"] == 0
    assert review.metadata["automated_approval_permitted"] is False


def test_summary_is_included_before_policy_summary(policy):
    review = evaluate_clean(
        policy,
        summary="Initial scanner report.",
    )

    assert review.summary.startswith("Initial scanner report.")
    assert "Security decision: NEEDS_REVIEW" in review.summary


def test_policy_summary_contains_scan_details(policy):
    review = evaluate_clean(
        policy,
        files_scanned=8,
        files_skipped=2,
    )

    assert "Files scanned: 8" in review.summary
    assert "Files skipped: 2" in review.summary
    assert "Human review is required" in review.summary


# Limitations and input validation


def test_limitations_are_cleaned_and_deduplicated(policy):
    review = evaluate_clean(
        policy,
        limitations=[
            "  Some files were skipped  ",
            "Some files were skipped",
            "",
            "Additional review required",
            None,
        ],
    )

    assert review.limitations == [
        "Some files were skipped",
        "Additional review required",
    ]


@pytest.mark.parametrize(
    "field_name",
    ["files_scanned", "files_skipped"],
)
def test_negative_file_counts_are_rejected(policy, field_name):
    with pytest.raises(ValueError):
        policy.evaluate(**{field_name: -1})


@pytest.mark.parametrize(
    "field_name",
    ["files_scanned", "files_skipped"],
)
def test_boolean_file_counts_are_rejected(policy, field_name):
    with pytest.raises(TypeError):
        policy.evaluate(**{field_name: True})


@pytest.mark.parametrize(
    "field_name",
    ["files_scanned", "files_skipped"],
)
def test_non_integer_file_counts_are_rejected(policy, field_name):
    with pytest.raises(TypeError):
        policy.evaluate(**{field_name: 1.5})


def test_invalid_review_status_is_rejected(policy):
    with pytest.raises(ValueError):
        policy.evaluate(status="IN_PROGRESS")


def test_non_security_finding_items_are_rejected(policy):
    with pytest.raises(TypeError):
        policy.evaluate(
            findings=["not a security finding"],
        )
