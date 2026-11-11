
"""
AgentOps - Test Report Parser Tests

Tests pytest summary parsing, execution counts, collection
errors, warnings, empty collections, and success evidence.
"""

import pytest

from agentops.tools.test_report_parser import (
    ParsedTestReport,
    TestReportParser,
)


@pytest.fixture
def parser():
    return TestReportParser()


# Parsed report model


def test_report_defaults():
    report = ParsedTestReport()

    assert report.total == 0
    assert report.passed == 0
    assert report.failed == 0
    assert report.skipped == 0
    assert report.errors == 0
    assert report.warnings == 0
    assert report.collection_errors == 0
    assert report.duration_seconds == 0.0
    assert report.summary_found is False
    assert report.no_tests_collected is False


@pytest.mark.parametrize(
    (
        "failed",
        "errors",
        "collection_errors",
        "expected",
    ),
    [
        (0, 0, 0, False),
        (1, 0, 0, True),
        (0, 1, 0, True),
        (0, 0, 1, True),
        (2, 1, 3, True),
    ],
)
def test_has_failures_property(
    failed,
    errors,
    collection_errors,
    expected,
):
    report = ParsedTestReport(
        failed=failed,
        errors=errors,
        collection_errors=collection_errors,
    )

    assert report.has_failures is expected


@pytest.mark.parametrize(
    (
        "report",
        "expected",
    ),
    [
        (
            ParsedTestReport(
                total=2,
                passed=2,
                summary_found=True,
            ),
            True,
        ),
        (
            ParsedTestReport(
                total=0,
                passed=0,
                summary_found=True,
            ),
            False,
        ),
        (
            ParsedTestReport(
                total=2,
                passed=1,
                failed=1,
                summary_found=True,
            ),
            False,
        ),
        (
            ParsedTestReport(
                total=2,
                passed=2,
                summary_found=False,
            ),
            False,
        ),
        (
            ParsedTestReport(
                total=2,
                passed=2,
                collection_errors=1,
                summary_found=True,
            ),
            False,
        ),
        (
            ParsedTestReport(
                total=2,
                passed=2,
                no_tests_collected=True,
                summary_found=True,
            ),
            False,
        ),
        (
            ParsedTestReport(
                total=2,
                passed=1,
                skipped=1,
                summary_found=True,
            ),
            True,
        ),
    ],
)
def test_successful_property(report, expected):
    assert report.successful is expected


# Empty output and basic parsing


@pytest.mark.parametrize(
    "output",
    [
        "",
        " ",
        "\n",
        "\n   \n",
    ],
)
def test_parse_empty_output(parser, output):
    report = parser.parse(output)

    assert report.total == 0
    assert report.summary_found is False
    assert report.no_tests_collected is False


def test_parse_empty_output_with_no_tests_exit_code(parser):
    report = parser.parse("", exit_code=5)

    assert report.no_tests_collected is True
    assert report.summary_found is False


def test_parse_empty_output_with_success_exit_code(parser):
    report = parser.parse("", exit_code=0)

    assert report.no_tests_collected is False
    assert report.successful is False


def test_parse_simple_passing_summary(parser):
    report = parser.parse(
        "========================= 5 passed in 0.42s ========================="
    )

    assert report.summary_found is True
    assert report.total == 5
    assert report.passed == 5
    assert report.failed == 0
    assert report.skipped == 0
    assert report.errors == 0
    assert report.duration_seconds == 0.42
    assert report.successful is True


def test_parse_mixed_test_summary(parser):
    report = parser.parse(
        "========================= "
        "5 passed, 2 failed, 1 skipped in 1.25s "
        "========================="
    )

    assert report.total == 8
    assert report.passed == 5
    assert report.failed == 2
    assert report.skipped == 1
    assert report.errors == 0
    assert report.duration_seconds == 1.25
    assert report.has_failures is True
    assert report.successful is False


def test_parse_summary_with_test_errors(parser):
    report = parser.parse(
        "========================= "
        "3 passed, 2 errors in 0.75s "
        "========================="
    )

    assert report.total == 5
    assert report.passed == 3
    assert report.errors == 2
    assert report.has_failures is True
    assert report.successful is False


def test_parse_summary_with_warnings(parser):
    report = parser.parse(
        "========================= "
        "4 passed, 3 warnings in 0.50s "
        "========================="
    )

    assert report.total == 4
    assert report.passed == 4
    assert report.warnings == 3
    assert report.successful is True


def test_parse_summary_with_decimal_duration(parser):
    report = parser.parse("2 passed in 12.345s")

    assert report.duration_seconds == 12.345


def test_parse_summary_with_integer_duration(parser):
    report = parser.parse("2 passed in 12s")

    assert report.duration_seconds == 12.0


def test_parse_summary_case_insensitively(parser):
    report = parser.parse("2 PASSED IN 1.50S")

    assert report.summary_found is True
    assert report.passed == 2
    assert report.duration_seconds == 1.5


def test_parse_summary_with_leading_and_trailing_whitespace(
    parser,
):
    report = parser.parse(
        "   2 passed in 0.25s   \n"
    )

    assert report.summary_found is True
    assert report.passed == 2


def test_parse_uses_final_summary_line(parser):
    output = (
        "1 passed in 0.10s\n"
        "Additional output\n"
        "3 passed, 1 skipped in 0.50s\n"
    )

    report = parser.parse(output)

    assert report.total == 4
    assert report.passed == 3
    assert report.skipped == 1
    assert report.duration_seconds == 0.5


# Count consistency and successful evidence


def test_parse_all_skipped_tests(parser):
    report = parser.parse("4 skipped in 0.30s")

    assert report.total == 4
    assert report.skipped == 4
    assert report.successful is True


def test_parse_zero_test_summary(parser):
    report = parser.parse("0 passed in 0.01s")

    assert report.summary_found is True
    assert report.total == 0
    assert report.no_tests_collected is True
    assert report.successful is False


def test_parse_inconsistent_counts_cannot_be_successful(
    parser,
):
    report = ParsedTestReport(
        total=5,
        passed=4,
        summary_found=True,
    )

    assert report.successful is False


def test_parse_failed_tests_cannot_be_successful(parser):
    report = parser.parse(
        "2 passed, 1 failed in 0.50s"
    )

    assert report.successful is False


# Collection errors


def test_parse_single_collection_error(parser):
    output = (
        "==================================== ERRORS "
        "====================================\n"
        "________________ ERROR collecting tests/test_bad.py "
        "________________\n"
        "ImportError: cannot import module\n"
        "========================= 1 error in 0.10s "
        "========================="
    )

    report = parser.parse(output)

    assert report.collection_errors == 1
    assert report.errors == 0
    assert report.total == 0
    assert report.has_failures is True
    assert report.successful is False


def test_parse_multiple_collection_errors(parser):
    output = (
        "ERROR collecting tests/test_one.py\n"
        "ImportError: first import failed\n"
        "ERROR collecting tests/test_two.py\n"
        "SyntaxError: second import failed\n"
        "2 errors in 0.20s"
    )

    report = parser.parse(output)

    assert report.collection_errors == 2
    assert report.errors == 0
    assert report.total == 0
    assert report.has_failures is True


def test_parse_collection_error_with_passing_tests(
    parser,
):
    output = (
        "ERROR collecting tests/test_broken.py\n"
        "ImportError: missing dependency\n"
        "========================= "
        "3 passed, 1 error in 0.40s "
        "========================="
    )

    report = parser.parse(output)

    assert report.collection_errors == 1
    assert report.passed == 3
    assert report.errors == 0
    assert report.total == 3
    assert report.has_failures is True
    assert report.successful is False


def test_parse_test_errors_without_collection_errors(
    parser,
):
    report = parser.parse("2 passed, 1 error in 0.30s")

    assert report.collection_errors == 0
    assert report.errors == 1
    assert report.total == 3
    assert report.has_failures is True


def test_parse_collection_error_without_summary(parser):
    output = (
        "ERROR collecting tests/test_broken.py\n"
        "ImportError: missing dependency"
    )

    report = parser.parse(output)

    assert report.collection_errors == 1
    assert report.summary_found is False
    assert report.has_failures is True
    assert report.successful is False


def test_parse_collection_error_is_not_counted_twice(
    parser,
):
    output = (
        "ERROR collecting tests/test_broken.py\n"
        "ImportError: cannot import module\n"
        "1 error in 0.10s"
    )

    report = parser.parse(output)

    assert report.collection_errors == 1
    assert report.errors == 0
    assert report.total == 0


# No tests collected detection


@pytest.mark.parametrize(
    "message",
    [
        "no tests ran",
        "collected 0 items",
        "collected 0 tests",
        "no tests collected",
    ],
)
def test_detect_no_tests_collected_messages(
    parser,
    message,
):
    report = parser.parse(message)

    assert report.no_tests_collected is True
    assert report.summary_found is False
    assert report.successful is False


def test_exit_code_five_marks_no_tests_collected(
    parser,
):
    report = parser.parse(
        "some pytest output",
        exit_code=5,
    )

    assert report.no_tests_collected is True


def test_exit_code_five_does_not_override_collection_errors(
    parser,
):
    output = (
        "ERROR collecting tests/test_bad.py\n"
        "ImportError: missing dependency"
    )

    report = parser.parse(output, exit_code=5)

    assert report.collection_errors == 1
    assert report.no_tests_collected is False


def test_exit_code_five_with_passing_summary_does_not_mark_empty(
    parser,
):
    report = parser.parse(
        "2 passed in 0.20s",
        exit_code=5,
    )

    assert report.total == 2
    assert report.no_tests_collected is False


# Warning detection


def test_parse_warning_count_from_summary(parser):
    report = parser.parse(
        "2 passed, 3 warnings in 0.20s"
    )

    assert report.warnings == 3


def test_parse_warning_records_from_warning_lines(parser):
    output = (
        "tests/test_example.py:10: UserWarning: first warning\n"
        "tests/test_example.py:20: DeprecationWarning: second warning\n"
        "2 passed in 0.20s"
    )

    report = parser.parse(output)

    assert report.warnings == 2


def test_parse_warning_line_without_summary(parser):
    output = (
        "tests/test_example.py:10: UserWarning: "
        "deprecated behavior"
    )

    report = parser.parse(output)

    assert report.warnings == 1
    assert report.summary_found is False


def test_warning_summary_fallback_detects_warning_presence(
    parser,
):
    output = (
        "2 passed in 0.20s\n"
        "========================= warnings summary "
        "=========================\n"
        "tests/test_example.py:10: warning details\n"
        "========================= short test summary "
        "========================="
    )

    report = parser.parse(output)

    assert report.warnings >= 1


def test_warning_summary_without_records_does_not_invent_count(
    parser,
):
    output = (
        "2 passed in 0.20s\n"
        "========================= warnings summary "
        "=========================\n"
        "========================= short test summary "
        "========================="
    )

    report = parser.parse(output)

    assert report.warnings == 0


def test_warning_count_is_not_double_counted(
    parser,
):
    output = (
        "tests/test_example.py:10: UserWarning: "
        "deprecated behavior\n"
        "2 passed, 1 warning in 0.20s"
    )

    report = parser.parse(output)

    assert report.warnings == 1


# Internal helper behavior


def test_find_summary_line_returns_final_matching_line(
    parser,
):
    lines = [
        "1 passed in 0.10s",
        "ordinary output",
        "2 passed in 0.20s",
    ]

    assert parser._find_summary_line(lines) == (
        "2 passed in 0.20s"
    )


def test_find_summary_line_returns_none_without_summary(
    parser,
):
    assert parser._find_summary_line(
        ["ordinary output", "no tests ran"]
    ) is None


def test_parse_summary_ignores_unrecognized_count_labels(
    parser,
):
    report = parser.parse(
        "2 passed, 4 custom-results in 0.20s"
    )

    assert report.passed == 2
    assert report.total == 2


def test_parse_summary_with_multiple_count_categories(
    parser,
):
    report = parser.parse(
        "5 passed, 2 failed, 3 skipped, "
        "1 error, 4 warnings in 2.75s"
    )

    assert report.passed == 5
    assert report.failed == 2
    assert report.skipped == 3
    assert report.errors == 1
    assert report.warnings == 4
    assert report.total == 11
    assert report.duration_seconds == 2.75


def test_parse_does_not_treat_import_error_detail_as_collection_error(
    parser,
):
    output = (
        "ImportError: cannot import name 'Example'\n"
        "2 passed in 0.20s"
    )

    report = parser.parse(output)

    assert report.collection_errors == 0
    assert report.passed == 2
    assert report.successful is True
