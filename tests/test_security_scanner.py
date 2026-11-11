
"""
AgentOps - Security Scanner Tests

Tests deterministic security detection, evidence redaction,
placeholder filtering, input validation, and safety limits.
"""

from threading import Event

import pytest

from agentops.tools.security_scanner import SecurityScanner


@pytest.fixture
def scanner():
    return SecurityScanner()


def test_detects_aws_access_key(scanner):
    secret = "AKIAIOSFODNN7EXAMPLE"

    findings = scanner.scan_files({
        "config.py": f'AWS_KEY = "{secret}"'
    })

    assert len(findings) == 1
    assert findings[0].category == "AWS Access Key"
    assert findings[0].severity == "HIGH"
    assert findings[0].file_path == "config.py"
    assert findings[0].line_number == 1
    assert secret not in findings[0].evidence
    assert "[REDACTED]" in findings[0].evidence


def test_detects_github_token(scanner):
    secret = "ghp_" + ("A" * 36)

    findings = scanner.scan_files({
        "settings.py": f'TOKEN = "{secret}"'
    })

    assert any(
        finding.category == "GitHub Token"
        for finding in findings
    )

    assert all(
        secret not in finding.evidence
        for finding in findings
    )


def test_detects_slack_token(scanner):
    secret = "xoxb-" + ("A" * 20)

    findings = scanner.scan_files({
        "settings.py": f'SLACK_TOKEN = "{secret}"'
    })

    assert any(
        finding.category == "Slack Token"
        for finding in findings
    )

    assert all(
        secret not in finding.evidence
        for finding in findings
    )


def test_detects_private_key_marker(scanner):
    marker = "-----BEGIN RSA PRIVATE KEY-----"

    findings = scanner.scan_files({
        "private.py": f'key = "{marker}"'
    })

    assert any(
        finding.category == "Private Key Material"
        and finding.severity == "CRITICAL"
        for finding in findings
    )


def test_detects_hardcoded_password(scanner):
    secret = "ProductionPassword123"

    findings = scanner.scan_files({
        "database.py": f'password = "{secret}"'
    })

    assert len(findings) == 1
    assert findings[0].category == "Hardcoded Credential"
    assert findings[0].severity == "HIGH"
    assert findings[0].confidence == "MEDIUM"
    assert secret not in findings[0].evidence
    assert "[REDACTED]" in findings[0].evidence


def test_detects_colon_assignment(scanner):
    findings = scanner.scan_files({
        "config.yaml": 'api_key: "ProductionKey123"'
    })

    assert len(findings) == 1
    assert findings[0].category == "Hardcoded Credential"


@pytest.mark.parametrize(
    "placeholder",
    [
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
    ],
)
def test_ignores_placeholder_values(scanner, placeholder):
    findings = scanner.scan_files({
        "config.py": f'password = "{placeholder}"'
    })

    assert findings == []


def test_ignores_environment_variable_reference(scanner):
    findings = scanner.scan_files({
        "config.py": 'password = "$DATABASE_PASSWORD"'
    })

    assert findings == []


def test_ignores_environment_expression(scanner):
    findings = scanner.scan_files({
        "config.py": 'password = "${DATABASE_PASSWORD}"'
    })

    assert findings == []


def test_redacts_secret_without_exposing_value(scanner):
    secret = "SuperSecretDatabasePassword"

    findings = scanner.scan_files({
        "config.py": f'database_password = "{secret}"'
    })

    assert len(findings) == 1

    finding = findings[0]

    assert secret not in finding.evidence
    assert secret not in finding.description
    assert secret not in finding.recommendation
    assert "[REDACTED]" in finding.evidence


def test_records_correct_line_number(scanner):
    findings = scanner.scan_files({
        "config.py": (
            "import os\n"
            "DEBUG = True\n"
            'password = "ProductionPassword123"\n'
        )
    })

    assert len(findings) == 1
    assert findings[0].line_number == 3


def test_scans_multiple_files(scanner):
    findings = scanner.scan_files({
        "first.py": 'password = "FirstSecret123"',
        "second.py": 'api_key = "SecondSecret123"',
    })

    assert len(findings) == 2

    paths = {finding.file_path for finding in findings}

    assert paths == {"first.py", "second.py"}


def test_rejects_non_dictionary_input(scanner):
    with pytest.raises(TypeError):
        scanner.scan_files(["password = 'secret123'"])


def test_ignores_non_string_file_content(scanner):
    findings = scanner.scan_files({
        "invalid.py": None,
        "valid.py": 'password = "ValidSecret123"',
    })

    assert len(findings) == 1
    assert findings[0].file_path == "valid.py"


def test_respects_max_files_limit(scanner):
    scanner.MAX_FILES = 1

    findings = scanner.scan_files({
        "first.py": 'password = "FirstSecret123"',
        "second.py": 'password = "SecondSecret123"',
    })

    assert len(findings) == 1
    assert findings[0].file_path == "first.py"


def test_respects_max_file_size_limit(scanner):
    scanner.MAX_FILE_SIZE_CHARS = 20

    content = (
        "# " + ("A" * 30) + "\n"
        'password = "Secret12345"'
    )

    findings = scanner.scan_files({
        "large.py": content
    })

    assert findings == []


def test_respects_max_findings_limit(scanner):
    scanner.MAX_FINDINGS = 2

    content = "\n".join([
        f'password = "SecretValue{i:03d}"'
        for i in range(10)
    ])

    findings = scanner.scan_files({
        "many_secrets.py": content
    })

    assert len(findings) == 2


def test_stops_when_cancellation_is_requested(scanner):
    cancel_event = Event()
    cancel_event.set()

    findings = scanner.scan_files(
        {
            "config.py": 'password = "ProductionSecret123"'
        },
        cancel_event=cancel_event,
    )

    assert findings == []


def test_deduplicates_same_line_detection(scanner):
    findings = scanner.scan_files({
        "config.py": 'password = "ProductionPassword123"'
    })

    identities = {
        (
            finding.file_path,
            finding.line_number,
            finding.category,
            finding.title,
        )
        for finding in findings
    }

    assert len(identities) == len(findings)
