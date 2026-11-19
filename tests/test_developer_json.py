
import json

import pytest

from agentops.agents.developer import DeveloperAgent


@pytest.fixture
def agent():
    return DeveloperAgent.__new__(DeveloperAgent)


def test_extract_json_accepts_valid_response(agent):
    response = json.dumps({
        "summary": "Created a calculator.",
        "files": [
            {
                "path": "app.py",
                "content": "print('Calculator')",
            }
        ],
    })

    result = agent._extract_json(response)

    assert result["summary"] == "Created a calculator."
    assert result["files"][0]["path"] == "app.py"


def test_extract_json_accepts_json_surrounded_by_markdown(agent):
    response = """Here is the implementation:

```json
{
    "summary": "Created the application.",
    "files": []
}
```

Implementation complete.
"""

    result = agent._extract_json(response)

    assert result == {
        "summary": "Created the application.",
        "files": [],
    }


def test_extract_json_rejects_non_string_response(agent):
    with pytest.raises(
        ValueError,
        match="invalid response",
    ):
        agent._extract_json(None)


def test_extract_json_rejects_missing_json_object(agent):
    with pytest.raises(
        ValueError,
        match="did not return a JSON object",
    ):
        agent._extract_json("No JSON was returned.")


def test_extract_json_rejects_malformed_json(agent):
    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        agent._extract_json(
            '{"summary": "Incomplete", "files": [}'
        )


def test_extract_json_rejects_non_object_json(agent):
    with pytest.raises(
        ValueError,
        match="did not return a JSON object",
    ):
        agent._extract_json('["one", "two"]')


def test_extract_json_requires_summary_string(agent):
    response = json.dumps({
        "summary": 123,
        "files": [],
    })

    with pytest.raises(
        ValueError,
        match="summary string",
    ):
        agent._extract_json(response)


def test_extract_json_requires_files_array(agent):
    response = json.dumps({
        "summary": "Created application.",
        "files": {},
    })

    with pytest.raises(
        ValueError,
        match="files array",
    ):
        agent._extract_json(response)


def test_extract_json_accepts_empty_files_array(agent):
    response = json.dumps({
        "summary": "No changes were necessary.",
        "files": [],
    })

    result = agent._extract_json(response)

    assert result["summary"] == "No changes were necessary."
    assert result["files"] == []
