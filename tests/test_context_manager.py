
import pytest

from agentops.tools.context_manager import (
    ProjectContext,
    ProjectContextManager,
)


def make_file(path, content="sample content"):
    return {
        "path": path,
        "content": content,
    }


def test_default_limits_are_configured():
    manager = ProjectContextManager()

    assert manager.max_context_chars == 5500
    assert manager.max_selected_files == 18


def test_custom_limits_are_configured():
    manager = ProjectContextManager(
        max_context_chars=1000,
        max_selected_files=5,
    )

    assert manager.max_context_chars == 1000
    assert manager.max_selected_files == 5


@pytest.mark.parametrize(
    "max_context_chars",
    [0, -1],
)
def test_invalid_context_character_limit_is_rejected(
    max_context_chars,
):
    with pytest.raises(
        ValueError,
        match="Context character limit must be greater than zero",
    ):
        ProjectContextManager(
            max_context_chars=max_context_chars,
        )


@pytest.mark.parametrize(
    "max_selected_files",
    [0, -1],
)
def test_invalid_selected_file_limit_is_rejected(
    max_selected_files,
):
    with pytest.raises(
        ValueError,
        match="Maximum selected files must be greater than zero",
    ):
        ProjectContextManager(
            max_selected_files=max_selected_files,
        )


def test_select_files_returns_project_context():
    manager = ProjectContextManager()

    result = manager.select_files(
        files=[make_file("app.py")],
        query="explain the application",
    )

    assert isinstance(result, ProjectContext)


def test_empty_file_list_returns_empty_context():
    manager = ProjectContextManager()

    result = manager.select_files(
        files=[],
        query="explain the application",
    )

    assert result.selected_files == []
    assert result.omitted_files == []
    assert result.total_characters == 0


def test_invalid_file_paths_are_skipped():
    manager = ProjectContextManager()

    files = [
        make_file(None),
        make_file(123),
        make_file("valid.py"),
    ]

    result = manager.select_files(
        files=files,
        query="explain the application",
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0]["path"] == "valid.py"


def test_invalid_file_content_is_skipped():
    manager = ProjectContextManager()

    files = [
        {"path": "missing.py"},
        {"path": "none.py", "content": None},
        {"path": "number.py", "content": 123},
        make_file("valid.py"),
    ]

    result = manager.select_files(
        files=files,
        query="explain the application",
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0]["path"] == "valid.py"


def test_empty_and_whitespace_only_files_are_skipped():
    manager = ProjectContextManager()

    files = [
        make_file("empty.py", ""),
        make_file("whitespace.py", "   \n\t "),
        make_file("valid.py", "actual content"),
    ]

    result = manager.select_files(
        files=files,
        query="explain the application",
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0]["path"] == "valid.py"


def test_entry_point_receives_higher_priority_than_documentation():
    manager = ProjectContextManager()

    files = [
        make_file("README.md"),
        make_file("main.py"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.selected_files[0]["path"] == "main.py"
    assert result.selected_files[1]["path"] == "README.md"


def test_documentation_receives_higher_priority_than_dependency_file():
    manager = ProjectContextManager()

    files = [
        make_file("requirements.txt"),
        make_file("README.md"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.selected_files[0]["path"] == "README.md"


def test_dependency_file_receives_higher_priority_than_configuration():
    manager = ProjectContextManager()

    files = [
        make_file("Dockerfile"),
        make_file("pyproject.toml"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.selected_files[0]["path"] == "pyproject.toml"


def test_important_directory_increases_file_priority():
    manager = ProjectContextManager()

    files = [
        make_file("misc/helper.py"),
        make_file("agents/helper.py"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.selected_files[0]["path"] == "agents/helper.py"


def test_test_file_receives_additional_priority():
    manager = ProjectContextManager()

    files = [
        make_file("helper.py"),
        make_file("test_helper.py"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.selected_files[0]["path"] == "test_helper.py"


def test_question_keywords_increase_path_relevance():
    manager = ProjectContextManager()

    files = [
        make_file("database.py", "generic content"),
        make_file("authentication.py", "generic content"),
    ]

    result = manager.select_files(
        files=files,
        query="explain authentication",
    )

    assert result.selected_files[0]["path"] == "authentication.py"


def test_question_keywords_increase_content_relevance():
    manager = ProjectContextManager()

    files = [
        make_file("first.py", "handles unrelated operations"),
        make_file("second.py", "handles authentication and authorization"),
    ]

    result = manager.select_files(
        files=files,
        query="explain authentication",
    )

    assert result.selected_files[0]["path"] == "second.py"


def test_stop_words_are_removed_from_question_keywords():
    manager = ProjectContextManager()

    keywords = manager._question_keywords(
        "How does the application work?"
    )

    assert "application" not in keywords
    assert "how" not in keywords
    assert "does" not in keywords
    assert "work" in keywords


def test_tokenization_is_case_insensitive():
    manager = ProjectContextManager()

    tokens = manager._tokenize(
        "Authentication AUTHENTICATION Database"
    )

    assert tokens == {
        "authentication",
        "database",
    }


def test_equal_priority_files_are_sorted_alphabetically():
    manager = ProjectContextManager()

    files = [
        make_file("zebra.py"),
        make_file("alpha.py"),
        make_file("middle.py"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    selected_paths = [
        item["path"]
        for item in result.selected_files
    ]

    assert selected_paths == [
        "alpha.py",
        "middle.py",
        "zebra.py",
    ]


def test_selection_respects_maximum_selected_file_limit():
    manager = ProjectContextManager(
        max_selected_files=2,
    )

    files = [
        make_file("first.py"),
        make_file("second.py"),
        make_file("third.py"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert len(result.selected_files) == 2
    assert len(result.omitted_files) == 1
    assert "third.py" in result.omitted_files[0]
    assert "maximum selected-file limit reached" in result.omitted_files[0]


def test_context_character_budget_is_enforced():
    manager = ProjectContextManager(
        max_context_chars=100,
    )

    files = [
        make_file("large.py", "x" * 500),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.total_characters <= 100
    assert result.total_characters > 0


def test_file_content_is_truncated_when_budget_is_exceeded():
    manager = ProjectContextManager(
        max_context_chars=40,
    )

    files = [
        make_file("large.py", "x" * 500),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert len(result.selected_files) == 1
    assert len(result.selected_files[0]["content"]) < 500
    assert "large.py: content truncated" in result.omitted_files


def test_truncated_content_reports_correct_character_count():
    manager = ProjectContextManager(
        max_context_chars=40,
    )

    result = manager.select_files(
        files=[
            make_file("large.py", "x" * 500),
        ],
        query="",
    )

    assert result.total_characters == 40


def test_insufficient_space_for_file_header_skips_file():
    manager = ProjectContextManager(
        max_context_chars=5,
    )

    result = manager.select_files(
        files=[
            make_file("application.py"),
        ],
        query="",
    )

    assert result.selected_files == []
    assert len(result.omitted_files) == 1
    assert "insufficient context space" in result.omitted_files[0]
    assert result.total_characters == 0


def test_context_character_count_includes_file_headers():
    manager = ProjectContextManager()

    result = manager.select_files(
        files=[
            make_file("app.py", "hello"),
        ],
        query="",
    )

    expected_header = "\n\nFILE: app.py\n"

    assert result.total_characters == (
        len(expected_header) + len("hello")
    )


def test_multiple_files_respect_combined_character_budget():
    manager = ProjectContextManager(
        max_context_chars=60,
    )

    files = [
        make_file("first.py", "a" * 20),
        make_file("second.py", "b" * 20),
        make_file("third.py", "c" * 20),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert result.total_characters <= 60
    assert len(result.selected_files) >= 1


def test_selection_is_deterministic():
    manager = ProjectContextManager()

    files = [
        make_file("zebra.py", "common content"),
        make_file("alpha.py", "common content"),
        make_file("main.py", "application entry point"),
    ]

    first_result = manager.select_files(
        files=files,
        query="explain the application",
    )

    second_result = manager.select_files(
        files=files,
        query="explain the application",
    )

    assert first_result == second_result


def test_original_file_data_is_not_modified():
    manager = ProjectContextManager()

    original = make_file(
        "app.py",
        "original content",
    )

    original_copy = original.copy()

    manager.select_files(
        files=[original],
        query="explain the application",
    )

    assert original == original_copy


def test_selected_files_contain_only_path_and_content():
    manager = ProjectContextManager()

    result = manager.select_files(
        files=[
            {
                "path": "app.py",
                "content": "application content",
                "extra": "should not be copied",
            }
        ],
        query="application",
    )

    assert result.selected_files == [
        {
            "path": "app.py",
            "content": "application content",
        }
    ]


def test_files_after_exhausted_budget_are_reported_as_omitted():
    manager = ProjectContextManager(
        max_context_chars=30,
    )

    files = [
        make_file("first.py", "x" * 100),
        make_file("second.py", "second content"),
    ]

    result = manager.select_files(
        files=files,
        query="",
    )

    assert len(result.selected_files) == 1
    assert any(
        "context character budget exhausted" in item
        for item in result.omitted_files
    )
