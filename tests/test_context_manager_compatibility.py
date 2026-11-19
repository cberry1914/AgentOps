
from agentops.tools.context_manager import (
    ProjectContextManager,
    SelectedContextFile,
)


def test_select_context_returns_attribute_based_selected_files():
    manager = ProjectContextManager()

    context = manager.select_context(
        task="Review the greeting function",
        files=[
            {
                "path": "hello.py",
                "content": (
                    "def greet(name):\n"
                    '    return f"Hello, {name}!"\n'
                ),
            }
        ],
    )

    assert len(context.selected_files) == 1

    selected_file = context.selected_files[0]

    assert isinstance(selected_file, SelectedContextFile)
    assert selected_file.path == "hello.py"
    assert "def greet(name):" in selected_file.content


def test_select_context_preserves_omitted_files_metadata():
    manager = ProjectContextManager(max_selected_files=1)

    context = manager.select_context(
        task="Review the application",
        files=[
            {
                "path": "first.py",
                "content": "def first():\n    return True\n",
            },
            {
                "path": "second.py",
                "content": "def second():\n    return False\n",
            },
        ],
    )

    assert len(context.selected_files) == 1
    assert len(context.omitted_files) == 1
    assert "maximum selected-file limit reached" in context.omitted_files[0]
    assert context.total_characters > 0


def test_select_context_preserves_character_budget():
    manager = ProjectContextManager(max_context_chars=100)

    context = manager.select_context(
        task="Review the application",
        files=[
            {
                "path": "large.py",
                "content": "x" * 500,
            }
        ],
    )

    assert len(context.selected_files) == 1
    assert context.total_characters <= 100
    assert context.total_characters > 0
    assert "large.py: content truncated" in context.omitted_files


def test_select_context_with_no_files_returns_empty_metadata():
    manager = ProjectContextManager()

    context = manager.select_context(
        task="Review the application",
        files=[],
    )

    assert context.selected_files == []
    assert context.omitted_files == []
    assert context.total_characters == 0


def test_select_files_preserves_dictionary_based_interface():
    manager = ProjectContextManager()

    context = manager.select_files(
        files=[
            {
                "path": "hello.py",
                "content": (
                    "def greet(name):\n"
                    '    return f"Hello, {name}!"\n'
                ),
            }
        ],
        query="Review the greeting function",
    )

    assert len(context.selected_files) == 1
    assert context.selected_files[0] == {
        "path": "hello.py",
        "content": (
            "def greet(name):\n"
            '    return f"Hello, {name}!"\n'
        ),
    }
    assert isinstance(context.selected_files[0], dict)


def test_select_context_and_select_files_select_same_content():
    manager = ProjectContextManager()

    files = [
        {
            "path": "main.py",
            "content": "def main():\n    return True\n",
        },
        {
            "path": "README.md",
            "content": "# Application\n\nA sample project.\n",
        },
        {
            "path": "services/helper.py",
            "content": "def helper():\n    return 'ready'\n",
        },
    ]

    dictionary_context = manager.select_files(
        files=files,
        query="Review the application",
    )

    object_context = manager.select_context(
        task="Review the application",
        files=files,
    )

    assert [
        (item["path"], item["content"])
        for item in dictionary_context.selected_files
    ] == [
        (item.path, item.content)
        for item in object_context.selected_files
    ]

    assert object_context.omitted_files == dictionary_context.omitted_files
    assert object_context.total_characters == dictionary_context.total_characters
