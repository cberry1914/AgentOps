
from threading import Event

import pytest

from agentops.tools.cancellation import TaskCancelledError
from agentops.tools.diagram_generator import ProjectDiagramGenerator


TEST_TASK_ID = "12345678-1234-1234-1234-123456789012"


class FakeWorkspaceManager:
    """In-memory workspace manager for diagram generator tests."""

    def __init__(self, files=None, read_errors=None):
        self.files = files or {}
        self.read_errors = read_errors or {}
        self.list_calls = []
        self.read_calls = []
        self.write_calls = []

    def list_files(self, task_id, relative_path=".", agent="developer"):
        self.list_calls.append(
            (task_id, relative_path, agent)
        )

        prefix = (
            ""
            if relative_path == "."
            else relative_path.rstrip("/") + "/"
        )

        entries = {}

        for path, content in self.files.items():
            if not path.startswith(prefix):
                continue

            remainder = path[len(prefix):]

            if not remainder:
                continue

            parts = remainder.split("/")
            name = parts[0]

            if len(parts) > 1:
                entries[name] = {
                    "name": name,
                    "path": prefix + name if prefix else name,
                    "type": "directory",
                    "size_bytes": 0,
                }
            else:
                entries[name] = {
                    "name": name,
                    "path": path,
                    "type": "file",
                    "size_bytes": len(content.encode("utf-8")),
                }

        return list(entries.values())

    def read_file(self, task_id, relative_path, agent="developer"):
        self.read_calls.append(
            (task_id, relative_path, agent)
        )

        if relative_path in self.read_errors:
            raise self.read_errors[relative_path]

        return self.files[relative_path]

    def write_file(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))
        raise AssertionError(
            "Diagram generation must not write to the workspace."
        )


@pytest.fixture
def workspace_manager():
    return FakeWorkspaceManager()


@pytest.fixture
def generator(workspace_manager):
    return ProjectDiagramGenerator(
        workspace_manager=workspace_manager,
    )


def test_generator_uses_supplied_workspace_manager(
    workspace_manager,
):
    generator = ProjectDiagramGenerator(
        workspace_manager=workspace_manager,
    )

    assert generator.workspace_manager is workspace_manager


def test_generator_creates_default_workspace_manager():
    generator = ProjectDiagramGenerator()

    assert generator.workspace_manager is not None


def test_node_id_is_deterministic():
    first = ProjectDiagramGenerator._node_id("src/main.py")
    second = ProjectDiagramGenerator._node_id("src/main.py")

    assert first == second


def test_node_id_changes_for_different_values():
    first = ProjectDiagramGenerator._node_id("src/main.py")
    second = ProjectDiagramGenerator._node_id("src/app.py")

    assert first != second


def test_node_id_uses_mermaid_safe_identifier():
    node_id = ProjectDiagramGenerator._node_id(
        "src/main.py"
    )

    assert node_id.startswith("node_")
    assert node_id.replace("_", "").isalnum()


def test_safe_label_replaces_backslashes():
    result = ProjectDiagramGenerator._safe_label(
        r"src\agentops\main.py"
    )

    assert result == "src/agentops/main.py"


def test_safe_label_replaces_quotes():
    result = ProjectDiagramGenerator._safe_label(
        'application "main"'
    )

    assert result == "application 'main'"


def test_safe_label_escapes_mermaid_special_characters():
    result = ProjectDiagramGenerator._safe_label(
        "a&b[c]{d}<e>"
    )

    assert result == (
        "a&amp;b&#91;c&#93;&#123;d&#125;&lt;e&gt;"
    )


def test_safe_label_replaces_control_whitespace():
    result = ProjectDiagramGenerator._safe_label(
        "first\nsecond\tthird"
    )

    assert result == "first second third"


def test_safe_label_trims_whitespace():
    result = ProjectDiagramGenerator._safe_label(
        "  application.py  "
    )

    assert result == "application.py"


def test_safe_label_limits_length_to_100_characters():
    result = ProjectDiagramGenerator._safe_label(
        "x" * 150
    )

    assert len(result) == 100


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "config/.env",
        "config/.env.local",
        "id_rsa",
        "keys/id_ed25519",
        "credentials",
        "credentials.json",
        "secrets.json",
        "secret.json",
        "terraform.tfstate",
        "terraform.tfstate.backup",
        "backup.tfstate",
        "certificates/server.pem",
        "certificates/server.key",
        "certificates/server.p12",
        "certificates/server.pfx",
    ],
)
def test_sensitive_files_are_identified(path):
    assert ProjectDiagramGenerator._is_sensitive_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "main.py",
        "README.md",
        "config.json",
        "requirements.txt",
        "src/application.py",
    ],
)
def test_normal_project_files_are_not_sensitive(path):
    assert not ProjectDiagramGenerator._is_sensitive_file(path)


@pytest.mark.parametrize(
    "directory",
    [
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".terraform",
        "terraform",
        "coverage",
        ".next",
        ".idea",
    ],
)
def test_skipped_directories_are_identified(directory):
    assert ProjectDiagramGenerator._should_skip_directory(
        directory
    )


def test_skipped_directory_matching_is_case_insensitive():
    assert ProjectDiagramGenerator._should_skip_directory(
        "NODE_MODULES"
    )


def test_normal_directory_is_not_skipped():
    assert not ProjectDiagramGenerator._should_skip_directory(
        "agentops"
    )


def test_discover_workspace_finds_root_files(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
        "README.md": "# Project",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    paths = {item["path"] for item in discovered}

    assert paths == {
        "main.py",
        "README.md",
    }
    assert warnings == []


def test_discover_workspace_finds_nested_files(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "src/main.py": "print('hello')",
        "src/agents/developer.py": "class Developer: pass",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    paths = {item["path"] for item in discovered}

    assert paths == {
        "src/main.py",
        "src/agents/developer.py",
    }
    assert warnings == []


def test_discover_workspace_skips_sensitive_files(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
        ".env": "SECRET=value",
        "config/credentials.json": "{}",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    paths = {item["path"] for item in discovered}

    assert paths == {"main.py"}
    assert len(warnings) == 2
    assert all("sensitive file excluded" in item for item in warnings)


def test_discover_workspace_skips_excluded_directories(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
        ".venv/lib/python.py": "excluded",
        "node_modules/package/index.js": "excluded",
        ".git/config": "excluded",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    paths = {item["path"] for item in discovered}

    assert paths == {"main.py"}
    assert warnings == []


def test_discover_workspace_uses_qa_read_only_role(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
    }

    generator._discover_workspace(TEST_TASK_ID)

    assert workspace_manager.list_calls
    assert all(
        call[2] == "qa"
        for call in workspace_manager.list_calls
    )


def test_discover_workspace_enforces_file_size_limit(
    workspace_manager,
    generator,
):
    generator.MAX_FILE_SIZE_BYTES = 5

    workspace_manager.files = {
        "small.py": "12345",
        "large.py": "123456",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    paths = {item["path"] for item in discovered}

    assert paths == {"small.py"}
    assert any("file exceeds size limit" in item for item in warnings)


def test_discover_workspace_enforces_file_count_limit(
    workspace_manager,
    generator,
):
    generator.MAX_FILES = 2

    workspace_manager.files = {
        "first.py": "first",
        "second.py": "second",
        "third.py": "third",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    assert len(discovered) == 2
    assert len(warnings) == 1
    assert "maximum file count reached" in warnings[0]


def test_discover_workspace_reports_excessive_directory_depth(
    workspace_manager,
    generator,
):
    generator.MAX_DIRECTORY_DEPTH = 1

    workspace_manager.files = {
        "level1/level2/main.py": "print('hello')",
    }

    discovered, warnings = generator._discover_workspace(
        TEST_TASK_ID
    )

    assert discovered == []
    assert any(
        "maximum directory depth reached" in item
        for item in warnings
    )


def test_discover_workspace_checks_cancellation(
    workspace_manager,
    generator,
):
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        generator._discover_workspace(
            TEST_TASK_ID,
            cancel_event,
        )


def test_read_python_sources_reads_only_python_files(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
        "README.md": "# Project",
        "config.json": "{}",
    }

    discovered = [
        {"path": path, "size_bytes": len(content), "content": None}
        for path, content in workspace_manager.files.items()
    ]

    python_files, warnings = generator._read_python_sources(
        TEST_TASK_ID,
        discovered,
    )

    assert python_files == [
        {
            "path": "main.py",
            "content": "print('hello')",
        }
    ]
    assert warnings == []


def test_read_python_sources_uses_qa_read_only_role(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
    }

    discovered = [
        {"path": "main.py", "size_bytes": 14, "content": None}
    ]

    generator._read_python_sources(
        TEST_TASK_ID,
        discovered,
    )

    assert workspace_manager.read_calls == [
        (TEST_TASK_ID, "main.py", "qa")
    ]


def test_read_python_sources_reports_read_errors(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
    }
    workspace_manager.read_errors = {
        "main.py": PermissionError("Access denied"),
    }

    discovered = [
        {"path": "main.py", "size_bytes": 14, "content": None}
    ]

    python_files, warnings = generator._read_python_sources(
        TEST_TASK_ID,
        discovered,
    )

    assert python_files == []
    assert len(warnings) == 1
    assert "main.py: read failed (PermissionError)" in warnings[0]


def test_read_python_sources_rejects_non_string_content(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
    }

    original_read_file = workspace_manager.read_file

    def invalid_read_file(*args, **kwargs):
        original_read_file(*args, **kwargs)
        return None

    workspace_manager.read_file = invalid_read_file

    discovered = [
        {"path": "main.py", "size_bytes": 14, "content": None}
    ]

    python_files, warnings = generator._read_python_sources(
        TEST_TASK_ID,
        discovered,
    )

    assert python_files == []
    assert any(
        "unsupported source content" in item
        for item in warnings
    )


def test_read_python_sources_checks_cancellation(
    workspace_manager,
    generator,
):
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        generator._read_python_sources(
            TEST_TASK_ID,
            [{"path": "main.py"}],
            cancel_event,
        )


def test_generate_project_structure_creates_flowchart(
    generator,
):
    diagram = generator.generate_project_structure(
        [
            {"path": "main.py"},
            {"path": "src/agent.py"},
        ]
    )

    assert diagram.startswith("flowchart TD")
    assert 'root["Project Workspace"]' in diagram
    assert "main.py" in diagram
    assert "src/" in diagram
    assert "agent.py" in diagram


def test_generate_project_structure_is_deterministic(
    generator,
):
    files = [
        {"path": "zebra.py"},
        {"path": "alpha.py"},
    ]

    first = generator.generate_project_structure(files)
    second = generator.generate_project_structure(files)

    assert first == second


def test_generate_project_structure_sanitizes_labels(
    generator,
):
    diagram = generator.generate_project_structure(
        [
            {"path": 'src/file["test"].py'},
        ]
    )

    assert "&#91;" in diagram
    assert "&#93;" in diagram
    assert '"test"' not in diagram


def test_generate_project_structure_handles_empty_input(
    generator,
):
    diagram = generator.generate_project_structure([])

    assert diagram == (
        "flowchart TD\n"
        '    root["Project Workspace"]'
    )


def test_generate_project_structure_enforces_node_limit(
    generator,
):
    generator.MAX_STRUCTURE_NODES = 2

    diagram = generator.generate_project_structure(
        [
            {"path": "first.py"},
            {"path": "second.py"},
        ]
    )

    assert "Additional files omitted from diagram" in diagram


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("main.py", "main"),
        ("src/agent.py", "src.agent"),
        ("src/agents/developer.py", "src.agents.developer"),
        ("src/agents/__init__.py", "src.agents"),
        ("package/__init__.py", "package"),
    ],
)
def test_module_name_resolves_valid_python_paths(
    path,
    expected,
):
    assert ProjectDiagramGenerator._module_name(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "main.txt",
        "invalid-name/file.py",
        "123/file.py",
        "__init__.py",
    ],
)
def test_module_name_rejects_invalid_or_non_python_paths(path):
    assert ProjectDiagramGenerator._module_name(path) is None


def test_resolve_relative_import_from_package():
    result = ProjectDiagramGenerator._resolve_relative_import(
        current_module="package.module",
        is_package=False,
        module="helpers",
        level=1,
    )

    assert result == "package.helpers"


def test_resolve_relative_import_from_package_initializer():
    result = ProjectDiagramGenerator._resolve_relative_import(
        current_module="package",
        is_package=True,
        module="helpers",
        level=1,
    )

    assert result == "package.helpers"


def test_resolve_relative_import_with_parent_directory():
    result = ProjectDiagramGenerator._resolve_relative_import(
        current_module="package.sub.module",
        is_package=False,
        module="helpers",
        level=2,
    )

    assert result == "package.helpers"


def test_extract_python_imports_identifies_direct_local_imports(
    generator,
):
    files = [
        {
            "path": "main.py",
            "content": "import helpers",
        },
        {
            "path": "helpers.py",
            "content": "def help_function(): pass",
        },
    ]

    dependencies, warnings = generator._extract_python_imports(
        files
    )

    assert dependencies == {
        "main": {"helpers"},
        "helpers": set(),
    }
    assert warnings == []


def test_extract_python_imports_identifies_from_imports(
    generator,
):
    files = [
        {
            "path": "main.py",
            "content": "from helpers import help_function",
        },
        {
            "path": "helpers.py",
            "content": "def help_function(): pass",
        },
    ]

    dependencies, warnings = generator._extract_python_imports(
        files
    )

    assert "helpers" in dependencies["main"]
    assert warnings == []


def test_extract_python_imports_identifies_relative_imports(
    generator,
):
    files = [
        {
            "path": "package/main.py",
            "content": "from . import helpers",
        },
        {
            "path": "package/helpers.py",
            "content": "def help_function(): pass",
        },
    ]

    dependencies, warnings = generator._extract_python_imports(
        files
    )

    assert "package.helpers" in dependencies["package.main"]
    assert warnings == []


def test_extract_python_imports_ignores_external_dependencies(
    generator,
):
    files = [
        {
            "path": "main.py",
            "content": "import os\nimport requests\nimport pytest",
        },
    ]

    dependencies, warnings = generator._extract_python_imports(
        files
    )

    assert dependencies == {"main": set()}
    assert warnings == []


def test_extract_python_imports_reports_invalid_python_syntax(
    generator,
):
    files = [
        {
            "path": "main.py",
            "content": "def broken(:\n    pass",
        },
    ]

    dependencies, warnings = generator._extract_python_imports(
        files
    )

    assert dependencies == {}
    assert len(warnings) == 1
    assert "Python syntax could not be parsed" in warnings[0]


def test_extract_python_imports_ignores_invalid_module_paths(
    generator,
):
    files = [
        {
            "path": "invalid-name.py",
            "content": "import helpers",
        },
        {
            "path": "helpers.py",
            "content": "pass",
        },
    ]

    dependencies, warnings = generator._extract_python_imports(
        files
    )

    assert "helpers" in dependencies
    assert "invalid-name" not in dependencies
    assert warnings == []


def test_extract_python_imports_checks_cancellation(
    generator,
):
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        generator._extract_python_imports(
            [
                {
                    "path": "main.py",
                    "content": "import os",
                }
            ],
            cancel_event,
        )


def test_generate_python_dependencies_creates_directed_edges(
    generator,
):
    diagram = generator.generate_python_dependencies(
        {
            "main": {"helpers"},
            "helpers": set(),
        }
    )

    assert diagram.startswith("flowchart LR")
    assert diagram.count("-->") == 1


def test_generate_python_dependencies_handles_empty_input(
    generator,
):
    diagram = generator.generate_python_dependencies({})

    assert diagram.startswith("flowchart LR")
    assert 'empty["No Python modules discovered"]' in diagram


def test_generate_python_dependencies_enforces_node_limit(
    generator,
):
    generator.MAX_DEPENDENCY_NODES = 2

    diagram = generator.generate_python_dependencies(
        {
            "alpha": set(),
            "beta": set(),
            "gamma": set(),
        }
    )

    assert "Additional modules omitted" in diagram


def test_generate_python_dependencies_enforces_edge_limit(
    generator,
):
    generator.MAX_DEPENDENCY_EDGES = 1

    diagram = generator.generate_python_dependencies(
        {
            "alpha": {"beta", "gamma"},
            "beta": set(),
            "gamma": set(),
        }
    )

    assert "Additional dependency edges omitted" in diagram


def test_component_for_path_identifies_agents():
    generator = ProjectDiagramGenerator()

    assert generator._component_for_path(
        "agents/developer.py"
    ) == "Agents"


def test_component_for_path_identifies_orchestration():
    generator = ProjectDiagramGenerator()

    assert generator._component_for_path(
        "orchestrator/workflow.py"
    ) == "Orchestration"


def test_component_for_path_identifies_api():
    generator = ProjectDiagramGenerator()

    assert generator._component_for_path(
        "routes/tasks.py"
    ) == "API"


def test_component_for_path_defaults_to_other():
    generator = ProjectDiagramGenerator()

    assert generator._component_for_path(
        "misc/helper.py"
    ) == "Other"


def test_generate_component_architecture_groups_modules(
    generator,
):
    python_files = [
        {
            "path": "agents/developer.py",
            "content": "pass",
        },
        {
            "path": "tools/file_manager.py",
            "content": "pass",
        },
    ]

    diagram = generator.generate_component_architecture(
        python_files,
        {},
    )

    assert diagram.startswith("flowchart LR")
    assert "Agents" in diagram
    assert "Tools" in diagram
    assert "agents.developer" in diagram
    assert "tools.file_manager" in diagram


def test_generate_component_architecture_uses_import_relationships(
    generator,
):
    python_files = [
        {
            "path": "agents/developer.py",
            "content": "pass",
        },
        {
            "path": "tools/file_manager.py",
            "content": "pass",
        },
    ]

    dependencies = {
        "agents.developer": {"tools.file_manager"},
        "tools.file_manager": set(),
    }

    diagram = generator.generate_component_architecture(
        python_files,
        dependencies,
    )

    assert "Agents" in diagram
    assert "Tools" in diagram
    assert diagram.count("-->") == 1


def test_generate_component_architecture_ignores_same_component_edges(
    generator,
):
    python_files = [
        {
            "path": "agents/developer.py",
            "content": "pass",
        },
        {
            "path": "agents/qa.py",
            "content": "pass",
        },
    ]

    dependencies = {
        "agents.developer": {"agents.qa"},
        "agents.qa": set(),
    }

    diagram = generator.generate_component_architecture(
        python_files,
        dependencies,
    )

    assert diagram.count("-->") == 0


def test_generate_component_architecture_handles_empty_input(
    generator,
):
    diagram = generator.generate_component_architecture(
        [],
        {},
    )

    assert 'empty["No recognizable Python components found"]' in diagram


def test_generate_all_returns_expected_structure(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "import helpers",
        "helpers.py": "def help_function(): pass",
        "README.md": "# Project",
    }

    result = generator.generate_all(TEST_TASK_ID)

    assert set(result) == {
        "diagrams",
        "metadata",
        "warnings",
        "workspace_modified",
    }

    assert set(result["diagrams"]) == {
        "project_structure",
        "python_dependencies",
        "component_architecture",
    }


def test_generate_all_reports_discovery_metadata(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "import helpers",
        "helpers.py": "pass",
        "README.md": "# Project",
    }

    result = generator.generate_all(TEST_TASK_ID)

    metadata = result["metadata"]

    assert metadata["files_discovered"] == 3
    assert metadata["python_files_read"] == 2
    assert metadata["python_modules_identified"] == 2
    assert metadata["local_import_relationships"] == 1
    assert metadata["warnings_count"] == 0


def test_generate_all_reports_workspace_as_unmodified(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
    }

    result = generator.generate_all(TEST_TASK_ID)

    assert result["workspace_modified"] is False
    assert workspace_manager.write_calls == []


def test_generate_all_reports_read_warnings(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "print('hello')",
    }
    workspace_manager.read_errors = {
        "main.py": PermissionError("Access denied"),
    }

    result = generator.generate_all(TEST_TASK_ID)

    assert result["metadata"]["warnings_count"] == 1
    assert any(
        "read failed" in warning
        for warning in result["warnings"]
    )


def test_generate_all_reports_parse_warnings(
    workspace_manager,
    generator,
):
    workspace_manager.files = {
        "main.py": "def broken(:\n    pass",
    }

    result = generator.generate_all(TEST_TASK_ID)

    assert result["metadata"]["warnings_count"] == 1
    assert any(
        "Python syntax could not be parsed" in warning
        for warning in result["warnings"]
    )


def test_generate_all_checks_cancellation_before_discovery(
    workspace_manager,
    generator,
):
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(TaskCancelledError):
        generator.generate_all(
            TEST_TASK_ID,
            cancel_event,
        )

    assert workspace_manager.list_calls == []
