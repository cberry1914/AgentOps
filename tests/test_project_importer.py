
from pathlib import Path

import pytest

from agentops.tools.file_manager import (
    WorkspaceError,
    WorkspaceManager,
)
from agentops.tools.project_importer import (
    ProjectImportError,
    ProjectImporter,
    ProjectImportResult,
)


TEST_TASK_ID = "12345678-1234-1234-1234-123456789012"


@pytest.fixture
def workspace_manager(tmp_path):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces"
    )
    manager.create_workspace(TEST_TASK_ID)
    return manager


@pytest.fixture
def importer(workspace_manager):
    return ProjectImporter(workspace_manager)


@pytest.fixture
def source_dir(tmp_path):
    source = tmp_path / "source-project"
    source.mkdir()
    return source


def create_file(directory, relative_path, content="sample content"):
    path = directory / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def workspace_files(manager):
    return manager.list_files(
        task_id=TEST_TASK_ID,
        relative_path=".",
        agent="orchestrator",
    )


# ---------------------------------------------------------
# Constructor and result model
# ---------------------------------------------------------


def test_importer_accepts_workspace_manager(workspace_manager):
    importer = ProjectImporter(workspace_manager)

    assert importer.workspace_manager is workspace_manager


def test_project_import_error_is_exception():
    assert issubclass(ProjectImportError, Exception)


def test_import_result_fields():
    result = ProjectImportResult(
        source_directory="/source",
        task_id=TEST_TASK_ID,
        files_imported=3,
        total_bytes=150,
        workspace_directory="/workspace",
    )

    assert result.source_directory == "/source"
    assert result.task_id == TEST_TASK_ID
    assert result.files_imported == 3
    assert result.total_bytes == 150
    assert result.workspace_directory == "/workspace"


def test_default_import_limits(importer):
    assert importer.MAX_FILES == 100
    assert importer.MAX_FILE_BYTES == 250_000
    assert importer.MAX_TOTAL_BYTES == 5_000_000
    assert importer.MAX_DIRECTORY_DEPTH == 10


# ---------------------------------------------------------
# Protected filename detection
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "credentials",
        "secrets",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
        "authorized_keys",
        "known_hosts",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "terraform.tfstate",
        "terraform.tfstate.backup",
    ],
)
def test_protected_filenames_are_rejected(importer, filename):
    assert importer._is_protected_file(filename)


def test_protected_filenames_are_case_insensitive(importer):
    assert importer._is_protected_file("SECRETS.JSON")


@pytest.mark.parametrize(
    "filename",
    [
        ".env",
        ".env.local",
        ".env.production",
        ".ENV",
        ".ENV.DEVELOPMENT",
    ],
)
def test_environment_files_are_protected(importer, filename):
    assert importer._is_protected_file(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "terraform.tfstate",
        "terraform.tfstate.backup",
        "terraform.tfstate.backup.old",
    ],
)
def test_terraform_state_variants_are_protected(
    importer,
    filename,
):
    assert importer._is_protected_file(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "private.pem",
        "server.key",
        "certificate.p12",
        "certificate.pfx",
        "private.ppk",
        "database.sqlite",
        "database.db",
        "database.dump",
        "backup.bak",
        "application.bin",
        "application.exe",
        "library.dll",
        "library.so",
        "library.dylib",
        "archive.zip",
        "archive.tar",
        "archive.gz",
        "archive.7z",
        "archive.rar",
        "document.pdf",
        "document.docx",
        "spreadsheet.xlsx",
        "presentation.pptx",
        "image.png",
        "image.jpg",
        "image.jpeg",
        "image.gif",
        "image.webp",
        "image.ico",
        "audio.mp3",
        "video.mp4",
        "video.mov",
        "audio.wav",
    ],
)
def test_protected_extensions_are_rejected(
    importer,
    filename,
):
    assert importer._is_protected_file(filename)


def test_ordinary_python_file_is_not_protected(importer):
    assert not importer._is_protected_file("service.py")


def test_similar_but_unprotected_filename_is_allowed(importer):
    assert not importer._is_protected_file("secrets_helper.py")


# ---------------------------------------------------------
# Allowed file detection
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "extension",
    sorted(ProjectImporter.ALLOWED_EXTENSIONS),
)
def test_allowed_extensions_are_recognized(
    importer,
    extension,
):
    assert importer._is_allowed_file(
        f"example{extension}"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "README",
        "LICENSE",
        "NOTICE",
        "Dockerfile",
        "Makefile",
        "Procfile",
        "Justfile",
        "Taskfile",
        "Gemfile",
        "Rakefile",
        "Jenkinsfile",
    ],
)
def test_allowed_extensionless_filenames(
    importer,
    filename,
):
    assert importer._is_allowed_file(filename)


def test_extension_matching_is_case_insensitive(importer):
    assert importer._is_allowed_file("SERVICE.PY")
    assert importer._is_allowed_file("CONFIG.YAML")


def test_extensionless_matching_is_case_insensitive(importer):
    assert importer._is_allowed_file("README")


def test_unknown_extension_is_rejected(importer):
    assert not importer._is_allowed_file("program.unknown")


def test_protected_extension_overrides_allowed_extension(
    importer,
):
    assert not importer._is_allowed_file("database.db")


def test_env_file_is_not_allowed(importer):
    assert not importer._is_allowed_file(".env")


def test_env_variant_is_not_allowed(importer):
    assert not importer._is_allowed_file(".env.local")


# ---------------------------------------------------------
# Skipped directory detection
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "directory",
    sorted(ProjectImporter.SKIPPED_DIRECTORIES),
)
def test_skipped_directories_are_defined(
    importer,
    directory,
):
    assert directory in importer.SKIPPED_DIRECTORIES


@pytest.mark.parametrize(
    "directory",
    [
        ".git",
        ".ssh",
        ".aws",
        ".azure",
        ".gnupg",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".terraform",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "workspaces",
        "site-packages",
        ".next",
        ".nuxt",
        "coverage",
    ],
)
def test_files_in_skipped_directories_are_excluded(
    importer,
    source_dir,
    directory,
):
    create_file(
        source_dir,
        f"{directory}/hidden.py",
        "secret content",
    )

    create_file(
        source_dir,
        "visible.py",
        "visible content",
    )

    collected = importer._collect_project_files(source_dir)

    paths = [item[0] for item in collected]

    assert "visible.py" in paths
    assert f"{directory}/hidden.py" not in paths


def test_skipped_directory_matching_is_case_insensitive(
    importer,
    source_dir,
):
    create_file(
        source_dir,
        ".GIT/hidden.py",
        "hidden",
    )

    create_file(
        source_dir,
        "visible.py",
        "visible",
    )

    collected = importer._collect_project_files(source_dir)

    assert [item[0] for item in collected] == ["visible.py"]


# ---------------------------------------------------------
# Source file collection
# ---------------------------------------------------------


def test_collects_eligible_python_file(importer, source_dir):
    create_file(
        source_dir,
        "service.py",
        "print('hello')",
    )

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 1
    assert collected[0][0] == "service.py"
    assert collected[0][1] == "print('hello')"
    assert collected[0][2] == len("print('hello')".encode())


def test_collects_nested_files(importer, source_dir):
    create_file(
        source_dir,
        "src/agents/developer.py",
        "class Developer: pass",
    )

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 1
    assert collected[0][0] == "src/agents/developer.py"


def test_uses_posix_relative_paths(importer, source_dir):
    create_file(
        source_dir,
        "src/services/api.py",
        "content",
    )

    collected = importer._collect_project_files(source_dir)

    assert collected[0][0] == "src/services/api.py"


def test_collects_multiple_eligible_files(importer, source_dir):
    create_file(source_dir, "main.py", "main")
    create_file(source_dir, "README.md", "readme")
    create_file(source_dir, "config.yaml", "config")

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 3


def test_collection_order_is_deterministic(importer, source_dir):
    create_file(source_dir, "zebra.py", "zebra")
    create_file(source_dir, "alpha.py", "alpha")
    create_file(source_dir, "middle.py", "middle")

    collected = importer._collect_project_files(source_dir)

    paths = [item[0] for item in collected]

    assert paths == [
        "alpha.py",
        "middle.py",
        "zebra.py",
    ]


def test_only_eligible_files_are_collected(importer, source_dir):
    create_file(source_dir, "main.py", "main")
    create_file(source_dir, "image.png", "not imported")
    create_file(source_dir, "unknown.xyz", "not imported")
    create_file(source_dir, ".env", "secret")

    collected = importer._collect_project_files(source_dir)

    assert [item[0] for item in collected] == ["main.py"]


def test_no_eligible_files_raises_error(importer, source_dir):
    create_file(source_dir, "image.png", "image")
    create_file(source_dir, "unknown.xyz", "unknown")

    with pytest.raises(
        ProjectImportError,
        match="No eligible source",
    ):
        importer._collect_project_files(source_dir)


def test_empty_directory_raises_error(importer, source_dir):
    with pytest.raises(
        ProjectImportError,
        match="No eligible source",
    ):
        importer._collect_project_files(source_dir)


# ---------------------------------------------------------
# Symbolic link protection
# ---------------------------------------------------------


def test_source_directory_symlink_is_rejected(
    importer,
    tmp_path,
    source_dir,
):
    symlink = tmp_path / "source-link"
    symlink.symlink_to(source_dir, target_is_directory=True)

    with pytest.raises(
        ProjectImportError,
        match="symbolic link",
    ):
        importer.import_project(symlink, TEST_TASK_ID)


def test_file_symlink_is_rejected(
    importer,
    source_dir,
    tmp_path,
):
    target = tmp_path / "target.py"
    target.write_text("safe content", encoding="utf-8")

    (source_dir / "linked.py").symlink_to(target)

    with pytest.raises(
        ProjectImportError,
        match="Symbolic links are not allowed",
    ):
        importer._collect_project_files(source_dir)


def test_directory_symlink_is_rejected(
    importer,
    source_dir,
    tmp_path,
):
    target = tmp_path / "target"
    target.mkdir()

    (source_dir / "linked").symlink_to(
        target,
        target_is_directory=True,
    )

    with pytest.raises(
        ProjectImportError,
        match="Symbolic links are not allowed",
    ):
        importer._collect_project_files(source_dir)


def test_symlink_inside_skipped_directory_is_ignored(
    importer,
    source_dir,
    tmp_path,
):
    target = tmp_path / "target"
    target.mkdir()

    git_dir = source_dir / ".git"
    git_dir.mkdir()

    (git_dir / "linked").symlink_to(
        target,
        target_is_directory=True,
    )

    create_file(
        source_dir,
        "visible.py",
        "visible content",
    )

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 1
    assert collected[0][0] == "visible.py"


# ---------------------------------------------------------
# File and project size limits
# ---------------------------------------------------------


def test_file_size_limit_is_enforced(
    importer,
    source_dir,
    monkeypatch,
):
    importer.MAX_FILE_BYTES = 10

    create_file(
        source_dir,
        "large.py",
        "X" * 11,
    )

    with pytest.raises(
        ProjectImportError,
        match="File exceeds the size limit",
    ):
        importer._collect_project_files(source_dir)


def test_file_at_exact_size_limit_is_allowed(
    importer,
    source_dir,
    monkeypatch,
):
    importer.MAX_FILE_BYTES = 10

    create_file(
        source_dir,
        "exact.py",
        "X" * 10,
    )

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 1
    assert collected[0][2] == 10


def test_total_project_size_limit_is_enforced(
    importer,
    source_dir,
):
    importer.MAX_FILE_BYTES = 10
    importer.MAX_TOTAL_BYTES = 15

    create_file(source_dir, "first.py", "A" * 10)
    create_file(source_dir, "second.py", "B" * 6)

    with pytest.raises(
        ProjectImportError,
        match="total import size limit",
    ):
        importer._collect_project_files(source_dir)


def test_total_size_at_exact_limit_is_allowed(
    importer,
    source_dir,
):
    importer.MAX_FILE_BYTES = 10
    importer.MAX_TOTAL_BYTES = 15

    create_file(source_dir, "first.py", "A" * 10)
    create_file(source_dir, "second.py", "B" * 5)

    collected = importer._collect_project_files(source_dir)

    assert sum(item[2] for item in collected) == 15


def test_file_count_limit_is_enforced(
    importer,
    source_dir,
):
    importer.MAX_FILES = 2

    create_file(source_dir, "first.py", "first")
    create_file(source_dir, "second.py", "second")
    create_file(source_dir, "third.py", "third")

    with pytest.raises(
        ProjectImportError,
        match="2-file limit",
    ):
        importer._collect_project_files(source_dir)


def test_file_count_at_exact_limit_is_allowed(
    importer,
    source_dir,
):
    importer.MAX_FILES = 2

    create_file(source_dir, "first.py", "first")
    create_file(source_dir, "second.py", "second")

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 2


# ---------------------------------------------------------
# Directory depth limits
# ---------------------------------------------------------


def test_directory_depth_limit_is_enforced(
    importer,
    source_dir,
):
    importer.MAX_DIRECTORY_DEPTH = 2

    create_file(
        source_dir,
        "one/two/three/service.py",
        "content",
    )

    with pytest.raises(
        ProjectImportError,
        match="maximum directory depth",
    ):
        importer._collect_project_files(source_dir)


def test_directory_at_exact_depth_limit_is_allowed(
    importer,
    source_dir,
):
    importer.MAX_DIRECTORY_DEPTH = 2

    create_file(
        source_dir,
        "one/two/service.py",
        "content",
    )

    collected = importer._collect_project_files(source_dir)

    assert len(collected) == 1
    assert collected[0][0] == "one/two/service.py"


def test_empty_deep_directory_also_counts_toward_depth(
    importer,
    source_dir,
):
    importer.MAX_DIRECTORY_DEPTH = 1

    (source_dir / "one" / "two").mkdir(parents=True)

    create_file(source_dir, "visible.py", "content")

    with pytest.raises(
        ProjectImportError,
        match="maximum directory depth",
    ):
        importer._collect_project_files(source_dir)


# ---------------------------------------------------------
# Text encoding and filesystem errors
# ---------------------------------------------------------


def test_invalid_utf8_file_is_rejected(
    importer,
    source_dir,
):
    path = source_dir / "invalid.py"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(
        ProjectImportError,
        match="not valid UTF-8",
    ):
        importer._collect_project_files(source_dir)


# ---------------------------------------------------------
# Successful project import
# ---------------------------------------------------------


def test_import_project_returns_result(
    importer,
    source_dir,
):
    create_file(source_dir, "main.py", "print('hello')")

    result = importer.import_project(
        source_dir,
        TEST_TASK_ID,
    )

    assert isinstance(result, ProjectImportResult)


def test_import_project_reports_imported_file_count(
    importer,
    source_dir,
):
    create_file(source_dir, "main.py", "main")
    create_file(source_dir, "README.md", "readme")

    result = importer.import_project(
        source_dir,
        TEST_TASK_ID,
    )

    assert result.files_imported == 2


def test_import_project_reports_original_byte_count(
    importer,
    source_dir,
):
    content = "print('hello')"
    create_file(source_dir, "main.py", content)

    result = importer.import_project(
        source_dir,
        TEST_TASK_ID,
    )

    assert result.total_bytes == len(content.encode("utf-8"))


def test_import_project_normalizes_task_id(
    importer,
    source_dir,
):
    create_file(source_dir, "main.py", "main")

    result = importer.import_project(
        source_dir,
        TEST_TASK_ID.upper(),
    )

    assert result.task_id == TEST_TASK_ID


def test_import_project_reports_resolved_source_directory(
    importer,
    source_dir,
):
    create_file(source_dir, "main.py", "main")

    result = importer.import_project(
        source_dir,
        TEST_TASK_ID,
    )

    assert result.source_directory == str(source_dir.resolve())


def test_import_project_reports_workspace_directory(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "main.py", "main")

    result = importer.import_project(
        source_dir,
        TEST_TASK_ID,
    )

    expected = str(
        workspace_manager.root_dir / TEST_TASK_ID
    )

    assert result.workspace_directory == expected


def test_import_project_writes_file_to_workspace(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(
        source_dir,
        "main.py",
        "print('hello')",
    )

    importer.import_project(source_dir, TEST_TASK_ID)

    imported = workspace_manager.read_file(
        task_id=TEST_TASK_ID,
        relative_path="main.py",
        agent="orchestrator",
    )

    assert imported == "print('hello')"


def test_import_project_creates_nested_directories(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(
        source_dir,
        "src/agents/developer.py",
        "class Developer: pass",
    )

    importer.import_project(source_dir, TEST_TASK_ID)

    imported = workspace_manager.read_file(
        task_id=TEST_TASK_ID,
        relative_path="src/agents/developer.py",
        agent="orchestrator",
    )

    assert imported == "class Developer: pass"


def test_import_project_preserves_relative_paths(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(
        source_dir,
        "src/api/main.py",
        "application",
    )

    importer.import_project(source_dir, TEST_TASK_ID)

    imported = workspace_manager.read_file(
        task_id=TEST_TASK_ID,
        relative_path="src/api/main.py",
        agent="orchestrator",
    )

    assert imported == "application"


def test_import_project_does_not_import_protected_files(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "main.py", "main")
    create_file(source_dir, ".env", "SECRET=value")
    create_file(source_dir, "credentials", "secret")
    create_file(source_dir, "private.pem", "private key")

    importer.import_project(source_dir, TEST_TASK_ID)

    paths = [
        item["path"]
        for item in workspace_files(workspace_manager)
    ]

    assert "main.py" in paths
    assert ".env" not in paths
    assert "credentials" not in paths
    assert "private.pem" not in paths


def test_import_project_does_not_modify_source(
    importer,
    source_dir,
):
    original = "original source content"
    create_file(source_dir, "main.py", original)

    importer.import_project(source_dir, TEST_TASK_ID)

    assert (
        source_dir / "main.py"
    ).read_text(encoding="utf-8") == original


# ---------------------------------------------------------
# Source and destination validation
# ---------------------------------------------------------


def test_missing_source_directory_is_rejected(
    importer,
    tmp_path,
):
    missing = tmp_path / "missing"

    with pytest.raises(
        ProjectImportError,
        match="does not exist",
    ):
        importer.import_project(missing, TEST_TASK_ID)


def test_source_file_is_rejected(
    importer,
    tmp_path,
):
    source_file = tmp_path / "source.py"
    source_file.write_text("content", encoding="utf-8")

    with pytest.raises(
        ProjectImportError,
        match="does not exist",
    ):
        importer.import_project(source_file, TEST_TASK_ID)


def test_import_from_inside_workspace_root_is_rejected(
    importer,
    workspace_manager,
):
    source = workspace_manager.root_dir / TEST_TASK_ID
    create_file(source, "main.py", "main")

    with pytest.raises(
        ProjectImportError,
        match="cannot be used as an import source",
    ):
        importer.import_project(source, TEST_TASK_ID)


def test_import_into_nonempty_workspace_is_rejected(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "main.py", "main")

    workspace_manager.write_file(
        task_id=TEST_TASK_ID,
        relative_path="existing.py",
        content="existing content",
        agent="orchestrator",
    )

    with pytest.raises(
        ProjectImportError,
        match="must be empty",
    ):
        importer.import_project(source_dir, TEST_TASK_ID)


def test_nonempty_workspace_is_not_overwritten(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "existing.py", "new content")

    workspace_manager.write_file(
        task_id=TEST_TASK_ID,
        relative_path="existing.py",
        content="original content",
        agent="orchestrator",
    )

    with pytest.raises(ProjectImportError):
        importer.import_project(source_dir, TEST_TASK_ID)

    content = workspace_manager.read_file(
        task_id=TEST_TASK_ID,
        relative_path="existing.py",
        agent="orchestrator",
    )

    assert content == "original content"


# ---------------------------------------------------------
# Preflight and partial-import protection
# ---------------------------------------------------------


def test_invalid_file_does_not_create_partial_import(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "valid.py", "valid")
    create_file(source_dir, "large.py", "X" * 300_000)

    with pytest.raises(
        ProjectImportError,
        match="File exceeds the size limit",
    ):
        importer.import_project(source_dir, TEST_TASK_ID)

    assert workspace_files(workspace_manager) == []


def test_invalid_utf8_does_not_create_partial_import(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "valid.py", "valid")
    (source_dir / "invalid.py").write_bytes(b"\xff\xfe")

    with pytest.raises(
        ProjectImportError,
        match="not valid UTF-8",
    ):
        importer.import_project(source_dir, TEST_TASK_ID)

    assert workspace_files(workspace_manager) == []


def test_symlink_does_not_create_partial_import(
    importer,
    source_dir,
    tmp_path,
    workspace_manager,
):
    create_file(source_dir, "valid.py", "valid")

    target = tmp_path / "target.py"
    target.write_text("target", encoding="utf-8")

    (source_dir / "linked.py").symlink_to(target)

    with pytest.raises(ProjectImportError):
        importer.import_project(source_dir, TEST_TASK_ID)

    assert workspace_files(workspace_manager) == []


def test_empty_import_does_not_create_files(
    importer,
    source_dir,
    workspace_manager,
):
    create_file(source_dir, "image.png", "image")

    with pytest.raises(ProjectImportError):
        importer.import_project(source_dir, TEST_TASK_ID)

    assert workspace_files(workspace_manager) == []


# ---------------------------------------------------------
# Workspace operation failures
# ---------------------------------------------------------


def test_workspace_errors_are_wrapped_as_import_errors(
    source_dir,
):
    class FailingWorkspaceManager:
        root_dir = Path("/tmp/agentops-test")

        def list_files(self, **kwargs):
            raise WorkspaceError("Workspace unavailable")

    importer = ProjectImporter(FailingWorkspaceManager())

    create_file(source_dir, "main.py", "main")

    with pytest.raises(
        ProjectImportError,
        match="Workspace operation failed",
    ):
        importer.import_project(source_dir, TEST_TASK_ID)


def test_workspace_path_validation_errors_are_wrapped(
    source_dir,
):
    class RejectingWorkspaceManager:
        root_dir = Path("/tmp/agentops-test")

        def _validate_relative_path(self, relative_path):
            raise WorkspaceError("Path rejected")

    importer = ProjectImporter(RejectingWorkspaceManager())

    create_file(source_dir, "main.py", "main")

    with pytest.raises(
        ProjectImportError,
        match="prohibited path",
    ):
        importer._collect_project_files(source_dir)
