
import uuid

import pytest

from agentops.tools.file_manager import (
    WorkspaceError,
    WorkspaceManager,
    WorkspacePermissionError,
)


@pytest.fixture
def manager(tmp_path):
    return WorkspaceManager(root_dir=tmp_path / "workspaces")


@pytest.fixture
def task_id():
    return str(uuid.uuid4())


@pytest.fixture
def workspace(manager, task_id):
    return manager.create_workspace(task_id)


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_default_root_directory_uses_workspaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manager = WorkspaceManager()

    assert manager.root_dir == (tmp_path / "workspaces").resolve()
    assert manager.root_dir.is_dir()


def test_custom_root_directory_is_created(tmp_path):
    root = tmp_path / "custom" / "workspaces"

    manager = WorkspaceManager(root_dir=root)

    assert manager.root_dir == root.resolve()
    assert root.is_dir()


@pytest.mark.parametrize("size", [0, -1, -100])
def test_invalid_max_file_size_is_rejected(tmp_path, size):
    with pytest.raises(ValueError):
        WorkspaceManager(
            root_dir=tmp_path / "workspaces",
            max_file_size_bytes=size,
        )


def test_configured_max_file_size_is_preserved(tmp_path):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces",
        max_file_size_bytes=500,
    )

    assert manager.max_file_size_bytes == 500


# ---------------------------------------------------------
# Task ID validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_id",
    [
        "",
        "not-a-uuid",
        "../12345678-1234-1234-1234-123456789abc",
        "12345678-1234-1234-1234-123456789ab",
        "12345678-1234-1234-1234-123456789abcd",
        None,
        12345,
    ],
)
def test_invalid_task_ids_are_rejected(manager, invalid_id):
    with pytest.raises(WorkspaceError):
        manager.create_workspace(invalid_id)


def test_uppercase_task_id_is_normalized(manager):
    task_id = "ABCDEF12-1234-5678-90AB-CDEF12345678"

    workspace = manager.create_workspace(task_id)

    assert workspace.name == task_id.lower()


def test_task_id_validation_returns_normalized_uuid(manager):
    task_id = "ABCDEF12-1234-5678-90AB-CDEF12345678"

    result = manager._validate_task_id(task_id)

    assert result == task_id.lower()


# ---------------------------------------------------------
# Workspace creation and isolation
# ---------------------------------------------------------

def test_create_workspace_creates_task_directory(manager, task_id):
    workspace = manager.create_workspace(task_id)

    assert workspace.exists()
    assert workspace.is_dir()
    assert workspace.parent == manager.root_dir


def test_duplicate_workspace_creation_is_rejected(
    manager,
    task_id,
):
    manager.create_workspace(task_id)

    with pytest.raises(WorkspaceError, match="already exists"):
        manager.create_workspace(task_id)


def test_workspace_does_not_exist_for_uncreated_task(
    manager,
    task_id,
):
    with pytest.raises(WorkspaceError, match="does not exist"):
        manager.list_files(task_id)


def test_tasks_receive_separate_workspaces(manager):
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())

    first_workspace = manager.create_workspace(first_id)
    second_workspace = manager.create_workspace(second_id)

    manager.write_file(first_id, "first.txt", "First task")

    assert first_workspace != second_workspace
    assert (first_workspace / "first.txt").exists()
    assert not (second_workspace / "first.txt").exists()


def test_workspace_symlink_is_rejected(manager, task_id, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()

    workspace_link = manager.root_dir / task_id
    workspace_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="symbolic link"):
        manager.list_files(task_id)


# ---------------------------------------------------------
# Role authorization
# ---------------------------------------------------------

@pytest.mark.parametrize(
    ("agent", "operation"),
    [
        ("developer", "list"),
        ("developer", "read"),
        ("developer", "write"),
        ("developer", "mkdir"),
        ("qa", "list"),
        ("qa", "read"),
        ("orchestrator", "list"),
        ("orchestrator", "read"),
        ("orchestrator", "write"),
        ("orchestrator", "mkdir"),
    ],
)
def test_allowed_role_permissions(manager, agent, operation):
    manager._authorize(agent, operation)


@pytest.mark.parametrize(
    ("agent", "operation"),
    [
        ("qa", "write"),
        ("qa", "mkdir"),
        ("developer", "delete"),
        ("qa", "delete"),
        ("orchestrator", "delete"),
        ("security", "read"),
        ("architect", "write"),
        ("unknown", "list"),
    ],
)
def test_disallowed_role_permissions_are_rejected(
    manager,
    agent,
    operation,
):
    with pytest.raises(WorkspacePermissionError):
        manager._authorize(agent, operation)


def test_qa_cannot_write_files(manager, task_id, workspace):
    with pytest.raises(WorkspacePermissionError):
        manager.write_file(
            task_id,
            "test.txt",
            "Not permitted",
            agent="qa",
        )


def test_qa_cannot_create_directories(manager, task_id, workspace):
    with pytest.raises(WorkspacePermissionError):
        manager.create_directory(
            task_id,
            "new_directory",
            agent="qa",
        )


def test_qa_can_read_files(manager, task_id, workspace):
    manager.write_file(task_id, "test.txt", "QA can read this")

    result = manager.read_file(
        task_id,
        "test.txt",
        agent="qa",
    )

    assert result == "QA can read this"


# ---------------------------------------------------------
# Relative path validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        None,
        "/etc/passwd",
        "../outside.txt",
        "folder/../outside.txt",
        "folder/./file.txt",
        "folder//file.txt",
        "./file.txt",
        " folder/file.txt",
        "folder/file.txt ",
        "folder\\file.txt",
        "C:/Windows/system.ini",
        "folder/\x00file.txt",
    ],
)
def test_invalid_relative_paths_are_rejected(
    manager,
    invalid_path,
):
    with pytest.raises(WorkspaceError):
        manager._validate_relative_path(invalid_path)


def test_workspace_root_relative_path_is_allowed(manager):
    assert manager._validate_relative_path(".") == "."


def test_normal_nested_relative_path_is_allowed(manager):
    path = "src/agentops/main.py"

    assert manager._validate_relative_path(path) == path


# ---------------------------------------------------------
# Protected directories and filenames
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "protected_path",
    [
        ".git/config",
        ".ssh/id_rsa",
        ".aws/credentials",
        ".azure/settings",
        ".gnupg/private-key",
        "src/.git/config",
        "src/.ssh/key",
    ],
)
def test_protected_directories_are_rejected(
    manager,
    protected_path,
):
    with pytest.raises(WorkspaceError):
        manager._validate_relative_path(protected_path)


@pytest.mark.parametrize(
    "protected_filename",
    [
        ".env",
        ".env.local",
        ".env.production",
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
        "private.pem",
        "certificate.key",
        "identity.p12",
        "certificate.pfx",
        "private.ppk",
        "terraform.tfstate",
        "terraform.tfstate.backup",
    ],
)
def test_protected_filenames_are_rejected(
    manager,
    protected_filename,
):
    with pytest.raises(WorkspaceError):
        manager._validate_relative_path(protected_filename)


def test_protected_filename_is_rejected_in_nested_directory(manager):
    with pytest.raises(WorkspaceError):
        manager._validate_relative_path("config/.env.local")


def test_protected_path_matching_is_case_insensitive(manager):
    protected_paths = [
        ".GIT/config",
        ".AWS/credentials",
        "CONFIG/.ENV",
        "private/SECRETS.JSON",
        "private/CERTIFICATE.PEM",
    ]

    for path in protected_paths:
        with pytest.raises(WorkspaceError):
            manager._validate_relative_path(path)


def test_similarly_named_unprotected_file_is_allowed(manager):
    assert (
        manager._validate_relative_path("environment.txt")
        == "environment.txt"
    )


# ---------------------------------------------------------
# Path resolution and symlink protection
# ---------------------------------------------------------

def test_path_outside_workspace_is_rejected(
    manager,
    task_id,
    workspace,
):
    with pytest.raises(WorkspaceError):
        manager.read_file(task_id, "../outside.txt")


def test_file_symlink_is_rejected(
    manager,
    task_id,
    workspace,
    tmp_path,
):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Outside content")

    link = workspace / "linked.txt"
    link.symlink_to(outside_file)

    with pytest.raises(WorkspaceError, match="(?i)symbolic link"):
        manager.read_file(task_id, "linked.txt")


def test_directory_symlink_is_rejected(
    manager,
    task_id,
    workspace,
    tmp_path,
):
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()

    link = workspace / "linked_directory"
    link.symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="(?i)symbolic link"):
        manager.list_files(task_id, "linked_directory")


def test_list_rejects_symlink_found_inside_directory(
    manager,
    task_id,
    workspace,
    tmp_path,
):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Outside")

    (workspace / "linked.txt").symlink_to(outside_file)

    with pytest.raises(WorkspaceError, match="symbolic link"):
        manager.list_files(task_id)


# ---------------------------------------------------------
# File listing
# ---------------------------------------------------------

def test_list_files_returns_sorted_entries(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "zebra.txt", "Z")
    manager.write_file(task_id, "apple.txt", "A")
    manager.create_directory(task_id, "middle")

    results = manager.list_files(task_id)

    names = [item["name"] for item in results]

    assert names == ["apple.txt", "middle", "zebra.txt"]


def test_list_files_returns_expected_entry_structure(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "Hello")

    results = manager.list_files(task_id)

    assert results == [
        {
            "name": "example.txt",
            "path": "example.txt",
            "type": "file",
            "size_bytes": 5,
        }
    ]


def test_list_files_reports_directory_type_and_no_size(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")

    results = manager.list_files(task_id)

    assert results == [
        {
            "name": "src",
            "path": "src",
            "type": "directory",
            "size_bytes": None,
        }
    ]


def test_list_files_hides_protected_entries(
    manager,
    task_id,
    workspace,
):
    (workspace / ".git").mkdir()
    (workspace / ".env").write_text("SECRET=value")
    (workspace / "visible.txt").write_text("Visible")

    results = manager.list_files(task_id)

    assert [item["name"] for item in results] == ["visible.txt"]


def test_list_files_rejects_file_as_directory(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "content")

    with pytest.raises(WorkspaceError, match="not a directory"):
        manager.list_files(task_id, "example.txt")


def test_list_files_supports_nested_directories(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src/agentops")
    manager.write_file(
        task_id,
        "src/agentops/main.py",
        "print('hello')",
    )

    results = manager.list_files(task_id, "src/agentops")

    assert len(results) == 1
    assert results[0]["name"] == "main.py"
    assert results[0]["path"] == "src/agentops/main.py"


# ---------------------------------------------------------
# File reading
# ---------------------------------------------------------

def test_read_file_returns_file_content(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "Hello AgentOps")

    result = manager.read_file(task_id, "example.txt")

    assert result == "Hello AgentOps"


def test_read_file_supports_nested_files(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")
    manager.write_file(task_id, "src/main.py", "print('hello')")

    result = manager.read_file(task_id, "src/main.py")

    assert result == "print('hello')"


def test_read_file_rejects_missing_file(
    manager,
    task_id,
    workspace,
):
    with pytest.raises(WorkspaceError, match="does not exist"):
        manager.read_file(task_id, "missing.txt")


def test_read_file_rejects_oversized_file(
    tmp_path,
    task_id,
):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces",
        max_file_size_bytes=5,
    )
    manager.create_workspace(task_id)

    (manager.root_dir / task_id / "large.txt").write_text("123456")

    with pytest.raises(WorkspaceError, match="size limit"):
        manager.read_file(task_id, "large.txt")


def test_read_file_accepts_file_at_size_limit(
    tmp_path,
    task_id,
):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces",
        max_file_size_bytes=5,
    )
    manager.create_workspace(task_id)

    (manager.root_dir / task_id / "valid.txt").write_text("12345")

    assert manager.read_file(task_id, "valid.txt") == "12345"


# ---------------------------------------------------------
# Directory creation
# ---------------------------------------------------------

def test_create_directory_creates_directory(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")

    assert (workspace / "src").is_dir()


def test_create_directory_creates_nested_parents(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src/agentops/tools")

    assert (workspace / "src" / "agentops" / "tools").is_dir()


def test_create_directory_rejects_existing_directory(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")

    with pytest.raises(WorkspaceError, match="already exists"):
        manager.create_directory(task_id, "src")


def test_create_directory_rejects_existing_file(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "content")

    with pytest.raises(WorkspaceError, match="already exists"):
        manager.create_directory(task_id, "example.txt")


# ---------------------------------------------------------
# File writing
# ---------------------------------------------------------

def test_write_file_creates_file(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "AgentOps")

    assert (workspace / "example.txt").read_text() == "AgentOps"


def test_write_file_replaces_existing_file(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "Original")
    manager.write_file(task_id, "example.txt", "Updated")

    assert (workspace / "example.txt").read_text() == "Updated"


def test_write_file_supports_nested_paths(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")

    manager.write_file(task_id, "src/main.py", "print('AgentOps')")

    assert (workspace / "src" / "main.py").read_text() == (
        "print('AgentOps')"
    )


def test_write_file_rejects_missing_parent_directory(
    manager,
    task_id,
    workspace,
):
    with pytest.raises(WorkspaceError, match="Parent directory"):
        manager.write_file(task_id, "missing/file.txt", "content")


def test_write_file_rejects_non_string_content(
    manager,
    task_id,
    workspace,
):
    with pytest.raises(WorkspaceError, match="must be a string"):
        manager.write_file(task_id, "example.txt", 123)


def test_write_file_rejects_content_over_size_limit(
    tmp_path,
    task_id,
):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces",
        max_file_size_bytes=5,
    )
    manager.create_workspace(task_id)

    with pytest.raises(WorkspaceError, match="size limit"):
        manager.write_file(task_id, "large.txt", "123456")


def test_write_file_accepts_content_at_size_limit(
    tmp_path,
    task_id,
):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces",
        max_file_size_bytes=5,
    )
    manager.create_workspace(task_id)

    manager.write_file(task_id, "valid.txt", "12345")

    assert manager.read_file(task_id, "valid.txt") == "12345"


def test_write_file_uses_utf8_byte_size(
    tmp_path,
    task_id,
):
    manager = WorkspaceManager(
        root_dir=tmp_path / "workspaces",
        max_file_size_bytes=4,
    )
    manager.create_workspace(task_id)

    with pytest.raises(WorkspaceError, match="size limit"):
        manager.write_file(task_id, "unicode.txt", "ééé")


def test_write_file_rejects_directory_as_target(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")

    with pytest.raises(WorkspaceError, match="not a regular file"):
        manager.write_file(task_id, "src", "content")


def test_write_file_rejects_symlink_target(
    manager,
    task_id,
    workspace,
    tmp_path,
):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Original")

    (workspace / "linked.txt").symlink_to(outside_file)

    with pytest.raises(WorkspaceError, match="(?i)symbolic link"):
        manager.write_file(task_id, "linked.txt", "Modified")

    assert outside_file.read_text() == "Original"


def test_failed_atomic_replace_preserves_original_file(
    manager,
    task_id,
    workspace,
    monkeypatch,
):
    manager.write_file(task_id, "example.txt", "Original")

    def fail_replace(source, destination):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(
        "agentops.tools.file_manager.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="Simulated"):
        manager.write_file(task_id, "example.txt", "Updated")

    assert (workspace / "example.txt").read_text() == "Original"

    temporary_files = list(workspace.glob(".agentops-*.tmp"))

    assert temporary_files == []


# ---------------------------------------------------------
# Audit history
# ---------------------------------------------------------

def test_workspace_creation_records_audit_event(
    manager,
    task_id,
):
    manager.create_workspace(task_id)

    events = manager.get_audit_events(task_id)

    assert len(events) == 1
    assert events[0]["operation"] == "create_workspace"
    assert events[0]["status"] == "SUCCESS"
    assert events[0]["agent"] == "orchestrator"


def test_successful_operations_are_audited(
    manager,
    task_id,
    workspace,
):
    manager.create_directory(task_id, "src")
    manager.write_file(task_id, "src/main.py", "print('hello')")
    manager.read_file(task_id, "src/main.py")
    manager.list_files(task_id, "src")

    events = manager.get_audit_events(task_id)

    operations = [event["operation"] for event in events]

    assert operations == [
        "create_workspace",
        "mkdir",
        "write",
        "read",
        "list",
    ]


def test_audit_events_include_expected_fields(
    manager,
    task_id,
    workspace,
):
    manager.write_file(task_id, "example.txt", "content")

    event = manager.get_audit_events(task_id)[-1]

    assert set(event) == {
        "timestamp",
        "task_id",
        "agent",
        "operation",
        "path",
        "status",
        "details",
    }


def test_audit_events_do_not_contain_file_contents(
    manager,
    task_id,
    workspace,
):
    secret_content = "Sensitive application information"

    manager.write_file(task_id, "example.txt", secret_content)

    events = manager.get_audit_events(task_id)

    assert secret_content not in str(events)


def test_audit_events_can_be_filtered_by_task(
    manager,
):
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())

    manager.create_workspace(first_id)
    manager.create_workspace(second_id)

    manager.write_file(first_id, "first.txt", "First")
    manager.write_file(second_id, "second.txt", "Second")

    first_events = manager.get_audit_events(first_id)
    second_events = manager.get_audit_events(second_id)

    assert all(
        event["task_id"] == first_id
        for event in first_events
    )
    assert all(
        event["task_id"] == second_id
        for event in second_events
    )


def test_get_all_audit_events_returns_all_tasks(manager):
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())

    manager.create_workspace(first_id)
    manager.create_workspace(second_id)

    events = manager.get_audit_events()

    assert len(events) == 2


def test_audit_event_list_is_a_separate_list(manager, task_id):
    manager.create_workspace(task_id)

    events = manager.get_audit_events()
    events.clear()

    assert len(manager.get_audit_events()) == 1


def test_audit_events_reject_invalid_filter_task_id(manager):
    with pytest.raises(WorkspaceError):
        manager.get_audit_events("../invalid")
