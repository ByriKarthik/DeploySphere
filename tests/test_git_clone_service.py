"""Unit tests for safe Git clone behavior; subprocesses are always mocked."""

from pathlib import Path
import subprocess

from bson import ObjectId
import pytest

from backend.services import git_clone


def _completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _successful_git_run(command, **kwargs):
    if "check-ref-format" in command:
        return _completed(command)
    if "ls-remote" in command:
        return _completed(command, stdout="abc refs/heads/main\n")
    if "clone" in command:
        return _completed(command)
    if "rev-parse" in command:
        return _completed(command, stdout="a" * 40 + "\n")
    raise AssertionError(f"Unexpected Git command: {command}")


def test_successful_clone_uses_safe_shallow_git_arguments(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        template = Path(kwargs["env"]["GIT_TEMPLATE_DIR"])
        assert template.is_dir()
        assert list(template.iterdir()) == []
        global_config = Path(kwargs["env"]["GIT_CONFIG_GLOBAL"])
        assert global_config.is_file()
        assert global_config.read_text() == ""
        commands.append((command, kwargs))
        return _successful_git_run(command, **kwargs)

    monkeypatch.setattr(git_clone.shutil, "which", lambda name: "git-test")
    monkeypatch.setattr(git_clone.subprocess, "run", fake_run)
    project_id = ObjectId()

    result = git_clone.clone_repository(
        "https://github.com/example/repo",
        "main",
        project_id,
        workspace_root=tmp_path,
        timeout_seconds=17,
    )

    assert result.commit_hash == "a" * 40
    assert Path(result.workspace_path) == tmp_path / str(project_id)
    assert Path(result.workspace_path).is_dir()
    clone_command, clone_options = next(
        (command, options) for command, options in commands if "clone" in command
    )
    assert "--depth" in clone_command
    assert clone_command[clone_command.index("--depth") + 1] == "1"
    assert "--single-branch" in clone_command
    assert "--no-tags" in clone_command
    assert "--no-recurse-submodules" in clone_command
    assert clone_command[1:5] == ["-c", "credential.helper=", "-c", "core.askPass="]
    assert clone_options["shell"] is False
    assert clone_options["timeout"] == 17
    assert clone_options["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert clone_options["env"]["GIT_CONFIG_GLOBAL"]
    assert "GIT_ASKPASS" not in clone_options["env"]


def test_invalid_repository_url_is_rejected_before_running_git(monkeypatch, tmp_path):
    monkeypatch.setattr(
        git_clone.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Git must not run for an invalid URL"),
    )

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.clone_repository(
            "https://user:secret@github.com/example/repo",
            "main",
            ObjectId(),
            workspace_root=tmp_path,
        )

    assert error.value.code == "invalid_repository_url"
    assert "secret" not in error.value.message


def test_invalid_branch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(git_clone.shutil, "which", lambda name: "git-test")
    monkeypatch.setattr(
        git_clone.subprocess,
        "run",
        lambda command, **kwargs: _completed(command, returncode=1),
    )

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.clone_repository(
            "https://github.com/example/repo",
            "bad branch",
            ObjectId(),
            workspace_root=tmp_path,
        )

    assert error.value.code == "invalid_branch"


def test_missing_git_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(git_clone.shutil, "which", lambda name: None)

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.clone_repository(
            "https://github.com/example/repo", "main", ObjectId(), workspace_root=tmp_path
        )

    assert error.value.code == "git_unavailable"


@pytest.mark.parametrize(
    ("returncode", "expected_code"),
    [(2, "branch_not_found"), (128, "network_error")],
)
def test_remote_branch_and_network_failures_are_categorized(
    tmp_path, monkeypatch, returncode, expected_code
):
    monkeypatch.setattr(git_clone.shutil, "which", lambda name: "git-test")

    def fake_run(command, **kwargs):
        if "check-ref-format" in command:
            return _completed(command)
        return _completed(command, returncode=returncode, stderr="private host detail")

    monkeypatch.setattr(git_clone.subprocess, "run", fake_run)

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.clone_repository(
            "https://github.com/example/repo", "main", ObjectId(), workspace_root=tmp_path
        )

    assert error.value.code == expected_code
    assert "private host detail" not in error.value.message
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_reported_without_raw_process_output(tmp_path, monkeypatch):
    monkeypatch.setattr(git_clone.shutil, "which", lambda name: "git-test")

    def fake_run(command, **kwargs):
        if "clone" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr="secret path")
        return _successful_git_run(command, **kwargs)

    monkeypatch.setattr(git_clone.subprocess, "run", fake_run)

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.clone_repository(
            "https://github.com/example/repo", "main", ObjectId(), workspace_root=tmp_path
        )

    assert error.value.code == "timeout"
    assert "secret path" not in error.value.message
    assert list(tmp_path.iterdir()) == []


def test_path_traversal_project_id_is_rejected(tmp_path):
    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.resolve_workspace_paths("../../outside", tmp_path)

    assert error.value.code == "invalid_project_id"
    assert not (tmp_path.parent / "outside").exists()


def test_existing_workspace_and_symlink_are_rejected(tmp_path, monkeypatch):
    project_id = ObjectId()
    destination = tmp_path / str(project_id)
    destination.mkdir()
    monkeypatch.setattr(git_clone.shutil, "which", lambda name: "git-test")
    monkeypatch.setattr(git_clone.subprocess, "run", _successful_git_run)

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.clone_repository(
            "https://github.com/example/repo", "main", project_id, workspace_root=tmp_path
        )

    assert error.value.code == "workspace_exists"


def test_workspace_symlink_is_rejected(tmp_path, monkeypatch):
    project_id = ObjectId()
    destination = tmp_path / str(project_id)
    original_is_symlink = Path.is_symlink

    def report_test_path_as_symlink(path):
        return path == destination or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_test_path_as_symlink)
    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.resolve_workspace_paths(project_id, tmp_path)

    assert error.value.code == "workspace_exists"


def test_configured_timeout_is_validated(monkeypatch):
    monkeypatch.setenv("DEPLOYSPHERE_GIT_TIMEOUT_SECONDS", "0")

    with pytest.raises(git_clone.GitCloneError) as error:
        git_clone.get_timeout_seconds()

    assert error.value.code == "invalid_configuration"
