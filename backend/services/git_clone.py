"""Safely clone a registered public GitHub repository into a local workspace."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

from bson import ObjectId
from bson.errors import InvalidId


DEFAULT_TIMEOUT_SECONDS = 300


class GitCloneError(Exception):
    """A clone failure with a safe code, message, and HTTP status."""

    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass(frozen=True)
class CloneResult:
    """Details returned after a repository has been cloned successfully."""

    workspace_path: str
    commit_hash: str


def get_timeout_seconds() -> int:
    """Read and validate the clone timeout from the environment."""
    configured_value = os.getenv("DEPLOYSPHERE_GIT_TIMEOUT_SECONDS")
    if configured_value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = int(configured_value)
    except ValueError as error:
        raise GitCloneError(
            "invalid_configuration", "The Git clone timeout setting is invalid.", 500
        ) from error
    if timeout < 1 or timeout > 3600:
        raise GitCloneError(
            "invalid_configuration", "The Git clone timeout must be from 1 to 3600 seconds.", 500
        )
    return timeout


def validate_repository_url(repository_url: str) -> str:
    """Accept only a public GitHub HTTPS URL without embedded credentials."""
    try:
        parsed = urlsplit(repository_url)
        port = parsed.port
    except (AttributeError, TypeError, ValueError) as error:
        raise GitCloneError(
            "invalid_repository_url", "The stored repository URL is invalid.", 422
        ) from error

    path_parts = parsed.path.strip("/").split("/")
    owner, repository = path_parts if len(path_parts) == 2 else ("", "")
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or len(path_parts) != 2
        or "//" in parsed.path
        or not re.fullmatch(r"[A-Za-z0-9-]+", owner)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", repository)
        or repository in ("", ".", "..")
    ):
        raise GitCloneError(
            "invalid_repository_url", "The stored repository must be a public GitHub HTTPS URL.", 422
        )
    return repository_url


def resolve_workspace_paths(
    project_id: ObjectId | str,
    workspace_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the configured workspace root and safe project destination."""
    try:
        object_id = project_id if isinstance(project_id, ObjectId) else ObjectId(project_id)
    except (InvalidId, TypeError, ValueError) as error:
        raise GitCloneError("invalid_project_id", "The project ID is invalid.", 400) from error

    configured_root = workspace_root or os.getenv("DEPLOYSPHERE_WORKSPACE_ROOT")
    if configured_root is None:
        configured_root = Path.home() / ".deploysphere" / "workspaces"

    try:
        root = Path(configured_root).expanduser().resolve()
        destination = root / str(object_id)
        if destination.is_symlink():
            raise GitCloneError(
                "workspace_exists", "A symlink already exists at this project's workspace path.", 409
            )
        destination = destination.resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GitCloneError(
            "invalid_workspace_root", "The configured workspace directory is invalid.", 500
        ) from error

    if not destination.is_relative_to(root) or destination == root:
        raise GitCloneError(
            "invalid_workspace_path", "The project workspace path is outside the workspace root.", 400
        )
    return root, destination


def _git_environment(template_directory: str, global_config: str) -> dict[str, str]:
    """Disable inherited Git configuration and all interactive credentials."""
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("GIT_") or key in {"SSH_AUTH_SOCK", "SSH_AGENT_PID"}:
            environment.pop(key, None)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = global_config
    environment["GIT_TEMPLATE_DIR"] = template_directory
    return environment


def _command_prefix(git_path: str) -> list[str]:
    """Disable Git credential helpers and askpass for every Git command."""
    return [git_path, "-c", "credential.helper=", "-c", "core.askPass="]


def _run_git(
    command: list[str],
    *,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run Git without a shell and translate process failures safely."""
    try:
        # An empty template directory prevents machine-level Git hooks from
        # being copied into the checkout and run during clone/checkout.
        with tempfile.TemporaryDirectory(prefix="deploysphere-git-template-") as template_directory:
            temporary_root = Path(template_directory)
            empty_template = temporary_root / "empty-template"
            empty_template.mkdir()
            empty_global_config = temporary_root / "empty.gitconfig"
            empty_global_config.touch()
            return subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                cwd=cwd,
                env=_git_environment(str(empty_template), str(empty_global_config)),
            )
    except subprocess.TimeoutExpired as error:
        raise GitCloneError(
            "timeout", "The Git operation exceeded its configured timeout.", 504
        ) from error
    except FileNotFoundError as error:
        raise GitCloneError("git_unavailable", "Git is not installed or not on PATH.", 503) from error
    except OSError as error:
        raise GitCloneError("git_unavailable", "Git could not be started.", 503) from error


def clone_repository(
    repository_url: str,
    branch: str,
    project_id: ObjectId | str,
    *,
    workspace_root: str | Path | None = None,
    timeout_seconds: int | None = None,
) -> CloneResult:
    """Shallow-clone one validated branch without running repository code."""
    repository_url = validate_repository_url(repository_url)
    if not isinstance(branch, str) or not branch:
        raise GitCloneError("invalid_branch", "The stored branch name is invalid.", 422)
    root, destination = resolve_workspace_paths(project_id, workspace_root)
    timeout = timeout_seconds if timeout_seconds is not None else get_timeout_seconds()
    if timeout < 1:
        raise GitCloneError("invalid_configuration", "The Git timeout must be positive.", 500)

    git_path = shutil.which("git")
    if git_path is None:
        raise GitCloneError("git_unavailable", "Git is not installed or not on PATH.", 503)

    branch_check = _run_git(
        _command_prefix(git_path) + ["check-ref-format", "--branch", branch],
        timeout_seconds=timeout,
    )
    if branch_check.returncode != 0:
        raise GitCloneError("invalid_branch", "The stored branch name is invalid.", 422)

    created_directory = False
    directory_identity = None
    clone_succeeded = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            raise GitCloneError(
                "workspace_exists", "A workspace already exists for this project.", 409
            )
        if not destination.is_relative_to(root.resolve()):
            raise GitCloneError(
                "invalid_workspace_path", "The project workspace path is outside the workspace root.", 400
            )
        # mkdir without exist_ok claims this generated path exclusively.
        destination.mkdir(mode=0o700)
        created_directory = True
        directory_info = destination.stat()
        directory_identity = (directory_info.st_dev, directory_info.st_ino)

        branch_check_remote = _run_git(
            _command_prefix(git_path)
            + ["ls-remote", "--exit-code", "--heads", "--", repository_url, f"refs/heads/{branch}"],
            timeout_seconds=timeout,
        )
        if branch_check_remote.returncode == 2:
            raise GitCloneError("branch_not_found", "The selected branch was not found on GitHub.", 404)
        if branch_check_remote.returncode != 0:
            raise GitCloneError("network_error", "GitHub could not be reached for this repository.", 502)

        clone = _run_git(
            _command_prefix(git_path)
            + [
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                branch,
                "--no-tags",
                "--no-recurse-submodules",
                "--",
                repository_url,
                str(destination),
            ],
            timeout_seconds=timeout,
            cwd=root,
        )
        if clone.returncode != 0:
            raise GitCloneError("network_error", "GitHub could not provide the selected branch.", 502)

        revision = _run_git(
            _command_prefix(git_path) + ["-C", str(destination), "rev-parse", "--verify", "HEAD"],
            timeout_seconds=timeout,
            cwd=root,
        )
        commit_hash = revision.stdout.strip().splitlines()[0] if revision.stdout.strip() else ""
        if (
            revision.returncode != 0
            or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit_hash)
        ):
            raise GitCloneError("git_error", "Git could not determine the cloned commit.", 502)

        clone_succeeded = True
        return CloneResult(workspace_path=str(destination), commit_hash=commit_hash)
    except FileExistsError as error:
        raise GitCloneError(
            "workspace_exists", "A workspace already exists for this project.", 409
        ) from error
    except GitCloneError:
        raise
    except OSError as error:
        raise GitCloneError(
            "workspace_unavailable", "The workspace directory could not be prepared.", 503
        ) from error
    finally:
        # Remove only this attempt's exclusively created path, and only if it
        # still resolves to the same directory beneath the configured root.
        if created_directory and not clone_succeeded and directory_identity is not None:
            try:
                if not destination.is_symlink():
                    current_path = destination.resolve()
                    current_info = current_path.stat()
                    current_identity = (current_info.st_dev, current_info.st_ino)
                    if (
                        current_path.is_relative_to(root.resolve())
                        and current_identity == directory_identity
                    ):
                        shutil.rmtree(current_path, ignore_errors=True)
            except OSError:
                pass
