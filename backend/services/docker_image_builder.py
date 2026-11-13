"""Build a Docker image from a bounded, sanitized repository context.

This service accepts detection output and workspace-relative paths. It never
accepts a host path for the build context and never runs repository code on the
host; Dockerfile instructions execute inside the Docker Engine during a build.
"""

from dataclasses import dataclass, field
import os
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import tempfile
from threading import Lock, Thread
import time
from typing import BinaryIO
from uuid import uuid4

from backend.database.initialize import MAX_BUILD_LOG_EXCERPT_BYTES
from backend.services.application_detector import DetectionResult


DEFAULT_MAX_CONTEXT_BYTES = 1_073_741_824  # 1 GiB
DEFAULT_MAX_CONTEXT_FILES = 20_000
DEFAULT_MAX_CONTEXT_ENTRIES = 50_000
DEFAULT_MAX_FILE_SIZE_BYTES = 104_857_600  # 100 MiB
DEFAULT_MAX_DEPTH = 32
DEFAULT_BUILD_TIMEOUT_SECONDS = 1_800
DEFAULT_MAX_OUTPUT_BYTES = 10_485_760  # 10 MiB of counted output
DEFAULT_MAX_OUTPUT_LINES = 100_000

READ_CHUNK_BYTES = 64 * 1024
TERMINATE_GRACE_SECONDS = 5
IMAGE_TAG_PATTERN = re.compile(r"^deploysphere/build:[a-f0-9]{32}$")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"""(?i)\b(password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key|client[_-]?secret|authorization)\b(\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|(?:bearer|basic)\s+[^\s,;]+|[^\s,;]+)"""
)
_ENVIRONMENT_ASSIGNMENT_PATTERN = re.compile(
    r"""(?m)(^|\s)([A-Za-z_][A-Za-z0-9_]{0,127})=(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)"""
)
_DOCKER_ENV_DIRECTIVE_PATTERN = re.compile(
    r"(?im)^([^\r\n]*?\b(?:ENV|ARG)\s+[A-Za-z_][A-Za-z0-9_]*(?:\s+|=)).*$"
)
_BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/-]+=*"
)
_CREDENTIAL_URL_PATTERN = re.compile(
    r"(?i)(https?://[^:/\s@]+:)[^@\s/]+@"
)
_COMMON_TOKEN_PATTERN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16})\b"
)
_RESERVED_TAGS: set[str] = set()
_RESERVED_TAGS_LOCK = Lock()

EXCLUDED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".cache",
    ".aws",
    ".ssh",
    ".gnupg",
    ".docker",
    "dist",
    "build",
    "coverage",
    "target",
}
EXCLUDED_FILES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
}
EXCLUDED_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


class BuildEngineError(Exception):
    """A build failure with safe details that contain no host paths or output."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class BuildLimits:
    """Resource limits for staging and running one Docker image build."""

    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES
    max_files: int = DEFAULT_MAX_CONTEXT_FILES
    max_entries: int = DEFAULT_MAX_CONTEXT_ENTRIES
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    max_depth: int = DEFAULT_MAX_DEPTH
    timeout_seconds: int = DEFAULT_BUILD_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES

    def __post_init__(self) -> None:
        """Reject limits that would disable resource protections."""
        if any(
            value < 1
            for value in (
                self.max_context_bytes,
                self.max_files,
                self.max_entries,
                self.max_file_size_bytes,
                self.timeout_seconds,
                self.max_output_bytes,
                self.max_output_lines,
            )
        ):
            raise ValueError("Build limits must be positive integers.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be zero or greater.")
        if self.timeout_seconds > 3600:
            raise ValueError("timeout_seconds cannot exceed one hour.")

    @classmethod
    def from_environment(cls) -> "BuildLimits":
        """Read optional build limits without accepting unbounded values."""
        names_and_defaults = {
            "max_context_bytes": ("DEPLOYSPHERE_BUILD_MAX_CONTEXT_BYTES", DEFAULT_MAX_CONTEXT_BYTES),
            "max_files": ("DEPLOYSPHERE_BUILD_MAX_FILES", DEFAULT_MAX_CONTEXT_FILES),
            "max_entries": ("DEPLOYSPHERE_BUILD_MAX_ENTRIES", DEFAULT_MAX_CONTEXT_ENTRIES),
            "max_file_size_bytes": (
                "DEPLOYSPHERE_BUILD_MAX_FILE_SIZE_BYTES",
                DEFAULT_MAX_FILE_SIZE_BYTES,
            ),
            "max_depth": ("DEPLOYSPHERE_BUILD_MAX_DEPTH", DEFAULT_MAX_DEPTH),
            "timeout_seconds": (
                "DEPLOYSPHERE_BUILD_TIMEOUT_SECONDS",
                DEFAULT_BUILD_TIMEOUT_SECONDS,
            ),
            "max_output_bytes": (
                "DEPLOYSPHERE_BUILD_MAX_OUTPUT_BYTES",
                DEFAULT_MAX_OUTPUT_BYTES,
            ),
            "max_output_lines": (
                "DEPLOYSPHERE_BUILD_MAX_OUTPUT_LINES",
                DEFAULT_MAX_OUTPUT_LINES,
            ),
        }
        try:
            values = {
                field: int(os.getenv(variable, str(default)))
                for field, (variable, default) in names_and_defaults.items()
            }
            return cls(**values)
        except (TypeError, ValueError) as error:
            raise BuildEngineError(
                "invalid_configuration", "Docker build resource limits are invalid."
            ) from error


@dataclass(frozen=True)
class BuildError:
    """A safe, short build failure description."""

    code: str
    message: str


@dataclass(frozen=True)
class BuildResult:
    """Structured result with bounded output counters and a safe log excerpt."""

    success: bool
    image_tag: str | None
    image_id: str | None
    duration_seconds: float
    log_bytes: int
    log_line_count: int
    log_truncated: bool
    outcome_known: bool = True
    error: BuildError | None = None
    log_excerpt: str = ""


@dataclass
class _OutputCounts:
    """Bounded counters and a rolling tail updated by the pipe reader."""

    bytes_seen: int = 0
    lines_seen: int = 0
    ends_with_newline: bool = True
    truncated: bool = False
    excerpt: bytearray = field(default_factory=bytearray)
    excerpt_truncated: bool = False

    def retain_tail(self, chunk: bytes) -> None:
        excess = len(self.excerpt) + len(chunk) - MAX_BUILD_LOG_EXCERPT_BYTES
        if excess > 0:
            del self.excerpt[:excess]
            self.truncated = True
            self.excerpt_truncated = True
        self.excerpt.extend(chunk[-MAX_BUILD_LOG_EXCERPT_BYTES:])


def _safe_log_excerpt(
    output: bytearray, *, truncated: bool = False
) -> str:
    """Decode and redact the retained output before it is persisted."""
    if not output:
        return ""
    excerpt = output.decode("utf-8", errors="replace")
    if truncated:
        first_newline = excerpt.find("\n")
        excerpt = excerpt[first_newline + 1 :] if first_newline >= 0 else ""
    excerpt = _SENSITIVE_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        excerpt,
    )
    excerpt = _ENVIRONMENT_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}=[REDACTED]",
        excerpt,
    )
    excerpt = _DOCKER_ENV_DIRECTIVE_PATTERN.sub(
        lambda match: f"{match.group(1)}[REDACTED]",
        excerpt,
    )
    excerpt = _BEARER_TOKEN_PATTERN.sub(
        lambda match: f"{match.group(1)} [REDACTED]",
        excerpt,
    )
    excerpt = _CREDENTIAL_URL_PATTERN.sub(r"\1[REDACTED]@", excerpt)
    excerpt = _COMMON_TOKEN_PATTERN.sub("[REDACTED]", excerpt)
    encoded = excerpt.encode("utf-8")
    if len(encoded) > MAX_BUILD_LOG_EXCERPT_BYTES:
        encoded = encoded[-MAX_BUILD_LOG_EXCERPT_BYTES:]
        excerpt = encoded.decode("utf-8", errors="replace")
    return excerpt


def create_unique_image_tag() -> str:
    """Create a tag in DeploySphere's reserved namespace with a random suffix."""
    return f"deploysphere/build:{uuid4().hex}"


def _failed_result(
    started_at: float,
    error: BuildEngineError,
    *,
    image_tag: str | None = None,
    counts: _OutputCounts | None = None,
    limits: BuildLimits | None = None,
    outcome_known: bool = True,
) -> BuildResult:
    """Build a failure response without copying untrusted exception text."""
    counts = counts or _OutputCounts()
    output_limits = limits or BuildLimits()
    line_count = counts.lines_seen + int(counts.bytes_seen > 0 and not counts.ends_with_newline)
    return BuildResult(
        success=False,
        image_tag=image_tag,
        image_id=None,
        duration_seconds=round(time.monotonic() - started_at, 3),
        log_bytes=min(counts.bytes_seen, output_limits.max_output_bytes),
        log_line_count=min(line_count, output_limits.max_output_lines),
        log_truncated=counts.truncated
        or counts.bytes_seen > output_limits.max_output_bytes
        or line_count > output_limits.max_output_lines,
        outcome_known=outcome_known,
        error=BuildError(error.code, error.message),
        log_excerpt=_safe_log_excerpt(
            counts.excerpt, truncated=counts.excerpt_truncated
        ),
    )


def _normalize_relative_path(value: str | Path, *, allow_root: bool = False) -> tuple[str, ...]:
    """Parse a canonical POSIX-style path and reject absolute/traversal paths."""
    try:
        raw_path = os.fspath(value)
    except TypeError as error:
        raise BuildEngineError("unsafe_path", "Build paths must be relative workspace paths.") from error
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise BuildEngineError("unsafe_path", "Build paths must be relative POSIX paths.")

    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise BuildEngineError("unsafe_path", "Absolute build paths are not allowed.")
    if raw_path == "." and allow_root:
        return ()
    parts = posix_path.parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or posix_path.as_posix() != raw_path
    ):
        raise BuildEngineError("unsafe_path", "Build paths cannot contain traversal or non-canonical segments.")
    return parts


def _verified_workspace_root(workspace_root: str | Path) -> Path:
    """Resolve an existing non-symlink workspace directory."""
    try:
        root = Path(workspace_root).expanduser()
        if not root.is_absolute() or root.is_symlink():
            raise BuildEngineError("unsafe_workspace", "The workspace must be an absolute, non-symlink directory.")
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise BuildEngineError("unsafe_workspace", "The workspace must be an existing directory.")
        return resolved
    except BuildEngineError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise BuildEngineError("unsafe_workspace", "The workspace directory is unavailable.") from error


def _resolve_inside_workspace(root: Path, parts: tuple[str, ...], *, expect_directory: bool) -> Path:
    """Check each path component for symlinks and workspace containment."""
    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise BuildEngineError("unsafe_path", "A requested build path is unavailable.") from error
        if stat.S_ISLNK(info.st_mode):
            raise BuildEngineError("unsafe_path", "Symlinks are not allowed in a build path.")
        is_final = index == len(parts) - 1
        if not is_final and not stat.S_ISDIR(info.st_mode):
            raise BuildEngineError("unsafe_path", "A build path component is not a directory.")

    try:
        resolved = current.resolve(strict=True) if parts else root
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise BuildEngineError("unsafe_path", "A build path escaped the workspace.") from error

    if parts:
        if expect_directory and not stat.S_ISDIR(current.lstat().st_mode):
            raise BuildEngineError("unsafe_path", "The build context must be a directory.")
        if not expect_directory and not stat.S_ISREG(current.lstat().st_mode):
            raise BuildEngineError("unsafe_path", "The Dockerfile must be a regular file.")
    return resolved


def _validate_detection(detection: DetectionResult, dockerfile_path: str) -> None:
    """Allow only one supported application type with a detected Dockerfile."""
    if detection.application_type not in {"python", "nodejs"}:
        raise BuildEngineError(
            "unsupported_application",
            "Image builds currently support one detected Python or Node.js application.",
        )
    if detection.build_method != "dockerfile":
        raise BuildEngineError(
            "unsupported_build_method",
            "Image builds currently require an unambiguous Dockerfile build method.",
        )
    if detection.detected_application_types != [detection.application_type]:
        raise BuildEngineError(
            "ambiguous_application",
            "The repository contains multiple or inconsistent application types.",
        )
    if "dockerfile" not in detection.detected_build_methods:
        raise BuildEngineError("dockerfile_not_detected", "The detection result does not include a Dockerfile.")
    detected_files = {PurePosixPath(item).as_posix() for item in detection.detected_files}
    if dockerfile_path not in detected_files:
        raise BuildEngineError(
            "dockerfile_not_detected", "The selected Dockerfile is not part of the saved detection result."
        )
    filename = PurePosixPath(dockerfile_path).name.casefold()
    if not (
        filename == "dockerfile"
        or filename.startswith("dockerfile.")
        or filename.startswith("dockerfile-")
        or filename.endswith(".dockerfile")
        or ".dockerfile." in filename
    ):
        raise BuildEngineError("invalid_dockerfile", "The selected file is not a recognized Dockerfile.")


def _is_excluded(name: str, *, is_directory: bool) -> bool:
    """Return whether a repository entry is unnecessary or sensitive."""
    lower_name = name.casefold()
    if lower_name == ".env" or lower_name.startswith((".env.", ".env-")):
        return True
    if is_directory:
        return lower_name in EXCLUDED_DIRECTORIES
    return (
        lower_name in EXCLUDED_FILES
        or Path(lower_name).suffix in EXCLUDED_SUFFIXES
    )


def _copy_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_info: os.stat_result,
    root: Path,
    max_file_size: int,
    bytes_already_copied: int,
    max_context_bytes: int,
) -> int:
    """Copy a bounded regular file without following symlinks."""
    try:
        current_info = source.lstat()
    except OSError as error:
        raise BuildEngineError("unsafe_file", "A repository file could not be inspected safely.") from error
    if stat.S_ISLNK(current_info.st_mode) or not stat.S_ISREG(current_info.st_mode):
        raise BuildEngineError("unsafe_file", "Only regular, non-symlink files may be staged.")
    # Windows DirEntry.stat can report zero inode/link metadata. Refresh with
    # Path.lstat before opening, then compare the opened handle against it.
    if expected_info.st_size != current_info.st_size or (
        expected_info.st_ino
        and expected_info.st_dev
        and (expected_info.st_dev, expected_info.st_ino)
        != (current_info.st_dev, current_info.st_ino)
    ):
        raise BuildEngineError("unsafe_file", "A repository file changed before it could be copied.")
    expected_info = current_info
    if expected_info.st_size > max_file_size:
        raise BuildEngineError("file_too_large", "A repository file exceeds the per-file size limit.")
    if expected_info.st_nlink > 1:
        raise BuildEngineError("unsafe_file", "Hard-linked files are not allowed in the build context.")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as error:
        raise BuildEngineError("unsafe_file", "A repository file could not be opened safely.") from error

    total_copied = 0
    try:
        opened_info = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened_info.st_mode)
            or (opened_info.st_dev, opened_info.st_ino) != (expected_info.st_dev, expected_info.st_ino)
            or opened_info.st_size > max_file_size
        ):
            raise BuildEngineError("unsafe_file", "A repository file changed before it could be copied.")
        try:
            source.resolve(strict=True).relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise BuildEngineError("unsafe_path", "A repository file escaped the build context.") from error

        destination.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(source_fd, "rb") as source_file, destination.open("xb") as destination_file:
            source_fd = -1
            while True:
                chunk = source_file.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                total_copied += len(chunk)
                if total_copied > max_file_size:
                    raise BuildEngineError("file_too_large", "A repository file exceeded the per-file size limit.")
                if bytes_already_copied + total_copied > max_context_bytes:
                    raise BuildEngineError("context_too_large", "The repository exceeds the build context size limit.")
                destination_file.write(chunk)
        if total_copied != opened_info.st_size:
            raise BuildEngineError("unsafe_file", "A repository file changed while it was copied.")
        return total_copied
    except BuildEngineError:
        raise
    except OSError as error:
        raise BuildEngineError("context_unavailable", "A repository file could not be staged.") from error
    finally:
        if source_fd >= 0:
            os.close(source_fd)


def _stage_context(
    source_root: Path,
    destination_root: Path,
    limits: BuildLimits,
) -> tuple[int, int]:
    """Copy a safe, size-limited tree into a fresh staging directory."""
    files_copied = 0
    entries_seen = 0
    bytes_copied = 0
    pending = [(source_root, destination_root, 0)]

    while pending:
        source_directory, destination_directory, depth = pending.pop()
        try:
            with os.scandir(source_directory) as iterator:
                entries = []
                for entry in iterator:
                    entries_seen += 1
                    if entries_seen > limits.max_entries:
                        raise BuildEngineError(
                            "too_many_entries", "The repository exceeds the build context entry limit."
                        )
                    if entry.is_symlink():
                        raise BuildEngineError(
                            "symlink_not_allowed", "Symlinks are not allowed in the build context."
                        )
                    is_directory = entry.is_dir(follow_symlinks=False)
                    if is_directory and _is_excluded(entry.name, is_directory=True):
                        continue
                    entries.append((entry, is_directory))
                entries.sort(key=lambda pair: (pair[0].name.casefold(), pair[0].name))
        except OSError as error:
            raise BuildEngineError("context_unavailable", "A repository directory could not be read safely.") from error

        for entry, is_directory in entries:
            source_path = Path(entry.path)
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise BuildEngineError("context_unavailable", "A repository entry could not be inspected safely.") from error

            if is_directory:
                if depth >= limits.max_depth:
                    raise BuildEngineError("context_too_deep", "The repository exceeds the build context depth limit.")
                destination_child = destination_directory / entry.name
                destination_child.mkdir()
                pending.append((source_path, destination_child, depth + 1))
                continue

            if not stat.S_ISREG(entry_info.st_mode):
                raise BuildEngineError("unsafe_file", "Non-regular files are not allowed in the build context.")
            if _is_excluded(entry.name, is_directory=False):
                continue
            files_copied += 1
            if files_copied > limits.max_files:
                raise BuildEngineError("too_many_files", "The repository exceeds the build context file limit.")
            bytes_copied += _copy_regular_file(
                source_path,
                destination_directory / entry.name,
                expected_info=entry_info,
                root=source_root,
                max_file_size=limits.max_file_size_bytes,
                bytes_already_copied=bytes_copied,
                max_context_bytes=limits.max_context_bytes,
            )

    return files_copied, bytes_copied


def _count_output(stream: BinaryIO, counts: _OutputCounts, limits: BuildLimits) -> None:
    """Drain Docker output while retaining bounded counters and its final excerpt."""
    try:
        while True:
            chunk = stream.read(READ_CHUNK_BYTES)
            if not chunk:
                return

            counts.retain_tail(chunk)
            remaining = max(0, limits.max_output_bytes - counts.bytes_seen)
            accepted = chunk[:remaining]
            counts.bytes_seen += len(accepted)
            newline_count = accepted.count(b"\n")
            next_line_count = counts.lines_seen + newline_count
            if next_line_count > limits.max_output_lines:
                counts.truncated = True
            counts.lines_seen = min(limits.max_output_lines, next_line_count)
            if accepted:
                counts.ends_with_newline = accepted.endswith(b"\n")
            if len(chunk) > len(accepted):
                counts.truncated = True
    except (OSError, ValueError):
        counts.truncated = True


def _terminate_process(process: subprocess.Popen) -> bool:
    """Terminate and reap only the Docker CLI process, if possible."""
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    except Exception:
        return False
    try:
        return process.poll() is not None
    except Exception:
        return False


class _BuildTemporaryDirectory:
    """Own one staging directory with optional cleanup after a child exits."""

    def __init__(self, prefix: str):
        self.name = tempfile.mkdtemp(prefix=prefix)
        self._cleanup_deferred = False

    def __fspath__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __enter__(self) -> "_BuildTemporaryDirectory":
        return self

    def __exit__(self, *_exc_info) -> None:
        if not self._cleanup_deferred:
            self._remove_owned_directory()

    def _remove_owned_directory(self) -> None:
        """Remove only the directory created by this build attempt."""
        owned_path = Path(self.name)
        if not owned_path.is_symlink():
            shutil.rmtree(owned_path, ignore_errors=True)

    def defer_cleanup_until_process_exit(self, process) -> None:
        """Retain staging files while the Docker CLI may still read them."""
        self._cleanup_deferred = True

        def clean_after_exit() -> None:
            try:
                process.wait()
            except Exception:
                return
            self._remove_owned_directory()

        Thread(target=clean_after_exit, daemon=True).start()


def _defer_staging_cleanup(temporary_directory: _BuildTemporaryDirectory, process) -> None:
    """Keep staged files until a Docker CLI that resisted termination exits."""
    temporary_directory.defer_cleanup_until_process_exit(process)


def _docker_environment(config_directory: Path | None = None) -> dict[str, str]:
    """Use an isolated CLI config while retaining only process essentials."""
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    if config_directory is not None:
        environment["DOCKER_CONFIG"] = str(config_directory)
    return environment


def _docker_endpoint(docker_executable: str) -> str:
    """Resolve the user's active Docker context to its daemon endpoint.

    Context inspection reads context metadata only. The actual build is then
    pinned to that endpoint while using a fresh credential-free CLI config.
    TLS and SSH contexts are rejected because their key material should not be
    copied into a build's temporary configuration.
    """
    discovery_environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST"}
    }
    try:
        endpoint_info = {}
        selected_context = os.environ.get("DOCKER_CONTEXT")
        explicit_host = os.environ.get("DOCKER_HOST")
        if explicit_host and not selected_context:
            endpoint = explicit_host
            if os.environ.get("DOCKER_TLS_VERIFY") or os.environ.get("DOCKER_CERT_PATH"):
                raise BuildEngineError("unsupported_docker_context", "TLS Docker endpoints require unsupported credentials.")
        else:
            if selected_context:
                context_name = selected_context
            else:
                context_result = subprocess.run(
                    [docker_executable, "context", "show"],
                    shell=False, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=10, env=discovery_environment,
                )
                if context_result.returncode != 0:
                    raise BuildEngineError("docker_unavailable", "The selected Docker context could not be read.")
                context_name = context_result.stdout.strip()
            if not context_name or any(character.isspace() for character in context_name):
                raise BuildEngineError("docker_unavailable", "The selected Docker context is invalid.")
            inspect_result = subprocess.run(
                [docker_executable, "context", "inspect", context_name, "--format", "{{json .Endpoints.docker}}"],
                shell=False, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=10, env=discovery_environment,
            )
            if inspect_result.returncode != 0:
                raise BuildEngineError("docker_unavailable", "The selected Docker endpoint could not be read.")
            endpoint_info = json.loads(inspect_result.stdout)
            endpoint = endpoint_info.get("Host") if isinstance(endpoint_info, dict) else None
        if not isinstance(endpoint, str) or not endpoint:
            raise BuildEngineError("docker_unavailable", "The selected Docker endpoint is invalid.")
        if endpoint.startswith(("ssh://", "tcp://")) or endpoint_info.get("SkipTLSVerify") is True:
            raise BuildEngineError("unsupported_docker_context", "This Docker context requires unsupported remote or TLS credentials.")
        if not endpoint.startswith(("npipe://", "unix://")):
            raise BuildEngineError("unsupported_docker_context", "Only local Docker Desktop or Unix socket contexts are supported.")
        return endpoint
    except BuildEngineError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        raise BuildEngineError("docker_unavailable", "The selected Docker context could not be resolved safely.")


def _check_image_tag_available(
    docker_executable: str, endpoint: str, config_directory: Path, image_tag: str
) -> None:
    """Check only the exact DeploySphere tag before attempting to build it."""
    try:
        result = subprocess.run(
            [docker_executable, "--config", str(config_directory), "--host", endpoint,
             "image", "ls", "--quiet", "--no-trunc", "--filter", f"reference={image_tag}"],
            shell=False, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=15, env=_docker_environment(config_directory),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BuildEngineError("tag_check_failed", "The Docker image tag could not be checked safely.") from error
    if result.returncode != 0:
        raise BuildEngineError("tag_check_failed", "The Docker image tag could not be checked safely.")
    if result.stdout.strip():
        raise BuildEngineError("image_tag_exists", "This DeploySphere image tag already exists; generate a new unique tag.")


def build_docker_image(
    workspace_root: str | Path,
    build_context: str | Path,
    dockerfile_path: str | Path,
    image_tag: str,
    detection: DetectionResult,
    *,
    limits: BuildLimits | None = None,
) -> BuildResult:
    """Build one supported detected application using a sanitized temp context.

    ``build_context`` and ``dockerfile_path`` are POSIX-style paths relative to
    ``workspace_root``. The Dockerfile must be listed by the saved detection
    result and must be inside the selected context. Build output is drained in
    fixed-size chunks; only a bounded, redacted tail excerpt is returned.
    """
    started_at = time.monotonic()
    configured_limits = limits or BuildLimits()
    validated_tag = image_tag if isinstance(image_tag, str) and IMAGE_TAG_PATTERN.fullmatch(image_tag) else None
    counts = _OutputCounts()
    tag_reserved = False
    tag_quarantined = False
    cli_started = False
    process_exit_confirmed = False
    process = None

    try:
        if limits is None:
            configured_limits = BuildLimits.from_environment()
        if validated_tag is None:
            raise BuildEngineError(
                "invalid_image_tag",
                "Use a unique tag generated in the deploysphere/build namespace.",
            )
        with _RESERVED_TAGS_LOCK:
            if validated_tag in _RESERVED_TAGS:
                raise BuildEngineError(
                    "image_tag_in_use",
                    "This DeploySphere image tag is active or quarantined after an indeterminate build.",
                )
            _RESERVED_TAGS.add(validated_tag)
            tag_reserved = True
        if not isinstance(detection, DetectionResult):
            raise BuildEngineError("invalid_detection", "A valid application detection result is required.")

        context_parts = _normalize_relative_path(build_context, allow_root=True)
        dockerfile_parts = _normalize_relative_path(dockerfile_path)
        normalized_dockerfile = PurePosixPath(*dockerfile_parts).as_posix()
        _validate_detection(detection, normalized_dockerfile)

        root = _verified_workspace_root(workspace_root)
        context_path = _resolve_inside_workspace(root, context_parts, expect_directory=True)
        dockerfile_full_path = _resolve_inside_workspace(root, dockerfile_parts, expect_directory=False)
        try:
            dockerfile_full_path.relative_to(context_path)
        except ValueError as error:
            raise BuildEngineError("unsafe_path", "The Dockerfile must be inside the build context.") from error

        docker_executable = shutil.which("docker")
        if docker_executable is None:
            raise BuildEngineError("docker_unavailable", "The Docker executable is not installed or not on PATH.")

        with _BuildTemporaryDirectory(prefix="deploysphere-build-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            docker_config = temporary_root / "docker-config"
            docker_config.mkdir()
            (docker_config / "config.json").write_text("{}", encoding="utf-8")
            staged_context = temporary_root / "context"
            staged_context.mkdir()
            _stage_context(context_path, staged_context, configured_limits)

            endpoint = _docker_endpoint(docker_executable)
            _check_image_tag_available(docker_executable, endpoint, docker_config, validated_tag)

            dockerfile_in_context = dockerfile_full_path.relative_to(context_path)
            staged_dockerfile = staged_context / dockerfile_in_context
            if staged_dockerfile.is_symlink() or not staged_dockerfile.is_file():
                raise BuildEngineError("invalid_dockerfile", "The selected Dockerfile was not staged safely.")

            iid_file = temporary_root / "image-id.txt"
            command = [
                docker_executable,
                "--config",
                str(docker_config),
                "--host",
                endpoint,
                "build",
                "--iidfile",
                str(iid_file),
                "--file",
                str(staged_dockerfile),
                "--tag",
                validated_tag,
                str(staged_context),
            ]

            try:
                process = subprocess.Popen(
                    command,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(temporary_root),
                    env=_docker_environment(docker_config),
                    bufsize=0,
                )
            except FileNotFoundError as error:
                raise BuildEngineError("docker_unavailable", "The Docker executable could not be started.") from error
            except OSError as error:
                raise BuildEngineError("docker_unavailable", "The Docker command could not be started.") from error
            cli_started = True

            timed_out = False
            process_stopped = True
            try:
                if process.stdout is None:
                    raise RuntimeError("Docker output pipe was unavailable")
                reader = Thread(
                    target=_count_output,
                    args=(process.stdout, counts, configured_limits),
                    daemon=True,
                )
                reader.start()
                try:
                    return_code = process.wait(timeout=configured_limits.timeout_seconds)
                    process_exit_confirmed = True
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process_stopped = _terminate_process(process)
                    process_exit_confirmed = process_stopped
                    return_code = process.returncode
                finally:
                    reader.join(timeout=TERMINATE_GRACE_SECONDS)
                    if reader.is_alive():
                        try:
                            process.stdout.close()
                        except OSError:
                            pass
                        reader.join(timeout=1)
                    try:
                        process.stdout.close()
                    except OSError:
                        pass
            except BuildEngineError:
                raise
            except Exception as error:
                if cli_started and not process_exit_confirmed:
                    # A client-side exception does not prove whether the daemon
                    # accepted or completed the request. Preserve both tag and
                    # staging data until the CLI is known to have exited.
                    tag_quarantined = True
                    process_stopped = _terminate_process(process)
                    process_exit_confirmed = process_stopped
                    if not process_stopped:
                        _defer_staging_cleanup(temporary_directory, process)
                    raise BuildEngineError(
                        "build_outcome_indeterminate",
                        "The Docker build ended unexpectedly; its daemon outcome is unknown.",
                    ) from error
                raise

            if timed_out:
                # Killing the CLI does not establish that the daemon stopped
                # the accepted build. Keep this tag reserved for this process.
                tag_quarantined = True
                if not process_stopped:
                    _defer_staging_cleanup(temporary_directory, process)
                timeout_message = (
                    "The Docker CLI was stopped after timeout; the daemon may still be processing the build, so its outcome is indeterminate."
                    if process_stopped
                    else "The Docker CLI could not be confirmed stopped; the daemon build outcome is indeterminate."
                )
                raise BuildEngineError("build_timeout", timeout_message)
            if return_code != 0:
                raise BuildEngineError("build_failed", "Docker reported that the image build failed.")

            image_id = None
            try:
                if iid_file.is_file() and not iid_file.is_symlink():
                    if iid_file.stat().st_size <= 128:
                        with iid_file.open("rb") as image_id_file:
                            candidate_id = image_id_file.read(128).decode("ascii", errors="ignore").strip()
                    else:
                        candidate_id = ""
                    if IMAGE_ID_PATTERN.fullmatch(candidate_id):
                        image_id = candidate_id
            except OSError:
                # The image may still have built; an unavailable ID file is optional metadata.
                image_id = None

            line_count = counts.lines_seen + int(counts.bytes_seen > 0 and not counts.ends_with_newline)
            return BuildResult(
                success=True,
                image_tag=validated_tag,
                image_id=image_id,
                duration_seconds=round(time.monotonic() - started_at, 3),
                log_bytes=min(counts.bytes_seen, configured_limits.max_output_bytes),
                log_line_count=min(line_count, configured_limits.max_output_lines),
                log_truncated=counts.truncated
                or counts.bytes_seen > configured_limits.max_output_bytes
                or line_count > configured_limits.max_output_lines,
                log_excerpt=_safe_log_excerpt(
                    counts.excerpt, truncated=counts.excerpt_truncated
                ),
            )
    except BuildEngineError as error:
        return _failed_result(
            started_at,
            error,
            image_tag=validated_tag,
            counts=counts,
            limits=configured_limits,
            outcome_known=error.code not in {"build_timeout", "build_outcome_indeterminate"},
        )
    except Exception:
        # Avoid reflecting paths or subprocess exception details to callers.
        safe_error = BuildEngineError("build_engine_error", "The Docker build could not be completed safely.")
        return _failed_result(
            started_at,
            safe_error,
            image_tag=validated_tag,
            counts=counts,
            limits=configured_limits,
            outcome_known=not (cli_started and not process_exit_confirmed),
        )
    finally:
        if tag_reserved and not tag_quarantined and validated_tag is not None:
            with _RESERVED_TAGS_LOCK:
                _RESERVED_TAGS.discard(validated_tag)
