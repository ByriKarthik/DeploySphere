"""Identify common application and build files without executing the checkout."""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import stat
import time
import tomllib


EXCLUDED_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}
PYTHON_MANIFESTS = {"requirements.txt", "pyproject.toml", "pipfile"}
COMPOSE_BASE_NAMES = {
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
}
MANIFESTS_TO_PARSE = {"package.json", "pyproject.toml", "pipfile"}


class DetectionError(Exception):
    """A safe error raised when the workspace cannot be scanned reliably."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ScanLimits:
    """Resource limits for a single detection scan."""

    max_depth: int = 12
    max_entries: int = 10_000
    max_file_size_bytes: int = 1_048_576
    max_scan_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Reject invalid limits instead of silently disabling protections."""
        if self.max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least one")
        if self.max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes must be at least one")
        if not math.isfinite(self.max_scan_seconds) or self.max_scan_seconds <= 0:
            raise ValueError("max_scan_seconds must be greater than zero")

    @classmethod
    def from_environment(cls) -> "ScanLimits":
        """Build scan limits from optional DeploySphere environment settings."""
        try:
            return cls(
                max_depth=int(os.getenv("DEPLOYSPHERE_DETECTION_MAX_DEPTH", "12")),
                max_entries=int(os.getenv("DEPLOYSPHERE_DETECTION_MAX_ENTRIES", "10000")),
                max_file_size_bytes=int(
                    os.getenv("DEPLOYSPHERE_DETECTION_MAX_FILE_SIZE_BYTES", "1048576")
                ),
                max_scan_seconds=float(
                    os.getenv("DEPLOYSPHERE_DETECTION_MAX_SCAN_SECONDS", "10")
                ),
            )
        except ValueError as error:
            raise DetectionError(
                "invalid_configuration", "The application detection limits are invalid."
            ) from error


@dataclass
class DetectionResult:
    """Structured, relative-path-only summary of repository indicators."""

    application_type: str
    build_method: str
    detected_application_types: list[str] = field(default_factory=list)
    detected_build_methods: list[str] = field(default_factory=list)
    detected_files: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _inside_workspace(path: Path, workspace_root: Path) -> bool:
    """Check a resolved path is beneath the canonical workspace root."""
    try:
        path.resolve(strict=True).relative_to(workspace_root)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _is_dockerfile(name: str) -> bool:
    """Recognize the default Dockerfile and common named variants."""
    lower_name = name.lower()
    return (
        lower_name == "dockerfile"
        or lower_name.startswith("dockerfile.")
        or lower_name.startswith("dockerfile-")
        or lower_name.endswith(".dockerfile")
        or ".dockerfile." in lower_name
    )


def _is_compose_file(name: str) -> bool:
    """Recognize Compose files and common environment/override variants."""
    lower_name = name.lower()
    if lower_name in COMPOSE_BASE_NAMES:
        return True
    if not lower_name.endswith((".yaml", ".yml")):
        return False
    return lower_name.startswith("compose.") or lower_name.startswith("docker-compose.")


def _file_kind(name: str) -> str | None:
    """Return the indicator family for a recognized filename."""
    lower_name = name.lower()
    if lower_name in PYTHON_MANIFESTS:
        return "python"
    if lower_name == "package.json":
        return "nodejs"
    if _is_dockerfile(name):
        return "dockerfile"
    if _is_compose_file(name):
        return "compose"
    return None


def _read_manifest(path: Path, root: Path, max_bytes: int) -> bytes:
    """Read a bounded regular file after checking symlinks and containment."""
    if path.is_symlink() or not _inside_workspace(path, root):
        raise OSError("manifest path is not a regular in-workspace file")

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise OSError("manifest is not a regular file")
    if before.st_size > max_bytes:
        raise OverflowError("manifest exceeds the configured file-size limit")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("manifest is not a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise OSError("manifest changed while being opened")
        if opened.st_size > max_bytes or not _inside_workspace(path, root):
            raise OverflowError("manifest changed or exceeds the configured file-size limit")

        with os.fdopen(descriptor, "rb") as manifest_file:
            descriptor = -1
            content = manifest_file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise OverflowError("manifest exceeds the configured file-size limit")
        return content
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _inspect_manifest(
    path: Path,
    root: Path,
    limits: ScanLimits,
    check_timeout,
    warnings: set[str],
) -> bool:
    """Parse JSON/TOML manifests as data and return whether parsing succeeded."""
    relative_path = path.relative_to(root).as_posix()
    check_timeout()
    try:
        content = _read_manifest(path, root, limits.max_file_size_bytes)
        text = content.decode("utf-8")
        name = path.name.lower()
        parsed = json.loads(text) if name == "package.json" else tomllib.loads(text)
        if not isinstance(parsed, dict):
            warnings.add(f"{relative_path}: configuration must contain an object/table.")
            return False
        check_timeout()
        return True
    except OverflowError:
        warnings.add(f"{relative_path}: file exceeds the configured size limit; it was not parsed.")
    except UnicodeDecodeError:
        warnings.add(f"{relative_path}: file is not valid UTF-8 and was not parsed.")
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, RecursionError):
        warnings.add(f"{relative_path}: malformed configuration; it was not parsed.")
    except OSError:
        warnings.add(f"{relative_path}: file could not be read safely.")
    check_timeout()
    return False


def _classify(
    file_kinds: set[str],
    valid_manifest_count: int,
    warning_count: int,
) -> DetectionResult:
    """Apply the documented deterministic application/build precedence rules."""
    application_types = []
    if "python" in file_kinds or "python_entrypoint" in file_kinds:
        application_types.append("python")
    if "nodejs" in file_kinds:
        application_types.append("nodejs")

    if not application_types:
        application_type = "unknown"
    elif len(application_types) == 1:
        application_type = application_types[0]
    else:
        application_type = "mixed"

    build_methods = []
    if "python" in file_kinds or "nodejs" in file_kinds:
        build_methods.append("native")
    if "dockerfile" in file_kinds:
        build_methods.append("dockerfile")
    if "compose" in file_kinds:
        build_methods.append("compose")

    # Compose plus Dockerfile configuration represents two available methods.
    # Otherwise container configuration takes precedence over native manifests.
    if "compose" in file_kinds and "dockerfile" in file_kinds:
        build_method = "mixed"
    elif "compose" in file_kinds:
        build_method = "compose"
    elif "dockerfile" in file_kinds:
        build_method = "dockerfile"
    elif "native" in build_methods:
        build_method = "native"
    else:
        build_method = "unknown"

    if not file_kinds:
        confidence = 0.0
    else:
        confidence = 0.25
        confidence += 0.18 * len(application_types)
        confidence += 0.12 * ("dockerfile" in file_kinds)
        confidence += 0.12 * ("compose" in file_kinds)
        confidence += 0.08 * min(valid_manifest_count, 2)
        confidence -= 0.08 * warning_count
        confidence = round(min(0.9, max(0.1, confidence)), 2)

    return DetectionResult(
        application_type=application_type,
        build_method=build_method,
        detected_application_types=application_types,
        detected_build_methods=build_methods,
        confidence=confidence,
    )


def detect_application(
    workspace_root: str | Path,
    *,
    limits: ScanLimits | None = None,
) -> DetectionResult:
    """Inspect a verified workspace and classify common application files.

    Classification is filename-based. Python and Node.js manifests determine
    ``application_type``; both together mean ``mixed``. ``build_method`` is
    ``mixed`` when Dockerfile and Compose files coexist, otherwise Compose
    takes precedence over Dockerfile, which takes precedence over native
    manifests. The detailed indicator lists always retain all observed types.
    No repository code, package scripts, Docker configuration, or dependencies
    are executed or installed.
    """
    scan_limits = limits or ScanLimits.from_environment()
    input_root = Path(workspace_root).expanduser()
    if not input_root.is_absolute():
        raise DetectionError("invalid_workspace", "The workspace root must be an absolute path.")
    if input_root.is_symlink():
        raise DetectionError("invalid_workspace", "The workspace root cannot be a symlink.")
    try:
        root = input_root.resolve(strict=True)
        if not root.is_dir():
            raise DetectionError("invalid_workspace", "The workspace root must be a directory.")
    except DetectionError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DetectionError("invalid_workspace", "The workspace root is unavailable.") from error

    started_at = time.monotonic()
    deadline = started_at + scan_limits.max_scan_seconds

    def check_timeout() -> None:
        if time.monotonic() > deadline:
            raise DetectionError("scan_timeout", "Application detection exceeded its time limit.")

    warnings: set[str] = {
        "Filename detection does not prove that the application is deployable."
    }
    file_kinds: set[str] = set()
    detected_files: set[str] = set()
    valid_manifest_count = 0
    has_root_python_entrypoint = False
    entries_seen = 0
    stop_scanning = False
    pending_directories: list[tuple[Path, int]] = [(root, 0)]

    while pending_directories and not stop_scanning:
        check_timeout()
        directory, depth = pending_directories.pop()
        if directory.is_symlink() or not _inside_workspace(directory, root):
            relative_directory = directory.relative_to(root).as_posix() or "."
            warnings.add(f"{relative_directory}: directory path escaped the workspace and was skipped.")
            continue
        try:
            with os.scandir(directory) as iterator:
                entries = []
                remaining_entries = scan_limits.max_entries - entries_seen
                for entry in iterator:
                    check_timeout()
                    if len(entries) >= remaining_entries:
                        stop_scanning = True
                        break
                    entries.append(entry)
                entries.sort(key=lambda entry: (entry.name.casefold(), entry.name))
        except OSError:
            relative_directory = directory.relative_to(root).as_posix() or "."
            warnings.add(f"{relative_directory}: directory could not be read.")
            continue

        child_directories = []
        for entry in entries:
            check_timeout()
            entries_seen += 1

            entry_path = Path(entry.path)
            relative_path = entry_path.relative_to(root).as_posix()
            try:
                if entry.is_symlink():
                    warnings.add(f"{relative_path}: symlink skipped for safety.")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name.casefold() in EXCLUDED_DIRECTORIES:
                        continue
                    if not _inside_workspace(entry_path, root):
                        warnings.add(f"{relative_path}: path escaped the workspace and was skipped.")
                        continue
                    if depth >= scan_limits.max_depth:
                        warnings.add(f"{relative_path}: maximum directory depth reached.")
                        continue
                    child_directories.append((entry_path, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if not _inside_workspace(entry_path, root):
                    warnings.add(f"{relative_path}: path escaped the workspace and was skipped.")
                    continue

                kind = _file_kind(entry.name)
                if kind is None:
                    if relative_path == "app.py":
                        has_root_python_entrypoint = True
                    continue
                file_kinds.add(kind)
                detected_files.add(relative_path)
                if entry.name.lower() in MANIFESTS_TO_PARSE:
                    if _inspect_manifest(
                        entry_path, root, scan_limits, check_timeout, warnings
                    ):
                        valid_manifest_count += 1
            except OSError:
                warnings.add(f"{relative_path}: filesystem entry could not be inspected safely.")

        pending_directories.extend(reversed(child_directories))
        if stop_scanning:
            warnings.add("Entry limit reached; detection results may be incomplete.")

    if has_root_python_entrypoint and "dockerfile" in file_kinds:
        file_kinds.add("python_entrypoint")
        detected_files.add("app.py")

    result = _classify(file_kinds, valid_manifest_count, len(warnings) - 1)
    result.detected_files = sorted(detected_files, key=lambda item: (item.casefold(), item))
    result.warnings = sorted(warnings, key=lambda item: (item.casefold(), item))
    return result
