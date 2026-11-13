"""Focused, filesystem-only tests for the application detection service."""

from pathlib import Path

import pytest

from backend.services import application_detector as detector


def _workspace(tmp_path, files=None):
    """Create an isolated workspace populated only with test fixtures."""
    root = tmp_path / "workspace"
    root.mkdir()
    for relative_name, contents in (files or {}).items():
        file_path = root / relative_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, bytes):
            file_path.write_bytes(contents)
        else:
            file_path.write_text(contents, encoding="utf-8")
    return root


def test_python_only_repository(tmp_path):
    root = _workspace(tmp_path, {"requirements.txt": "requests==2.0\n"})

    result = detector.detect_application(root)

    assert result.application_type == "python"
    assert result.build_method == "native"
    assert result.detected_application_types == ["python"]
    assert result.detected_build_methods == ["native"]
    assert result.detected_files == ["requirements.txt"]
    assert 0 <= result.confidence <= 1
    assert any("does not prove" in warning for warning in result.warnings)


def test_node_only_repository(tmp_path):
    root = _workspace(tmp_path, {"package.json": '{"name":"sample"}'})

    result = detector.detect_application(root)

    assert result.application_type == "nodejs"
    assert result.build_method == "native"
    assert result.detected_files == ["package.json"]


def test_dockerfile_only_repository(tmp_path):
    root = _workspace(tmp_path, {"Dockerfile": "FROM scratch\n"})

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.build_method == "dockerfile"
    assert result.detected_build_methods == ["dockerfile"]


def test_root_app_py_with_dockerfile_is_python_application(tmp_path):
    root = _workspace(
        tmp_path,
        {"app.py": "raise RuntimeError('source is not executed')\n", "Dockerfile": "FROM scratch\n"},
    )

    result = detector.detect_application(root)

    assert result.application_type == "python"
    assert result.build_method == "dockerfile"
    assert result.detected_application_types == ["python"]
    assert result.detected_build_methods == ["dockerfile"]
    assert result.detected_files == ["app.py", "Dockerfile"]


def test_root_app_py_without_python_manifest_or_dockerfile_remains_unknown(tmp_path):
    root = _workspace(tmp_path, {"app.py": "application source"})

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.build_method == "unknown"
    assert result.detected_files == []


def test_nested_app_py_with_dockerfile_remains_unknown(tmp_path):
    root = _workspace(
        tmp_path,
        {"services/api/app.py": "application source", "Dockerfile": "FROM scratch\n"},
    )

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.build_method == "dockerfile"
    assert result.detected_files == ["Dockerfile"]


def test_compose_only_repository(tmp_path):
    root = _workspace(tmp_path, {"compose.yaml": "services: {}\n"})

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.build_method == "compose"
    assert result.detected_files == ["compose.yaml"]


def test_python_manifest_with_dockerfile_uses_dockerfile_method(tmp_path):
    root = _workspace(
        tmp_path,
        {"pyproject.toml": '[project]\nname = "sample"\n', "Dockerfile": "FROM python\n"},
    )

    result = detector.detect_application(root)

    assert result.application_type == "python"
    assert result.build_method == "dockerfile"
    assert result.detected_build_methods == ["native", "dockerfile"]


def test_node_manifest_with_dockerfile_uses_dockerfile_method(tmp_path):
    root = _workspace(
        tmp_path,
        {"package.json": '{"name":"sample"}', "Dockerfile.dev": "FROM node\n"},
    )

    result = detector.detect_application(root)

    assert result.application_type == "nodejs"
    assert result.build_method == "dockerfile"
    assert "Dockerfile.dev" in result.detected_files


def test_python_and_node_manifests_are_mixed_application_type(tmp_path):
    root = _workspace(
        tmp_path,
        {"requirements.txt": "", "package.json": '{"name":"sample"}'},
    )

    result = detector.detect_application(root)

    assert result.application_type == "mixed"
    assert result.build_method == "native"
    assert result.detected_application_types == ["python", "nodejs"]


def test_dockerfile_and_compose_classify_as_mixed_and_keep_both(tmp_path):
    root = _workspace(
        tmp_path,
        {"Dockerfile.production": "FROM scratch\n", "docker-compose.override.yml": "services: {}\n"},
    )

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.build_method == "mixed"
    assert result.detected_build_methods == ["dockerfile", "compose"]
    assert result.detected_files == ["docker-compose.override.yml", "Dockerfile.production"]


def test_nested_application_directories_are_detected_with_relative_paths(tmp_path):
    root = _workspace(tmp_path, {"services/api/requirements.txt": ""})

    result = detector.detect_application(root)

    assert result.application_type == "python"
    assert result.detected_files == ["services/api/requirements.txt"]
    assert all(not Path(path).is_absolute() for path in result.detected_files)


def test_multiple_manifests_and_dockerfile_variants_are_all_reported(tmp_path):
    root = _workspace(
        tmp_path,
        {
            "requirements.txt": "",
            "pyproject.toml": '[project]\nname = "sample"\n',
            "Pipfile": '[packages]\nrequests = "*"\n',
            "Dockerfile.dev": "FROM python\n",
            "Dockerfile.production": "FROM python\n",
            "api.Dockerfile": "FROM python\n",
        },
    )

    result = detector.detect_application(root)

    assert result.application_type == "python"
    assert result.build_method == "dockerfile"
    assert len(result.detected_files) == 6


def test_unknown_repository_has_zero_confidence(tmp_path):
    root = _workspace(tmp_path, {"README.md": "No recognized indicators.\n"})

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.build_method == "unknown"
    assert result.detected_files == []
    assert result.confidence == 0.0


def test_malformed_package_json_becomes_warning(tmp_path):
    root = _workspace(tmp_path, {"package.json": '{"name": '})

    result = detector.detect_application(root)

    assert result.application_type == "nodejs"
    assert result.build_method == "native"
    assert any("package.json" in warning and "malformed" in warning for warning in result.warnings)


@pytest.mark.parametrize("filename", ["pyproject.toml", "Pipfile"])
def test_malformed_toml_becomes_warning(tmp_path, filename):
    root = _workspace(tmp_path, {filename: "[invalid\nnot toml"})

    result = detector.detect_application(root)

    assert result.application_type == "python"
    assert any(filename in warning and "malformed" in warning for warning in result.warnings)


def test_invalid_text_encoding_becomes_warning(tmp_path):
    root = _workspace(tmp_path, {"package.json": b"\xff\xfe"})

    result = detector.detect_application(root)

    assert result.application_type == "nodejs"
    assert any("not valid UTF-8" in warning for warning in result.warnings)


def test_unreadable_manifest_becomes_warning(tmp_path, monkeypatch):
    root = _workspace(tmp_path, {"package.json": '{"name":"sample"}'})
    manifest = root / "package.json"
    original_read = detector._read_manifest

    def unreadable(path, workspace, max_bytes):
        if path == manifest:
            raise PermissionError("test unreadable file")
        return original_read(path, workspace, max_bytes)

    monkeypatch.setattr(detector, "_read_manifest", unreadable)

    result = detector.detect_application(root)

    assert result.application_type == "nodejs"
    assert any("package.json" in warning and "could not be read" in warning for warning in result.warnings)


class _FakeEntry:
    """Minimal directory entry for symlink cases unavailable on this host."""

    def __init__(self, name, path):
        self.name = name
        self.path = str(path)

    def is_symlink(self):
        return True

    def is_dir(self, *, follow_symlinks=True):
        raise AssertionError("Symlink targets must not be inspected")

    def is_file(self, *, follow_symlinks=True):
        raise AssertionError("Symlink targets must not be inspected")


class _FakeScandir:
    def __init__(self, entries):
        self.entries = entries

    def __enter__(self):
        return iter(self.entries)

    def __exit__(self, *args):
        return False


@pytest.mark.parametrize("entry_name", ["linked-package.json", "linked-directory"])
def test_symlink_entries_are_skipped_without_reading_targets(tmp_path, monkeypatch, entry_name):
    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "package.json").write_text('{"private":"secret"}', encoding="utf-8")
    fake_entry = _FakeEntry(entry_name, root / entry_name)
    real_scandir = detector.os.scandir
    monkeypatch.setattr(
        detector.os,
        "scandir",
        lambda directory: _FakeScandir([fake_entry]) if Path(directory) == root else real_scandir(directory),
    )

    result = detector.detect_application(root)

    assert result.detected_files == []
    assert any(entry_name in warning and "symlink skipped" in warning for warning in result.warnings)


def test_symlink_workspace_root_is_rejected(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == root or original_is_symlink(path),
    )

    with pytest.raises(detector.DetectionError) as error:
        detector.detect_application(root)

    assert error.value.code == "invalid_workspace"


def test_workspace_escape_is_skipped(tmp_path, monkeypatch):
    root = _workspace(tmp_path, {"package.json": '{"name":"outside"}'})
    package_path = root / "package.json"
    original_check = detector._inside_workspace

    def reject_package(path, workspace):
        if path == package_path:
            return False
        return original_check(path, workspace)

    monkeypatch.setattr(detector, "_inside_workspace", reject_package)

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.detected_files == []
    assert any("path escaped" in warning for warning in result.warnings)


def test_maximum_depth_stops_nested_scan_with_warning(tmp_path):
    root = _workspace(tmp_path, {"level1/level2/requirements.txt": ""})

    result = detector.detect_application(root, limits=detector.ScanLimits(max_depth=1))

    assert result.detected_files == []
    assert any("maximum directory depth" in warning for warning in result.warnings)


def test_maximum_entry_limit_returns_partial_result_and_warning(tmp_path):
    root = _workspace(
        tmp_path,
        {"requirements.txt": "", "package.json": '{"name":"sample"}', "Dockerfile": "FROM scratch"},
    )

    result = detector.detect_application(root, limits=detector.ScanLimits(max_entries=2))

    assert len(result.detected_files) <= 2
    assert any("Entry limit reached" in warning for warning in result.warnings)


def test_maximum_file_size_skips_parse_and_warns(tmp_path):
    root = _workspace(tmp_path, {"package.json": '{"name":"sample"}'})

    result = detector.detect_application(
        root, limits=detector.ScanLimits(max_file_size_bytes=4)
    )

    assert result.application_type == "nodejs"
    assert any("size limit" in warning for warning in result.warnings)


def test_scan_timeout_aborts_with_safe_error(tmp_path, monkeypatch):
    root = _workspace(tmp_path, {"requirements.txt": ""})
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(detector.time, "monotonic", lambda: next(ticks))

    with pytest.raises(detector.DetectionError) as error:
        detector.detect_application(
            root, limits=detector.ScanLimits(max_scan_seconds=1.0)
        )

    assert error.value.code == "scan_timeout"


def test_excluded_directories_are_not_scanned(tmp_path):
    root = _workspace(
        tmp_path,
        {
            "node_modules/package.json": '{"name":"ignored"}',
            ".venv/requirements.txt": "",
            "dist/Dockerfile": "FROM scratch",
            "build/compose.yml": "services: {}",
            ".git/package.json": '{"name":"ignored"}',
        },
    )

    result = detector.detect_application(root)

    assert result.application_type == "unknown"
    assert result.detected_files == []


def test_environment_limits_are_configurable(monkeypatch):
    monkeypatch.setenv("DEPLOYSPHERE_DETECTION_MAX_DEPTH", "3")
    monkeypatch.setenv("DEPLOYSPHERE_DETECTION_MAX_ENTRIES", "25")
    monkeypatch.setenv("DEPLOYSPHERE_DETECTION_MAX_FILE_SIZE_BYTES", "2048")
    monkeypatch.setenv("DEPLOYSPHERE_DETECTION_MAX_SCAN_SECONDS", "4.5")

    limits = detector.ScanLimits.from_environment()

    assert limits == detector.ScanLimits(3, 25, 2048, 4.5)


def test_non_finite_scan_timeout_is_rejected():
    with pytest.raises(ValueError, match="max_scan_seconds"):
        detector.ScanLimits(max_scan_seconds=float("inf"))
