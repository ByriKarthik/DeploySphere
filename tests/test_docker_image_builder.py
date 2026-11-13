"""Mocked tests for the Docker image-build engine; no Docker daemon is used."""

from io import BytesIO
from pathlib import Path
import subprocess
from threading import Event

import pytest

from backend.services import docker_image_builder as builder
from backend.services.application_detector import DetectionResult


def _workspace(root: Path) -> Path:
    """Create a disposable repository fixture without executing its contents."""
    root.mkdir()
    (root / "Dockerfile").write_text("FROM scratch\nCOPY app.txt /app.txt\n", encoding="utf-8")
    (root / "app.txt").write_text("test data", encoding="utf-8")
    return root


def _detection(**overrides) -> DetectionResult:
    """Return supported detector output for a Python Dockerfile repository."""
    fields = {
        "application_type": "python",
        "build_method": "dockerfile",
        "detected_application_types": ["python"],
        "detected_build_methods": ["native", "dockerfile"],
        "detected_files": ["Dockerfile", "requirements.txt"],
        "confidence": 0.8,
        "warnings": [],
    }
    fields.update(overrides)
    return DetectionResult(**fields)


def _tag() -> str:
    """Use the service's unique tag helper in tests."""
    return builder.create_unique_image_tag()


class FakeProcess:
    """Small Popen replacement that never launches Docker or repository code."""

    def __init__(
        self,
        command,
        *,
        returncode=0,
        output=b"step done\n",
        timeout=False,
        wait_error=None,
    ):
        self.command = command
        self.returncode = returncode
        self.stdout = BytesIO(output)
        self.timeout = timeout
        self.wait_error = wait_error
        self.wait_error_raised = False
        self.terminated = False
        iid_path = Path(command[command.index("--iidfile") + 1])
        if returncode == 0:
            iid_path.write_text("sha256:" + "a" * 64, encoding="ascii")

    def wait(self, timeout=None):
        if self.wait_error is not None and not self.wait_error_raised:
            self.wait_error_raised = True
            raise self.wait_error
        if self.timeout and not self.terminated:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9

    def poll(self):
        return self.returncode


def _configure_staging(monkeypatch, tmp_path):
    """Make all temporary staging directories visible to the test."""
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    monkeypatch.setattr(builder.tempfile, "tempdir", str(staging_root))
    return staging_root


def _mock_docker(monkeypatch, *, returncode=0, output=b"step done\n", timeout=False,
                 existing_image=False, processes=None, wait_errors=None):
    """Patch Docker discovery and process launch with an inert fake."""
    calls = []
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker-test.exe")

    def fake_run(command, **kwargs):
        if command[1:3] == ["context", "show"]:
            return subprocess.CompletedProcess(command, 0, "desktop-linux\n", "")
        if "inspect" in command:
            return subprocess.CompletedProcess(
                command, 0, '{"Host":"npipe:////./pipe/docker_engine","SkipTLSVerify":false}', ""
            )
        if "image" in command and existing_image:
            return subprocess.CompletedProcess(command, 0, "sha256:existing\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        wait_error = wait_errors.pop(0) if wait_errors else None
        process = FakeProcess(
            command,
            returncode=returncode,
            output=output,
            timeout=timeout,
            wait_error=wait_error,
        )
        if processes is not None:
            processes.append(process)
        return process

    monkeypatch.setattr(builder.subprocess, "Popen", fake_popen)
    return calls


def test_successful_build_uses_safe_arguments_and_returns_image_metadata(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path / "workspace")
    (workspace / ".env").write_text("SECRET=do-not-copy", encoding="utf-8")
    (workspace / ".env.example").write_text("SAFE=example", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("private", encoding="utf-8")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "large.js").write_text("unused", encoding="utf-8")
    staging_root = _configure_staging(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    original_config = tmp_path / "user-docker-config"
    original_config.mkdir()
    (original_config / "config.json").write_text(
        '{"auths":{"registry.example":{"auth":"secret"}},"credsStore":"desktop"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(original_config))
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    monkeypatch.setenv("DOCKER_HOST", "tcp://credential-host.invalid:2376")
    built_tags = set()

    def fake_run(command, **kwargs):
        if command[1:3] == ["context", "show"]:
            return subprocess.CompletedProcess(command, 0, "desktop-linux\n", "")
        if "inspect" in command:
            return subprocess.CompletedProcess(
                command, 0, '{"Host":"npipe:////./pipe/docker_engine","SkipTLSVerify":false}', ""
            )
        if "image" in command:
            requested_tag = command[-1].split("=", 1)[1]
            return subprocess.CompletedProcess(
                command, 0, "sha256:existing\n" if requested_tag in built_tags else "", ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    calls = []
    isolated_configs = []
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker-test.exe")

    def fake_popen(command, **kwargs):
        config_dir = Path(command[2])
        isolated_configs.append((config_dir, (config_dir / "config.json").read_text(encoding="utf-8")))
        staged_context = Path(command[-1])
        assert (staged_context / "app.txt").exists()
        assert (staged_context / "Dockerfile").is_file()
        assert Path(command[command.index("--file") + 1]).resolve() == (
            staged_context / "Dockerfile"
        ).resolve()
        assert not (staged_context / ".env").exists()
        assert not (staged_context / ".env.example").exists()
        assert not (staged_context / ".git").exists()
        assert not (staged_context / "node_modules").exists()
        calls.append((command, kwargs))
        built_tags.add(command[command.index("--tag") + 1])
        return FakeProcess(command)

    monkeypatch.setattr(builder.subprocess, "Popen", fake_popen)

    image_tag = _tag()
    result = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        image_tag,
        _detection(),
    )

    assert result.success is True
    assert result.image_id == "sha256:" + "a" * 64
    assert result.log_bytes > 0
    assert result.log_line_count == 1
    assert result.log_excerpt == "step done\n"
    assert result.error is None
    assert len(calls) == 1
    with builder._RESERVED_TAGS_LOCK:
        assert image_tag not in builder._RESERVED_TAGS
    command, options = calls[0]
    assert command[1] == "--config"
    config_path = Path(command[2])
    assert isolated_configs[0][1] == "{}"
    assert command[3:5] == ["--host", "npipe:////./pipe/docker_engine"]
    assert command[command.index("build") + 1] == "--iidfile"
    assert not any(argument == "--progress" or argument.startswith("--progress=") for argument in command)
    assert options["shell"] is False
    assert options["stderr"] == subprocess.STDOUT
    assert options["stdin"] == subprocess.DEVNULL
    assert "--file" in command
    assert "--iidfile" in command
    assert "--tag" in command
    assert not any("run" == argument for argument in command)
    assert str(workspace) not in command
    assert options["env"].get("AWS_SECRET_ACCESS_KEY") is None
    assert str(original_config) not in command
    assert options["env"]["DOCKER_CONFIG"] != str(original_config)
    assert "auths" not in isolated_configs[0][1]
    assert not config_path.exists()
    assert options["env"]["DOCKER_CONFIG"] == str(config_path)
    assert "DOCKER_CONTEXT" not in options["env"]
    assert "DOCKER_HOST" not in options["env"]
    assert not list(staging_root.iterdir())

    repeated = builder.build_docker_image(workspace, ".", "Dockerfile", image_tag, _detection())
    assert repeated.success is False
    assert repeated.error.code == "image_tag_exists"
    assert len(calls) == 1


def test_nested_dockerfile_argument_points_into_staged_context(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "Dockerfile").unlink()
    nested_dockerfile = workspace / "services" / "api" / "Dockerfile"
    nested_dockerfile.parent.mkdir(parents=True)
    nested_dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    _configure_staging(monkeypatch, tmp_path)
    calls = _mock_docker(monkeypatch)

    result = builder.build_docker_image(
        workspace,
        ".",
        "services/api/Dockerfile",
        _tag(),
        _detection(detected_files=["services/api/Dockerfile"]),
    )

    assert result.success is True
    command, options = calls[0]
    staged_context = Path(command[-1])
    assert options["shell"] is False
    assert Path(command[command.index("--file") + 1]).resolve() == (
        staged_context / "services" / "api" / "Dockerfile"
    ).resolve()


def test_generated_tag_is_unique_and_in_the_reserved_namespace():
    first = builder.create_unique_image_tag()
    second = builder.create_unique_image_tag()

    assert first.startswith("deploysphere/build:")
    assert first != second


def test_docker_failure_returns_safe_error_and_cleans_staging(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    staging_root = _configure_staging(monkeypatch, tmp_path)
    calls = _mock_docker(monkeypatch, returncode=1, output=b"secret-token C:/host/private\n")
    tag = _tag()

    result = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())

    assert result.success is False
    assert result.error.code == "build_failed"
    with builder._RESERVED_TAGS_LOCK:
        assert result.image_tag not in builder._RESERVED_TAGS
    assert "secret-token" not in result.error.message
    assert "C:/host/private" not in result.error.message
    assert not list(staging_root.iterdir())

    repeated = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())
    assert repeated.error.code == "build_failed"
    assert len(calls) == 2


def test_prelaunch_failure_is_known_and_releases_tag(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    _configure_staging(monkeypatch, tmp_path)
    _mock_docker(monkeypatch)
    calls = []

    def fail_then_start(command, **kwargs):
        if not calls:
            calls.append("failed")
            raise OSError("private launch detail")
        calls.append(command)
        return FakeProcess(command)

    monkeypatch.setattr(builder.subprocess, "Popen", fail_then_start)
    tag = _tag()

    first = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())
    second = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())

    assert first.success is False
    assert first.outcome_known is True
    assert first.error.code == "docker_unavailable"
    assert second.success is True
    assert len(calls) == 2


def test_unexpected_wait_error_quarantines_tag_but_allows_another_tag(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path / "workspace")
    _configure_staging(monkeypatch, tmp_path)
    processes = []
    calls = _mock_docker(
        monkeypatch,
        processes=processes,
        wait_errors=[RuntimeError("private process detail")],
    )
    tag = _tag()

    result = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())

    assert result.success is False
    assert result.outcome_known is False
    assert result.error.code == "build_outcome_indeterminate"
    assert "private process detail" not in result.error.message
    assert processes[0].terminated is True

    repeated = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())
    assert repeated.success is False
    assert repeated.error.code == "image_tag_in_use"
    assert len(calls) == 1

    different_tag_result = builder.build_docker_image(
        workspace, ".", "Dockerfile", _tag(), _detection()
    )
    assert different_tag_result.success is True
    assert len(calls) == 2


def test_unconfirmed_cli_exit_defers_staging_cleanup(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    staging_root = _configure_staging(monkeypatch, tmp_path)
    _mock_docker(monkeypatch)
    release_process = Event()
    cleanup_finished = Event()
    original_rmtree = builder.shutil.rmtree

    class UnconfirmedProcess(FakeProcess):
        def __init__(self, command):
            super().__init__(command)
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise RuntimeError("unexpected initial wait failure")
            if self.wait_calls <= 3:
                raise subprocess.TimeoutExpired(self.command, timeout)
            assert release_process.wait(timeout=5)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminated = True
            self.returncode = -9

    def track_cleanup(path, *args, **kwargs):
        result = original_rmtree(path, *args, **kwargs)
        if Path(path).parent == staging_root:
            cleanup_finished.set()
        return result

    monkeypatch.setattr(builder.shutil, "rmtree", track_cleanup)
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda command, **kwargs: UnconfirmedProcess(command),
    )

    result = builder.build_docker_image(workspace, ".", "Dockerfile", _tag(), _detection())

    assert result.success is False
    assert result.outcome_known is False
    assert len(list(staging_root.iterdir())) == 1
    release_process.set()
    assert cleanup_finished.wait(timeout=5)
    assert not list(staging_root.iterdir())


def test_build_timeout_terminates_only_cli_and_cleans_staging(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    staging_root = _configure_staging(monkeypatch, tmp_path)
    processes = []
    calls = _mock_docker(monkeypatch, timeout=True, processes=processes)
    timed_out_tag = _tag()

    result = builder.build_docker_image(workspace, ".", "Dockerfile", timed_out_tag, _detection())

    assert result.success is False
    assert result.error.code == "build_timeout"
    assert result.outcome_known is False
    assert "indeterminate" in result.error.message
    assert processes[0].terminated is True
    assert not list(staging_root.iterdir())
    assert len(calls) == 1

    repeated = builder.build_docker_image(
        workspace, ".", "Dockerfile", timed_out_tag, _detection()
    )
    assert repeated.success is False
    assert repeated.error.code == "image_tag_in_use"
    assert len(calls) == 1

    different_calls = []

    def successful_popen(command, **kwargs):
        different_calls.append(command)
        return FakeProcess(command)

    monkeypatch.setattr(builder.subprocess, "Popen", successful_popen)
    different_tag_result = builder.build_docker_image(
        workspace, ".", "Dockerfile", _tag(), _detection()
    )
    assert different_tag_result.success is True
    assert len(different_calls) == 1


def test_existing_image_tag_is_rejected_without_starting_build(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    _configure_staging(monkeypatch, tmp_path)
    calls = _mock_docker(monkeypatch, existing_image=True)

    result = builder.build_docker_image(workspace, ".", "Dockerfile", _tag(), _detection())

    assert result.success is False
    assert result.error.code == "image_tag_exists"
    assert not any(command[-1] == "build" for command, _ in calls if len(command) > 2)


def test_same_tag_cannot_be_used_by_two_active_builds(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    tag = _tag()
    with builder._RESERVED_TAGS_LOCK:
        builder._RESERVED_TAGS.add(tag)

    result = builder.build_docker_image(workspace, ".", "Dockerfile", tag, _detection())

    assert result.error.code == "image_tag_in_use"
    with builder._RESERVED_TAGS_LOCK:
        builder._RESERVED_TAGS.discard(tag)


def test_missing_docker_executable_returns_safe_failure(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: None)
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start when missing"),
    )

    result = builder.build_docker_image(workspace, ".", "Dockerfile", _tag(), _detection())

    assert result.success is False
    assert result.error.code == "docker_unavailable"


@pytest.mark.parametrize(
    "build_context,dockerfile_path",
    [
        ("../outside", "Dockerfile"),
        ("C:/host/workspace", "Dockerfile"),
        (".", "../Dockerfile"),
        (".", "/etc/Dockerfile"),
        (".", "C:/host/Dockerfile"),
        (".", "Dockerfile\\..\\Dockerfile"),
    ],
)
def test_unsafe_relative_paths_are_rejected_before_docker(
    tmp_path, monkeypatch, build_context, dockerfile_path
):
    workspace = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker")
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start for an unsafe path"),
    )

    result = builder.build_docker_image(
        workspace, build_context, dockerfile_path, _tag(), _detection()
    )

    assert result.success is False
    assert result.error.code == "unsafe_path"


def test_workspace_root_symlink_is_rejected(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    link = tmp_path / "workspace-link"
    try:
        link.symlink_to(workspace, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This Windows environment does not allow creating directory symlinks.")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker")
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start for a symlink workspace"),
    )

    result = builder.build_docker_image(link, ".", "Dockerfile", _tag(), _detection())

    assert result.success is False
    assert result.error.code == "unsafe_workspace"


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_symlinks_inside_build_context_are_rejected(tmp_path, monkeypatch, entry_kind):
    workspace = _workspace(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("private", encoding="utf-8")
    link = workspace / ("linked-dir" if entry_kind == "directory" else "linked-file")
    try:
        link.symlink_to(outside if entry_kind == "directory" else target,
                        target_is_directory=entry_kind == "directory")
    except (OSError, NotImplementedError):
        pytest.skip("This Windows environment does not allow creating symlinks.")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker")
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start for a symlink context"),
    )

    result = builder.build_docker_image(workspace, ".", "Dockerfile", _tag(), _detection())

    assert result.success is False
    assert result.error.code == "symlink_not_allowed"


def test_oversized_context_and_individual_files_are_rejected(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker")
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start for an oversized context"),
    )

    context_result = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        _tag(),
        _detection(),
        limits=builder.BuildLimits(max_context_bytes=2),
    )
    file_result = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        _tag(),
        _detection(),
        limits=builder.BuildLimits(max_file_size_bytes=2),
    )

    assert context_result.error.code == "context_too_large"
    assert file_result.error.code == "file_too_large"


def test_file_count_limit_is_enforced(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "second.txt").write_text("extra", encoding="utf-8")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker")
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start for too many files"),
    )

    result = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        _tag(),
        _detection(),
        limits=builder.BuildLimits(max_files=2),
    )

    assert result.success is False
    assert result.error.code == "too_many_files"


def test_output_capture_is_bounded_and_does_not_return_raw_output(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    output = (b"sensitive-build-output\n" * 1000)
    staging_root = _configure_staging(monkeypatch, tmp_path)
    _mock_docker(monkeypatch, output=output)

    result = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        _tag(),
        _detection(),
        limits=builder.BuildLimits(max_output_bytes=128, max_output_lines=3),
    )

    assert result.success is True
    assert result.log_bytes == 128
    assert result.log_line_count == 3
    assert result.log_truncated is True
    assert result.log_excerpt.endswith("sensitive-build-output\n")
    assert len(result.log_excerpt.encode("utf-8")) <= builder.MAX_BUILD_LOG_EXCERPT_BYTES
    assert not list(staging_root.iterdir())


def test_failed_build_captures_output_and_redacts_secrets(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    _configure_staging(monkeypatch, tmp_path)
    _mock_docker(
        monkeypatch,
        returncode=1,
        output=(
            b"failed to resolve image\n"
            b"PASSWORD=private-value\n"
            b"Authorization: Bearer abcdef123456\n"
            b"https://user:private-password@example.test/repo\n"
            b"PATH=/private/config\n"
        ),
    )

    result = builder.build_docker_image(
        workspace, ".", "Dockerfile", _tag(), _detection()
    )

    assert result.success is False
    assert result.error.code == "build_failed"
    assert "failed to resolve image" in result.log_excerpt
    assert "private-value" not in result.log_excerpt
    assert "abcdef123456" not in result.log_excerpt
    assert "private-password" not in result.log_excerpt
    assert "/private/config" not in result.log_excerpt
    assert "[REDACTED]" in result.log_excerpt


def test_output_excerpt_keeps_the_final_bounded_output(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    _configure_staging(monkeypatch, tmp_path)
    _mock_docker(
        monkeypatch,
        returncode=1,
        output=b"old output\n" + b"x" * 20_000 + b"\nfinal build error\n",
    )

    result = builder.build_docker_image(
        workspace, ".", "Dockerfile", _tag(), _detection()
    )

    assert result.log_truncated is True
    assert result.log_excerpt.endswith("final build error\n")
    assert "old output" not in result.log_excerpt
    assert len(result.log_excerpt.encode("utf-8")) <= builder.MAX_BUILD_LOG_EXCERPT_BYTES


def test_empty_build_output_has_an_empty_excerpt(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    _configure_staging(monkeypatch, tmp_path)
    _mock_docker(monkeypatch, output=b"")

    result = builder.build_docker_image(
        workspace, ".", "Dockerfile", _tag(), _detection()
    )

    assert result.success is True
    assert result.log_bytes == 0
    assert result.log_line_count == 0
    assert result.log_excerpt == ""


def test_unsupported_or_ambiguous_detection_is_rejected(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    monkeypatch.setattr(builder.shutil, "which", lambda executable: "docker")
    monkeypatch.setattr(
        builder.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Docker must not start for unsupported detection"),
    )

    mixed = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        _tag(),
        _detection(
            application_type="mixed",
            detected_application_types=["python", "nodejs"],
        ),
    )
    compose = builder.build_docker_image(
        workspace,
        ".",
        "Dockerfile",
        _tag(),
        _detection(build_method="mixed"),
    )

    assert mixed.error.code == "unsupported_application"
    assert compose.error.code == "unsupported_build_method"


def test_staging_failure_does_not_modify_workspace(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path / "workspace")
    original_files = sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*"))
    _configure_staging(monkeypatch, tmp_path)
    _mock_docker(monkeypatch, returncode=1)

    builder.build_docker_image(workspace, ".", "Dockerfile", _tag(), _detection())

    current_files = sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*"))
    assert current_files == original_files
    assert (workspace / "Dockerfile").read_text(encoding="utf-8").startswith("FROM scratch")
