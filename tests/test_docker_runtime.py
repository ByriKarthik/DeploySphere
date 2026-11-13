"""Mocked tests for the Docker runtime helper; no Docker daemon is used."""

from io import BytesIO
import builtins
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread
import time

import pytest
from bson import ObjectId

from backend.services import docker_runtime as runtime


PROJECT_ID = "a" * 24
DEPLOYMENT_ID = "b" * 24
CONTAINER_ID = "c" * 64
IMAGE_TAG = "deploysphere/build:" + "d" * 32


class FakeProcess:
    """Controlled process object that never starts Docker."""

    def __init__(self, command, stdout=b"", stderr=b"", returncode=0, timeout=False):
        self.command = command
        self.stdout = BytesIO(stdout)
        self.stderr = BytesIO(stderr)
        self.returncode = returncode
        self.timeout = timeout
        self.terminated = False

    def wait(self, timeout=None):
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


class DelayedPipe:
    def __init__(self, release, output=b""):
        self.release = release
        self.output = output
        self.read_started = Event()
        self.closed_event = Event()
        self._read_output = False

    def read(self, _size):
        self.read_started.set()
        if not self.release.wait(10):
            raise OSError("test pipe was not released")
        if self._read_output:
            return b""
        self._read_output = True
        return self.output

    def close(self):
        self.closed_event.set()


class PermanentPipe:
    def __init__(self, release, output=b""):
        self.release = release
        self.output = output
        self.read_started = Event()
        self.closed_event = Event()
        self._read_output = False

    def read(self, _size):
        self.read_started.set()
        self.release.wait()
        if self._read_output:
            return b""
        self._read_output = True
        return self.output

    def close(self):
        self.closed_event.set()


class ReadErrorPipe:
    def read(self, _size):
        raise OSError("mocked pipe read failure")

    def close(self):
        pass


class TrackedBytesIO(BytesIO):
    def __init__(self, value=b""):
        super().__init__(value)
        self.closed_event = Event()

    def close(self):
        super().close()
        self.closed_event.set()


class UnconfirmedProcess(FakeProcess):
    def __init__(self, command):
        super().__init__(command)
        self.exit_event = Event()
        self.killed = False

    def wait(self, timeout=None):
        if not self.exit_event.wait(timeout):
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def poll(self):
        return self.returncode if self.exit_event.is_set() else None


class InterruptibleProcess(FakeProcess):
    def __init__(self, command, *, confirm_termination):
        super().__init__(command, returncode=None)
        self.exit_event = Event()
        self.confirm_termination = confirm_termination
        self.interrupt_wait = True
        self.killed = False

    def wait(self, timeout=None):
        if self.interrupt_wait:
            self.interrupt_wait = False
            raise KeyboardInterrupt()
        if not self.exit_event.wait(timeout):
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.poll()

    def terminate(self):
        self.terminated = True
        if self.confirm_termination:
            self.exit_event.set()

    def kill(self):
        self.killed = True

    def poll(self):
        if not self.exit_event.is_set():
            return None
        self.returncode = -15 if self.terminated else -9
        return self.returncode


def _session(tmp_path, temporary_directory=None):
    return runtime._DockerSession(
        executable="mock-docker",
        endpoint="mock-endpoint",
        config_directory=str(tmp_path),
        environment={},
        temporary_directory=temporary_directory,
    )


def _track_captures(monkeypatch):
    capture_type = runtime._Capture
    captures = []

    def create_capture(data):
        capture = capture_type(data)
        captures.append(capture)
        return capture

    monkeypatch.setattr(runtime, "_Capture", create_capture)
    return captures


def _wait_for_reserved_count(manager, count, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.reserved_count == count:
            return True
        Event().wait(0.01)
    return manager.reserved_count == count


def test_read_pipe_captures_output_below_limit():
    output = b"docker output"
    pipe = BytesIO(output)
    capture = runtime._Capture(bytearray())

    runtime._read_pipe(pipe, capture)

    assert capture.data == output
    assert capture.total_bytes == len(output)
    assert not capture.truncated


def test_read_pipe_bounds_output_and_marks_truncated():
    output = b"x" * (runtime.MAX_CAPTURE_BYTES + 1)
    capture = runtime._Capture(bytearray())

    runtime._read_pipe(BytesIO(output), capture)

    assert len(capture.data) == runtime.MAX_CAPTURE_BYTES
    assert capture.truncated
    assert capture.total_bytes == len(output)


def test_read_pipe_distinguishes_read_error_from_capture_truncation():
    capture = runtime._Capture(bytearray())

    runtime._read_pipe(ReadErrorPipe(), capture)

    assert capture.read_error
    assert not capture.truncated


def test_run_command_reports_pipe_read_errors(monkeypatch, tmp_path):
    process = FakeProcess(["mock-docker"])
    process.stdout = ReadErrorPipe()
    process.stderr = BytesIO()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)

    result = runtime._run_command(_session(tmp_path), ["version"], timeout_seconds=1)

    assert result.outcome == "indeterminate"
    assert result.error.code == "docker_output_read_failed"


def test_read_pipe_continues_draining_after_capture_limit():
    output = b"x" * (runtime.MAX_CAPTURE_BYTES + runtime.READ_CHUNK_BYTES * 3)
    pipe = BytesIO(output)
    capture = runtime._Capture(bytearray())

    runtime._read_pipe(pipe, capture)

    assert pipe.tell() == len(output)
    assert capture.truncated
    assert len(capture.data) == runtime.MAX_CAPTURE_BYTES


def test_run_command_completes_stdout_and_stderr(monkeypatch, tmp_path):
    process = FakeProcess(
        ["mock-docker"],
        stdout=b"stdout complete",
        stderr=b"stderr complete",
    )
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    captures = _track_captures(monkeypatch)

    result = runtime._run_command(_session(tmp_path), ["version"], timeout_seconds=1)

    assert result.outcome == "succeeded"
    assert result.stdout == b"stdout complete"
    assert [bytes(capture.data) for capture in captures] == [
        b"stdout complete",
        b"stderr complete",
    ]
    assert all(not capture.read_error for capture in captures)
    assert process.stdout.closed and process.stderr.closed


def test_run_command_drains_large_subprocess_output_before_success(monkeypatch, tmp_path):
    original_popen = subprocess.Popen
    output_size = runtime.MAX_CAPTURE_BYTES * 16
    script = (
        "import sys\n"
        f"sys.stdout.buffer.write(b'o' * {output_size})\n"
        "sys.stdout.flush()\n"
        f"sys.stderr.buffer.write(b'e' * {output_size})\n"
        "sys.stderr.flush()\n"
    )

    def python_subprocess(_command, **kwargs):
        return original_popen(
            [sys.executable, "-c", script],
            **kwargs,
        )

    monkeypatch.setattr(runtime.subprocess, "Popen", python_subprocess)
    captures = _track_captures(monkeypatch)

    result = runtime._run_command(_session(tmp_path), ["version"], timeout_seconds=10)

    assert result.outcome == "succeeded"
    assert result.stdout == b"o" * runtime.MAX_CAPTURE_BYTES
    assert [capture.total_bytes for capture in captures] == [output_size, output_size]
    assert all(capture.truncated for capture in captures)
    assert all(not capture.read_error for capture in captures)


def test_run_command_defers_reader_that_exceeds_completion_deadline(
    monkeypatch, tmp_path
):
    release = Event()
    stdout = DelayedPipe(release, b"late stdout")
    process = FakeProcess(["mock-docker"])
    process.stdout = stdout
    process.stderr = BytesIO(b"complete stderr")
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    temporary_directory = runtime.tempfile.TemporaryDirectory(dir=tmp_path)
    cleanup_complete = Event()
    original_cleanup = temporary_directory.cleanup

    def tracked_cleanup():
        original_cleanup()
        cleanup_complete.set()

    temporary_directory.cleanup = tracked_cleanup
    session = _session(tmp_path, temporary_directory)

    result = runtime._run_command(session, ["version"], timeout_seconds=1)

    assert stdout.read_started.is_set()
    assert result.outcome == "indeterminate"
    assert result.error.code == "docker_output_incomplete"
    assert session.cleanup_deferred
    assert not stdout.closed_event.is_set()

    release.set()
    assert stdout.closed_event.wait(2)
    assert cleanup_complete.wait(2)
    assert not Path(temporary_directory.name).exists()


def test_discovery_defers_reader_for_inherited_pipe_handle(monkeypatch, tmp_path):
    release = Event()
    stdout = DelayedPipe(release, b"late context")
    process = FakeProcess(["mock-docker"])
    process.stdout = stdout
    process.stderr = BytesIO()
    monkeypatch.setattr(
        runtime,
        "PIPE_READER_COMPLETION_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(runtime._SafeRuntimeFailure, match="fully read"):
        runtime._run_discovery("mock-docker", ["mock-docker", "context"], {})

    assert not stdout.closed_event.is_set()
    release.set()
    assert stdout.closed_event.wait(2)


def test_permanent_inherited_pipe_keeps_bounded_cleanup_ownership(
    monkeypatch, tmp_path
):
    manager = runtime._DEFERRED_PIPE_CLEANUP_MANAGER
    assert _wait_for_reserved_count(manager, 0)
    monkeypatch.setattr(
        runtime,
        "PIPE_READER_COMPLETION_TIMEOUT_SECONDS",
        0.01,
    )

    release_primary = Event()
    stdout = PermanentPipe(release_primary, b"output held by inherited handle")
    process = FakeProcess(["mock-docker"])
    process.stdout = stdout
    process.stderr = BytesIO()
    popen_calls = []

    def fake_popen(*_args, **_kwargs):
        popen_calls.append(True)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    temporary_directory = runtime.tempfile.TemporaryDirectory(dir=tmp_path)
    config_directory = Path(temporary_directory.name)
    cleanup_complete = Event()
    original_cleanup = temporary_directory.cleanup

    def tracked_cleanup():
        original_cleanup()
        cleanup_complete.set()

    temporary_directory.cleanup = tracked_cleanup
    session = _session(tmp_path, temporary_directory)
    release_secondary = Event()
    secondary_pipe = PermanentPipe(release_secondary, b"secondary output")
    secondary_process = FakeProcess(["mock-docker-secondary"])
    secondary_reader = Thread(
        target=runtime._read_pipe,
        args=(secondary_pipe, runtime._Capture(bytearray())),
        daemon=True,
    )
    secondary_reservation = None
    filler_reservations = []

    try:
        result = runtime._run_command(session, ["version"], timeout_seconds=1)

        assert stdout.read_started.is_set()
        assert result.outcome == "indeterminate"
        assert result.error.code == "docker_output_incomplete"
        assert manager.reserved_count == 1
        assert not stdout.closed_event.is_set()
        assert cleanup_complete.wait(2)
        assert process.poll() is not None
        assert not config_directory.exists()
        worker = manager._worker
        assert worker is not None and worker.is_alive()

        secondary_reservation = manager.reserve()
        assert secondary_reservation is not None
        secondary_reader.start()
        assert secondary_pipe.read_started.wait(1)
        manager.defer(
            secondary_reservation,
            secondary_process,
            [secondary_reader],
            [secondary_pipe],
        )
        filler_reservations = [
            manager.reserve()
            for _ in range(manager.maximum - 2)
        ]
        assert all(reservation is not None for reservation in filler_reservations)
        assert manager.reserved_count == manager.maximum

        rejected = runtime._run_command(
            _session(tmp_path),
            ["version"],
            timeout_seconds=1,
        )

        assert rejected.outcome == "failed"
        assert rejected.error.code == "docker_cleanup_capacity"
        assert len(popen_calls) == 1
        assert manager._worker is worker
        assert not stdout.closed_event.is_set()

        release_secondary.set()
        assert secondary_pipe.closed_event.wait(2)
        assert _wait_for_reserved_count(manager, manager.maximum - 1)
        assert not stdout.closed_event.is_set()
    finally:
        release_secondary.set()
        release_primary.set()
        if secondary_reservation is not None:
            manager.release(secondary_reservation)
        for reservation in filler_reservations:
            if reservation is not None:
                manager.release(reservation)

    assert stdout.closed_event.wait(2)
    assert _wait_for_reserved_count(manager, 0)


def test_cleanup_worker_retries_callback_failure(caplog):
    manager = runtime._DeferredPipeCleanupManager(1)
    reservation = manager.reserve()
    assert reservation is not None
    process = FakeProcess(["mock-docker"])
    pipe = TrackedBytesIO()
    attempts = []
    callback_complete = Event()

    def cleanup_callback():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("temporary cleanup failure")
        callback_complete.set()

    manager.defer(
        reservation,
        process,
        [],
        [pipe],
        on_process_exit=cleanup_callback,
    )

    assert callback_complete.wait(3)
    assert pipe.closed_event.wait(1)
    assert manager._worker is not None and manager._worker.is_alive()
    assert _wait_for_reserved_count(manager, 0)
    assert len(attempts) >= 2
    assert "retaining its reservation" in caplog.text


def test_cleanup_reservations_are_thread_safe_and_bounded():
    manager = runtime._DeferredPipeCleanupManager(
        runtime.MAX_RESERVED_PIPE_CLEANUPS
    )
    worker_count = manager.maximum * 3
    barrier = Barrier(worker_count)

    def reserve_simultaneously():
        barrier.wait(timeout=3)
        return manager.reserve()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        reservations = list(
            executor.map(lambda _: reserve_simultaneously(), range(worker_count))
        )

    successful = [item for item in reservations if item is not None]
    failed = [item for item in reservations if item is None]
    assert len(successful) <= runtime.MAX_RESERVED_PIPE_CLEANUPS
    assert len(failed) == worker_count - len(successful)
    assert manager.reserved_count == len(successful)

    with ThreadPoolExecutor(max_workers=len(successful)) as executor:
        list(executor.map(lambda item: manager.release(item), successful))
    assert manager.reserved_count == 0
    for reservation in successful:
        manager.release(reservation)
    assert manager.reserved_count == 0


@pytest.mark.parametrize("operation", ["command", "discovery"])
@pytest.mark.parametrize("termination_confirmed", [True, False])
def test_keyboard_interrupt_terminates_or_defers_live_process(
    monkeypatch, tmp_path, operation, termination_confirmed
):
    manager = runtime._DEFERRED_PIPE_CLEANUP_MANAGER
    assert _wait_for_reserved_count(manager, 0)
    monkeypatch.setattr(
        runtime,
        "PIPE_READER_COMPLETION_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(runtime, "TERMINATE_GRACE_SECONDS", 0.01)
    release = Event()
    stdout = PermanentPipe(release, b"interrupted output")
    process = InterruptibleProcess(
        ["mock-docker"], confirm_termination=termination_confirmed
    )
    process.stdout = stdout
    process.stderr = BytesIO()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)

    temporary_directory = None
    cleanup_complete = Event()
    if operation == "command":
        temporary_directory = runtime.tempfile.TemporaryDirectory(dir=tmp_path)
        original_cleanup = temporary_directory.cleanup

        def tracked_cleanup():
            original_cleanup()
            cleanup_complete.set()

        temporary_directory.cleanup = tracked_cleanup
        session = _session(tmp_path, temporary_directory)

    try:
        with pytest.raises(KeyboardInterrupt):
            if operation == "command":
                runtime._run_command(session, ["version"], timeout_seconds=1)
            else:
                runtime._run_discovery(
                    "mock-docker", ["mock-docker", "context"], {}
                )

        assert stdout.read_started.is_set()
        assert process.terminated
        if termination_confirmed:
            assert process.poll() is not None
        else:
            assert process.poll() is None
        assert process.killed is (not termination_confirmed)
        assert manager.reserved_count == 1
        assert not stdout.closed_event.is_set()
        if operation == "command":
            assert session.cleanup_deferred
            if termination_confirmed:
                assert cleanup_complete.wait(2)
                assert not Path(temporary_directory.name).exists()
            else:
                assert Path(temporary_directory.name).exists()
                process.exit_event.set()
                assert cleanup_complete.wait(2)
                assert not Path(temporary_directory.name).exists()
        elif not termination_confirmed:
            process.exit_event.set()
    finally:
        release.set()

    assert stdout.closed_event.wait(2)
    assert _wait_for_reserved_count(manager, 0)


@pytest.mark.parametrize("operation", ["command", "discovery"])
def test_keyboard_interrupt_before_process_creation_releases_reservation(
    monkeypatch, tmp_path, operation
):
    manager = runtime._DEFERRED_PIPE_CLEANUP_MANAGER
    assert _wait_for_reserved_count(manager, 0)

    def interrupt_popen(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runtime.subprocess, "Popen", interrupt_popen)

    with pytest.raises(KeyboardInterrupt):
        if operation == "command":
            runtime._run_command(_session(tmp_path), ["version"], timeout_seconds=1)
        else:
            runtime._run_discovery(
                "mock-docker", ["mock-docker", "context"], {}
            )

    assert manager.reserved_count == 0


def test_discovery_completes_both_output_streams(monkeypatch):
    process = FakeProcess(
        ["mock-docker"],
        stdout=b"context output",
        stderr=b"context diagnostics",
    )
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    captures = _track_captures(monkeypatch)

    output = runtime._run_discovery("mock-docker", ["mock-docker", "context"], {})

    assert output == b"context output"
    assert [bytes(capture.data) for capture in captures] == [
        b"context output",
        b"context diagnostics",
    ]
    assert all(not capture.read_error for capture in captures)
    assert process.stdout.closed and process.stderr.closed


def test_discovery_reports_pipe_read_errors(monkeypatch):
    process = FakeProcess(["mock-docker"])
    process.stdout = ReadErrorPipe()
    process.stderr = BytesIO()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(runtime._SafeRuntimeFailure, match="output could not be read"):
        runtime._run_discovery("mock-docker", ["mock-docker", "context"], {})


def test_command_timeout_with_confirmed_termination_closes_pipes(
    monkeypatch, tmp_path
):
    process = FakeProcess(
        ["mock-docker"],
        stdout=b"partial",
        stderr=b"diagnostics",
        timeout=True,
    )
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    session = _session(tmp_path)

    result = runtime._run_command(session, ["version"], timeout_seconds=1)

    assert result.outcome == "indeterminate"
    assert process.terminated
    assert process.stdout.closed and process.stderr.closed
    assert not session.cleanup_deferred


def test_command_timeout_with_unconfirmed_termination_defers_pipe_cleanup(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runtime, "TERMINATE_GRACE_SECONDS", 0.01)
    temporary_directory = runtime.tempfile.TemporaryDirectory(dir=tmp_path)
    cleanup_complete = Event()
    original_cleanup = temporary_directory.cleanup

    def tracked_cleanup():
        original_cleanup()
        cleanup_complete.set()

    temporary_directory.cleanup = tracked_cleanup
    process = UnconfirmedProcess(["mock-docker"])
    process.stdout = TrackedBytesIO()
    process.stderr = TrackedBytesIO()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)
    session = _session(tmp_path, temporary_directory)

    result = runtime._run_command(session, ["version"], timeout_seconds=0.01)

    assert result.outcome == "indeterminate"
    assert process.terminated and process.killed
    assert session.cleanup_deferred
    assert not process.stdout.closed and not process.stderr.closed
    assert Path(temporary_directory.name).exists()

    process.exit_event.set()
    assert process.stdout.closed_event.wait(2)
    assert process.stderr.closed_event.is_set()
    assert cleanup_complete.wait(2)
    assert not Path(temporary_directory.name).exists()


def _inspect_output(
    *,
    container_id=CONTAINER_ID,
    container_name=f"/deploysphere-{DEPLOYMENT_ID}",
    managed="true",
    project_id=PROJECT_ID,
    deployment_id=DEPLOYMENT_ID,
    state="created",
    container_port=8080,
    host_ip="127.0.0.1",
    host_port="49152",
    ports=None,
    port_bindings=None,
):
    if ports is None:
        ports = (
            {f"{container_port}/tcp": [{"HostIp": host_ip, "HostPort": host_port}]}
            if host_port is not None
            else {f"{container_port}/tcp": None}
        )
    if port_bindings is None:
        port_bindings = {
            f"{container_port}/tcp": [{"HostIp": host_ip, "HostPort": ""}]
        }
    values = [
        container_id,
        container_name,
        state,
        managed,
        project_id,
        deployment_id,
        ports,
        port_bindings,
    ]
    return "\n".join(json.dumps(value) for value in values).encode("utf-8")


def _mock_docker(
    monkeypatch,
    *,
    failure=None,
    timeout_action=None,
    inspect_output=None,
    final_inspect_output=None,
    lifecycle=None,
    context_failure=False,
    context_output=None,
    stderr_output=b"secret host path /private/host and registry token",
):
    """Mock context discovery and each Docker CLI invocation."""
    calls = []
    processes = []
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "docker-test")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_TLS_VERIFY", raising=False)
    monkeypatch.delenv("DOCKER_CERT_PATH", raising=False)

    inspect_count = 0
    initial_state = {
        "start": "created",
        "stop": "running",
        "restart": "running",
    }.get(lifecycle, "created")
    simulated_state = initial_state

    def fake_popen(command, **kwargs):
        nonlocal inspect_count, simulated_state
        calls.append((command, kwargs))
        if command[1:] == ["context", "show"]:
            return FakeProcess(
                command,
                stdout=b"desktop-linux\n",
                returncode=1 if context_failure else 0,
            )
        if "context" in command and "inspect" in command:
            output = context_output or b'{"Host":"npipe:////./pipe/docker_engine","SkipTLSVerify":false}'
            return FakeProcess(command, stdout=output, returncode=1 if context_failure else 0)
        config_directory = command[command.index("--config") + 1]
        kwargs["config_contents"] = Path(config_directory, "config.json").read_text(encoding="utf-8")
        verb = next(
            item for item in ("create", "inspect", "start", "stop", "restart")
            if item in command
        )
        if verb == "create":
            text = CONTAINER_ID.encode("ascii") + b"\n"
        elif verb == "inspect":
            inspect_count += 1
            if inspect_count > 1 and final_inspect_output is not None:
                text = final_inspect_output
            elif inspect_output is not None:
                text = inspect_output
            else:
                text = _inspect_output(state=simulated_state)
        elif verb in ("start", "restart"):
            simulated_state = "running"
            text = b""
        elif verb == "stop":
            simulated_state = "exited"
            text = b""
        else:
            text = b""

        is_final_inspect = (verb == "inspect" and inspect_count > 1)
        if failure == "final_inspect":
            returncode = 1 if is_final_inspect else 0
        elif failure == "inspect":
            returncode = 1 if (verb == "inspect" and inspect_count == 1) else 0
        else:
            returncode = 1 if failure == verb else 0

        if timeout_action == "final_inspect":
            timed_out = is_final_inspect
        elif timeout_action == "inspect":
            timed_out = (verb == "inspect" and inspect_count == 1)
        else:
            timed_out = (timeout_action == verb)

        process = FakeProcess(
            command,
            stdout=text,
            stderr=stderr_output,
            returncode=returncode,
            timeout=timed_out,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    return calls, processes


def _docker_calls(calls):
    return [call for call in calls if call[0][0] == "docker-test"]


def _runtime_calls(calls):
    return [call for call in _docker_calls(calls) if "--config" in call[0]]


def _verb(command):
    return command[command.index("--host") + 2]


def test_create_uses_deterministic_name_labels_security_limits_and_loopback_binding(
    monkeypatch,
):
    calls, _ = _mock_docker(monkeypatch)

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "succeeded"
    assert result.container_id == CONTAINER_ID
    assert result.container_name == f"deploysphere-{DEPLOYMENT_ID}"
    assert result.container_port == 8080
    assert result.allocated_port is None
    create_call = next(command for command, _ in _docker_calls(calls) if "create" in command)
    assert create_call[create_call.index("--name") + 1] == f"deploysphere-{DEPLOYMENT_ID}"
    assert "127.0.0.1::8080/tcp" in create_call
    assert "--memory" in create_call and create_call[create_call.index("--memory") + 1] == "512m"
    assert "--cpus" in create_call and create_call[create_call.index("--cpus") + 1] == "1.0"
    assert "--pids-limit" in create_call and create_call[create_call.index("--pids-limit") + 1] == "256"
    assert "--cap-drop" in create_call and create_call[create_call.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in create_call and create_call[create_call.index("--security-opt") + 1] == "no-new-privileges"
    assert "--restart" in create_call and create_call[create_call.index("--restart") + 1] == "no"
    assert "--privileged" not in create_call
    assert not any(
        item in create_call
        for item in (
            "--mount", "-v", "--device", "--network", "--network=host",
            "--pid", "--pid=host", "--env", "--env-file",
        )
    )
    assert not any(word in create_call for word in ("rm", "prune"))
    assert f"{runtime.MANAGED_LABEL}=true" in create_call
    assert f"{runtime.PROJECT_LABEL}={PROJECT_ID}" in create_call
    assert f"{runtime.DEPLOYMENT_LABEL}={DEPLOYMENT_ID}" in create_call


def test_runtime_commands_use_shell_false_and_isolated_empty_docker_config(
    monkeypatch,
):
    calls, _ = _mock_docker(monkeypatch)

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "succeeded"
    for command, kwargs in _runtime_calls(calls):
        assert kwargs["shell"] is False
        assert "--config" in command
        config_index = command.index("--config") + 1
        config_path = Path(command[config_index]) / "config.json"
        assert config_path.parent.name == "docker-config"
        assert kwargs["config_contents"] == "{}"
        assert kwargs["env"]["DOCKER_CONFIG"] == str(config_path.parent)
        assert "DOCKER_HOST" not in kwargs["env"]
        assert "DOCKER_CONTEXT" not in kwargs["env"]
        assert "DOCKER_AUTH_CONFIG" not in kwargs["env"]
    assert all(kwargs["shell"] is False for _, kwargs in _docker_calls(calls))


@pytest.mark.parametrize("bad_tag", ["nginx:latest", "deploysphere/build:ABC", "deploysphere/build:" + "x" * 32])
def test_rejects_non_deploysphere_image_tags_before_docker(monkeypatch, bad_tag):
    calls, _ = _mock_docker(monkeypatch)

    with pytest.raises(ValueError, match="image_tag"):
        runtime.create_container(bad_tag, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert not _docker_calls(calls)


@pytest.mark.parametrize("bad_id", ["bad", "a" * 23, "g" * 24])
def test_rejects_invalid_project_or_deployment_object_ids(monkeypatch, bad_id):
    calls, _ = _mock_docker(monkeypatch)

    with pytest.raises(ValueError, match="ObjectId"):
        runtime.create_container(IMAGE_TAG, bad_id, DEPLOYMENT_ID, 8080)
    with pytest.raises(ValueError, match="ObjectId"):
        runtime.create_container(IMAGE_TAG, PROJECT_ID, bad_id, 8080)

    assert not _docker_calls(calls)


@pytest.mark.parametrize("bad_port", [0, 65536, "8080", True, 8080.0])
def test_rejects_invalid_container_ports(monkeypatch, bad_port):
    calls, _ = _mock_docker(monkeypatch)

    with pytest.raises(ValueError, match="container_port"):
        runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, bad_port)

    assert not _docker_calls(calls)


def test_create_does_not_require_allocated_host_port_before_start(monkeypatch):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(
            container_port=3000,
            ports={},
            port_bindings={"3000/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]},
        ),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 3000)

    assert result.outcome == "succeeded"
    assert result.container_port == 3000
    assert result.allocated_port is None


@pytest.mark.parametrize("host_ip", ["0.0.0.0", "::", "192.168.1.1"])
def test_create_does_not_accept_non_loopback_port_binding(monkeypatch, host_ip):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(
            port_bindings={"8080/tcp": [{"HostIp": host_ip, "HostPort": ""}]}
        ),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_create_uses_supplied_container_port_in_dynamic_tcp_mapping(monkeypatch):
    calls, _ = _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(container_port=3000),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 3000)

    create_call = next(command for command, _ in _docker_calls(calls) if "create" in command)
    assert "127.0.0.1::3000/tcp" in create_call
    assert result.container_port == 3000


def test_create_reports_confirmed_created_state_when_port_is_assigned_on_start(
    monkeypatch,
):
    _mock_docker(monkeypatch, inspect_output=_inspect_output(host_port=None))

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "succeeded"
    assert result.container_id == CONTAINER_ID
    assert result.container_port == 8080
    assert result.allocated_port is None
    assert result.state == "created"


def test_create_requires_requested_tcp_port_key_even_if_another_port_exists(monkeypatch):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(
            container_port=3000,
            port_bindings={"3000/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]},
        ),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_create_rejects_missing_port_bindings_key(monkeypatch):
    _mock_docker(monkeypatch, inspect_output=_inspect_output(port_bindings={}))

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_create_rejects_multiple_port_mappings(monkeypatch):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(
            port_bindings={
                "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}],
                "9000/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}],
            }
        ),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


@pytest.mark.parametrize("bad_host_port", ["0", "65536", "notaport", "-1"])
def test_create_rejects_invalid_allocated_port_values(monkeypatch, bad_host_port):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(
            port_bindings={
                "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": bad_host_port}]
            }
        ),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_create_rejects_multiple_bindings_for_requested_port(monkeypatch):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(
            port_bindings={
                "8080/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": ""},
                    {"HostIp": "127.0.0.1", "HostPort": ""},
                ]
            }
        ),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_lifecycle_operations_verify_and_succeed_for_owned_container(monkeypatch, operation):
    calls, _ = _mock_docker(monkeypatch, lifecycle=operation)

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "succeeded"
    assert result.owned is True
    assert result.container_port == 8080
    assert result.allocated_port == (None if operation == "stop" else 49152)
    commands = [command for command, _ in _runtime_calls(calls)]
    assert sum(_verb(command) == "inspect" for command in commands) == 2
    assert sum(_verb(command) == operation for command in commands) == 1


def test_stop_succeeds_when_post_stop_network_ports_are_empty(monkeypatch):
    calls, _ = _mock_docker(
        monkeypatch,
        lifecycle="stop",
        final_inspect_output=_inspect_output(state="exited", ports={}),
    )

    result = runtime.stop_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "succeeded"
    assert result.owned is True
    assert result.state == "exited"
    assert result.container_port == 8080
    assert result.allocated_port is None
    assert sum(
        _verb(command) == "stop" for command, _ in _runtime_calls(calls)
    ) == 1


def test_stop_does_not_issue_command_if_pre_stop_state_is_not_running(monkeypatch):
    calls, _ = _mock_docker(
        monkeypatch,
        lifecycle="stop",
        inspect_output=_inspect_output(state="created"),
    )

    result = runtime.stop_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "failed"
    assert result.state == "created"
    assert result.error.code == "unexpected_initial_state"
    assert not any(
        _verb(command) == "stop" for command, _ in _runtime_calls(calls)
    )


def test_stop_rejects_post_stop_ownership_mismatch(monkeypatch):
    _mock_docker(
        monkeypatch,
        lifecycle="stop",
        final_inspect_output=_inspect_output(
            state="exited",
            deployment_id="e" * 24,
            ports={},
        ),
    )

    result = runtime.stop_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.owned is None
    assert result.error.code == "container_not_owned"


@pytest.mark.parametrize(
    ("operation", "wrong_final_state"),
    [
        ("start", "exited"),
        ("start", "paused"),
        ("start", "dead"),
        ("restart", "created"),
        ("restart", "paused"),
        ("restart", "dead"),
        ("stop", "running"),
        ("stop", "paused"),
        ("stop", "restarting"),
    ],
)
def test_unexpected_final_lifecycle_state_is_indeterminate(
    monkeypatch, operation, wrong_final_state
):
    _mock_docker(
        monkeypatch,
        lifecycle=operation,
        final_inspect_output=_inspect_output(state=wrong_final_state),
    )

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "indeterminate"
    assert result.state == wrong_final_state
    assert result.error.code == "unexpected_final_state"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_lifecycle_final_inspect_timeout_is_indeterminate(monkeypatch, operation):
    _mock_docker(monkeypatch, lifecycle=operation, timeout_action="final_inspect")

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "indeterminate"
    assert result.error.code == "docker_outcome_indeterminate"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_lifecycle_final_inspect_failure_is_indeterminate(monkeypatch, operation):
    _mock_docker(monkeypatch, lifecycle=operation, failure="final_inspect")

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "indeterminate"
    assert result.error.code == "docker_command_failed"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_lifecycle_final_inspect_malformed_is_indeterminate(monkeypatch, operation):
    _mock_docker(
        monkeypatch,
        lifecycle=operation,
        final_inspect_output=b"corrupted inspection\n",
    )

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_docker_response"


def test_start_requires_a_loopback_binding_and_allocated_host_port(monkeypatch):
    _mock_docker(
        monkeypatch,
        lifecycle="start",
        final_inspect_output=_inspect_output(state="running", host_ip="0.0.0.0"),
    )

    result = runtime.start_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_start_verifies_allocated_loopback_port_after_start(monkeypatch):
    calls, _ = _mock_docker(
        monkeypatch,
        lifecycle="start",
        inspect_output=_inspect_output(state="created", ports={}),
        final_inspect_output=_inspect_output(
            state="running",
            container_port=8080,
            host_ip="127.0.0.1",
            host_port="49152",
        ),
    )

    result = runtime.start_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "succeeded"
    assert result.state == "running"
    assert result.container_port == 8080
    assert result.allocated_port == 49152
    inspections = [
        command
        for command, _ in _runtime_calls(calls)
        if _verb(command) == "inspect"
    ]
    assert len(inspections) == 2


def test_start_rejects_missing_port_binding_after_start(monkeypatch):
    _mock_docker(
        monkeypatch,
        lifecycle="start",
        final_inspect_output=_inspect_output(state="running", ports={}),
    )

    result = runtime.start_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


@pytest.mark.parametrize("bad_port", ["0", "65536", "abc"])
def test_start_rejects_invalid_allocated_port_after_start(monkeypatch, bad_port):
    _mock_docker(
        monkeypatch,
        lifecycle="start",
        final_inspect_output=_inspect_output(
            state="running",
            ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": bad_port}]},
        ),
    )

    result = runtime.start_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_start_requires_allocated_port_after_docker_accepts_start(monkeypatch):
    _mock_docker(
        monkeypatch,
        lifecycle="start",
        final_inspect_output=_inspect_output(state="running", host_port=None),
    )

    result = runtime.start_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.state == "running"
    assert result.error.code == "port_binding_unavailable"


def test_restart_requires_a_loopback_binding_and_allocated_host_port(monkeypatch):
    _mock_docker(
        monkeypatch,
        lifecycle="restart",
        final_inspect_output=_inspect_output(state="running", host_ip="0.0.0.0"),
    )

    result = runtime.restart_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_port_binding"


def test_restart_requires_allocated_port_after_docker_accepts_restart(monkeypatch):
    _mock_docker(
        monkeypatch,
        lifecycle="restart",
        final_inspect_output=_inspect_output(state="running", host_port=None),
    )

    result = runtime.restart_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.state == "running"
    assert result.error.code == "port_binding_unavailable"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_lifecycle_operation_safe_against_unbound_local_error_on_pre_command_failure(
    monkeypatch, operation
):
    _mock_docker(monkeypatch, lifecycle=operation, failure="inspect")

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "failed"
    assert result.error.code == "docker_command_failed"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_lifecycle_operation_safe_against_unbound_local_error_on_unexpected_exception(
    monkeypatch, operation
):
    _mock_docker(monkeypatch, lifecycle=operation)
    inspect_call_count = 0
    orig_inspect = runtime._verified_inspection

    def fail_second_inspect(*args, **kwargs):
        nonlocal inspect_call_count
        inspect_call_count += 1
        if inspect_call_count > 1:
            raise RuntimeError("synthetic_post_accepted_error")
        return orig_inspect(*args, **kwargs)

    monkeypatch.setattr(runtime, "_verified_inspection", fail_second_inspect)

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "indeterminate"
    assert result.error.code == "docker_outcome_indeterminate"


@pytest.mark.parametrize(
    "overrides",
    [
        {"container_id": "e" * 64},
        {"container_name": "/deploysphere-wrong"},
        {"managed": "false"},
        {"project_id": "f" * 24},
        {"deployment_id": "e" * 24},
    ],
)
@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_unowned_container_is_rejected_without_lifecycle_command(
    monkeypatch, operation, overrides
):
    _mock_docker(monkeypatch, inspect_output=_inspect_output(**overrides))
    calls = runtime.subprocess.Popen
    # Keep a direct call counter independent of context-discovery calls.
    commands = []
    original = calls

    def recording_popen(command, **kwargs):
        commands.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", recording_popen)

    result = getattr(runtime, f"{operation}_container")(
        CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
    )

    assert result.outcome == "failed"
    assert result.owned is False
    assert result.error.code == "container_not_owned"
    assert not any(operation in command for command in commands)


def test_ownership_check_and_inspect_return_only_verified_container_metadata(monkeypatch):
    _mock_docker(monkeypatch)

    verified = runtime.verify_deploysphere_ownership(
        CONTAINER_ID, ObjectId(PROJECT_ID), ObjectId(DEPLOYMENT_ID)
    )
    inspected = runtime.inspect_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert verified.outcome == inspected.outcome == "succeeded"
    assert verified.owned is inspected.owned is True
    assert inspected.container_name == f"deploysphere-{DEPLOYMENT_ID}"
    assert inspected.state == "created"
    assert inspected.container_port == 8080
    assert inspected.allocated_port == 49152


def test_startup_inspection_discovers_only_the_exact_deterministic_name(monkeypatch):
    calls, _ = _mock_docker(monkeypatch, inspect_output=_inspect_output())

    result = runtime.inspect_container_for_startup(
        container_id=None,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.error is None
    assert result.observation is not None
    assert result.observation.container_id == CONTAINER_ID
    assert result.observation.container_name == f"deploysphere-{DEPLOYMENT_ID}"
    assert result.observation.identity_verified is True
    inspect_call = next(
        command
        for command, _kwargs in calls
        if "inspect" in command and "context" not in command
    )
    assert inspect_call[-3:] == [
        "--format",
        runtime._CONTAINER_INSPECT_TEMPLATE,
        f"deploysphere-{DEPLOYMENT_ID}",
    ]
    assert inspect_call[-1] == f"deploysphere-{DEPLOYMENT_ID}"
    assert "--type" in inspect_call
    assert not any(
        operation in command
        for command, _kwargs in calls
        for operation in ("create", "start", "stop", "restart", "rm", "prune")
    )


def test_startup_inspection_rejects_stored_name_before_docker(monkeypatch):
    calls, _ = _mock_docker(monkeypatch)

    result = runtime.inspect_container_for_startup(
        container_id=None,
        container_name="deploysphere-some-other-deployment",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.error is not None
    assert result.error.code == "identity_mismatch"
    assert calls == []


def test_startup_inspection_rejects_returned_full_id_mismatch(monkeypatch):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(container_id="e" * 64),
    )

    result = runtime.inspect_container_for_startup(
        container_id=CONTAINER_ID,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.observation is not None
    assert result.observation.identity_verified is False


@pytest.mark.parametrize(
    "label_overrides",
    [
        {"managed": "false"},
        {"project_id": "f" * 24},
        {"deployment_id": "e" * 24},
    ],
)
def test_startup_inspection_rejects_each_ownership_label_mismatch(
    monkeypatch, label_overrides
):
    _mock_docker(
        monkeypatch,
        inspect_output=_inspect_output(**label_overrides),
    )

    result = runtime.inspect_container_for_startup(
        container_id=CONTAINER_ID,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.observation is not None
    assert result.observation.identity_verified is False


def test_startup_inspection_rejects_multiple_results(monkeypatch):
    output = _inspect_output()
    _mock_docker(monkeypatch, inspect_output=output + b"\n" + output)

    result = runtime.inspect_container_for_startup(
        container_id=None,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.observation is None
    assert result.error is not None
    assert result.error.code == "ambiguous_lookup"


@pytest.mark.parametrize(
    ("stderr_output", "expected_code"),
    [
        (
            f"Error: No such object: {CONTAINER_ID}".encode("ascii"),
            "container_not_found",
        ),
        (
            b"Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
            b"Is the docker daemon running?",
            "docker_unavailable",
        ),
        (b"unrecognized Docker error", "inspection_inconclusive"),
    ],
)
def test_startup_inspection_classifies_only_explicit_command_errors(
    monkeypatch, stderr_output, expected_code
):
    _mock_docker(
        monkeypatch,
        failure="inspect",
        stderr_output=stderr_output,
    )

    result = runtime.inspect_container_for_startup(
        container_id=CONTAINER_ID,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.observation is None
    assert result.error is not None
    assert result.error.code == expected_code


def test_startup_inspection_timeout_is_inconclusive(monkeypatch):
    _mock_docker(monkeypatch, timeout_action="inspect")

    result = runtime.inspect_container_for_startup(
        container_id=CONTAINER_ID,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    assert result.observation is None
    assert result.error is not None
    assert result.error.code == "inspection_inconclusive"


def test_rejects_short_arbitrary_container_id_before_inspection(monkeypatch):
    calls, _ = _mock_docker(monkeypatch)

    with pytest.raises(ValueError, match="full Docker container ID"):
        runtime.stop_container("c" * 12, PROJECT_ID, DEPLOYMENT_ID)

    assert not _docker_calls(calls)


def test_missing_docker_executable_returns_safe_failure(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "failed"
    assert result.error.code == "docker_unavailable"


def test_remote_or_tls_endpoint_is_rejected(monkeypatch):
    _mock_docker(monkeypatch)
    monkeypatch.setenv("DOCKER_HOST", "tcp://example.invalid:2376")

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "failed"
    assert result.error.code == "unsupported_docker_context"


def test_known_nonzero_command_exit_is_failed_and_sanitized(monkeypatch):
    _mock_docker(monkeypatch, failure="create")

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "failed"
    assert result.error.code == "docker_command_failed"
    assert "secret" not in result.error.message
    assert "/private/host" not in result.error.message


def test_known_lifecycle_failure_is_not_reported_as_indeterminate(monkeypatch):
    _mock_docker(monkeypatch, failure="stop", lifecycle="stop")

    result = runtime.stop_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "failed"
    assert result.owned is True
    assert result.error.code == "docker_command_failed"


def test_malformed_inspection_after_creation_is_indeterminate_and_sanitized(monkeypatch):
    _mock_docker(monkeypatch, inspect_output=b"host path and credential text")

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code == "invalid_docker_response"
    assert "credential" not in result.error.message


def test_timeout_while_verifying_ownership_does_not_issue_stop(monkeypatch):
    calls, _ = _mock_docker(monkeypatch, timeout_action="inspect")

    result = runtime.stop_container(CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID)

    assert result.outcome == "indeterminate"
    assert result.owned is None
    assert not any(
        _verb(command) == "stop" for command, _ in _runtime_calls(calls)
    )


@pytest.mark.parametrize("action", ["create", "start", "stop", "restart"])
def test_timeout_is_indeterminate_and_never_claims_cancellation(monkeypatch, action):
    _mock_docker(
        monkeypatch,
        timeout_action=action,
        lifecycle=action if action in {"start", "stop", "restart"} else None,
    )

    if action == "create":
        result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)
    else:
        result = getattr(runtime, f"{action}_container")(
            CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID
        )

    assert result.outcome == "indeterminate"
    assert result.error.code == "docker_outcome_indeterminate"
    assert "cancel" not in result.error.message.casefold()


def test_unavailable_context_is_safe_failure(monkeypatch):
    _mock_docker(monkeypatch, context_failure=True)

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "failed"
    assert result.error.code == "docker_unavailable"
    assert "credential" not in result.error.message


def test_context_discovery_output_is_bounded(monkeypatch):
    _mock_docker(
        monkeypatch,
        context_output=b"x" * (runtime.MAX_CAPTURE_BYTES + 1),
    )

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "failed"
    assert result.error.code == "docker_unavailable"


def test_temp_docker_config_is_cleaned_if_session_entry_fails(monkeypatch, tmp_path):
    _mock_docker(monkeypatch)
    original_temporary_directory = runtime.tempfile.TemporaryDirectory
    temporary_roots = []

    def tracked_temporary_directory(*args, **kwargs):
        temporary = original_temporary_directory(*args, dir=tmp_path, **kwargs)
        temporary_roots.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(runtime.tempfile, "TemporaryDirectory", tracked_temporary_directory)
    original_open = builtins.open

    def fail_config_creation(file, *args, **kwargs):
        if str(file).endswith("config.json"):
            raise OSError("mocked config initialization failure")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(runtime, "open", fail_config_creation, raising=False)

    with pytest.raises(OSError, match="mocked config initialization failure"):
        with runtime._SessionOwner():
            pass

    assert temporary_roots
    assert not temporary_roots[0].exists()


def test_inspection_output_is_bounded_and_not_returned(monkeypatch):
    _mock_docker(monkeypatch, inspect_output=b"x" * (runtime.MAX_CAPTURE_BYTES + 10))

    result = runtime.create_container(IMAGE_TAG, PROJECT_ID, DEPLOYMENT_ID, 8080)

    assert result.outcome == "indeterminate"
    assert result.error.code in {"docker_output_too_large", "invalid_docker_response"}
    assert not hasattr(result, "stdout")
    assert "x" * 100 not in repr(result)


def test_command_timeout_bounds_are_validated():
    for invalid in (0, 301, "5", True):
        with pytest.raises(ValueError, match="timeout_seconds"):
            runtime.inspect_container(
                CONTAINER_ID, PROJECT_ID, DEPLOYMENT_ID, timeout_seconds=invalid
            )
