"""Focused tests for single-process ownership and FastAPI lifespan."""

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from backend import main
from backend.database import initialize as database_initializer
from backend.process_lock import (
    BackendProcessLock,
    BackendProcessLockError,
    default_process_lock_path,
)
from backend.services import deployment_coordinator


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HOLDER_SCRIPT = """
from pathlib import Path
import sys
from backend.process_lock import BackendProcessLock

lock = BackendProcessLock(sys.argv[1])
lock.acquire()
Path(sys.argv[2]).write_text("ready", encoding="ascii")
sys.stdin.readline()
lock.release()
"""
_ACQUIRE_SCRIPT = """
import sys
from backend.process_lock import BackendProcessLock, BackendProcessLockError

lock = BackendProcessLock(sys.argv[1])
try:
    lock.acquire()
except BackendProcessLockError as error:
    print(f"BUSY: {error}")
    raise SystemExit(2)
print("ACQUIRED")
lock.release()
"""


def _wait_for_ready(process: subprocess.Popen, ready_file: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if ready_file.is_file():
            return
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"Lock-holder subprocess exited early: {stderr}")
        time.sleep(0.01)
    raise AssertionError("Lock-holder subprocess did not acquire the lock in time.")


def _start_holder(lock_path: Path, ready_file: Path) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOLDER_SCRIPT,
            str(lock_path),
            str(ready_file),
        ],
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_ready(process, ready_file)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        raise
    return process


def _release_holder(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            if process.stdin is not None:
                process.stdin.write("release\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    if process.stderr is not None:
        process.stderr.close()


@contextmanager
def _holding_process(lock_path: Path, ready_file: Path):
    process = _start_holder(lock_path, ready_file)
    try:
        yield process
    finally:
        _release_holder(process)


def _try_acquire_in_subprocess(lock_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _ACQUIRE_SCRIPT, str(lock_path)],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def test_first_process_acquires_and_second_process_cannot_steal_lock(tmp_path):
    lock_path = tmp_path / "DeploySphere" / "backend.lock"
    ready_file = tmp_path / "holder-ready"

    with _holding_process(lock_path, ready_file):
        assert lock_path.is_file()
        first_attempt = _try_acquire_in_subprocess(lock_path)
        second_attempt = _try_acquire_in_subprocess(lock_path)

        assert first_attempt.returncode == 2
        assert "BUSY:" in first_attempt.stdout
        assert second_attempt.returncode == 2
        assert "BUSY:" in second_attempt.stdout
        assert lock_path.is_file()


def test_second_process_can_acquire_after_normal_release_and_lock_file_persists(
    tmp_path,
):
    lock_path = tmp_path / "backend.lock"
    process = _start_holder(lock_path, tmp_path / "holder-ready")
    _release_holder(process)

    assert lock_path.is_file()
    result = _try_acquire_in_subprocess(lock_path)

    assert result.returncode == 0
    assert "ACQUIRED" in result.stdout
    assert lock_path.is_file()


def test_second_process_can_acquire_after_holder_exits_unexpectedly(tmp_path):
    lock_path = tmp_path / "backend.lock"
    process = _start_holder(lock_path, tmp_path / "holder-ready")
    try:
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.stderr is not None:
            process.stderr.close()

    assert lock_path.is_file()
    result = _try_acquire_in_subprocess(lock_path)

    assert result.returncode == 0
    assert "ACQUIRED" in result.stdout
    assert lock_path.is_file()


def test_repeated_release_and_reacquire_are_safe(tmp_path):
    lock_path = tmp_path / "backend.lock"
    lock = BackendProcessLock(lock_path)

    lock.acquire()
    lock.release()
    lock.release()

    assert lock_path.is_file()
    lock.acquire()
    lock.release()
    assert lock_path.is_file()


def test_default_lock_path_uses_local_application_state(monkeypatch, tmp_path):
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        expected = tmp_path / "DeploySphere" / "backend.lock"
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        expected = tmp_path / "deploysphere" / "backend.lock"

    assert default_process_lock_path() == expected


def test_lifespan_holds_lock_without_database_initialization_or_recovery(
    monkeypatch,
):
    events = []
    lock = Mock()
    lock.__enter__ = Mock(side_effect=lambda: events.append("acquired") or lock)
    lock.__exit__ = Mock(
        side_effect=lambda *_args: events.append("released") or False
    )
    monkeypatch.setattr(main, "BackendProcessLock", lambda: lock)
    inspect = Mock(
        side_effect=lambda _database: events.append("inspected")
        or {
            "outcome": "completed",
            "classification_counts": {},
            "results": [],
        }
    )
    monkeypatch.setattr(main, "inspect_active_deployments", inspect)
    initialize = Mock(side_effect=AssertionError("database initialization is forbidden"))
    reconcile = Mock(side_effect=AssertionError("startup recovery is forbidden"))
    monkeypatch.setattr(database_initializer, "initialize_database", initialize)
    monkeypatch.setattr(
        deployment_coordinator,
        "reconcile_unknown_deployment_as_failed",
        reconcile,
    )

    with TestClient(main.app) as client:
        assert events == ["acquired", "inspected"]
        response = client.get("/")
        assert response.status_code == 200
        assert events == ["acquired", "inspected"]

    assert events == ["acquired", "inspected", "released"]
    inspect.assert_called_once_with(main.connection.database)
    snapshot = main.app.state.startup_inspection_snapshot
    assert snapshot["outcome"] == "completed"
    assert datetime.fromisoformat(snapshot["captured_at"]).tzinfo is not None
    initialize.assert_not_called()
    reconcile.assert_not_called()


def test_lifespan_aborts_startup_when_process_lock_is_held(monkeypatch):
    def fail_acquisition():
        raise BackendProcessLockError("Another backend owns the process lock.")

    monkeypatch.setattr(main, "BackendProcessLock", fail_acquisition)

    with pytest.raises(BackendProcessLockError, match="Another backend owns"):
        with TestClient(main.app):
            pytest.fail("The application served despite a held process lock.")


def test_existing_routers_remain_registered():
    paths = set(main.app.openapi()["paths"])

    assert {
        "/",
        "/api/health",
        "/api/diagnostics/startup-inspection",
        "/api/projects/{project_id}/deployments",
        "/api/projects/{project_id}/deployments/active",
        "/api/projects/{project_id}/deployments/{deployment_id}",
    } <= paths
