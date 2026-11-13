"""Integration tests for detection API and MongoDB reservation behavior."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from uuid import uuid4

from bson import ObjectId
from fastapi.testclient import TestClient
import pytest
from pymongo.errors import ServerSelectionTimeoutError

from backend.database.connection import client
from backend.database import connection
from backend.database.initialize import initialize_database
from backend.main import app
from backend.routers import project_detections
from backend.services import application_detection
from backend.services.application_detector import DetectionError, DetectionResult


http = TestClient(app)


def _setup_project(monkeypatch, tmp_path, *, commit_hash="a" * 40):
    """Create an isolated DB, project, and verified temporary workspace."""
    name = f"ds_detect_{uuid4().hex}"
    database = client[name]
    initialize_database(database)
    monkeypatch.setattr(connection, "database", database)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path.resolve()))

    project_id = ObjectId()
    now = datetime.now(timezone.utc)
    database.projects.insert_one(
        {
            "_id": project_id,
            "name": "Detection test project",
            "repository_url": f"https://github.com/example/{project_id}",
            "branch": "main",
            "project_type": "unknown",
            "deployment_config": {},
            "created_at": now,
            "updated_at": now,
        }
    )
    workspace_path = tmp_path.resolve() / str(project_id)
    workspace_path.mkdir()
    (workspace_path / "requirements.txt").write_text("", encoding="utf-8")
    workspace_id = ObjectId()
    database.repository_workspaces.insert_one(
        {
            "_id": workspace_id,
            "project_id": project_id,
            "status": "cloned",
            "repository_url": f"https://github.com/example/{project_id}",
            "branch": "main",
            "workspace_path": str(workspace_path),
            "commit_hash": commit_hash,
            "created_at": now,
            "updated_at": now,
        }
    )
    return name, database, project_id, workspace_id, workspace_path


def _result(application_type="python"):
    """Return predictable detector output without invoking repository code."""
    return DetectionResult(
        application_type=application_type,
        build_method="native",
        detected_application_types=[application_type],
        detected_build_methods=["native"],
        detected_files=["requirements.txt"],
        confidence=0.5,
        warnings=["Filename detection does not prove deployability."],
    )


def test_first_and_repeated_detection_requests_both_scan_and_save(
    monkeypatch, tmp_path
):
    name, database, project_id, _, workspace_path = _setup_project(monkeypatch, tmp_path)
    calls = []

    def scan(path):
        calls.append(path)
        return _result("python" if len(calls) == 1 else "nodejs")

    monkeypatch.setattr(application_detection, "detect_application", scan)
    try:
        first = http.post(f"/api/projects/{project_id}/detect")
        repeated = http.post(f"/api/projects/{project_id}/detect")
        saved = database.application_detections.find_one({"project_id": project_id})

        assert first.status_code == 201
        assert first.json()["status"] == "completed"
        assert first.json()["application_type"] == "python"
        assert "attempt_id" not in first.json()
        assert "reservation_expires_at" not in first.json()
        assert repeated.status_code == 200
        assert repeated.json()["id"] == first.json()["id"]
        assert repeated.json()["application_type"] == "nodejs"
        assert saved["application_type"] == "nodejs"
        assert calls == [str(workspace_path), str(workspace_path)]
        assert database.application_detections.count_documents({}) == 1
    finally:
        client.drop_database(name)


def test_commit_change_runs_detection_again(monkeypatch, tmp_path):
    name, database, project_id, _, _ = _setup_project(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        application_detection,
        "detect_application",
        lambda path: (calls.append(path) or _result("nodejs")),
    )
    try:
        first = http.post(f"/api/projects/{project_id}/detect")
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"commit_hash": "b" * 40}}
        )
        changed = http.post(f"/api/projects/{project_id}/detect")

        assert first.status_code == 201
        assert changed.status_code == 200
        assert changed.json()["commit_hash"] == "b" * 40
        assert changed.json()["application_type"] == "nodejs"
        assert len(calls) == 2
        assert database.application_detections.count_documents({}) == 1
    finally:
        client.drop_database(name)


def test_active_concurrent_detection_returns_conflict(monkeypatch, tmp_path):
    name, _, project_id, _, _ = _setup_project(monkeypatch, tmp_path)
    started = Event()
    finish = Event()

    def slow_scan(path):
        started.set()
        assert finish.wait(timeout=10)
        return _result()

    monkeypatch.setattr(application_detection, "detect_application", slow_scan)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_future = executor.submit(http.post, f"/api/projects/{project_id}/detect")
            assert started.wait(timeout=5)
            second = http.post(f"/api/projects/{project_id}/detect")
            finish.set()
            first = first_future.result(timeout=10)

        assert second.status_code == 409
        assert first.status_code == 201
    finally:
        finish.set()
        client.drop_database(name)


def test_failed_detection_can_be_retried(monkeypatch, tmp_path):
    name, database, project_id, _, _ = _setup_project(monkeypatch, tmp_path)
    outcomes = iter(
        [DetectionError("scan_timeout", "Application detection timed out."), _result()]
    )

    def scan(path):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(application_detection, "detect_application", scan)
    try:
        first = http.post(f"/api/projects/{project_id}/detect")
        failed = database.application_detections.find_one({"project_id": project_id})
        retry = http.post(f"/api/projects/{project_id}/detect")

        assert first.status_code == 422
        assert failed["status"] == "failed"
        assert "error" in failed
        assert retry.status_code == 200
        assert retry.json()["status"] == "completed"
        assert database.application_detections.count_documents({}) == 1
    finally:
        client.drop_database(name)


def test_stale_detection_reservation_is_recovered(monkeypatch, tmp_path):
    name, database, project_id, workspace_id, _ = _setup_project(monkeypatch, tmp_path)
    old = datetime.now(timezone.utc) - timedelta(minutes=4)
    database.application_detections.insert_one(
        {
            "project_id": project_id,
            "workspace_id": workspace_id,
            "attempt_id": "old-attempt",
            "commit_hash": "a" * 40,
            "status": "detecting",
            "application_type": "unknown",
            "build_method": "unknown",
            "detected_application_types": [],
            "detected_build_methods": [],
            "detected_files": [],
            "confidence": 0.0,
            "warnings": [],
            "reservation_expires_at": old,
            "created_at": old,
            "updated_at": old,
        }
    )
    monkeypatch.setattr(application_detection, "detect_application", lambda path: _result())
    try:
        response = http.post(f"/api/projects/{project_id}/detect")
        saved = database.application_detections.find_one({"project_id": project_id})

        assert response.status_code == 200
        assert saved["status"] == "completed"
        assert saved["attempt_id"] != "old-attempt"
    finally:
        client.drop_database(name)


def test_old_attempt_cannot_overwrite_newer_reservation(monkeypatch, tmp_path):
    name, database, project_id, _, workspace_path = _setup_project(monkeypatch, tmp_path)

    def supersede_attempt(path):
        database.application_detections.update_one(
            {"project_id": project_id}, {"$set": {"attempt_id": "newer-attempt"}}
        )
        return _result("nodejs")

    monkeypatch.setattr(application_detection, "detect_application", supersede_attempt)
    try:
        response = http.post(f"/api/projects/{project_id}/detect")
        saved = database.application_detections.find_one({"project_id": project_id})

        assert response.status_code == 409
        assert saved["status"] == "detecting"
        assert saved["attempt_id"] == "newer-attempt"
        assert saved["application_type"] == "unknown"
    finally:
        client.drop_database(name)


def test_unknown_and_malformed_project_ids(monkeypatch, tmp_path):
    name, _, _, _, _ = _setup_project(monkeypatch, tmp_path)
    try:
        malformed = http.post("/api/projects/not-an-object-id/detect")
        missing = http.post(f"/api/projects/{ObjectId()}/detect")
        assert malformed.status_code == 400
        assert missing.status_code == 404
    finally:
        client.drop_database(name)


def test_missing_or_uncloned_workspace_returns_conflict(monkeypatch, tmp_path):
    name, database, project_id, _, _ = _setup_project(monkeypatch, tmp_path)
    database.repository_workspaces.delete_many({"project_id": project_id})
    try:
        missing = http.post(f"/api/projects/{project_id}/detect")
        assert missing.status_code == 409
        database.repository_workspaces.insert_one(
            {
                "project_id": project_id,
                "status": "cloning",
                "repository_url": "https://github.com/example/project",
                "branch": "main",
                "workspace_path": str(tmp_path / str(project_id)),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        uncloned = http.post(f"/api/projects/{project_id}/detect")
        assert uncloned.status_code == 409
    finally:
        client.drop_database(name)


def test_recorded_path_mismatch_and_symlink_are_rejected(monkeypatch, tmp_path):
    name, database, project_id, _, workspace_path = _setup_project(monkeypatch, tmp_path)
    try:
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"workspace_path": str(tmp_path / "outside")}}
        )
        mismatch = http.post(f"/api/projects/{project_id}/detect")
        assert mismatch.status_code == 409
        assert database.application_detections.count_documents({}) == 0

        outside = tmp_path / "outside"
        outside.mkdir()
        (workspace_path / "requirements.txt").unlink()
        workspace_path.rmdir()
        try:
            workspace_path.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("This Windows environment does not allow creating directory symlinks.")
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"workspace_path": str(workspace_path)}}
        )
        symlink = http.post(f"/api/projects/{project_id}/detect")
        assert symlink.status_code == 409
        assert database.application_detections.count_documents({}) == 0
    finally:
        client.drop_database(name)


def test_get_detection_is_read_only_and_returns_saved_result(monkeypatch, tmp_path):
    name, _, project_id, _, _ = _setup_project(monkeypatch, tmp_path)
    calls = []

    def scan(path):
        calls.append(path)
        return _result()

    monkeypatch.setattr(application_detection, "detect_application", scan)
    try:
        missing = http.get(f"/api/projects/{project_id}/detection")
        created = http.post(f"/api/projects/{project_id}/detect")
        fetched = http.get(f"/api/projects/{project_id}/detection")

        assert missing.status_code == 404
        assert created.status_code == 201
        assert fetched.status_code == 200
        assert fetched.json()["commit_hash"] == "a" * 40
        assert fetched.json()["application_type"] == created.json()["application_type"]
        assert len(calls) == 1
    finally:
        client.drop_database(name)


def test_unexpected_detector_exception_is_safe_and_retryable(monkeypatch, tmp_path):
    name, database, project_id, _, _ = _setup_project(monkeypatch, tmp_path)

    def broken_scan(path):
        raise RuntimeError("private C:/host/path and secret")

    monkeypatch.setattr(application_detection, "detect_application", broken_scan)
    try:
        response = http.post(f"/api/projects/{project_id}/detect")
        saved = database.application_detections.find_one({"project_id": project_id})
        assert response.status_code == 500
        assert "private" not in response.text
        assert "secret" not in response.text
        assert saved["error"]["code"] == "detection_failed"
        assert "private" not in saved["error"]["message"]
    finally:
        client.drop_database(name)


def test_mongodb_errors_during_reservation_and_result_update_are_503(
    monkeypatch, tmp_path
):
    name, database, project_id, _, _ = _setup_project(monkeypatch, tmp_path)
    original_reserve = application_detection._reserve_detection

    def unavailable(*args, **kwargs):
        raise application_detection.DetectionStoreError

    monkeypatch.setattr(application_detection, "_reserve_detection", unavailable)
    try:
        reservation_failure = http.post(f"/api/projects/{project_id}/detect")
        assert reservation_failure.status_code == 503
    finally:
        monkeypatch.setattr(application_detection, "_reserve_detection", original_reserve)

    monkeypatch.setattr(application_detection, "detect_application", lambda path: _result())
    collection = database.application_detections
    collection_type = type(collection)
    original_update = collection_type.find_one_and_update

    def fail_result_update(self, *args, **kwargs):
        # Allow reservation's existing-record operation, fail the completion update.
        if self.name == "application_detections" and args and args[0].get("status") == "detecting":
            raise ServerSelectionTimeoutError("private database address")
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(collection_type, "find_one_and_update", fail_result_update)
    try:
        result_failure = http.post(f"/api/projects/{project_id}/detect")
        assert result_failure.status_code == 503
        assert "private database address" not in result_failure.text
    finally:
        monkeypatch.setattr(collection_type, "find_one_and_update", original_update)
        client.drop_database(name)
