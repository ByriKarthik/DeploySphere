"""Integration tests for image-build MongoDB orchestration and API routes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from uuid import uuid4

from bson import ObjectId
from fastapi.testclient import TestClient
import pytest
from pymongo.errors import ServerSelectionTimeoutError

from backend.database import connection
from backend.database.connection import client
from backend.database.initialize import initialize_database
from backend.main import app
from backend.services import image_builds
from backend.services.docker_image_builder import BuildError, BuildResult


http = TestClient(app)


def _setup_project(monkeypatch, tmp_path):
    """Create a disposable database and a workspace with nested Dockerfile."""
    database_name = f"ds_image_build_api_{uuid4().hex}"
    database = client[database_name]
    initialize_database(database)
    monkeypatch.setattr(connection, "database", database)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path.resolve()))

    now = datetime.now(timezone.utc)
    project_id = ObjectId()
    workspace_id = ObjectId()
    workspace_path = tmp_path.resolve() / str(project_id)
    (workspace_path / "services" / "api").mkdir(parents=True)
    (workspace_path / "services" / "api" / "Dockerfile").write_text(
        "FROM scratch\n", encoding="utf-8"
    )
    database.projects.insert_one(
        {
            "_id": project_id,
            "name": "Build test project",
            "repository_url": f"https://github.com/example/{project_id}",
            "branch": "main",
            "project_type": "unknown",
            "deployment_config": {},
            "created_at": now,
            "updated_at": now,
        }
    )
    database.repository_workspaces.insert_one(
        {
            "_id": workspace_id,
            "project_id": project_id,
            "status": "cloned",
            "repository_url": f"https://github.com/example/{project_id}",
            "branch": "main",
            "workspace_path": str(workspace_path),
            "commit_hash": "a" * 40,
            "created_at": now,
            "updated_at": now,
        }
    )
    detection_id = ObjectId()
    database.application_detections.insert_one(
        {
            "_id": detection_id,
            "project_id": project_id,
            "workspace_id": workspace_id,
            "attempt_id": uuid4().hex,
            "commit_hash": "a" * 40,
            "status": "completed",
            "application_type": "python",
            "build_method": "dockerfile",
            "detected_application_types": ["python"],
            "detected_build_methods": ["native", "dockerfile"],
            "detected_files": ["requirements.txt", "services/api/Dockerfile"],
            "confidence": 0.9,
            "warnings": [],
            "created_at": now,
            "updated_at": now,
        }
    )
    return database_name, database, project_id, workspace_id, detection_id, workspace_path


def _engine_result(*, success=True, outcome_known=True, log_excerpt=""):
    """Return a safe fake engine result without starting Docker."""
    return BuildResult(
        success=success,
        image_tag=None,
        image_id="sha256:" + "b" * 64 if success else None,
        duration_seconds=2.5,
        log_bytes=123,
        log_line_count=4,
        log_truncated=False,
        outcome_known=outcome_known,
        error=None
        if success
        else BuildError(
            "build_timeout" if not outcome_known else "build_failed",
            "The build timed out; its daemon outcome is unknown."
            if not outcome_known
            else "Docker could not complete this build.",
        ),
        log_excerpt=log_excerpt,
    )


def _cleanup(name):
    """Drop only the disposable database created by a test."""
    client.drop_database(name)


def test_successful_build_persists_nested_dockerfile_and_safe_response(
    monkeypatch, tmp_path
):
    name, database, project_id, workspace_id, detection_id, workspace_path = _setup_project(
        monkeypatch, tmp_path
    )
    calls = []

    def fake_build(root, context, dockerfile, tag, detection):
        calls.append((root, context, dockerfile, tag, detection.application_type))
        return _engine_result()

    monkeypatch.setattr(image_builds, "build_docker_image", fake_build)
    try:
        response = http.post(f"/api/projects/{project_id}/build")
        payload = response.json()
        saved = database.image_builds.find_one({"_id": ObjectId(payload["id"])})

        assert response.status_code == 201
        assert payload["status"] == "succeeded"
        assert payload["image_id"] == "sha256:" + "b" * 64
        assert payload["dockerfile_path"] == "services/api/Dockerfile"
        assert calls == [
            (str(workspace_path), ".", "services/api/Dockerfile", saved["image_tag"], "python")
        ]
        assert saved["workspace_id"] == workspace_id
        assert saved["detection_id"] == detection_id
        assert saved["status"] == "succeeded"
        assert "attempt_id" not in payload
        assert "workspace_id" not in payload
        assert "detection_id" not in payload
        assert "build_context" not in payload
        assert "reservation_expires_at" not in payload
        assert str(workspace_path) not in response.text
    finally:
        _cleanup(name)


def test_successful_build_is_idempotent_for_same_detection(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        image_builds,
        "build_docker_image",
        lambda *args: (calls.append(args) or _engine_result()),
    )
    try:
        first = http.post(f"/api/projects/{project_id}/build")
        repeated = http.post(f"/api/projects/{project_id}/build")

        assert first.status_code == 201
        assert repeated.status_code == 200
        assert first.json()["id"] == repeated.json()["id"]
        assert len(calls) == 1
        assert database.image_builds.count_documents({}) == 1
    finally:
        _cleanup(name)


def test_concurrent_build_reservation_returns_conflict(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    started = Event()
    finish = Event()

    def slow_build(*args):
        started.set()
        assert finish.wait(timeout=10)
        return _engine_result()

    monkeypatch.setattr(image_builds, "build_docker_image", slow_build)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_future = executor.submit(http.post, f"/api/projects/{project_id}/build")
            assert started.wait(timeout=5)
            second = http.post(f"/api/projects/{project_id}/build")
            finish.set()
            first = first_future.result(timeout=10)

        assert second.status_code == 409
        assert first.status_code == 201
        assert database.image_builds.count_documents({"status": "succeeded"}) == 1
    finally:
        finish.set()
        _cleanup(name)


def test_known_failure_can_be_retried(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    results = iter([_engine_result(success=False), _engine_result()])
    calls = []

    def alternating_build(*args):
        calls.append(args[3])
        return next(results)

    monkeypatch.setattr(image_builds, "build_docker_image", alternating_build)
    try:
        failed = http.post(f"/api/projects/{project_id}/build")
        succeeded = http.post(f"/api/projects/{project_id}/build")

        assert failed.status_code == 201
        assert failed.json()["status"] == "failed"
        assert succeeded.status_code == 201
        assert succeeded.json()["status"] == "succeeded"
        assert len(calls) == 2
        assert calls[0] != calls[1]
        assert database.image_builds.count_documents({}) == 2
    finally:
        _cleanup(name)


def test_stale_detection_commit_is_rejected(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(image_builds, "build_docker_image", lambda *args: calls.append(args))
    database.repository_workspaces.update_one(
        {"project_id": project_id}, {"$set": {"commit_hash": "c" * 40}}
    )
    try:
        response = http.post(f"/api/projects/{project_id}/build")
        assert response.status_code == 409
        assert "detection" in response.json()["detail"].lower()
        assert calls == []
        assert database.image_builds.count_documents({}) == 0
    finally:
        _cleanup(name)


def test_missing_project_workspace_and_detection_are_rejected(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    try:
        assert http.post(f"/api/projects/{ObjectId()}/build").status_code == 404
        workspace = database.repository_workspaces.find_one({"project_id": project_id})
        database.repository_workspaces.delete_one({"project_id": project_id})
        assert http.post(f"/api/projects/{project_id}/build").status_code == 409
        database.repository_workspaces.insert_one(workspace)
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"status": "failed"}}
        )
        assert http.post(f"/api/projects/{project_id}/build").status_code == 409
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"status": "cloned"}}
        )
        database.application_detections.delete_many({"project_id": project_id})
        assert http.post(f"/api/projects/{project_id}/build").status_code == 409
        assert http.post("/api/projects/not-an-object-id/build").status_code == 400
    finally:
        _cleanup(name)


def test_recorded_workspace_path_mismatch_and_symlink_are_rejected(
    monkeypatch, tmp_path
):
    name, database, project_id, *_ , workspace_path = _setup_project(monkeypatch, tmp_path)
    try:
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"workspace_path": str(tmp_path / "elsewhere")}}
        )
        mismatch = http.post(f"/api/projects/{project_id}/build")
        assert mismatch.status_code == 409

        outside = tmp_path / "outside"
        outside.mkdir()
        workspace_path.rename(tmp_path / "original-workspace")
        try:
            workspace_path.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("This environment does not permit creating directory symlinks.")
        database.repository_workspaces.update_one(
            {"project_id": project_id}, {"$set": {"workspace_path": str(workspace_path)}}
        )
        assert http.post(f"/api/projects/{project_id}/build").status_code == 409
        assert database.image_builds.count_documents({}) == 0
    finally:
        _cleanup(name)


@pytest.mark.parametrize("dockerfiles", [[], ["services/api/Dockerfile", "Dockerfile.dev"]])
def test_missing_or_ambiguous_dockerfiles_are_rejected(monkeypatch, tmp_path, dockerfiles):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    database.application_detections.update_one(
        {"project_id": project_id}, {"$set": {"detected_files": dockerfiles}}
    )
    calls = []
    monkeypatch.setattr(image_builds, "build_docker_image", lambda *args: calls.append(args))
    try:
        response = http.post(f"/api/projects/{project_id}/build")
        assert response.status_code == 422
        assert calls == []
        assert database.image_builds.count_documents({}) == 0
    finally:
        _cleanup(name)


def test_indeterminate_timeout_keeps_build_active_and_blocks_retry(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        image_builds,
        "build_docker_image",
        lambda *args: (calls.append(args) or _engine_result(success=False, outcome_known=False)),
    )
    try:
        timed_out = http.post(f"/api/projects/{project_id}/build")
        retry = http.post(f"/api/projects/{project_id}/build")
        record = database.image_builds.find_one({"project_id": project_id})

        assert timed_out.status_code == 201
        assert timed_out.json()["status"] == "building"
        assert timed_out.json()["outcome_known"] is False
        assert timed_out.json()["error"]["code"] == "build_timeout"
        assert retry.status_code == 409
        assert len(calls) == 1
        assert record["status"] == "building"
        assert record["error"]["code"] == "build_timeout"
    finally:
        _cleanup(name)


def _insert_history(database, project_id, index, status="succeeded"):
    """Insert one valid historical build record with a deterministic time."""
    now = datetime.now(timezone.utc) + timedelta(seconds=index)
    document = {
        "project_id": project_id,
        "workspace_id": ObjectId(),
        "detection_id": ObjectId(),
        "commit_hash": "a" * 40,
        "attempt_id": uuid4().hex,
        "status": status,
        "image_tag": f"deploysphere/build:{uuid4().hex}",
        "build_context": ".",
        "dockerfile_path": "Dockerfile",
        "build_duration": float(index),
        "created_at": now,
        "updated_at": now,
    }
    return database.image_builds.insert_one(document).inserted_id


def test_build_history_pagination_and_project_scoped_detail(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    ids = [_insert_history(database, project_id, value) for value in range(3)]
    other_project = ObjectId()
    now = datetime.now(timezone.utc)
    database.projects.insert_one(
        {
            "_id": other_project,
            "name": "Other project",
            "repository_url": f"https://github.com/example/{other_project}",
            "branch": "main",
            "project_type": "unknown",
            "deployment_config": {},
            "created_at": now,
            "updated_at": now,
        }
    )
    other_build = _insert_history(database, other_project, 5)
    try:
        page = http.get(f"/api/projects/{project_id}/builds?skip=1&limit=1")
        detail = http.get(f"/api/projects/{project_id}/builds/{ids[2]}")
        wrong_owner = http.get(f"/api/projects/{project_id}/builds/{other_build}")
        invalid = http.get(f"/api/projects/{project_id}/builds/not-an-object-id")

        assert page.status_code == 200
        assert len(page.json()) == 1
        assert page.json()[0]["id"] == str(ids[1])
        assert detail.status_code == 200
        assert detail.json()["id"] == str(ids[2])
        assert wrong_owner.status_code == 404
        assert invalid.status_code == 400
    finally:
        _cleanup(name)


def test_failed_build_log_excerpt_is_saved_and_project_scoped(monkeypatch, tmp_path):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    monkeypatch.setattr(
        image_builds,
        "build_docker_image",
        lambda *args: _engine_result(
            success=False,
            log_excerpt="Dockerfile:3: invalid instruction\n",
        ),
    )
    try:
        build_response = http.post(f"/api/projects/{project_id}/build")
        build_id = build_response.json()["id"]
        log_response = http.get(
            f"/api/projects/{project_id}/builds/{build_id}/logs"
        )
        saved = database.image_builds.find_one({"_id": ObjectId(build_id)})
        wrong_project = http.get(
            f"/api/projects/{ObjectId()}/builds/{build_id}/logs"
        )

        assert build_response.status_code == 201
        assert saved["log_excerpt"] == "Dockerfile:3: invalid instruction\n"
        assert log_response.status_code == 200
        assert log_response.json() == {
            "log_excerpt": "Dockerfile:3: invalid instruction\n"
        }
        assert wrong_project.status_code == 404
    finally:
        _cleanup(name)


def test_mongodb_failure_during_reservation_and_finalization_is_safe(
    monkeypatch, tmp_path
):
    name, database, project_id, *_ = _setup_project(monkeypatch, tmp_path)
    collection_type = type(database.image_builds)
    original_insert = collection_type.insert_one

    def fail_reservation(self, *args, **kwargs):
        if self.name == "image_builds":
            raise ServerSelectionTimeoutError("private database detail")
        return original_insert(self, *args, **kwargs)

    monkeypatch.setattr(collection_type, "insert_one", fail_reservation)
    try:
        response = http.post(f"/api/projects/{project_id}/build")
        assert response.status_code == 503
        assert "private database detail" not in response.text
    finally:
        monkeypatch.setattr(collection_type, "insert_one", original_insert)

    monkeypatch.setattr(image_builds, "build_docker_image", lambda *args: _engine_result())
    original_update = collection_type.find_one_and_update

    def fail_finalization(self, filter, *args, **kwargs):
        if self.name == "image_builds" and filter.get("status") == "building":
            raise ServerSelectionTimeoutError("private finalization detail")
        return original_update(self, filter, *args, **kwargs)

    monkeypatch.setattr(collection_type, "find_one_and_update", fail_finalization)
    try:
        response = http.post(f"/api/projects/{project_id}/build")
        assert response.status_code == 503
        assert "private finalization detail" not in response.text
        record = database.image_builds.find_one({"project_id": project_id})
        assert record["status"] == "building"
    finally:
        monkeypatch.setattr(collection_type, "find_one_and_update", original_update)
        _cleanup(name)
