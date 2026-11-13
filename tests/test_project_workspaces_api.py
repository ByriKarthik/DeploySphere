"""API tests for project workspace reservations and clone lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from uuid import uuid4

from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

from backend.database import connection
from backend.database.connection import client
from backend.database.initialize import initialize_database
from backend.main import app
from backend.routers import project_workspaces
from backend.services.git_clone import CloneResult, GitCloneError


http = TestClient(app)


def _make_project_database(monkeypatch):
    """Create an isolated database and direct the API to use it."""
    name = f"ds_ws_{uuid4().hex}"
    database = client[name]
    initialize_database(database)
    monkeypatch.setattr(connection, "database", database)
    return name, database


def _register_project():
    response = http.post(
        "/api/projects",
        json={
            "name": "Workspace test project",
            "repository_url": "https://github.com/example/workspace-test",
            "branch": "main",
        },
    )
    assert response.status_code == 201
    return response.json()


def _clone_success(repository_url, branch, project_id, **kwargs):
    return CloneResult(
        workspace_path=f"private-path-for-{project_id}",
        commit_hash="b" * 40,
    )


def test_successful_clone_returns_status_without_absolute_path(monkeypatch, tmp_path):
    name, database = _make_project_database(monkeypatch)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(project_workspaces, "clone_repository", _clone_success)
    try:
        project = _register_project()

        response = http.post(f"/api/projects/{project['id']}/clone")

        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "cloned"
        assert payload["project_id"] == project["id"]
        assert payload["commit_hash"] == "b" * 40
        assert "workspace_path" not in payload
        workspace = database.repository_workspaces.find_one(
            {"project_id": ObjectId(project["id"])}
        )
        assert workspace["status"] == "cloned"
        assert workspace["workspace_path"] == str(tmp_path / project["id"])

        listed = http.get("/api/projects").json()
        assert listed[0]["workspace"]["status"] == "cloned"
    finally:
        client.drop_database(name)


def test_invalid_project_id_and_missing_project(monkeypatch):
    name, _ = _make_project_database(monkeypatch)
    try:
        invalid_response = http.post("/api/projects/not-an-object-id/clone")
        missing_response = http.post(f"/api/projects/{ObjectId()}/clone")

        assert invalid_response.status_code == 400
        assert missing_response.status_code == 404
    finally:
        client.drop_database(name)


def test_invalid_stored_repository_is_rejected_before_reservation(monkeypatch):
    name, database = _make_project_database(monkeypatch)
    project_id = ObjectId()
    now = datetime.now(timezone.utc)
    database.projects.insert_one(
        {
            "_id": project_id,
            "name": "Invalid stored URL",
            "repository_url": "https://github.com/example/valid",
            "branch": "main",
            "project_type": "unknown",
            "deployment_config": {},
            "created_at": now,
            "updated_at": now,
        }
    )
    database.projects.update_one(
        {"_id": project_id}, {"$set": {"repository_url": "https://example.org/repo"}}
    )
    try:
        response = http.post(f"/api/projects/{project_id}/clone")

        assert response.status_code == 422
        assert database.repository_workspaces.count_documents({}) == 0
    finally:
        client.drop_database(name)


def test_duplicate_clone_returns_409(monkeypatch, tmp_path):
    name, _ = _make_project_database(monkeypatch)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))
    calls = []

    def clone(*args, **kwargs):
        calls.append(args)
        return _clone_success(*args, **kwargs)

    monkeypatch.setattr(project_workspaces, "clone_repository", clone)
    try:
        project = _register_project()
        first = http.post(f"/api/projects/{project['id']}/clone")
        second = http.post(f"/api/projects/{project['id']}/clone")

        assert first.status_code == 201
        assert second.status_code == 409
        assert len(calls) == 1
    finally:
        client.drop_database(name)


def test_failed_clone_can_be_retried_without_duplicate_record(monkeypatch, tmp_path):
    name, database = _make_project_database(monkeypatch)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))
    outcomes = iter(
        [
            GitCloneError("network_error", "GitHub could not be reached.", 502),
            CloneResult(str(tmp_path / "unused"), "c" * 40),
        ]
    )
    def clone_attempt(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(project_workspaces, "clone_repository", clone_attempt)
    try:
        project = _register_project()
        first = http.post(f"/api/projects/{project['id']}/clone")
        failed_workspace = database.repository_workspaces.find_one(
            {"project_id": ObjectId(project["id"])}
        )
        second = http.post(f"/api/projects/{project['id']}/clone")

        assert first.status_code == 502
        assert failed_workspace["status"] == "failed"
        assert failed_workspace["error"]["code"] == "network_error"
        assert second.status_code == 201
        assert database.repository_workspaces.count_documents({}) == 1
        assert database.repository_workspaces.find_one(
            {"project_id": ObjectId(project["id"])}
        )["status"] == "cloned"
    finally:
        client.drop_database(name)


def test_concurrent_clone_requests_only_reserve_once(monkeypatch, tmp_path):
    name, database = _make_project_database(monkeypatch)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))
    clone_started = Event()
    allow_clone_to_finish = Event()

    def slow_clone(*args, **kwargs):
        clone_started.set()
        assert allow_clone_to_finish.wait(timeout=5)
        return _clone_success(*args, **kwargs)

    monkeypatch.setattr(project_workspaces, "clone_repository", slow_clone)
    try:
        project = _register_project()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(http.post, f"/api/projects/{project['id']}/clone")
            assert clone_started.wait(timeout=5)
            second = http.post(f"/api/projects/{project['id']}/clone")
            allow_clone_to_finish.set()
            first = first_future.result(timeout=5)

        assert first.status_code == 201
        assert second.status_code == 409
        assert database.repository_workspaces.count_documents(
            {"project_id": ObjectId(project["id"])}
        ) == 1
    finally:
        allow_clone_to_finish.set()
        client.drop_database(name)


class _UnavailableProjects:
    def find_one(self, query):
        raise ServerSelectionTimeoutError("MongoDB is unavailable")


class _UnavailableDatabase:
    projects = _UnavailableProjects()


def test_mongodb_lookup_failure_returns_503(monkeypatch):
    monkeypatch.setattr(connection, "database", _UnavailableDatabase())

    response = http.post(f"/api/projects/{ObjectId()}/clone")

    assert response.status_code == 503


class _ReservationFailureWorkspaces:
    def find_one_and_update(self, *args, **kwargs):
        raise ServerSelectionTimeoutError("MongoDB is unavailable")


class _ReservationFailureDatabase:
    projects = type("Projects", (), {"find_one": lambda self, query: {"repository_url": "https://github.com/example/repo", "branch": "main"}})()
    repository_workspaces = _ReservationFailureWorkspaces()


def test_mongodb_reservation_failure_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(connection, "database", _ReservationFailureDatabase())
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))

    response = http.post(f"/api/projects/{ObjectId()}/clone")

    assert response.status_code == 503


class _FinalUpdateFailureWorkspaces:
    def find_one_and_update(self, *args, **kwargs):
        return None

    def insert_one(self, document):
        return None

    def update_one(self, *args, **kwargs):
        raise ServerSelectionTimeoutError("MongoDB is unavailable")


class _FinalUpdateFailureDatabase:
    projects = type("Projects", (), {"find_one": lambda self, query: {"repository_url": "https://github.com/example/repo", "branch": "main"}})()
    repository_workspaces = _FinalUpdateFailureWorkspaces()


def test_mongodb_final_update_failure_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(connection, "database", _FinalUpdateFailureDatabase())
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(project_workspaces, "clone_repository", _clone_success)

    response = http.post(f"/api/projects/{ObjectId()}/clone")

    assert response.status_code == 503


class _FailureStatusUpdateWorkspaces:
    def find_one_and_update(self, *args, **kwargs):
        return None

    def insert_one(self, document):
        return None

    def update_one(self, *args, **kwargs):
        raise ServerSelectionTimeoutError("MongoDB is unavailable")


class _FailureStatusUpdateDatabase:
    projects = type("Projects", (), {"find_one": lambda self, query: {"repository_url": "https://github.com/example/repo", "branch": "main"}})()
    repository_workspaces = _FailureStatusUpdateWorkspaces()


def test_mongodb_failure_status_update_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(connection, "database", _FailureStatusUpdateDatabase())
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path))

    def fail_clone(*args, **kwargs):
        raise GitCloneError("network_error", "GitHub could not be reached.", 502)

    monkeypatch.setattr(project_workspaces, "clone_repository", fail_clone)

    response = http.post(f"/api/projects/{ObjectId()}/clone")

    assert response.status_code == 503
