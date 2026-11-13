"""API tests for GitHub project registration and listing."""

from uuid import uuid4
from datetime import datetime, timezone

from bson import ObjectId
from fastapi.testclient import TestClient
import pytest
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from backend.database import connection
from backend.database.connection import client
from backend.database.initialize import initialize_database
from backend.main import app


http = TestClient(app)


def _create_test_database(monkeypatch):
    """Create an isolated, schema-initialized database for an API test."""
    name = f"deploysphere_api_test_{uuid4().hex}"
    database = client[name]
    initialize_database(database)
    monkeypatch.setattr(connection, "database", database)
    return name, database


def test_successful_project_registration(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    try:
        response = http.post(
            "/api/projects",
            json={
                "name": "Example App",
                "repository_url": "https://github.com/Example/DeploySphere.git",
            },
        )

        assert response.status_code == 201
        project = response.json()
        assert project["id"]
        assert project["name"] == "Example App"
        assert project["repository_url"] == "https://github.com/example/deploysphere"
        assert project["branch"] == "main"
        assert project["project_type"] == "unknown"
        assert project["deployment_config"] == {}
        assert "detection" not in project
        assert project["created_at"].endswith("Z")
        assert project["created_at"] == project["updated_at"]
        assert database.projects.count_documents({}) == 1
    finally:
        client.drop_database(name)


def test_invalid_repository_url_is_rejected(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    try:
        response = http.post(
            "/api/projects",
            json={
                "name": "Example App",
                "repository_url": "https://gitlab.com/example/deploysphere",
            },
        )

        assert response.status_code == 422
        assert database.projects.count_documents({}) == 0
    finally:
        client.drop_database(name)


def test_duplicate_repository_and_branch_returns_conflict(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    payload = {
        "name": "Example App",
        "repository_url": "https://github.com/example/deploysphere",
        "branch": "main",
    }
    try:
        assert http.post("/api/projects", json=payload).status_code == 201
        response = http.post("/api/projects", json=payload)

        assert response.status_code == 409
        assert database.projects.count_documents({}) == 1
    finally:
        client.drop_database(name)


def test_project_listing_returns_registered_projects(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    try:
        http.post(
            "/api/projects",
            json={
                "name": "First Project",
                "repository_url": "https://github.com/example/first",
                "branch": "main",
            },
        )
        http.post(
            "/api/projects",
            json={
                "name": "Second Project",
                "repository_url": "https://github.com/example/second",
                "branch": "develop",
            },
        )

        response = http.get("/api/projects")

        assert response.status_code == 200
        assert {item["name"] for item in response.json()} == {
            "First Project", "Second Project"
        }
    finally:
        client.drop_database(name)


class UnavailableProjects:
    """Small fake collection that behaves like an unavailable MongoDB server."""

    def insert_one(self, document):
        raise ServerSelectionTimeoutError("MongoDB is unavailable")

    def find(self):
        raise ServerSelectionTimeoutError("MongoDB is unavailable")


class UnavailableDatabase:
    projects = UnavailableProjects()


def test_mongodb_failure_returns_503_for_create_and_list(monkeypatch):
    monkeypatch.setattr(connection, "database", UnavailableDatabase())
    payload = {
        "name": "Example App",
        "repository_url": "https://github.com/example/deploysphere",
    }

    create_response = http.post("/api/projects", json=payload)
    list_response = http.get("/api/projects")

    assert create_response.status_code == 503
    assert list_response.status_code == 503


def _insert_listing_project(database, name: str) -> tuple[ObjectId, datetime]:
    """Insert a project directly for focused listing tests."""
    project_id = ObjectId()
    now = datetime.now(timezone.utc)
    database.projects.insert_one(
        {
            "_id": project_id,
            "name": name,
            "repository_url": f"https://github.com/example/{name.lower().replace(' ', '-')}",
            "branch": "main",
            "project_type": "unknown",
            "deployment_config": {},
            "created_at": now,
            "updated_at": now,
        }
    )
    return project_id, now


def _insert_listing_workspace(
    database,
    project_id: ObjectId,
    now: datetime,
    *,
    status: str = "cloned",
    commit_hash: str = "a" * 40,
) -> ObjectId:
    """Insert a workspace with a private path that must not appear in the API."""
    return database.repository_workspaces.insert_one(
        {
            "project_id": project_id,
            "status": status,
            "repository_url": "https://github.com/example/project",
            "branch": "main",
            "workspace_path": f"C:/private/workspaces/{project_id}",
            **({"commit_hash": commit_hash} if status == "cloned" else {}),
            "created_at": now,
            "updated_at": now,
        }
    ).inserted_id


def _insert_listing_detection(
    database,
    project_id: ObjectId,
    workspace_id: ObjectId,
    now: datetime,
    *,
    status: str = "completed",
    commit_hash: str = "a" * 40,
) -> ObjectId:
    """Insert a detection with internal fields to check response filtering."""
    return database.application_detections.insert_one(
        {
            "project_id": project_id,
            "workspace_id": workspace_id,
            "attempt_id": "private-attempt-token",
            "status": status,
            "application_type": "python",
            "build_method": "dockerfile",
            "detected_application_types": ["python"],
            "detected_build_methods": ["native", "dockerfile"],
            "detected_files": ["api/requirements.txt", "Dockerfile"],
            "confidence": 0.82,
            "warnings": ["Filename detection does not prove deployability."],
            "commit_hash": commit_hash,
            "created_at": now,
            "updated_at": now,
            "reservation_expires_at": now,
            **(
                {
                    "error": {
                        "code": "scan_failed",
                        "message": "Safe failure summary.",
                        "occurred_at": now,
                    }
                }
                if status == "failed"
                else {}
            ),
        }
    ).inserted_id


def test_project_without_workspace_has_null_workspace_and_detection(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    _insert_listing_project(database, "No workspace")
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        assert response.json()[0]["workspace"] is None
        assert response.json()[0]["detection"] is None
    finally:
        client.drop_database(name)


def test_cloned_project_without_detection_has_null_detection(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "No detection")
    _insert_listing_workspace(database, project_id, now)
    # A record tied to another workspace must not be associated with this one.
    _insert_listing_detection(database, project_id, ObjectId(), now)
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        assert response.json()[0]["workspace"]["status"] == "cloned"
        assert response.json()[0]["detection"] is None
    finally:
        client.drop_database(name)


def test_failed_clone_has_no_detection_summary(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "Failed clone")
    _insert_listing_workspace(database, project_id, now, status="failed")
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        assert response.json()[0]["workspace"]["status"] == "failed"
        assert response.json()[0]["detection"] is None
    finally:
        client.drop_database(name)


def test_completed_detection_summary_is_safe_and_complete(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "Completed detection")
    workspace_id = _insert_listing_workspace(database, project_id, now)
    _insert_listing_detection(database, project_id, workspace_id, now)
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        item = response.json()[0]
        detection = item["detection"]
        assert detection["status"] == "completed"
        assert detection["application_type"] == "python"
        assert detection["build_method"] == "dockerfile"
        assert detection["detected_application_types"] == ["python"]
        assert detection["detected_build_methods"] == ["native", "dockerfile"]
        assert detection["detected_files"] == ["api/requirements.txt", "Dockerfile"]
        assert detection["confidence"] == 0.82
        assert detection["commit_hash"] == "a" * 40
        assert "error" not in detection
        serialized = response.text
        assert "private-attempt-token" not in serialized
        assert "reservation_expires_at" not in serialized
        assert "private/workspaces" not in serialized
    finally:
        client.drop_database(name)


def test_failed_detection_summary_excludes_internal_error(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "Failed detection")
    workspace_id = _insert_listing_workspace(database, project_id, now)
    _insert_listing_detection(database, project_id, workspace_id, now, status="failed")
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        detection = response.json()[0]["detection"]
        assert detection["status"] == "failed"
        assert "error" not in detection
    finally:
        client.drop_database(name)


def test_listing_filters_unsafe_stored_detection_paths(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "Unsafe stored paths")
    workspace_id = _insert_listing_workspace(database, project_id, now)
    _insert_listing_detection(database, project_id, workspace_id, now)
    database.application_detections.update_one(
        {"project_id": project_id},
        {"$set": {"detected_files": ["Dockerfile", "C:\\private\\host", "../../outside"]}},
    )
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        assert response.json()[0]["detection"]["detected_files"] == ["Dockerfile"]
        assert "C:\\private\\host" not in response.text
    finally:
        client.drop_database(name)


def test_detection_for_an_older_commit_is_suppressed(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "Stale detection")
    workspace_id = _insert_listing_workspace(
        database, project_id, now, commit_hash="b" * 40
    )
    _insert_listing_detection(
        database, project_id, workspace_id, now, commit_hash="a" * 40
    )
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        assert response.json()[0]["detection"] is None
    finally:
        client.drop_database(name)


def test_detection_summaries_are_associated_with_their_projects(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    first_id, first_now = _insert_listing_project(database, "First association")
    second_id, second_now = _insert_listing_project(database, "Second association")
    first_workspace = _insert_listing_workspace(database, first_id, first_now)
    second_workspace = _insert_listing_workspace(database, second_id, second_now)
    _insert_listing_detection(database, first_id, first_workspace, first_now)
    _insert_listing_detection(
        database,
        second_id,
        second_workspace,
        second_now,
        status="failed",
        commit_hash="a" * 40,
    )
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        by_id = {item["id"]: item for item in response.json()}
        assert by_id[str(first_id)]["detection"]["status"] == "completed"
        assert by_id[str(second_id)]["detection"]["status"] == "failed"
    finally:
        client.drop_database(name)


def test_project_listing_uses_constant_batch_query_count(monkeypatch):
    name, database = _create_test_database(monkeypatch)
    for number in range(4):
        project_id, now = _insert_listing_project(database, f"Batch project {number}")
        workspace_id = _insert_listing_workspace(database, project_id, now)
        _insert_listing_detection(database, project_id, workspace_id, now)

    counts = {"projects": 0, "repository_workspaces": 0, "application_detections": 0}
    collection_type = type(database.projects)
    original_find = collection_type.find

    def count_find(collection, *args, **kwargs):
        if collection.name in counts:
            counts[collection.name] += 1
        return original_find(collection, *args, **kwargs)

    monkeypatch.setattr(collection_type, "find", count_find)
    try:
        response = http.get("/api/projects")

        assert response.status_code == 200
        assert len(response.json()) == 4
        assert counts == {
            "projects": 1,
            "repository_workspaces": 1,
            "application_detections": 1,
        }
    finally:
        client.drop_database(name)


@pytest.mark.parametrize("failed_collection", ["repository_workspaces", "application_detections"])
def test_mongodb_failure_loading_workspaces_or_detections_returns_503(
    monkeypatch, failed_collection
):
    name, database = _create_test_database(monkeypatch)
    project_id, now = _insert_listing_project(database, "Database failure")
    workspace_id = _insert_listing_workspace(database, project_id, now)
    _insert_listing_detection(database, project_id, workspace_id, now)
    collection_type = type(database.projects)
    original_find = collection_type.find

    def fail_find(collection, *args, **kwargs):
        if collection.name == failed_collection:
            raise PyMongoError("database details must not be returned")
        return original_find(collection, *args, **kwargs)

    monkeypatch.setattr(collection_type, "find", fail_find)
    try:
        response = http.get("/api/projects")
        assert response.status_code == 503
        assert "database details" not in response.text
    finally:
        client.drop_database(name)
