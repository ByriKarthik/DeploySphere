"""Mocked API tests for project-scoped deployment endpoints."""

from datetime import datetime, timezone
from uuid import uuid4

from bson import ObjectId
from fastapi.testclient import TestClient
import pytest

from backend.database import connection
from backend.database.connection import client
from backend.database.initialize import initialize_database
from backend.main import app
from backend.services import deployment_coordinator as coordinator
from backend.services import docker_runtime


http = TestClient(app)


@pytest.fixture
def api_database(monkeypatch):
    """Provide an isolated project database for the deployment API."""
    name = f"ds_deployment_api_{uuid4().hex}"
    database = client[name]
    initialize_database(database)
    monkeypatch.setattr(connection, "database", database)

    now = datetime.now(timezone.utc)
    project_id = ObjectId()
    database.projects.insert_one(
        {
            "_id": project_id,
            "name": "Deployment API project",
            "repository_url": f"https://github.com/example/{project_id}",
            "branch": "main",
            "project_type": "unknown",
            "deployment_config": {},
            "created_at": now,
            "updated_at": now,
        }
    )
    try:
        yield database, project_id
    finally:
        client.drop_database(name)


def _deployment_document(project_id: ObjectId, *, status: str = "running") -> dict:
    now = datetime.now(timezone.utc)
    return {
        "_id": ObjectId(),
        "project_id": project_id,
        "image_build_id": ObjectId(),
        "workspace_id": ObjectId(),
        "detection_id": ObjectId(),
        "attempt_id": "internal-attempt",
        "idempotency_key": "internal-idempotency-key",
        "status": status,
        "commit_hash": "a" * 40,
        "docker_image": "deploysphere/build:" + "b" * 32,
        "container_id": "c" * 64,
        "container_name": "deploysphere-private",
        "container_port": 8080,
        "allocated_port": 49152,
        "deployment_url": "http://127.0.0.1:49152",
        "created_at": now,
        "updated_at": now,
        "started_at": now,
    }


@pytest.mark.parametrize(("replay", "expected_status"), [(False, 201), (True, 200)])
def test_create_deployment_uses_explicit_build_and_idempotency(
    api_database, monkeypatch, replay, expected_status
):
    _, project_id = api_database
    document = _deployment_document(project_id)
    calls = []

    def fake_deploy(database, **kwargs):
        calls.append((database, kwargs))
        return document, replay

    monkeypatch.setattr(coordinator, "deploy_project", fake_deploy)
    response = http.post(
        f"/api/projects/{project_id}/deployments",
        json={"image_build_id": str(document["image_build_id"]), "container_port": 8080},
        headers={"Idempotency-Key": "request-1"},
    )

    assert response.status_code == expected_status
    assert len(calls) == 1
    assert calls[0][1] == {
        "project_id": project_id,
        "image_build_id": str(document["image_build_id"]),
        "container_port": 8080,
        "idempotency_key": "request-1",
    }
    payload = response.json()
    assert payload["id"] == str(document["_id"])
    assert payload["image_build_id"] == str(document["image_build_id"])
    assert "attempt_id" not in payload
    assert "idempotency_key" not in payload
    assert "container_id" not in payload
    assert "workspace_id" not in payload
    assert "detection_id" not in payload


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        ({"container_port": 8080}, {"Idempotency-Key": "request-1"}),
        ({"image_build_id": str(ObjectId())}, {"Idempotency-Key": "request-1"}),
        (
            {"image_build_id": str(ObjectId()), "container_port": True},
            {"Idempotency-Key": "request-1"},
        ),
        (
            {"image_build_id": str(ObjectId()), "container_port": 8080},
            {},
        ),
    ],
)
def test_create_deployment_requires_valid_explicit_request(
    api_database, monkeypatch, body, headers
):
    _, project_id = api_database
    monkeypatch.setattr(
        coordinator,
        "deploy_project",
        lambda *args, **kwargs: pytest.fail("Coordinator must not run for invalid requests"),
    )

    response = http.post(
        f"/api/projects/{project_id}/deployments",
        json=body,
        headers=headers,
    )

    assert response.status_code == 422


def test_deployment_history_is_project_scoped_and_bounded(api_database, monkeypatch):
    database, project_id = api_database
    document = _deployment_document(project_id)
    calls = []

    def fake_list(database_arg, project_arg, **kwargs):
        calls.append((database_arg, project_arg, kwargs))
        return [document]

    monkeypatch.setattr(coordinator, "list_deployments", fake_list)
    response = http.get(
        f"/api/projects/{project_id}/deployments?skip=10&limit=5"
    )

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert calls == [(database, project_id, {"skip": 10, "limit": 5})]

    invalid = http.get(f"/api/projects/{project_id}/deployments?limit=101")
    assert invalid.status_code == 422


def test_deployment_history_accepts_maximum_skip_and_rejects_larger_value(
    api_database, monkeypatch
):
    database, project_id = api_database
    calls = []

    def fake_list(database_arg, project_arg, **kwargs):
        calls.append((database_arg, project_arg, kwargs))
        return []

    monkeypatch.setattr(coordinator, "list_deployments", fake_list)
    maximum = http.get(f"/api/projects/{project_id}/deployments?skip=10000")
    too_large = http.get(f"/api/projects/{project_id}/deployments?skip=10001")

    assert maximum.status_code == 200
    assert calls == [(database, project_id, {"skip": 10000, "limit": 50})]
    assert too_large.status_code == 422
    assert len(calls) == 1


def test_get_deployment_is_scoped_and_not_found_is_404(api_database, monkeypatch):
    _, project_id = api_database
    deployment_id = ObjectId()
    calls = []

    def fake_get(database, requested_project, requested_deployment):
        calls.append((requested_project, requested_deployment))
        return None

    monkeypatch.setattr(coordinator, "get_deployment", fake_get)
    response = http.get(
        f"/api/projects/{project_id}/deployments/{deployment_id}"
    )

    assert response.status_code == 404
    assert calls == [(project_id, deployment_id)]


def test_active_deployment_route_precedes_deployment_id_route(api_database, monkeypatch):
    _, project_id = api_database
    document = _deployment_document(project_id)
    calls = []
    monkeypatch.setattr(
        coordinator,
        "get_active_deployment",
        lambda database, requested_project: (
            calls.append(requested_project) or document
        ),
    )

    response = http.get(f"/api/projects/{project_id}/deployments/active")

    assert response.status_code == 200
    assert response.json()["id"] == str(document["_id"])
    assert calls == [project_id]


@pytest.mark.parametrize(
    ("action", "service_name", "expected_status"),
    [
        ("stop", "stop_deployment", "stopped"),
        ("restart", "restart_deployment", "running"),
    ],
)
def test_lifecycle_routes_delegate_to_coordinator(
    api_database, monkeypatch, action, service_name, expected_status
):
    database, project_id = api_database
    document = _deployment_document(project_id, status=expected_status)
    deployment_id = document["_id"]
    calls = []

    def fake_lifecycle(database_arg, requested_project, requested_deployment):
        calls.append((database_arg, requested_project, requested_deployment))
        return document

    monkeypatch.setattr(coordinator, service_name, fake_lifecycle)
    response = http.post(
        f"/api/projects/{project_id}/deployments/{deployment_id}/{action}"
    )

    assert response.status_code == 200
    assert response.json()["status"] == expected_status
    assert calls == [(database, project_id, deployment_id)]


def test_missing_project_is_rejected_before_deployment_lookup(api_database, monkeypatch):
    _, project_id = api_database
    monkeypatch.setattr(
        coordinator,
        "get_active_deployment",
        lambda *args, **kwargs: pytest.fail("Deployment service must not run"),
    )

    response = http.get(f"/api/projects/{ObjectId()}/deployments/active")

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("operation", "initial_status", "runtime_result", "expected_status"),
    [
        (
            "stop",
            "running",
            docker_runtime.RuntimeResult(
                operation="stop",
                outcome="succeeded",
                container_id="c" * 64,
                state="exited",
                owned=True,
            ),
            "stopped",
        ),
        (
            "restart",
            "stopped",
            docker_runtime.RuntimeResult(
                operation="restart",
                outcome="succeeded",
                container_id="c" * 64,
                allocated_port=49153,
                state="running",
                owned=True,
            ),
            "running",
        ),
        (
            "stop",
            "running",
            docker_runtime.RuntimeResult(
                operation="stop",
                outcome="indeterminate",
                container_id="c" * 64,
                owned=True,
                error=docker_runtime.RuntimeError(
                    "docker_outcome_indeterminate",
                    "The Docker stop outcome could not be confirmed.",
                ),
            ),
            "unknown",
        ),
    ],
)
def test_coordinator_lifecycle_uses_mocked_runtime_once(
    api_database,
    monkeypatch,
    operation,
    initial_status,
    runtime_result,
    expected_status,
):
    database, project_id = api_database
    document = _deployment_document(project_id, status=initial_status)
    document["container_name"] = f"deploysphere-{document['_id']}"
    database.deployments.insert_one(document)
    calls = []

    def fake_runtime(**kwargs):
        calls.append(kwargs)
        return runtime_result

    runtime_function = (
        "stop_container" if operation == "stop" else "restart_container"
    )
    monkeypatch.setattr(coordinator.docker_runtime, runtime_function, fake_runtime)
    service_function = (
        coordinator.stop_deployment
        if operation == "stop"
        else coordinator.restart_deployment
    )

    saved = service_function(database, project_id, document["_id"])

    assert saved["status"] == expected_status
    assert len(calls) == 1
    assert calls[0]["container_id"] == document["container_id"]
    if operation == "stop" and expected_status == "stopped":
        assert "allocated_port" not in saved
        assert "deployment_url" not in saved
    if operation == "restart":
        assert saved["allocated_port"] == 49153
        assert saved["deployment_url"] == "http://127.0.0.1:49153"
