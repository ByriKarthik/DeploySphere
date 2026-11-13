"""End-to-end deployment API tests using isolated MongoDB and mocked engines."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4

from bson import ObjectId
from fastapi.testclient import TestClient
import pytest
import pymongo.collection
from pymongo.errors import PyMongoError

from backend.database import connection
from backend.database.connection import client
from backend.database.initialize import initialize_database
from backend.main import app
from backend.routers import project_workspaces
from backend.services import application_detection, deployment_coordinator, image_builds
from backend.services.application_detector import DetectionResult
from backend.services.docker_image_builder import BuildResult
from backend.services.docker_runtime import RuntimeResult
from backend.services.git_clone import CloneResult


http = TestClient(app)
IMAGE_TAG = "deploysphere/build:" + "d" * 32
CONTAINER_ID = "c" * 64


@pytest.fixture
def isolated_api_database(monkeypatch, tmp_path):
    """Direct API access to a uniquely named disposable database."""
    database_name = f"ds_deployment_e2e_{uuid4().hex}"
    assert database_name not in client.list_database_names()
    database = client[database_name]
    initialize_database(database)
    monkeypatch.setattr(connection, "database", database)
    monkeypatch.setenv("DEPLOYSPHERE_WORKSPACE_ROOT", str(tmp_path.resolve()))
    try:
        yield database, tmp_path.resolve()
    finally:
        client.drop_database(database_name)


def _successful_build_result():
    return BuildResult(
        success=True,
        image_tag=None,
        image_id="sha256:" + "b" * 64,
        duration_seconds=0.25,
        log_bytes=0,
        log_line_count=0,
        log_truncated=False,
    )


def _runtime_result(operation, *, state, allocated_port=None):
    return RuntimeResult(
        operation=operation,
        outcome="succeeded",
        container_id=CONTAINER_ID,
        container_name="deploysphere-mocked",
        container_port=8080,
        allocated_port=allocated_port,
        state=state,
        owned=True,
    )


def test_complete_deployment_api_workflow(isolated_api_database, monkeypatch):
    database, workspace_root = isolated_api_database
    clone_calls = []
    detection_calls = []
    build_calls = []
    runtime_calls = {"create": [], "start": [], "stop": [], "restart": []}

    def fake_clone(repository_url, branch, project_id, *, workspace_root):
        destination = Path(workspace_root) / str(project_id)
        destination.mkdir(parents=True)
        (destination / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        clone_calls.append((repository_url, branch, str(project_id)))
        return CloneResult(str(destination), "a" * 40)

    def fake_detect(path):
        detection_calls.append(path)
        return DetectionResult(
            application_type="python",
            build_method="dockerfile",
            detected_application_types=["python"],
            detected_build_methods=["dockerfile", "native"],
            detected_files=["Dockerfile", "requirements.txt"],
            confidence=0.95,
            warnings=[],
        )

    def fake_build(root, context, dockerfile, tag, detection):
        build_calls.append((root, context, dockerfile, tag, detection.application_type))
        return _successful_build_result()

    def fake_create(image_tag, project_id, deployment_id, container_port, timeout_seconds=30):
        runtime_calls["create"].append((image_tag, project_id, deployment_id, container_port))
        return _runtime_result("create", state="created")

    def fake_start(container_id, project_id, deployment_id, timeout_seconds=30):
        runtime_calls["start"].append((container_id, project_id, deployment_id))
        return _runtime_result("start", state="running", allocated_port=49152)

    def fake_stop(container_id, project_id, deployment_id, timeout_seconds=30):
        runtime_calls["stop"].append((container_id, project_id, deployment_id))
        return _runtime_result("stop", state="exited")

    def fake_restart(container_id, project_id, deployment_id, timeout_seconds=30):
        runtime_calls["restart"].append((container_id, project_id, deployment_id))
        return _runtime_result("restart", state="running", allocated_port=49153)

    monkeypatch.setattr(project_workspaces, "clone_repository", fake_clone)
    monkeypatch.setattr(application_detection, "detect_application", fake_detect)
    monkeypatch.setattr(image_builds, "build_docker_image", fake_build)
    monkeypatch.setattr(deployment_coordinator.docker_runtime, "create_container", fake_create)
    monkeypatch.setattr(deployment_coordinator.docker_runtime, "start_container", fake_start)
    monkeypatch.setattr(deployment_coordinator.docker_runtime, "stop_container", fake_stop)
    monkeypatch.setattr(deployment_coordinator.docker_runtime, "restart_container", fake_restart)

    registration = http.post(
        "/api/projects",
        json={
            "name": "End-to-end deployment",
            "repository_url": f"https://github.com/example/workflow-{uuid4().hex}",
            "branch": "main",
        },
    )
    assert registration.status_code == 201
    project = registration.json()
    project_id = project["id"]

    cloned = http.post(f"/api/projects/{project_id}/clone")
    assert cloned.status_code == 201
    assert cloned.json()["status"] == "cloned"

    detection = http.post(f"/api/projects/{project_id}/detect")
    assert detection.status_code == 201
    assert detection.json()["status"] == "completed"

    build = http.post(f"/api/projects/{project_id}/build")
    assert build.status_code == 201
    assert build.json()["status"] == "succeeded"
    build_id = build.json()["id"]

    deployment_request = {
        "image_build_id": build_id,
        "container_port": 8080,
    }
    headers = {"Idempotency-Key": "e2e-deployment-request"}
    created = http.post(
        f"/api/projects/{project_id}/deployments",
        json=deployment_request,
        headers=headers,
    )
    assert created.status_code == 201
    deployment = created.json()
    deployment_id = deployment["id"]
    assert deployment["status"] == "running"
    assert deployment["deployment_url"] == "http://127.0.0.1:49152"

    replay = http.post(
        f"/api/projects/{project_id}/deployments",
        json=deployment_request,
        headers=headers,
    )
    assert replay.status_code == 200
    assert replay.json()["id"] == deployment_id

    active = http.get(f"/api/projects/{project_id}/deployments/active")
    history = http.get(f"/api/projects/{project_id}/deployments")
    details = http.get(f"/api/projects/{project_id}/deployments/{deployment_id}")
    assert active.status_code == 200
    assert active.json()["id"] == deployment_id
    assert history.status_code == 200
    assert [item["id"] for item in history.json()] == [deployment_id]
    assert details.status_code == 200
    assert details.json()["status"] == "running"

    stopped = http.post(
        f"/api/projects/{project_id}/deployments/{deployment_id}/stop"
    )
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert http.get(f"/api/projects/{project_id}/deployments/active").status_code == 404

    restarted = http.post(
        f"/api/projects/{project_id}/deployments/{deployment_id}/restart"
    )
    assert restarted.status_code == 200
    assert restarted.json()["status"] == "running"
    assert restarted.json()["deployment_url"] == "http://127.0.0.1:49153"
    refreshed_details = http.get(
        f"/api/projects/{project_id}/deployments/{deployment_id}"
    )
    assert refreshed_details.json()["status"] == "running"
    assert refreshed_details.json()["deployment_url"] == "http://127.0.0.1:49153"

    assert clone_calls == [
        (project["repository_url"], "main", project_id)
    ]
    assert len(detection_calls) == 1
    assert len(build_calls) == 1
    assert len(runtime_calls["create"]) == len(runtime_calls["start"]) == 1
    assert len(runtime_calls["stop"]) == len(runtime_calls["restart"]) == 1
    assert database.deployments.count_documents({"project_id": ObjectId(project_id)}) == 1


def test_deployment_api_rejects_malformed_and_foreign_resource_ids(
    isolated_api_database, monkeypatch
):
    database, _ = isolated_api_database
    first_project = http.post(
        "/api/projects",
        json={
            "name": "First project",
            "repository_url": f"https://github.com/example/first-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    second_project = http.post(
        "/api/projects",
        json={
            "name": "Second project",
            "repository_url": f"https://github.com/example/second-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    now = datetime.now(timezone.utc)
    build_id = ObjectId()
    database.image_builds.insert_one(
        {
            "_id": build_id,
            "project_id": ObjectId(second_project["id"]),
            "workspace_id": ObjectId(),
            "detection_id": ObjectId(),
            "commit_hash": "a" * 40,
            "attempt_id": uuid4().hex,
            "status": "succeeded",
            "image_tag": IMAGE_TAG,
            "build_context": ".",
            "dockerfile_path": "Dockerfile",
            "build_duration": 1.0,
            "created_at": now,
            "updated_at": now,
        }
    )
    monkeypatch.setattr(
        deployment_coordinator.docker_runtime,
        "create_container",
        lambda *args, **kwargs: pytest.fail("Docker must not run for foreign build"),
    )

    malformed_project = http.post(
        "/api/projects/not-an-object-id/deployments",
        json={"image_build_id": str(build_id), "container_port": 8080},
        headers={"Idempotency-Key": "malformed-project"},
    )
    malformed_build = http.post(
        f"/api/projects/{first_project['id']}/deployments",
        json={"image_build_id": "not-an-object-id", "container_port": 8080},
        headers={"Idempotency-Key": "malformed-build"},
    )
    foreign_build = http.post(
        f"/api/projects/{first_project['id']}/deployments",
        json={"image_build_id": str(build_id), "container_port": 8080},
        headers={"Idempotency-Key": "foreign-build"},
    )
    malformed_deployment = http.get(
        f"/api/projects/{first_project['id']}/deployments/not-an-object-id"
    )
    foreign_deployment = _insert_running_deployment(
        database, ObjectId(second_project["id"])
    )
    foreign_deployment_id = foreign_deployment["_id"]
    database.deployments.update_one(
        {"_id": foreign_deployment_id}, {"$set": {"status": "failed"}}
    )
    foreign_deployment = http.get(
        f"/api/projects/{first_project['id']}/deployments/{foreign_deployment_id}"
    )
    scoped_history = http.get(f"/api/projects/{first_project['id']}/deployments")

    assert malformed_project.status_code == 400
    assert malformed_build.status_code == 400
    assert foreign_build.status_code == 409
    assert malformed_deployment.status_code == 400
    assert foreign_deployment.status_code == 404
    assert scoped_history.status_code == 200
    assert scoped_history.json() == []


def test_image_build_cannot_use_another_projects_workspace(
    isolated_api_database, monkeypatch
):
    database, workspace_root = isolated_api_database
    requesting_project = http.post(
        "/api/projects",
        json={
            "name": "Workspace owner",
            "repository_url": f"https://github.com/example/workspace-owner-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    workspace_owner = http.post(
        "/api/projects",
        json={
            "name": "Different workspace owner",
            "repository_url": f"https://github.com/example/workspace-other-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    now = datetime.now(timezone.utc)
    owner_id = ObjectId(workspace_owner["id"])
    owner_workspace_path = workspace_root / workspace_owner["id"]
    owner_workspace_path.mkdir()
    (owner_workspace_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    database.repository_workspaces.insert_one(
        {
            "_id": ObjectId(),
            "project_id": owner_id,
            "status": "cloned",
            "repository_url": workspace_owner["repository_url"],
            "branch": "main",
            "workspace_path": str(owner_workspace_path),
            "commit_hash": "a" * 40,
            "created_at": now,
            "updated_at": now,
        }
    )
    monkeypatch.setattr(
        image_builds,
        "build_docker_image",
        lambda *args, **kwargs: pytest.fail("A foreign workspace must not be built"),
    )

    response = http.post(f"/api/projects/{requesting_project['id']}/build")

    assert response.status_code == 409
    assert database.image_builds.count_documents({}) == 0


def _insert_running_deployment(database, project_id):
    now = datetime.now(timezone.utc)
    deployment_id = ObjectId()
    document = {
        "_id": deployment_id,
        "project_id": project_id,
        "image_build_id": ObjectId(),
        "workspace_id": ObjectId(),
        "detection_id": ObjectId(),
        "attempt_id": uuid4().hex,
        "idempotency_key": uuid4().hex,
        "status": "running",
        "docker_image": IMAGE_TAG,
        "container_id": CONTAINER_ID,
        "container_name": f"deploysphere-{deployment_id}",
        "container_port": 8080,
        "allocated_port": 49152,
        "deployment_url": "http://127.0.0.1:49152",
        "created_at": now,
        "updated_at": now,
        "started_at": now,
    }
    database.deployments.insert_one(document)
    return document


def test_stop_restart_concurrency_allows_only_one_runtime_operation(
    isolated_api_database, monkeypatch
):
    database, _ = isolated_api_database
    project = http.post(
        "/api/projects",
        json={
            "name": "Lifecycle concurrency",
            "repository_url": f"https://github.com/example/lifecycle-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    project_id = ObjectId(project["id"])
    deployment = _insert_running_deployment(database, project_id)
    stop_started = Event()
    allow_stop_to_finish = Event()
    runtime_calls = []

    def blocking_stop(**kwargs):
        runtime_calls.append("stop")
        stop_started.set()
        assert allow_stop_to_finish.wait(timeout=10)
        return _runtime_result("stop", state="exited")

    def unexpected_restart(**kwargs):
        runtime_calls.append("restart")
        return _runtime_result("restart", state="running", allocated_port=49153)

    monkeypatch.setattr(deployment_coordinator.docker_runtime, "stop_container", blocking_stop)
    monkeypatch.setattr(
        deployment_coordinator.docker_runtime, "restart_container", unexpected_restart
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            stop_future = pool.submit(
                http.post,
                f"/api/projects/{project['id']}/deployments/{deployment['_id']}/stop",
            )
            assert stop_started.wait(timeout=5)
            concurrent_restart = http.post(
                f"/api/projects/{project['id']}/deployments/{deployment['_id']}/restart"
            )
            allow_stop_to_finish.set()
            stopped = stop_future.result(timeout=10)
    finally:
        allow_stop_to_finish.set()

    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert concurrent_restart.status_code == 409
    assert runtime_calls == ["stop"]


def test_persistence_failure_after_accepted_stop_marks_outcome_unknown(
    isolated_api_database, monkeypatch
):
    database, _ = isolated_api_database
    project = http.post(
        "/api/projects",
        json={
            "name": "Stop persistence failure",
            "repository_url": f"https://github.com/example/stop-failure-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    project_id = ObjectId(project["id"])
    deployment = _insert_running_deployment(database, project_id)
    runtime_calls = []
    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update

    def fail_final_stop(self, filter_dict, update, *args, **kwargs):
        if (
            self.name == "deployments"
            and filter_dict.get("status") == "stopping"
            and update.get("$set", {}).get("status") == "stopped"
        ):
            raise PyMongoError("Simulated state persistence failure")
        return original_find_one_and_update(self, filter_dict, update, *args, **kwargs)

    def accepted_stop(**kwargs):
        runtime_calls.append(kwargs)
        return _runtime_result("stop", state="exited")

    monkeypatch.setattr(
        pymongo.collection.Collection, "find_one_and_update", fail_final_stop
    )
    monkeypatch.setattr(deployment_coordinator.docker_runtime, "stop_container", accepted_stop)

    response = http.post(
        f"/api/projects/{project['id']}/deployments/{deployment['_id']}/stop"
    )

    saved = database.deployments.find_one({"_id": deployment["_id"]})
    assert response.status_code == 503
    assert saved["status"] == "unknown"
    assert saved["error"]["code"] == "mongodb_persistence_failed"
    assert len(runtime_calls) == 1


def test_invalid_lifecycle_state_does_not_invoke_runtime(
    isolated_api_database, monkeypatch
):
    database, _ = isolated_api_database
    project = http.post(
        "/api/projects",
        json={
            "name": "Invalid lifecycle state",
            "repository_url": f"https://github.com/example/state-{uuid4().hex}",
            "branch": "main",
        },
    ).json()
    deployment = _insert_running_deployment(database, ObjectId(project["id"]))
    database.deployments.update_one(
        {"_id": deployment["_id"]}, {"$set": {"status": "stopped"}}
    )
    monkeypatch.setattr(
        deployment_coordinator.docker_runtime,
        "stop_container",
        lambda *args, **kwargs: pytest.fail("Stop must not run for a stopped deployment"),
    )

    response = http.post(
        f"/api/projects/{project['id']}/deployments/{deployment['_id']}/stop"
    )

    assert response.status_code == 409
    assert database.deployments.find_one({"_id": deployment["_id"]})["status"] == "stopped"
