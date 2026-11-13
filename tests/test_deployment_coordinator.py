"""Mocked tests for the deployment coordinator; no Docker daemon is used."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from bson import ObjectId
import pymongo.collection
import pytest
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.database.connection import client
from backend.database.initialize import initialize_database
from backend.services import deployment_coordinator as coordinator
from backend.services import docker_runtime

IMAGE_TAG = "deploysphere/build:" + "d" * 32
CONTAINER_ID = "c" * 64


@pytest.fixture
def disposable_database():
    """Create and clean up a uniquely named test database."""
    name = f"deploysphere_coord_test_{uuid4().hex}"
    database = client[name]
    initialize_database(database)
    try:
        yield database
    finally:
        client.drop_database(name)


def _seed_project_and_build(database, *, status="succeeded", image_tag=IMAGE_TAG, project_id=None):
    """Seed valid project and image-build records in MongoDB."""
    now = datetime.now(timezone.utc)
    p_id = project_id or ObjectId()
    w_id = ObjectId()
    d_id = ObjectId()
    b_id = ObjectId()

    database.projects.insert_one({
        "_id": p_id,
        "name": "Deploy test project",
        "repository_url": f"https://github.com/example/{p_id}",
        "branch": "main",
        "project_type": "unknown",
        "deployment_config": {"port": 8080},
        "created_at": now,
        "updated_at": now,
    })

    database.image_builds.insert_one({
        "_id": b_id,
        "project_id": p_id,
        "workspace_id": w_id,
        "detection_id": d_id,
        "commit_hash": "a" * 40,
        "attempt_id": uuid4().hex,
        "status": status,
        "image_tag": image_tag,
        "build_context": ".",
        "dockerfile_path": "Dockerfile",
        "build_duration": 2.5,
        "created_at": now,
        "updated_at": now,
    })

    return p_id, b_id, w_id, d_id


def _seed_unknown_deployment(database):
    """Insert an isolated unknown deployment for reconciliation tests."""
    p_id, b_id, w_id, d_id = _seed_project_and_build(database)
    deployment_id = ObjectId()
    attempt_id = uuid4().hex
    container_id = "e" * 64
    now = datetime.now(timezone.utc)
    database.deployments.insert_one({
        "_id": deployment_id,
        "project_id": p_id,
        "image_build_id": b_id,
        "workspace_id": w_id,
        "detection_id": d_id,
        "attempt_id": attempt_id,
        "idempotency_key": "unknown-deployment-key",
        "status": "unknown",
        "commit_hash": "a" * 40,
        "docker_image": IMAGE_TAG,
        "container_id": container_id,
        "container_name": f"deploysphere-{deployment_id}",
        "container_port": 8080,
        "error": {
            "code": "docker_outcome_indeterminate",
            "message": "The container start outcome is unknown.",
            "occurred_at": now,
        },
        "created_at": now,
        "updated_at": now,
    })
    return p_id, b_id, deployment_id, container_id, attempt_id


def _forbid_docker_runtime(monkeypatch):
    """Ensure reconciliation and reservation tests never invoke Docker."""
    for operation in (
        "create_container",
        "start_container",
        "stop_container",
        "restart_container",
    ):
        monkeypatch.setattr(
            coordinator.docker_runtime,
            operation,
            lambda *args, _operation=operation, **kwargs: pytest.fail(
                f"Docker runtime operation {_operation} must not be called"
            ),
        )


def _mock_runtime(
    monkeypatch,
    *,
    create_outcome="succeeded",
    create_error=None,
    create_container_id=CONTAINER_ID,
    start_outcome="succeeded",
    start_error=None,
    allocated_port=49152,
    start_state="running",
    on_start_hook=None,
):
    """Mock Docker runtime functions without invoking Docker."""
    create_calls = []
    start_calls = []

    def fake_create(image_tag, project_id, deployment_id, container_port, timeout_seconds=30):
        create_calls.append({
            "image_tag": image_tag,
            "project_id": project_id,
            "deployment_id": deployment_id,
            "container_port": container_port,
            "timeout_seconds": timeout_seconds,
        })
        return docker_runtime.RuntimeResult(
            operation="create",
            outcome=create_outcome,
            container_id=create_container_id if create_outcome == "succeeded" else (create_container_id if create_outcome == "indeterminate" else None),
            container_name=f"deploysphere-{deployment_id}",
            container_port=container_port,
            state="created" if create_outcome == "succeeded" else None,
            owned=True if create_outcome == "succeeded" else None,
            error=create_error,
        )

    def fake_start(container_id, project_id, deployment_id, timeout_seconds=30):
        start_calls.append({
            "container_id": container_id,
            "project_id": project_id,
            "deployment_id": deployment_id,
            "timeout_seconds": timeout_seconds,
        })
        if on_start_hook:
            on_start_hook(container_id, project_id, deployment_id)
        return docker_runtime.RuntimeResult(
            operation="start",
            outcome=start_outcome,
            container_id=container_id,
            container_name=f"deploysphere-{deployment_id}",
            allocated_port=allocated_port if start_outcome == "succeeded" else None,
            state=start_state if start_outcome == "succeeded" else None,
            owned=True if start_outcome == "succeeded" else None,
            error=start_error,
        )

    monkeypatch.setattr(coordinator.docker_runtime, "create_container", fake_create)
    monkeypatch.setattr(coordinator.docker_runtime, "start_container", fake_start)
    return create_calls, start_calls


# 1. Successful deployment
def test_successful_deployment_flow(disposable_database, monkeypatch):
    p_id, b_id, w_id, d_id = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(monkeypatch)

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-success-1",
        timeout_seconds=45,
    )

    assert is_replay is False
    assert doc["status"] == "running"
    assert doc["container_id"] == CONTAINER_ID
    assert doc["container_name"] == f"deploysphere-{doc['_id']}"
    assert doc["container_port"] == 8080
    assert doc["allocated_port"] == 49152
    assert doc["deployment_url"] == "http://127.0.0.1:49152"
    assert doc["started_at"] is not None
    assert doc["image_build_id"] == b_id
    assert doc["workspace_id"] == w_id
    assert doc["detection_id"] == d_id

    assert len(creates) == 1
    assert creates[0]["image_tag"] == IMAGE_TAG
    assert creates[0]["container_port"] == 8080
    assert creates[0]["timeout_seconds"] == 45

    assert len(starts) == 1
    assert starts[0]["container_id"] == CONTAINER_ID


# 2. Missing project
def test_deploy_project_missing_project(disposable_database, monkeypatch):
    _mock_runtime(monkeypatch)
    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=ObjectId(),
            image_build_id=ObjectId(),
            container_port=8080,
            idempotency_key="key-missing-proj",
        )
    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "project_not_found"


# 3. Missing build
def test_deploy_project_missing_build(disposable_database, monkeypatch):
    p_id, _, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)
    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=ObjectId(),
            container_port=8080,
            idempotency_key="key-missing-build",
        )
    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "image_build_not_found"


# 4. Build belonging to another project
def test_deploy_project_build_belonging_to_another_project(disposable_database, monkeypatch):
    p_id1, b_id1, _, _ = _seed_project_and_build(disposable_database)
    p_id2, _, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id2,
            image_build_id=b_id1,
            container_port=8080,
            idempotency_key="key-wrong-proj",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "image_build_mismatch"


# 5. Build not succeeded
@pytest.mark.parametrize("bad_status", ["failed", "building", "queued", "cancelled"])
def test_deploy_project_build_not_succeeded(disposable_database, monkeypatch, bad_status):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database, status=bad_status)
    _mock_runtime(monkeypatch)

    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-not-succeeded",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "image_build_not_succeeded"


# 6. Invalid image tag
@pytest.mark.parametrize("bad_tag", ["nginx:latest", "deploysphere/build:XYZ", "deploysphere/test:123"])
def test_deploy_project_invalid_image_tag(disposable_database, monkeypatch, bad_tag):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database, image_tag=bad_tag)
    _mock_runtime(monkeypatch)

    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-bad-tag",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "invalid_image_tag"


# 7. Invalid container port
@pytest.mark.parametrize("bad_port", [0, 65536, -1, "8080", True, 8080.5])
def test_deploy_project_invalid_container_port(disposable_database, bad_port):
    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=ObjectId(),
            image_build_id=ObjectId(),
            container_port=bad_port,
            idempotency_key="key-bad-port",
        )
    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "invalid_container_port"


# 8. Missing required build metadata
def test_deploy_project_missing_required_build_metadata(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    disposable_database.image_builds.update_one(
        {"_id": b_id}, {"$unset": {"workspace_id": ""}},
        bypass_document_validation=True,
    )
    _mock_runtime(monkeypatch)

    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-missing-meta",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "missing_build_metadata"


# 9. Same idempotency key and same parameters (idempotent replay)
def test_deploy_project_same_idempotency_key_same_parameters(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(monkeypatch)

    doc1, is_replay1 = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-replay-1",
    )
    assert is_replay1 is False

    doc2, is_replay2 = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-replay-1",
    )
    assert is_replay2 is True
    assert doc1["_id"] == doc2["_id"]
    assert len(creates) == 1
    assert len(starts) == 1


# 10. Same key with different parameters
def test_deploy_project_same_key_with_different_parameters(disposable_database, monkeypatch):
    p_id, b_id1, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id1,
        container_port=8080,
        idempotency_key="key-conflict-1",
    )

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id1,
            container_port=3000,
            idempotency_key="key-conflict-1",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "idempotency_conflict"


# 11. Two concurrent identical requests
def test_deploy_project_concurrent_identical_requests(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(monkeypatch)

    def run():
        return coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-concurrent-identical",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run)
        f2 = executor.submit(run)
        res1 = f1.result()
        res2 = f2.result()

    assert {res1[1], res2[1]} == {False, True}
    assert res1[0]["_id"] == res2[0]["_id"]
    assert len(creates) == 1
    assert len(starts) == 1


# 12. Two concurrent requests with different idempotency keys
def test_deploy_project_concurrent_different_keys_rejected(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    def run(key):
        try:
            return coordinator.deploy_project(
                disposable_database,
                project_id=p_id,
                image_build_id=b_id,
                container_port=8080,
                idempotency_key=key,
            )
        except coordinator.DeploymentConflictError as err:
            return err

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run, "key-diff-1")
        f2 = executor.submit(run, "key-diff-2")
        r1 = f1.result()
        r2 = f2.result()

    # One succeeds, the other gets active_deployment_exists
    types = {type(r1), type(r2)}
    assert coordinator.DeploymentConflictError in types
    assert tuple in types


# 13. Duplicate active deployment
def test_deploy_project_duplicate_active_deployment(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-active-1",
    )

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-active-2",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "active_deployment_exists"


def test_reconcile_unknown_deployment_as_failed_preserves_identity(
    disposable_database, monkeypatch
):
    p_id, _, deployment_id, container_id, attempt_id = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)
    before = disposable_database.deployments.find_one({"_id": deployment_id})

    reconciled = coordinator.reconcile_unknown_deployment_as_failed(
        disposable_database,
        p_id,
        deployment_id,
        container_id,
        attempt_id=attempt_id,
    )

    assert reconciled["status"] == "failed"
    assert reconciled["updated_at"] >= before["updated_at"]
    assert reconciled["error"]["code"] == coordinator.RECONCILIATION_ERROR_CODE
    assert reconciled["error"]["message"] == coordinator.RECONCILIATION_ERROR_MESSAGE
    for field in (
        "_id",
        "project_id",
        "image_build_id",
        "workspace_id",
        "detection_id",
        "attempt_id",
        "docker_image",
        "container_id",
        "container_name",
        "container_port",
    ):
        assert reconciled.get(field) == before.get(field)
    assert "image_id" not in reconciled


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": ObjectId()},
        {"deployment_id": ObjectId()},
        {"container_id": "f" * 64},
        {"attempt_id": "f" * 32},
    ],
)
def test_reconciliation_identifier_mismatch_conflicts_without_update(
    disposable_database, monkeypatch, overrides
):
    p_id, _, deployment_id, container_id, attempt_id = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)
    before = disposable_database.deployments.find_one({"_id": deployment_id})
    args = {
        "project_id": p_id,
        "deployment_id": deployment_id,
        "container_id": container_id,
        "attempt_id": attempt_id,
    }
    args.update(overrides)

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.reconcile_unknown_deployment_as_failed(
            disposable_database,
            args["project_id"],
            args["deployment_id"],
            args["container_id"],
            args["attempt_id"],
        )

    after = disposable_database.deployments.find_one({"_id": deployment_id})
    assert exc_info.value.code == "deployment_reconciliation_conflict"
    assert after == before


@pytest.mark.parametrize("attempt_id", ["", "not-an-attempt-id", None])
def test_reconciliation_rejects_invalid_attempt_ids(
    disposable_database, monkeypatch, attempt_id
):
    p_id, _, deployment_id, container_id, _ = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)
    before = disposable_database.deployments.find_one({"_id": deployment_id})

    with pytest.raises(coordinator.DeploymentRequestError) as exc_info:
        coordinator.reconcile_unknown_deployment_as_failed(
            disposable_database,
            p_id,
            deployment_id,
            container_id,
            attempt_id,
        )

    after = disposable_database.deployments.find_one({"_id": deployment_id})
    assert exc_info.value.code == "invalid_attempt_id"
    assert after == before


def test_reconciliation_requires_attempt_id(disposable_database, monkeypatch):
    p_id, _, deployment_id, container_id, _ = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)
    before = disposable_database.deployments.find_one({"_id": deployment_id})

    with pytest.raises(TypeError, match="attempt_id"):
        coordinator.reconcile_unknown_deployment_as_failed(
            disposable_database,
            p_id,
            deployment_id,
            container_id,
        )

    assert disposable_database.deployments.find_one({"_id": deployment_id}) == before


def test_reconciliation_rejects_non_unknown_status_without_overwrite(
    disposable_database, monkeypatch
):
    p_id, _, deployment_id, container_id, attempt_id = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)
    disposable_database.deployments.update_one(
        {"_id": deployment_id},
        {"$set": {"status": "running"}},
    )
    before = disposable_database.deployments.find_one({"_id": deployment_id})

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.reconcile_unknown_deployment_as_failed(
            disposable_database,
            p_id,
            deployment_id,
            container_id,
            attempt_id,
        )

    assert exc_info.value.code == "deployment_reconciliation_conflict"
    assert disposable_database.deployments.find_one({"_id": deployment_id}) == before


def test_concurrent_reconciliations_only_transition_matching_record_once(
    disposable_database, monkeypatch
):
    p_id, _, deployment_id, container_id, attempt_id = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)

    def reconcile():
        try:
            return coordinator.reconcile_unknown_deployment_as_failed(
                disposable_database,
                p_id,
                deployment_id,
                container_id,
                attempt_id,
            )
        except coordinator.DeploymentConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reconcile(), range(2)))

    assert sum(isinstance(result, dict) for result in results) == 1
    conflicts = [
        result
        for result in results
        if isinstance(result, coordinator.DeploymentConflictError)
    ]
    assert len(conflicts) == 1
    assert conflicts[0].code == "deployment_reconciliation_conflict"
    record = disposable_database.deployments.find_one({"_id": deployment_id})
    assert record["status"] == "failed"
    assert record["container_id"] == container_id


def test_reconciliation_releases_active_index_and_allows_next_reservation(
    disposable_database, monkeypatch
):
    p_id, build_id, deployment_id, container_id, attempt_id = _seed_unknown_deployment(
        disposable_database
    )
    _forbid_docker_runtime(monkeypatch)
    build = disposable_database.image_builds.find_one({"_id": build_id})

    with pytest.raises(coordinator.DeploymentConflictError) as active_conflict:
        coordinator._reserve_deployment(
            disposable_database, p_id, build, 8080, "next-deployment-key"
        )
    assert active_conflict.value.code == "active_deployment_exists"

    coordinator.reconcile_unknown_deployment_as_failed(
        disposable_database,
        p_id,
        deployment_id,
        container_id,
        attempt_id=attempt_id,
    )
    reservation, replay = coordinator._reserve_deployment(
        disposable_database, p_id, build, 8080, "next-deployment-key"
    )

    assert replay is False
    assert reservation["status"] == "queued"
    assert disposable_database.deployments.find_one({"_id": deployment_id})[
        "container_id"
    ] == container_id


@pytest.mark.parametrize(
    ("runtime_outcome", "runtime_state", "expected_status"),
    [
        ("succeeded", "exited", "stopped"),
        ("succeeded", "running", "unknown"),
        ("indeterminate", "exited", "unknown"),
    ],
)
def test_stop_coordinator_marks_stopped_only_after_confirmed_exit(
    monkeypatch, runtime_outcome, runtime_state, expected_status
):
    project_id = ObjectId()
    deployment_id = ObjectId()
    attempt_id = uuid4().hex
    container_id = "c" * 64
    record = {
        "_id": deployment_id,
        "project_id": project_id,
        "attempt_id": attempt_id,
        "status": "running",
        "container_id": container_id,
        "container_name": f"deploysphere-{deployment_id}",
        "container_port": 5000,
        "allocated_port": 49152,
        "deployment_url": "http://127.0.0.1:49152",
    }

    class InMemoryDeployments:
        def find_one(self, filter_document):
            return deepcopy(record) if all(
                record.get(key) == value for key, value in filter_document.items()
            ) else None

        def find_one_and_update(self, filter_document, update, **kwargs):
            if not all(
                record.get(key) == value for key, value in filter_document.items()
            ):
                return None
            record.update(deepcopy(update.get("$set", {})))
            for field in update.get("$unset", {}):
                record.pop(field, None)
            return deepcopy(record)

    database = SimpleNamespace(deployments=InMemoryDeployments())
    stop_runtime = Mock(return_value=docker_runtime.RuntimeResult(
        operation="stop",
        outcome=runtime_outcome,
        container_id=container_id,
        container_name=record["container_name"],
        state=runtime_state,
        owned=True,
    ))
    monkeypatch.setattr(coordinator.docker_runtime, "stop_container", stop_runtime)
    for operation in ("create_container", "start_container", "restart_container"):
        monkeypatch.setattr(
            coordinator.docker_runtime,
            operation,
            Mock(side_effect=AssertionError(f"{operation} must not be called")),
        )

    result = coordinator.stop_deployment(database, project_id, deployment_id)

    assert result["status"] == expected_status
    assert result["container_id"] == container_id
    assert stop_runtime.call_count == 1


# 14. Docker create failure
def test_deploy_project_docker_create_failure(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(
        monkeypatch,
        create_outcome="failed",
        create_error=docker_runtime.RuntimeError("docker_command_failed", "Failed to create container"),
    )

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-create-fail",
    )

    assert is_replay is False
    assert doc["status"] == "failed"
    assert doc["error"]["code"] == "docker_command_failed"
    assert len(creates) == 1
    assert len(starts) == 0


# 15. Docker create indeterminate outcome
def test_deploy_project_docker_create_indeterminate(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(
        monkeypatch,
        create_outcome="indeterminate",
        create_container_id="c" * 64,
        create_error=docker_runtime.RuntimeError("docker_outcome_indeterminate", "Timeout on create"),
    )

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-create-indet",
    )

    assert is_replay is False
    assert doc["status"] == "unknown"
    assert doc["container_id"] == "c" * 64
    assert doc["error"]["code"] == "docker_outcome_indeterminate"
    assert len(creates) == 1
    assert len(starts) == 0


# 16. MongoDB failure before Docker create
def test_deploy_project_mongodb_failure_before_docker_create(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, _ = _mock_runtime(monkeypatch)

    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update

    def fail_on_creating(self, filter_dict, *args, **kwargs):
        if self.name == "deployments" and filter_dict.get("status") == "queued":
            raise PyMongoError("Simulated write failure")
        return original_find_one_and_update(self, filter_dict, *args, **kwargs)

    monkeypatch.setattr(pymongo.collection.Collection, "find_one_and_update", fail_on_creating)

    with pytest.raises(coordinator.DeploymentStoreError):
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-mongo-pre-create",
        )
    assert len(creates) == 0


# 17. MongoDB failure persisting container ID — start must never be called
def test_deploy_project_mongodb_failure_persisting_container_id_halts_start(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(monkeypatch)

    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update

    def fail_on_starting(self, filter_dict, *args, **kwargs):
        if self.name == "deployments" and filter_dict.get("status") == "creating":
            raise PyMongoError("Simulated write failure on starting")
        return original_find_one_and_update(self, filter_dict, *args, **kwargs)

    monkeypatch.setattr(pymongo.collection.Collection, "find_one_and_update", fail_on_starting)

    with pytest.raises(coordinator.DeploymentStoreError):
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-mongo-save-cid",
        )

    assert len(creates) == 1
    assert len(starts) == 0  # CRITICAL: start was never called!

    saved = disposable_database.deployments.find_one({"project_id": p_id})
    assert saved["status"] == "unknown"
    assert saved["error"]["code"] == "mongodb_persistence_failed"


def test_emergency_unknown_does_not_overwrite_newer_terminal_deployment(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(monkeypatch)

    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update
    original_update_one = pymongo.collection.Collection.update_one
    newer_attempt_id = "newer-attempt"

    def supersede_before_persisting_container_id(self, filter_dict, *args, **kwargs):
        if self.name == "deployments" and filter_dict.get("status") == "creating":
            original_update_one(
                self,
                {
                    "_id": filter_dict["_id"],
                    "attempt_id": filter_dict["attempt_id"],
                    "status": "creating",
                },
                {"$set": {"attempt_id": newer_attempt_id, "status": "failed"}},
            )
            raise PyMongoError("Simulated ambiguous persistence failure")
        return original_find_one_and_update(self, filter_dict, *args, **kwargs)

    monkeypatch.setattr(
        pymongo.collection.Collection,
        "find_one_and_update",
        supersede_before_persisting_container_id,
    )

    with pytest.raises(coordinator.DeploymentStoreError):
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-emergency-newer-attempt",
        )

    saved = disposable_database.deployments.find_one({"project_id": p_id})
    assert saved["attempt_id"] == newer_attempt_id
    assert saved["status"] == "failed"
    assert "container_id" not in saved
    assert len(creates) == 1
    assert len(starts) == 0


# 18. Docker start failure
def test_deploy_project_docker_start_failure(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(
        monkeypatch,
        start_outcome="failed",
        start_error=docker_runtime.RuntimeError("docker_command_failed", "Failed to start container"),
    )

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-start-fail",
    )

    assert is_replay is False
    assert doc["status"] == "failed"
    assert doc["error"]["code"] == "docker_command_failed"
    assert len(creates) == 1
    assert len(starts) == 1


# 19. Docker start indeterminate outcome
def test_deploy_project_docker_start_indeterminate(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(
        monkeypatch,
        start_outcome="indeterminate",
        start_error=docker_runtime.RuntimeError("docker_outcome_indeterminate", "Timeout on start"),
    )

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-start-indet",
    )

    assert is_replay is False
    assert doc["status"] == "unknown"
    assert doc["error"]["code"] == "docker_outcome_indeterminate"


# 20. Running result without valid allocated port
@pytest.mark.parametrize("bad_port", [None, 0, 65536, "invalid"])
def test_deploy_project_running_result_without_valid_allocated_port(disposable_database, monkeypatch, bad_port):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(
        monkeypatch,
        start_outcome="succeeded",
        allocated_port=bad_port,
        start_state="running",
    )

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-bad-allocated",
    )

    assert is_replay is False
    assert doc["status"] == "unknown"
    assert doc["error"]["code"] == "port_binding_unavailable"
    assert doc.get("allocated_port") is None
    assert doc.get("deployment_url") is None


# 21. MongoDB failure after Docker start
def test_deploy_project_mongodb_failure_after_docker_start(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(monkeypatch)

    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update

    def fail_on_running(self, filter_dict, update_dict, *args, **kwargs):
        if self.name == "deployments" and filter_dict.get("status") == "starting" and update_dict.get("$set", {}).get("status") == "running":
            raise PyMongoError("Simulated write failure on running")
        return original_find_one_and_update(self, filter_dict, update_dict, *args, **kwargs)

    monkeypatch.setattr(pymongo.collection.Collection, "find_one_and_update", fail_on_running)

    with pytest.raises(coordinator.DeploymentStoreError):
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-mongo-post-start",
        )

    assert len(creates) == 1
    assert len(starts) == 1
    saved = disposable_database.deployments.find_one({"project_id": p_id})
    assert saved["status"] == "unknown"


# 22. Emergency unknown-state write failure
def test_deploy_project_emergency_unknown_state_write_failure_does_not_mask(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update
    original_update_one = pymongo.collection.Collection.update_one

    def fail_find_one_and_update(self, *args, **kwargs):
        if self.name == "deployments":
            raise PyMongoError("Database down")
        return original_find_one_and_update(self, *args, **kwargs)

    def fail_update_one(self, *args, **kwargs):
        if self.name == "deployments":
            raise PyMongoError("Database down")
        return original_update_one(self, *args, **kwargs)

    monkeypatch.setattr(pymongo.collection.Collection, "find_one_and_update", fail_find_one_and_update)
    monkeypatch.setattr(pymongo.collection.Collection, "update_one", fail_update_one)

    with pytest.raises(coordinator.DeploymentStoreError):
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-double-fail",
        )


# 23. Compare-and-set update returns no document (superseded)
def test_deploy_project_compare_and_set_returns_no_document(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    original_find_one_and_update = pymongo.collection.Collection.find_one_and_update

    def return_none_on_creating(self, filter_dict, *args, **kwargs):
        if self.name == "deployments" and filter_dict.get("status") == "queued":
            return None
        return original_find_one_and_update(self, filter_dict, *args, **kwargs)

    monkeypatch.setattr(pymongo.collection.Collection, "find_one_and_update", return_none_on_creating)

    with pytest.raises(coordinator.DeploymentConflictError) as exc_info:
        coordinator.deploy_project(
            disposable_database,
            project_id=p_id,
            image_build_id=b_id,
            container_port=8080,
            idempotency_key="key-superseded",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "deployment_superseded"


# 24. Unexpected exceptions during execution
def test_deploy_project_unexpected_exceptions_treated_safely(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)

    def crash_docker(*args, **kwargs):
        raise RuntimeError("Unexpected OS crash")

    monkeypatch.setattr(coordinator.docker_runtime, "create_container", crash_docker)

    doc, is_replay = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-crash-create",
    )

    assert doc["status"] == "unknown"
    assert doc["error"]["code"] == "docker_outcome_indeterminate"


# 25. Project-scoped get/list/active queries
def test_project_scoped_queries(disposable_database, monkeypatch):
    p_id1, b_id1, _, _ = _seed_project_and_build(disposable_database)
    p_id2, b_id2, _, _ = _seed_project_and_build(disposable_database)
    _mock_runtime(monkeypatch)

    doc1, _ = coordinator.deploy_project(
        disposable_database, p_id1, b_id1, 8080, "key-query-1"
    )
    doc2, _ = coordinator.deploy_project(
        disposable_database, p_id2, b_id2, 8080, "key-query-2"
    )

    # get_deployment respects project scoping
    assert coordinator.get_deployment(disposable_database, p_id1, doc1["_id"]) is not None
    assert coordinator.get_deployment(disposable_database, p_id2, doc1["_id"]) is None

    # list_deployments respects project scoping and pagination
    list1 = coordinator.list_deployments(disposable_database, p_id1)
    assert len(list1) == 1
    assert list1[0]["_id"] == doc1["_id"]

    list2 = coordinator.list_deployments(disposable_database, p_id2)
    assert len(list2) == 1
    assert list2[0]["_id"] == doc2["_id"]

    # get_active_deployment
    active1 = coordinator.get_active_deployment(disposable_database, p_id1)
    assert active1 is not None
    assert active1["_id"] == doc1["_id"]

    # Pagination validation
    with pytest.raises(coordinator.DeploymentRequestError):
        coordinator.list_deployments(disposable_database, p_id1, skip=-1)
    assert coordinator.list_deployments(
        disposable_database, p_id1, skip=10_000
    ) == []
    with pytest.raises(coordinator.DeploymentRequestError):
        coordinator.list_deployments(disposable_database, p_id1, skip=10_001)
    with pytest.raises(coordinator.DeploymentRequestError):
        coordinator.list_deployments(disposable_database, p_id1, limit=0)
    with pytest.raises(coordinator.DeploymentRequestError):
        coordinator.list_deployments(disposable_database, p_id1, limit=101)


# 26. No Docker retry on any uncertain outcome
def test_no_docker_retry_on_uncertain_outcomes(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    creates, starts = _mock_runtime(
        monkeypatch,
        start_outcome="indeterminate",
        start_error=docker_runtime.RuntimeError("docker_outcome_indeterminate", "Timeout on start"),
    )

    doc, _ = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-no-retry",
    )

    assert doc["status"] == "unknown"
    assert len(creates) == 1
    assert len(starts) == 1


# 27. Container ID written before start invoked
def test_persisted_container_id_written_before_start_invoked(disposable_database, monkeypatch):
    p_id, b_id, _, _ = _seed_project_and_build(disposable_database)
    verified_state_during_start = []

    def on_start(container_id, project_id, deployment_id):
        # Query database right inside start_container
        record = disposable_database.deployments.find_one({"_id": deployment_id})
        verified_state_during_start.append({
            "status": record.get("status"),
            "container_id": record.get("container_id"),
        })

    _mock_runtime(monkeypatch, on_start_hook=on_start)

    doc, _ = coordinator.deploy_project(
        disposable_database,
        project_id=p_id,
        image_build_id=b_id,
        container_port=8080,
        idempotency_key="key-order-check",
    )

    assert len(verified_state_during_start) == 1
    assert verified_state_during_start[0]["status"] == "starting"
    assert verified_state_during_start[0]["container_id"] == CONTAINER_ID
