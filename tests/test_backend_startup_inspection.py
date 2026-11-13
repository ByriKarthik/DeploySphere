"""Mocked tests for read-only deployment inspection during backend startup."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError
import pytest

from backend import main
from backend.database import initialize as database_initializer
from backend.services import deployment_coordinator, deployment_inspector
from backend.services import docker_runtime


PROJECT_ID = ObjectId("6af466ba7f1e6d2f280a5177")
DEPLOYMENT_ID = ObjectId("6af4a61fa05c38e44da51b55")
CONTAINER_ID = "c" * 64
ATTEMPT_ID = "a" * 32


def _document(**overrides):
    document = {
        "_id": DEPLOYMENT_ID,
        "project_id": PROJECT_ID,
        "status": "running",
        "container_id": CONTAINER_ID,
        "container_name": f"deploysphere-{DEPLOYMENT_ID}",
        "attempt_id": ATTEMPT_ID,
        "container_port": 5000,
        "allocated_port": 62869,
        "deployment_url": "http://127.0.0.1:62869",
    }
    document.update(overrides)
    return document


def _observation(**overrides):
    observation = docker_runtime.StartupContainerObservation(
        container_id=CONTAINER_ID,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        managed_label="true",
        project_id_label=str(PROJECT_ID),
        deployment_id_label=str(DEPLOYMENT_ID),
        state="running",
        network_ports={
            "5000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "62869"}]
        },
        port_bindings={
            "5000/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]
        },
        identity_verified=True,
    )
    return replace(observation, **overrides)


class FakeDeployments:
    def __init__(
        self,
        documents,
        *,
        find_error=None,
        find_one_error=None,
    ):
        self.documents = {document["_id"]: deepcopy(document) for document in documents}
        self.find_error = find_error
        self.find_one_error = find_one_error
        self.mutations = []

    @staticmethod
    def _project(document, projection):
        return {
            field: deepcopy(document[field])
            for field, include in projection.items()
            if include and field in document
        }

    def find(self, query, *, projection):
        if self.find_error is not None:
            raise self.find_error
        active_statuses = query["status"]["$in"]
        return [
            self._project(document, projection)
            for document in self.documents.values()
            if document.get("status") in active_statuses
        ]

    def find_one(self, query, *, projection):
        if self.find_one_error is not None:
            raise self.find_one_error
        document = self.documents.get(query["_id"])
        return self._project(document, projection) if document is not None else None

    def insert_one(self, *_args, **_kwargs):
        self.mutations.append("insert")
        raise AssertionError("Startup inspection must not insert MongoDB records.")

    def update_one(self, *_args, **_kwargs):
        self.mutations.append("update")
        raise AssertionError("Startup inspection must not update MongoDB records.")

    def delete_one(self, *_args, **_kwargs):
        self.mutations.append("delete")
        raise AssertionError("Startup inspection must not delete MongoDB records.")


def _database(*documents, **collection_options):
    return SimpleNamespace(
        deployments=FakeDeployments(documents, **collection_options)
    )


@pytest.fixture
def forbid_lifecycle_operations(monkeypatch):
    for operation in (
        "create_container",
        "start_container",
        "stop_container",
        "restart_container",
    ):
        monkeypatch.setattr(
            docker_runtime,
            operation,
            Mock(side_effect=AssertionError(f"{operation} is forbidden during inspection")),
        )


def _mock_inspection(monkeypatch, observation=None, error=None):
    result = docker_runtime.StartupContainerInspection(
        observation=observation,
        error=error,
    )
    return monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        Mock(return_value=result),
    )


def _inspect(database):
    return deployment_inspector.inspect_active_deployments(database)


def test_running_deployment_requires_verified_identity_ports_and_url(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document())
    _mock_inspection(monkeypatch, _observation())

    summary = _inspect(database)

    assert summary["candidate_count"] == 1
    assert summary["classification_counts"] == {"running_verified": 1}
    assert summary["results"][0]["classification"] == "running_verified"
    assert summary["outcome"] == "completed"
    assert database.deployments.mutations == []


def test_missing_container_id_uses_only_the_stored_deterministic_name(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document(container_id=None, status="starting"))
    inspect = Mock(
        return_value=docker_runtime.StartupContainerInspection(
            observation=_observation(state="created", network_ports=None, port_bindings=None)
        )
    )
    monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        inspect,
    )

    summary = _inspect(database)

    inspect.assert_called_once_with(
        container_id=None,
        container_name=f"deploysphere-{DEPLOYMENT_ID}",
        project_id=PROJECT_ID,
        deployment_id=DEPLOYMENT_ID,
    )
    assert summary["results"][0]["classification"] == "created_verified"


def test_stored_name_mismatch_is_rejected_without_docker_inspection(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document(container_name="deploysphere-other"))
    inspect = Mock()
    monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        inspect,
    )

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "identity_mismatch"
    inspect.assert_not_called()


def test_missing_or_malformed_deployment_identity_is_not_inspected(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document(attempt_id="not-an-attempt"))
    inspect = Mock()
    monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        inspect,
    )

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "missing_identity"
    inspect.assert_not_called()


@pytest.mark.parametrize(
    "observation",
    [
        _observation(container_id="e" * 64, identity_verified=False),
        _observation(managed_label="false", identity_verified=False),
        _observation(project_id_label="f" * 24, identity_verified=False),
        _observation(deployment_id_label="e" * 24, identity_verified=False),
        _observation(container_name="deploysphere-other", identity_verified=False),
    ],
)
def test_full_id_name_and_each_ownership_mismatch_are_rejected(
    monkeypatch, forbid_lifecycle_operations, observation
):
    database = _database(_document())
    _mock_inspection(monkeypatch, observation)

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "identity_mismatch"


@pytest.mark.parametrize(
    "observation,document_overrides",
    [
        (
            _observation(
                network_ports={
                    "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "62869"}]
                },
                port_bindings={
                    "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]
                },
            ),
            {},
        ),
        (
            _observation(
                network_ports={
                    "5000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "62869"}]
                }
            ),
            {},
        ),
        (
            _observation(
                network_ports={
                    "5000/tcp": [
                        {"HostIp": "127.0.0.1", "HostPort": "62869"},
                        {"HostIp": "127.0.0.1", "HostPort": "62870"},
                    ]
                }
            ),
            {},
        ),
        (_observation(network_ports={}), {}),
        (_observation(), {"allocated_port": 62870}),
        (_observation(), {"deployment_url": "http://127.0.0.1:62870"}),
    ],
)
def test_inconsistent_running_port_observations_remain_unresolved(
    monkeypatch,
    forbid_lifecycle_operations,
    observation,
    document_overrides,
):
    database = _database(_document(**document_overrides))
    _mock_inspection(monkeypatch, observation)

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "state_mismatch"


@pytest.mark.parametrize(
    ("state", "status", "classification"),
    [
        ("exited", "starting", "state_mismatch"),
        ("exited", "restarting", "state_mismatch"),
        ("exited", "stopping", "exited_verified"),
        ("created", "starting", "created_verified"),
        ("exited", "running", "state_mismatch"),
    ],
)
def test_transitional_state_observations_remain_conservative(
    monkeypatch, forbid_lifecycle_operations, state, status, classification
):
    database = _database(_document(status=status))
    _mock_inspection(
        monkeypatch,
        _observation(state=state, network_ports=None, port_bindings=None),
    )

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == classification


def test_unsupported_docker_state_is_not_interpreted_as_success(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document())
    _mock_inspection(monkeypatch, _observation(state="paused"))

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "unsupported_state"


@pytest.mark.parametrize(
    ("error_code", "classification"),
    [
        ("container_not_found", "container_not_found"),
        ("docker_unavailable", "docker_unavailable"),
        ("identity_mismatch", "identity_mismatch"),
        ("ambiguous_lookup", "ambiguous_lookup"),
        ("missing_identity", "missing_identity"),
        ("docker_timeout", "inspection_inconclusive"),
        ("inspection_inconclusive", "inspection_inconclusive"),
    ],
)
def test_runtime_errors_map_to_explicit_safe_classifications(
    monkeypatch, forbid_lifecycle_operations, error_code, classification
):
    database = _database(_document())
    _mock_inspection(
        monkeypatch,
        error=docker_runtime.RuntimeError(error_code, "mocked inspection result"),
    )

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == classification


def test_database_change_during_inspection_marks_snapshot_stale(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document())

    def inspect_then_change(**_kwargs):
        database.deployments.documents[DEPLOYMENT_ID]["status"] = "unknown"
        return docker_runtime.StartupContainerInspection(
            observation=_observation()
        )

    monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        inspect_then_change,
    )

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "stale_database_snapshot"
    assert summary["outcome"] == "partial"
    assert database.deployments.mutations == []


def test_mongodb_candidate_retrieval_failure_is_logged_and_nonfatal(
    caplog, monkeypatch, forbid_lifecycle_operations
):
    database = _database(
        _document(),
        find_error=PyMongoError("mock database unavailable"),
    )
    inspect = Mock()
    monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        inspect,
    )

    summary = _inspect(database)

    assert summary["database_error"] == "PyMongoError"
    assert summary["candidate_count"] == 0
    assert summary["outcome"] == "unavailable"
    assert "continuing without inspection" in caplog.text
    inspect.assert_not_called()


def test_mongodb_snapshot_reread_failure_is_inconclusive(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(
        _document(),
        find_one_error=PyMongoError("mock reread failure"),
    )
    _mock_inspection(monkeypatch, _observation())

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "inspection_inconclusive"
    assert database.deployments.mutations == []


def test_unexpected_runtime_exception_is_inconclusive_and_nonfatal(
    monkeypatch, forbid_lifecycle_operations
):
    database = _database(_document())
    inspect = Mock(side_effect=RuntimeError("unexpected mock failure"))
    monkeypatch.setattr(
        deployment_inspector.docker_runtime,
        "inspect_container_for_startup",
        inspect,
    )

    summary = _inspect(database)

    assert summary["results"][0]["classification"] == "inspection_inconclusive"


def test_startup_inspection_runs_under_lock_and_releases_after_exception(
    monkeypatch,
):
    events = []
    lock_held = False

    class ProcessLock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True
            events.append("lock_acquired")
            return self

        def __exit__(self, *_exc_info):
            nonlocal lock_held
            lock_held = False
            events.append("lock_released")

    def unexpected_inspection(database):
        assert database is marker_database
        assert lock_held
        events.append("inspection")
        raise RuntimeError("unexpected inspector failure")

    marker_database = object()
    monkeypatch.setattr(main, "BackendProcessLock", ProcessLock)
    monkeypatch.setattr(main.connection, "database", marker_database)
    monkeypatch.setattr(main, "inspect_active_deployments", unexpected_inspection)
    initialize = Mock(side_effect=AssertionError("database initialization is forbidden"))
    reconcile = Mock(side_effect=AssertionError("reconciliation is forbidden"))
    monkeypatch.setattr(database_initializer, "initialize_database", initialize)
    monkeypatch.setattr(
        deployment_coordinator,
        "reconcile_unknown_deployment_as_failed",
        reconcile,
    )

    with TestClient(main.app) as client:
        assert lock_held
        assert client.get("/").status_code == 200
        response = client.get("/api/diagnostics/startup-inspection")
        assert response.status_code == 200
        assert response.json()["outcome"] == "failed"
        assert response.json()["results"] == []
        assert lock_held

    assert events == ["lock_acquired", "inspection", "lock_released"]
    assert not lock_held
    snapshot = main.app.state.startup_inspection_snapshot
    assert snapshot["outcome"] == "failed"
    assert "unexpected inspector failure" not in repr(snapshot)
    initialize.assert_not_called()
    reconcile.assert_not_called()
