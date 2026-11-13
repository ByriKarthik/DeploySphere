"""Mocked tests for the read-only startup diagnostics API."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import main
from backend.routers.startup_diagnostics import router
from backend.services import docker_runtime


def test_startup_diagnostics_returns_safe_snapshot_without_runtime_reads(monkeypatch):
    database = SimpleNamespace(
        deployments=SimpleNamespace(
            find=Mock(side_effect=AssertionError("MongoDB reads are forbidden")),
            find_one=Mock(side_effect=AssertionError("MongoDB reads are forbidden")),
            update_one=Mock(side_effect=AssertionError("MongoDB writes are forbidden")),
        )
    )
    summary = {
        "outcome": "partial",
        "classification_counts": {
            "inspection_inconclusive": 1,
            "identity_mismatch": 1,
        },
        "results": [
            {
                "deployment_id": "6af4a61fa05c38e44da51b55",
                "status": "starting",
                "classification": "inspection_inconclusive",
                "reason": "Private output C:\\private\\secret",
                "container_id": "c" * 64,
                "container_name": "private-container-name",
                "docker_state": "created",
            },
            {
                "deployment_id": "6af4a61fa05c38e44da51b56",
                "status": "C:\\private\\secret",
                "classification": "identity_mismatch",
                "reason": "Private output C:\\private\\secret",
                "container_id": "d" * 64,
                "container_name": "another-private-name",
                "docker_state": "running",
            },
        ],
        "database_error": None,
    }
    lock = Mock()
    lock.__enter__ = Mock(return_value=lock)
    lock.__exit__ = Mock(return_value=False)
    inspect = Mock(return_value=summary)
    docker_inspect = Mock(
        side_effect=AssertionError("Docker inspection is forbidden in the API route")
    )
    monkeypatch.setattr(main, "BackendProcessLock", lambda: lock)
    monkeypatch.setattr(main.connection, "database", database)
    monkeypatch.setattr(main, "inspect_active_deployments", inspect)
    monkeypatch.setattr(docker_runtime, "inspect_container_for_startup", docker_inspect)

    with TestClient(main.app) as client:
        response = client.get("/api/diagnostics/startup-inspection")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "captured_at",
        "outcome",
        "classification_counts",
        "results",
        "historical",
        "warning",
    }
    datetime.fromisoformat(body["captured_at"])
    assert body["outcome"] == "partial"
    assert body["classification_counts"] == {
        "inspection_inconclusive": 1,
        "identity_mismatch": 1,
    }
    assert body["results"] == [
        {
            "deployment_id": "6af4a61fa05c38e44da51b55",
            "status": "starting",
            "classification": "inspection_inconclusive",
            "reason": "Startup inspection could not establish a conclusive result.",
        },
        {
            "deployment_id": "6af4a61fa05c38e44da51b56",
            "status": "unknown",
            "classification": "identity_mismatch",
            "reason": "Container identity or ownership could not be verified.",
        },
    ]
    assert body["historical"] is True
    assert "not current deployment health" in body["warning"]
    serialized = response.text
    assert "private" not in serialized
    assert "secret" not in serialized
    assert "c" * 64 not in serialized
    inspect.assert_called_once_with(database)
    database.deployments.find.assert_not_called()
    database.deployments.find_one.assert_not_called()
    database.deployments.update_one.assert_not_called()
    docker_inspect.assert_not_called()


def test_startup_diagnostics_reports_missing_snapshot():
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get("/api/diagnostics/startup-inspection")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "startup_snapshot_unavailable",
        "message": "No startup inspection snapshot is available.",
    }
