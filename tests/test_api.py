"""Tests for the initial FastAPI application and health route."""

from fastapi.testclient import TestClient

from backend.database import connection
from backend.main import app
from backend.routers import health


client = TestClient(app)


def test_health_returns_connected_status(monkeypatch):
    monkeypatch.setattr(
        health,
        "check_mongodb_connection",
        lambda: {"connected": True, "database": "deploysphere"},
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "api_status": "ok",
        "mongodb_status": "connected",
        "database": connection.DATABASE_NAME,
    }


def test_health_returns_503_when_mongodb_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        health,
        "check_mongodb_connection",
        lambda: {
            "connected": False,
            "database": "deploysphere",
            "message": "MongoDB unavailable",
        },
    )

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {
        "api_status": "degraded",
        "mongodb_status": "unavailable",
        "database": connection.DATABASE_NAME,
    }


def test_root_returns_application_information():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["application"] == "DeploySphere API"
    assert response.json()["status"] == "running"


def test_vite_origin_is_allowed_by_cors():
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
