"""DeploySphere FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import connection
from backend.process_lock import BackendProcessLock
from backend.routers.health import router as health_router
from backend.routers.projects import router as projects_router
from backend.routers.project_workspaces import router as project_workspaces_router
from backend.routers.project_detections import router as project_detections_router
from backend.routers.project_builds import router as project_builds_router
from backend.routers.deployments import router as deployments_router
from backend.routers.startup_diagnostics import router as startup_diagnostics_router
from backend.services.deployment_inspector import inspect_active_deployments


logger = logging.getLogger(__name__)
STARTUP_INSPECTION_OUTCOMES = {"completed", "partial", "unavailable"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lock backend ownership and inspect deployment state before serving."""
    with BackendProcessLock():
        app.state.startup_inspection_snapshot = None
        try:
            summary = inspect_active_deployments(connection.database)
            if (
                not isinstance(summary, dict)
                or summary.get("outcome") not in STARTUP_INSPECTION_OUTCOMES
                or not isinstance(summary.get("classification_counts"), dict)
                or not isinstance(summary.get("results"), list)
            ):
                raise RuntimeError("Startup inspection returned an invalid summary.")
            app.state.startup_inspection_snapshot = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "outcome": summary["outcome"],
                "classification_counts": summary.get("classification_counts", {}),
                "results": summary.get("results", []),
            }
        except Exception:
            logger.exception(
                "Startup deployment inspection failed unexpectedly; continuing without recovery."
            )
            app.state.startup_inspection_snapshot = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "outcome": "failed",
                "classification_counts": {},
                "results": [],
            }
        yield


app = FastAPI(
    title="DeploySphere API",
    description="API for the DeploySphere application.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(projects_router)
app.include_router(project_workspaces_router)
app.include_router(project_detections_router)
app.include_router(project_builds_router)
app.include_router(deployments_router)
app.include_router(startup_diagnostics_router)


@app.get("/")
def get_application_info() -> dict[str, str]:
    """Return basic information about the running API."""
    return {
        "application": "DeploySphere API",
        "status": "running",
        "docs": "/docs",
    }
