"""API endpoints for detecting application files in cloned workspaces."""

from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from pymongo.errors import PyMongoError

from backend.database import connection
from backend.services.application_detection import (
    DetectionConflict,
    DetectionRunError,
    DetectionStoreError,
    run_detection,
)
from backend.services.git_clone import GitCloneError, resolve_workspace_paths


router = APIRouter(prefix="/api/projects", tags=["application detection"])


class DetectionErrorResponse(BaseModel):
    """Safe error information saved for a failed detection."""

    code: str
    message: str
    occurred_at: datetime


class DetectionResponse(BaseModel):
    """Public detection fields; internal reservation details are omitted."""

    id: str
    project_id: str
    workspace_id: str
    commit_hash: str
    status: str
    application_type: str
    build_method: str
    detected_application_types: list[str]
    detected_build_methods: list[str]
    detected_files: list[str]
    confidence: float
    warnings: list[str]
    error: DetectionErrorResponse | None = None
    created_at: datetime
    updated_at: datetime


def _as_utc(value: datetime) -> datetime:
    """Treat MongoDB's naive UTC dates as UTC for API serialization."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _format_detection(document: dict) -> DetectionResponse:
    """Convert a stored detection document without exposing lease internals."""
    values = {key: document.get(key) for key in DetectionResponse.model_fields}
    values["id"] = str(document["_id"])
    values["project_id"] = str(document["project_id"])
    values["workspace_id"] = str(document["workspace_id"])
    values["created_at"] = _as_utc(document["created_at"])
    values["updated_at"] = _as_utc(document["updated_at"])
    if values.get("error"):
        values["error"] = {
            **values["error"],
            "occurred_at": _as_utc(values["error"]["occurred_at"]),
        }
    return DetectionResponse(**values)


def _load_project_and_workspace(project_id: str) -> tuple[ObjectId, dict, dict]:
    """Load the project and require its successfully cloned workspace."""
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="The project ID is invalid.")
    object_id = ObjectId(project_id)
    try:
        database = connection.database
        project = database.projects.find_one({"_id": object_id})
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        workspace = database.repository_workspaces.find_one({"project_id": object_id})
    except PyMongoError as error:
        raise HTTPException(status_code=503, detail="MongoDB could not load the project workspace.") from error
    if workspace is None or workspace.get("status") != "cloned":
        raise HTTPException(status_code=409, detail="The project does not have a successfully cloned workspace.")
    return object_id, project, workspace


def _verified_workspace_path(project_id: ObjectId, workspace: dict) -> Path:
    """Derive the expected path and reject any path stored outside that location."""
    try:
        _, expected_path = resolve_workspace_paths(project_id)
    except GitCloneError as error:
        raise HTTPException(status_code=409, detail=error.message) from error

    recorded_path = workspace.get("workspace_path")
    if not isinstance(recorded_path, str):
        raise HTTPException(status_code=409, detail="The cloned workspace path is invalid.")
    try:
        recorded = Path(recorded_path).expanduser()
        if not recorded.is_absolute() or recorded.is_symlink():
            raise ValueError("unsafe workspace path")
        if recorded.resolve(strict=False) != expected_path:
            raise ValueError("workspace path does not match the project")
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail="The cloned workspace path failed verification.") from error
    return expected_path


@router.post("/{project_id}/detect", response_model=DetectionResponse)
def detect_project(project_id: str, response: Response) -> DetectionResponse:
    """Detect manifests in the project's verified, cloned workspace."""
    object_id, _, workspace = _load_project_and_workspace(project_id)
    workspace_path = _verified_workspace_path(object_id, workspace)
    commit_hash = workspace.get("commit_hash")
    if not isinstance(commit_hash, str) or not commit_hash:
        raise HTTPException(status_code=409, detail="The cloned workspace has no valid commit hash.")

    try:
        document, response_status = run_detection(
            connection.database,
            object_id,
            workspace["_id"],
            commit_hash,
            str(workspace_path),
        )
    except DetectionConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except DetectionStoreError as error:
        raise HTTPException(status_code=503, detail="MongoDB could not reserve or save the detection.") from error
    except DetectionRunError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from error
    except PyMongoError as error:
        raise HTTPException(status_code=503, detail="MongoDB could not save the detection.") from error

    # The route's response code depends on whether this was a first run,
    # cached result, or re-detection after a prior record.
    response.status_code = response_status
    return _format_detection(document)


@router.get("/{project_id}/detection", response_model=DetectionResponse)
def get_project_detection(project_id: str) -> DetectionResponse:
    """Return the saved detection for a cloned project.

    A project without a saved detection returns 404. A missing or uncloned
    workspace returns 409, distinguishing setup state from absent results.
    """
    object_id, _, workspace = _load_project_and_workspace(project_id)
    try:
        document = connection.database.application_detections.find_one(
            {"project_id": object_id, "workspace_id": workspace["_id"]}
        )
    except PyMongoError as error:
        raise HTTPException(status_code=503, detail="MongoDB could not load the detection.") from error
    if document is None:
        raise HTTPException(status_code=404, detail="No application detection has been saved for this project.")
    return _format_detection(document)
