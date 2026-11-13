"""API endpoints for Docker image-build history and orchestration."""

from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel
from pymongo.errors import PyMongoError

from backend.database import connection
from backend.services.image_builds import (
    ImageBuildConflict,
    ImageBuildRequestError,
    ImageBuildStoreError,
    get_image_build,
    list_image_builds,
    run_image_build,
)


router = APIRouter(prefix="/api/projects", tags=["image builds"])


class ImageBuildErrorResponse(BaseModel):
    """Safe error details persisted for a failed or indeterminate build."""

    code: str
    message: str
    occurred_at: datetime


class ImageBuildLogResponse(BaseModel):
    """Bounded output excerpt for one project-owned image build."""

    log_excerpt: str


class ImageBuildResponse(BaseModel):
    """Fields safe for API clients; reservation internals are omitted."""

    id: str
    project_id: str
    status: str
    commit_hash: str
    image_tag: str
    image_id: str | None = None
    dockerfile_path: str
    build_duration: float
    log_bytes: int = 0
    log_line_count: int = 0
    log_truncated: bool = False
    outcome_known: bool | None = None
    error: ImageBuildErrorResponse | None = None
    created_at: datetime
    updated_at: datetime


def _as_utc(value: datetime) -> datetime:
    """Treat MongoDB's naive dates as UTC for JSON responses."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _format_build(document: dict) -> ImageBuildResponse:
    """Return only public build information, never internal IDs or paths."""
    error = document.get("error")
    formatted_error = None
    if error:
        formatted_error = {
            "code": error["code"],
            "message": error["message"],
            "occurred_at": _as_utc(error["occurred_at"]),
        }

    if document.get("status") == "building":
        outcome_known = False if error else None
    else:
        outcome_known = True

    return ImageBuildResponse(
        id=str(document["_id"]),
        project_id=str(document["project_id"]),
        status=document["status"],
        commit_hash=document["commit_hash"],
        image_tag=document["image_tag"],
        image_id=document.get("image_id"),
        dockerfile_path=document["dockerfile_path"],
        build_duration=float(document.get("build_duration", 0.0)),
        log_bytes=document.get("log_bytes", 0),
        log_line_count=document.get("log_line_count", 0),
        log_truncated=document.get("log_truncated", False),
        outcome_known=outcome_known,
        error=formatted_error,
        created_at=_as_utc(document["created_at"]),
        updated_at=_as_utc(document["updated_at"]),
    )


def _project_object_id(project_id: str) -> ObjectId:
    """Validate and convert a route project ID."""
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="The project ID is invalid.")
    return ObjectId(project_id)


def _require_project(project_id: ObjectId) -> None:
    """Return 404 for unknown projects and 503 for MongoDB failures."""
    try:
        if connection.database.projects.find_one({"_id": project_id}) is None:
            raise HTTPException(status_code=404, detail="Project not found.")
    except PyMongoError as error:
        raise HTTPException(status_code=503, detail="MongoDB could not load the project.") from error


def _raise_service_error(error: Exception) -> None:
    """Translate orchestration errors into the established HTTP convention."""
    if isinstance(error, ImageBuildRequestError):
        raise HTTPException(status_code=error.status_code, detail=error.message) from error
    if isinstance(error, ImageBuildConflict):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, (ImageBuildStoreError, PyMongoError)):
        raise HTTPException(status_code=503, detail="MongoDB could not load or save the image build.") from error
    raise error


@router.post("/{project_id}/build", response_model=ImageBuildResponse)
def build_project_image(project_id: str, response: Response) -> ImageBuildResponse:
    """Build the current successfully detected Dockerfile for a project."""
    object_id = _project_object_id(project_id)
    try:
        document, cached = run_image_build(connection.database, object_id)
    except (ImageBuildRequestError, ImageBuildConflict, ImageBuildStoreError, PyMongoError) as error:
        _raise_service_error(error)

    response.status_code = 200 if cached else 201
    return _format_build(document)


@router.get("/{project_id}/builds", response_model=list[ImageBuildResponse])
def get_project_builds(
    project_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[ImageBuildResponse]:
    """Return a bounded, newest-first page of project build history."""
    object_id = _project_object_id(project_id)
    _require_project(object_id)
    try:
        return [
            _format_build(item)
            for item in list_image_builds(connection.database, object_id, skip, limit)
        ]
    except (ImageBuildStoreError, PyMongoError) as error:
        _raise_service_error(error)


@router.get("/{project_id}/builds/{build_id}", response_model=ImageBuildResponse)
def get_project_build(project_id: str, build_id: str) -> ImageBuildResponse:
    """Return one build only when it belongs to the requested project."""
    project_object_id = _project_object_id(project_id)
    if not ObjectId.is_valid(build_id):
        raise HTTPException(status_code=400, detail="The build ID is invalid.")
    _require_project(project_object_id)
    try:
        document = get_image_build(connection.database, project_object_id, ObjectId(build_id))
    except (ImageBuildStoreError, PyMongoError) as error:
        _raise_service_error(error)
    if document is None:
        raise HTTPException(status_code=404, detail="Image build not found.")
    return _format_build(document)


@router.get(
    "/{project_id}/builds/{build_id}/logs",
    response_model=ImageBuildLogResponse,
)
def get_project_build_logs(project_id: str, build_id: str) -> ImageBuildLogResponse:
    """Return a bounded log excerpt only for a build owned by this project."""
    project_object_id = _project_object_id(project_id)
    if not ObjectId.is_valid(build_id):
        raise HTTPException(status_code=400, detail="The build ID is invalid.")
    _require_project(project_object_id)
    try:
        document = get_image_build(
            connection.database, project_object_id, ObjectId(build_id)
        )
    except (ImageBuildStoreError, PyMongoError) as error:
        _raise_service_error(error)
    if document is None:
        raise HTTPException(status_code=404, detail="Image build not found.")
    return ImageBuildLogResponse(log_excerpt=document.get("log_excerpt", ""))
