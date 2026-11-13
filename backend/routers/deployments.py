"""API endpoints for project deployment history and lifecycle operations."""

from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from backend.database import connection
from backend.services import deployment_coordinator as coordinator


router = APIRouter(prefix="/api/projects", tags=["deployments"])


class DeploymentCreate(BaseModel):
    """Explicit image build and container port for a deployment request."""

    image_build_id: str = Field(min_length=1)
    container_port: int = Field(ge=1, le=65535, strict=True)


class DeploymentErrorResponse(BaseModel):
    """Safe error details persisted for an uncertain or failed deployment."""

    code: str
    message: str
    occurred_at: datetime


class DeploymentResponse(BaseModel):
    """Public deployment fields; reservation internals are omitted."""

    id: str
    project_id: str
    image_build_id: str | None = None
    status: str
    commit_hash: str | None = None
    image_tag: str | None = None
    container_port: int | None = None
    allocated_port: int | None = None
    deployment_url: str | None = None
    error: DeploymentErrorResponse | None = None
    created_at: datetime
    updated_at: datetime | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None


def _as_utc(value: datetime) -> datetime:
    """Treat MongoDB's naive UTC dates as UTC for JSON responses."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _format_deployment(document: dict) -> DeploymentResponse:
    """Return only public deployment fields."""
    error = document.get("error")
    return DeploymentResponse(
        id=str(document["_id"]),
        project_id=str(document["project_id"]),
        image_build_id=(
            str(document["image_build_id"])
            if document.get("image_build_id") is not None
            else None
        ),
        status=document["status"],
        commit_hash=document.get("commit_hash"),
        image_tag=document.get("docker_image"),
        container_port=document.get("container_port"),
        allocated_port=document.get("allocated_port"),
        deployment_url=document.get("deployment_url"),
        error=(
            {
                "code": error["code"],
                "message": error["message"],
                "occurred_at": _as_utc(error["occurred_at"]),
            }
            if error
            else None
        ),
        created_at=_as_utc(document["created_at"]),
        updated_at=(
            _as_utc(document["updated_at"])
            if isinstance(document.get("updated_at"), datetime)
            else None
        ),
        started_at=(
            _as_utc(document["started_at"])
            if isinstance(document.get("started_at"), datetime)
            else None
        ),
        stopped_at=(
            _as_utc(document["stopped_at"])
            if isinstance(document.get("stopped_at"), datetime)
            else None
        ),
    )


def _project_object_id(project_id: str) -> ObjectId:
    """Validate and convert a route project ID."""
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="The project ID is invalid.")
    return ObjectId(project_id)


def _deployment_object_id(deployment_id: str) -> ObjectId:
    """Validate and convert a route deployment ID."""
    if not ObjectId.is_valid(deployment_id):
        raise HTTPException(status_code=400, detail="The deployment ID is invalid.")
    return ObjectId(deployment_id)


def _require_project(project_id: ObjectId) -> None:
    """Return 404 for unknown projects and 503 for MongoDB failures."""
    try:
        if connection.database.projects.find_one({"_id": project_id}) is None:
            raise HTTPException(status_code=404, detail="Project not found.")
    except PyMongoError as error:
        raise HTTPException(status_code=503, detail="MongoDB could not load the project.") from error


def _raise_service_error(error: Exception) -> None:
    """Translate coordinator and database errors into HTTP responses."""
    if isinstance(error, coordinator.DeploymentRequestError):
        raise HTTPException(status_code=error.status_code, detail=error.message) from error
    if isinstance(error, coordinator.DeploymentConflictError):
        raise HTTPException(status_code=409, detail=error.message) from error
    if isinstance(error, (coordinator.DeploymentStoreError, PyMongoError)):
        raise HTTPException(status_code=503, detail="MongoDB could not load or save the deployment.") from error
    raise error


@router.post(
    "/{project_id}/deployments",
    response_model=DeploymentResponse,
)
def create_project_deployment(
    project_id: str,
    request: DeploymentCreate,
    response: Response,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=128),
) -> DeploymentResponse:
    """Create one deployment using the explicitly requested image build."""
    object_id = _project_object_id(project_id)
    try:
        document, replay = coordinator.deploy_project(
            connection.database,
            project_id=object_id,
            image_build_id=request.image_build_id,
            container_port=request.container_port,
            idempotency_key=idempotency_key,
        )
    except (
        coordinator.DeploymentRequestError,
        coordinator.DeploymentConflictError,
        coordinator.DeploymentStoreError,
        PyMongoError,
    ) as error:
        _raise_service_error(error)
    response.status_code = 200 if replay else 201
    return _format_deployment(document)


@router.get(
    "/{project_id}/deployments",
    response_model=list[DeploymentResponse],
)
def get_project_deployments(
    project_id: str,
    skip: int = Query(default=0, ge=0, le=10_000),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[DeploymentResponse]:
    """Return a bounded page of deployment history for a project."""
    object_id = _project_object_id(project_id)
    _require_project(object_id)
    try:
        documents = coordinator.list_deployments(
            connection.database, object_id, skip=skip, limit=limit
        )
    except (
        coordinator.DeploymentRequestError,
        coordinator.DeploymentStoreError,
        PyMongoError,
    ) as error:
        _raise_service_error(error)
    return [_format_deployment(document) for document in documents]


@router.get(
    "/{project_id}/deployments/active",
    response_model=DeploymentResponse,
)
def get_project_active_deployment(project_id: str) -> DeploymentResponse:
    """Return the active deployment for a project."""
    object_id = _project_object_id(project_id)
    _require_project(object_id)
    try:
        document = coordinator.get_active_deployment(connection.database, object_id)
    except (coordinator.DeploymentStoreError, PyMongoError) as error:
        _raise_service_error(error)
    if document is None:
        raise HTTPException(status_code=404, detail="No active deployment was found.")
    return _format_deployment(document)


@router.get(
    "/{project_id}/deployments/{deployment_id}",
    response_model=DeploymentResponse,
)
def get_project_deployment(project_id: str, deployment_id: str) -> DeploymentResponse:
    """Return one deployment only when it belongs to the requested project."""
    project_object_id = _project_object_id(project_id)
    deployment_object_id = _deployment_object_id(deployment_id)
    _require_project(project_object_id)
    try:
        document = coordinator.get_deployment(
            connection.database, project_object_id, deployment_object_id
        )
    except (coordinator.DeploymentStoreError, PyMongoError) as error:
        _raise_service_error(error)
    if document is None:
        raise HTTPException(status_code=404, detail="Deployment not found.")
    return _format_deployment(document)


@router.post(
    "/{project_id}/deployments/{deployment_id}/stop",
    response_model=DeploymentResponse,
)
def stop_project_deployment(project_id: str, deployment_id: str) -> DeploymentResponse:
    """Stop the project-owned deployment without removing its container."""
    project_object_id = _project_object_id(project_id)
    deployment_object_id = _deployment_object_id(deployment_id)
    _require_project(project_object_id)
    try:
        document = coordinator.stop_deployment(
            connection.database, project_object_id, deployment_object_id
        )
    except (
        coordinator.DeploymentRequestError,
        coordinator.DeploymentConflictError,
        coordinator.DeploymentStoreError,
        PyMongoError,
    ) as error:
        _raise_service_error(error)
    return _format_deployment(document)


@router.post(
    "/{project_id}/deployments/{deployment_id}/restart",
    response_model=DeploymentResponse,
)
def restart_project_deployment(project_id: str, deployment_id: str) -> DeploymentResponse:
    """Restart the project-owned container without recreating it."""
    project_object_id = _project_object_id(project_id)
    deployment_object_id = _deployment_object_id(deployment_id)
    _require_project(project_object_id)
    try:
        document = coordinator.restart_deployment(
            connection.database, project_object_id, deployment_object_id
        )
    except (
        coordinator.DeploymentRequestError,
        coordinator.DeploymentConflictError,
        coordinator.DeploymentStoreError,
        PyMongoError,
    ) as error:
        _raise_service_error(error)
    return _format_deployment(document)
