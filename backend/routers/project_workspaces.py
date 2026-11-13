"""API endpoints for cloning registered projects into local workspaces."""

from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.database import connection
from backend.services.git_clone import (
    GitCloneError,
    clone_repository,
    resolve_workspace_paths,
    validate_repository_url,
)


router = APIRouter(prefix="/api/projects", tags=["project workspaces"])


class WorkspaceError(BaseModel):
    """Safe error details for the most recent clone attempt."""

    code: str
    message: str
    occurred_at: datetime


class WorkspaceSummary(BaseModel):
    """Workspace information that is safe to return to API clients."""

    status: str
    repository_url: str
    branch: str
    commit_hash: str | None = None
    error: WorkspaceError | None = None
    created_at: datetime
    updated_at: datetime


class CloneResponse(WorkspaceSummary):
    """Successful clone response without the host filesystem path."""

    project_id: str


def _reserve_workspace(
    project_id: ObjectId,
    repository_url: str,
    branch: str,
    workspace_path: str,
    now: datetime,
) -> dict:
    """Atomically create a reservation or claim a previous failed attempt."""
    workspaces = connection.database.repository_workspaces
    reservation_fields = {
        "repository_url": repository_url,
        "branch": branch,
        "workspace_path": workspace_path,
        "status": "cloning",
        "updated_at": now,
    }
    try:
        existing_failed = workspaces.find_one_and_update(
            {"project_id": project_id, "status": "failed"},
            {
                "$set": reservation_fields,
                "$unset": {"commit_hash": "", "error": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A clone is already running or a workspace already exists for this project.",
        ) from error
    if existing_failed is not None:
        return existing_failed

    reservation = {
        "project_id": project_id,
        **reservation_fields,
        "created_at": now,
    }
    try:
        workspaces.insert_one(reservation)
    except DuplicateKeyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A clone is already running or a workspace already exists for this project.",
        ) from error
    return reservation


def _record_failure(project_id: ObjectId, error: GitCloneError) -> None:
    """Mark the active reservation failed so the user can retry it."""
    now = datetime.now(timezone.utc)
    try:
        result = connection.database.repository_workspaces.update_one(
            {"project_id": project_id, "status": "cloning"},
            {
                "$set": {
                    "status": "failed",
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "occurred_at": now,
                    },
                    "updated_at": now,
                },
                "$unset": {"commit_hash": ""},
            },
        )
    except PyMongoError as database_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The clone failed and MongoDB could not save its status.",
        ) from database_error
    if result.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The clone failed but its workspace status could not be updated.",
        )


@router.post(
    "/{project_id}/clone",
    response_model=CloneResponse,
    status_code=status.HTTP_201_CREATED,
)
def clone_project(project_id: str) -> CloneResponse:
    """Clone a project's stored repository and branch into its safe workspace."""
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="The project ID is invalid.")
    object_id = ObjectId(project_id)

    try:
        project = connection.database.projects.find_one({"_id": object_id})
    except PyMongoError as error:
        raise HTTPException(
            status_code=503, detail="MongoDB could not load the project."
        ) from error
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    try:
        validate_repository_url(project.get("repository_url"))
    except GitCloneError as error:
        raise HTTPException(status_code=error.http_status, detail=error.message) from error
    if not isinstance(project.get("branch"), str) or not project["branch"]:
        raise HTTPException(status_code=422, detail="The stored branch name is invalid.")

    try:
        root, destination = resolve_workspace_paths(object_id)
    except GitCloneError as error:
        raise HTTPException(status_code=error.http_status, detail=error.message) from error

    now = datetime.now(timezone.utc)
    try:
        reservation = _reserve_workspace(
            object_id,
            project.get("repository_url"),
            project.get("branch"),
            str(destination),
            now,
        )
    except HTTPException:
        raise
    except PyMongoError as error:
        raise HTTPException(
            status_code=503, detail="MongoDB could not reserve this project workspace."
        ) from error

    try:
        clone_result = clone_repository(
            project.get("repository_url"),
            project.get("branch"),
            object_id,
            workspace_root=root,
        )
    except GitCloneError as error:
        _record_failure(object_id, error)
        raise HTTPException(status_code=error.http_status, detail=error.message) from error

    updated_at = datetime.now(timezone.utc)
    try:
        result = connection.database.repository_workspaces.update_one(
            {"project_id": object_id, "status": "cloning"},
            {
                "$set": {
                    "status": "cloned",
                    "commit_hash": clone_result.commit_hash,
                    "updated_at": updated_at,
                },
                "$unset": {"error": ""},
            },
        )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail="The repository was cloned, but MongoDB could not save its status.",
        ) from error
    if result.matched_count != 1:
        raise HTTPException(
            status_code=503,
            detail="The repository was cloned, but its workspace status could not be updated.",
        )

    return CloneResponse(
        project_id=str(object_id),
        status="cloned",
        repository_url=reservation["repository_url"],
        branch=reservation["branch"],
        commit_hash=clone_result.commit_hash,
        created_at=reservation["created_at"],
        updated_at=updated_at,
    )
