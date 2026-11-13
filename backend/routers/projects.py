"""Endpoints for registering and listing GitHub projects."""

from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
import re
from urllib.parse import urlsplit

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.database import connection


router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    """Fields required to register a GitHub repository."""

    name: str = Field(min_length=1, max_length=100)
    repository_url: str = Field(min_length=1, max_length=300)
    branch: str = Field(default="main", min_length=1, max_length=250)

    @field_validator("name", "branch", mode="before")
    @classmethod
    def trim_text(cls, value: object) -> object:
        """Trim surrounding whitespace from user-entered text."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("repository_url", mode="before")
    @classmethod
    def validate_github_repository(cls, value: object) -> object:
        """Accept only HTTPS URLs pointing directly to a GitHub repository."""
        if not isinstance(value, str):
            raise ValueError("Repository URL must be a string.")

        value = value.strip()
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise ValueError("Enter a valid GitHub repository URL.") from error

        parts = parsed.path.strip("/").split("/")
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.hostname.lower() != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
            or parsed.query
            or parsed.fragment
            or len(parts) != 2
        ):
            raise ValueError("Use a repository URL like https://github.com/owner/repository.")

        owner, repository = parts
        if repository.lower().endswith(".git"):
            repository = repository[:-4]
        if (
            not re.fullmatch(r"[A-Za-z0-9-]+", owner)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", repository)
            or repository in ("", ".", "..")
        ):
            raise ValueError("Enter a valid GitHub owner and repository name.")

        # Normalize case and the optional .git suffix so equivalent URLs share
        # the same value in the unique repository/branch index.
        return f"https://github.com/{owner.lower()}/{repository.lower()}"


class ProjectResponse(BaseModel):
    """Public project fields returned by the API."""

    id: str
    name: str
    repository_url: str
    branch: str
    project_type: str
    deployment_config: dict[str, object]
    created_at: datetime
    updated_at: datetime
    workspace: "WorkspaceSummary | None" = None


class DetectionSummary(BaseModel):
    """Safe, compact application-detection information for a project."""

    status: str
    application_type: str
    build_method: str
    detected_application_types: list[str]
    detected_build_methods: list[str]
    detected_files: list[str]
    confidence: float
    warnings: list[str]
    commit_hash: str
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(ProjectResponse):
    """Project list fields, including the current workspace's detection."""

    detection: DetectionSummary | None = None


class WorkspaceError(BaseModel):
    """Safe summary of a failed clone attempt."""

    code: str
    message: str
    occurred_at: datetime


class WorkspaceSummary(BaseModel):
    """Clone status included with each project listing."""

    status: str
    repository_url: str
    branch: str
    commit_hash: str | None = None
    error: WorkspaceError | None = None
    created_at: datetime
    updated_at: datetime


def _format_project(document: dict) -> dict:
    """Convert a MongoDB project document into its API representation."""
    created_at = document["created_at"]
    updated_at = document["updated_at"]
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    return {
        "id": str(document["_id"]),
        "name": document["name"],
        "repository_url": document["repository_url"],
        "branch": document["branch"],
        "project_type": document["project_type"],
        "deployment_config": document.get("deployment_config", {}),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _format_detection_summary(document: dict) -> dict:
    """Keep only public summary fields and normalize MongoDB UTC dates."""
    summary = {
        field: document[field]
        for field in (
            "status",
            "application_type",
            "build_method",
            "detected_application_types",
            "detected_build_methods",
            "detected_files",
            "confidence",
            "warnings",
            "commit_hash",
            "created_at",
            "updated_at",
        )
    }
    summary["detected_files"] = [
        file_path
        for file_path in summary["detected_files"]
        if _is_workspace_relative_path(file_path)
    ]
    for field in ("created_at", "updated_at"):
        if summary[field].tzinfo is None:
            summary[field] = summary[field].replace(tzinfo=timezone.utc)
    return summary


def _is_workspace_relative_path(file_path: object) -> bool:
    """Reject absolute, drive-qualified, or parent-traversing stored paths."""
    if not isinstance(file_path, str) or not file_path:
        return False
    posix_path = PurePosixPath(file_path)
    windows_path = PureWindowsPath(file_path)
    return (
        not posix_path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and ".." not in posix_path.parts
        and ".." not in windows_path.parts
    )


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate) -> ProjectResponse:
    """Register a GitHub repository without cloning or contacting GitHub."""
    now = datetime.now(timezone.utc)
    document = {
        "_id": ObjectId(),
        "name": project.name,
        "repository_url": project.repository_url,
        "branch": project.branch,
        # These fields are required by the existing MongoDB validator.
        "project_type": "unknown",
        "deployment_config": {},
        "created_at": now,
        "updated_at": now,
    }

    try:
        connection.database.projects.insert_one(document)
    except DuplicateKeyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A project is already registered for this repository and branch.",
        ) from error
    except PyMongoError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The project could not be saved because MongoDB is unavailable.",
        ) from error

    return ProjectResponse(**_format_project(document))


@router.get("", response_model=list[ProjectListResponse])
def list_projects() -> list[ProjectListResponse]:
    """Return projects with workspace and current-commit detection summaries."""
    try:
        documents = list(connection.database.projects.find().sort("created_at", -1))
        project_ids = [document["_id"] for document in documents]
        workspace_by_project: dict[ObjectId, dict] = {}
        detection_by_pair: dict[tuple[ObjectId, ObjectId], dict] = {}
        if project_ids:
            workspaces = list(
                connection.database.repository_workspaces.find(
                    {"project_id": {"$in": project_ids}}
                )
            )
            workspace_by_project = {
                workspace["project_id"]: workspace for workspace in workspaces
            }

            cloned_workspaces = [
                workspace
                for workspace in workspaces
                if workspace.get("status") == "cloned"
                and isinstance(workspace.get("commit_hash"), str)
            ]
            if cloned_workspaces:
                detections = connection.database.application_detections.find(
                    {
                        "project_id": {"$in": [item["project_id"] for item in cloned_workspaces]},
                        "workspace_id": {"$in": [item["_id"] for item in cloned_workspaces]},
                    }
                )
                detection_by_pair = {
                    (detection["project_id"], detection["workspace_id"]): detection
                    for detection in detections
                }

        projects = []
        for document in documents:
            project_data = _format_project(document)
            workspace = workspace_by_project.get(document["_id"])
            if workspace is not None:
                workspace_data = {
                    "status": workspace["status"],
                    "repository_url": workspace["repository_url"],
                    "branch": workspace["branch"],
                    "commit_hash": workspace.get("commit_hash"),
                    "error": workspace.get("error"),
                    "created_at": workspace["created_at"],
                    "updated_at": workspace["updated_at"],
                }
                project_data["workspace"] = workspace_data
                if workspace.get("status") == "cloned":
                    detection = detection_by_pair.get(
                        (document["_id"], workspace["_id"])
                    )
                    # A previous detection is stale once the cloned commit changes.
                    if (
                        detection is not None
                        and detection.get("commit_hash") == workspace.get("commit_hash")
                    ):
                        project_data["detection"] = _format_detection_summary(detection)
            projects.append(ProjectListResponse(**project_data))
        return projects
    except PyMongoError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Projects could not be loaded because MongoDB is unavailable.",
        ) from error
