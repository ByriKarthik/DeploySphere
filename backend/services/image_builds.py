"""Coordinate image-build reservations, execution, and MongoDB persistence."""

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.database.initialize import MAX_BUILD_LOG_BYTES, MAX_BUILD_LOG_LINE_COUNT
from backend.services.application_detector import DetectionResult
from backend.services.docker_image_builder import (
    BuildError,
    BuildResult,
    build_docker_image,
    create_unique_image_tag,
)
from backend.services.git_clone import GitCloneError, resolve_workspace_paths


class ImageBuildRequestError(Exception):
    """A safe validation or state error suitable for an HTTP response."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ImageBuildConflict(Exception):
    """Another build owns the project and workspace reservation."""


class ImageBuildStoreError(Exception):
    """MongoDB could not reserve or persist an image build."""


def _now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _verified_workspace_path(project_id: ObjectId, workspace: dict) -> Path:
    """Compare the recorded workspace path to its internally derived path."""
    try:
        _, expected_path = resolve_workspace_paths(project_id)
        recorded_value = workspace.get("workspace_path")
        if not isinstance(recorded_value, str):
            raise ValueError("missing workspace path")
        recorded_path = Path(recorded_value).expanduser()
        if not recorded_path.is_absolute() or recorded_path.is_symlink():
            raise ValueError("unsafe workspace path")
        if expected_path.is_symlink() or not expected_path.is_dir():
            raise ValueError("workspace directory unavailable")
        if recorded_path.resolve(strict=True) != expected_path.resolve(strict=True):
            raise ValueError("workspace path mismatch")
        return expected_path
    except (GitCloneError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ImageBuildRequestError(
            "workspace_path_invalid",
            "The cloned workspace path failed verification.",
            409,
        ) from error


def _is_dockerfile_path(value: object) -> bool:
    """Accept only safe, workspace-relative paths with a recognized Dockerfile name."""
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return False
    name = path.name.casefold()
    return (
        name == "dockerfile"
        or name.startswith("dockerfile.")
        or name.startswith("dockerfile-")
        or name.endswith(".dockerfile")
        or ".dockerfile." in name
    )


def _select_dockerfile(detection: dict) -> str:
    """Require a supported, unambiguous detection and exactly one Dockerfile."""
    if detection.get("application_type") not in {"python", "nodejs"}:
        raise ImageBuildRequestError(
            "unsupported_application",
            "Image builds support one detected Python or Node.js application.",
            422,
        )
    if detection.get("build_method") != "dockerfile":
        raise ImageBuildRequestError(
            "unsupported_build_method",
            "Image builds currently require a Dockerfile build method.",
            422,
        )
    if "dockerfile" not in detection.get("detected_build_methods", []):
        raise ImageBuildRequestError(
            "dockerfile_not_detected",
            "The completed detection does not include a Dockerfile build method.",
            422,
        )
    if detection.get("detected_application_types") != [detection["application_type"]]:
        raise ImageBuildRequestError(
            "ambiguous_application",
            "The repository has multiple or inconsistent application types.",
            422,
        )

    files = detection.get("detected_files", [])
    dockerfiles = sorted({item for item in files if _is_dockerfile_path(item)})
    if len(dockerfiles) != 1:
        raise ImageBuildRequestError(
            "ambiguous_dockerfile" if dockerfiles else "dockerfile_not_found",
            "Detection must identify exactly one supported Dockerfile before building.",
            422,
        )
    return dockerfiles[0]


def _load_build_inputs(database, project_id: ObjectId) -> tuple[dict, dict, dict, Path, str]:
    """Load and validate the current project, workspace, detection, and Dockerfile."""
    try:
        project = database.projects.find_one({"_id": project_id})
        if project is None:
            raise ImageBuildRequestError("project_not_found", "Project not found.", 404)
        workspace = database.repository_workspaces.find_one({"project_id": project_id})
        if workspace is None or workspace.get("status") != "cloned":
            raise ImageBuildRequestError(
                "workspace_unavailable",
                "The project does not have a successfully cloned workspace.",
                409,
            )
        workspace_id = workspace.get("_id")
        detection = database.application_detections.find_one(
            {"project_id": project_id, "workspace_id": workspace_id}
        )
    except ImageBuildRequestError:
        raise
    except PyMongoError as error:
        raise ImageBuildStoreError from error

    if detection is None or detection.get("status") != "completed":
        raise ImageBuildRequestError(
            "detection_required",
            "Run successful application detection before building an image.",
            409,
        )
    commit_hash = workspace.get("commit_hash")
    if (
        not isinstance(commit_hash, str)
        or not commit_hash
        or detection.get("commit_hash") != commit_hash
    ):
        raise ImageBuildRequestError(
            "detection_stale",
            "Application detection does not match the current workspace commit. Run detection again.",
            409,
        )
    if detection.get("project_id") != project_id or detection.get("workspace_id") != workspace_id:
        raise ImageBuildRequestError(
            "detection_mismatch", "The detection does not belong to this workspace.", 409
        )

    workspace_path = _verified_workspace_path(project_id, workspace)
    dockerfile_path = _select_dockerfile(detection)
    return project, workspace, detection, workspace_path, dockerfile_path


def _latest_build(collection, project_id: ObjectId, workspace_id: ObjectId) -> dict | None:
    """Return the newest build for one project workspace."""
    try:
        return next(
            iter(
                collection.find(
                    {"project_id": project_id, "workspace_id": workspace_id}
                ).sort("created_at", -1).limit(1)
            ),
            None,
        )
    except PyMongoError as error:
        raise ImageBuildStoreError from error


def _reserve_build(
    collection,
    project_id: ObjectId,
    workspace_id: ObjectId,
    detection: dict,
    dockerfile_path: str,
) -> tuple[dict, bool]:
    """Return a matching success or atomically insert an active reservation."""
    latest = _latest_build(collection, project_id, workspace_id)
    if latest is not None and latest.get("status") in {"queued", "building"}:
        raise ImageBuildConflict("An image build is already active for this workspace.")
    if (
        latest is not None
        and latest.get("status") == "succeeded"
        and latest.get("detection_id") == detection["_id"]
        and latest.get("commit_hash") == detection["commit_hash"]
    ):
        return latest, True

    now = _now()
    attempt_id = uuid4().hex
    record = {
        "_id": ObjectId(),
        "project_id": project_id,
        "workspace_id": workspace_id,
        "detection_id": detection["_id"],
        "commit_hash": detection["commit_hash"],
        "attempt_id": attempt_id,
        "status": "queued",
        "image_tag": create_unique_image_tag(),
        # The whole repository is the context. The detector supplies a path
        # relative to that root, matching build_docker_image's contract.
        "build_context": ".",
        "dockerfile_path": dockerfile_path,
        "build_duration": 0.0,
        "log_bytes": 0,
        "log_line_count": 0,
        "log_truncated": False,
        "created_at": now,
        "updated_at": now,
    }
    try:
        collection.insert_one(record)
    except DuplicateKeyError as error:
        raise ImageBuildConflict(
            "Another image build already reserved this project workspace."
        ) from error
    except PyMongoError as error:
        raise ImageBuildStoreError from error
    return record, False


def _safe_build_metadata(result: BuildResult) -> dict:
    """Keep only bounded, schema-compatible engine metadata."""
    log_bytes = min(result.log_bytes, MAX_BUILD_LOG_BYTES)
    line_count = min(result.log_line_count, MAX_BUILD_LOG_LINE_COUNT)
    return {
        "build_duration": max(0.0, result.duration_seconds),
        "log_bytes": log_bytes,
        "log_line_count": line_count,
        "log_truncated": result.log_truncated
        or result.log_bytes > MAX_BUILD_LOG_BYTES
        or result.log_line_count > MAX_BUILD_LOG_LINE_COUNT,
        "log_excerpt": result.log_excerpt,
    }


def _save_result(collection, reservation: dict, result: BuildResult) -> dict:
    """Persist the engine result with compare-and-set ownership checks."""
    now = _now()
    fields = {**_safe_build_metadata(result), "updated_at": now}
    if result.outcome_known:
        if result.success:
            fields["status"] = "succeeded"
            if result.image_id is not None:
                fields["image_id"] = result.image_id
        else:
            fields["status"] = "failed"
            safe_error = result.error or BuildError(
                "build_failed", "Docker could not complete the image build."
            )
            fields["error"] = {
                "code": safe_error.code,
                "message": safe_error.message[:500],
                "occurred_at": now,
            }
    else:
        # Keep status=building so the partial unique index blocks retries. The
        # timeout does not establish whether the daemon is still building.
        safe_error = result.error or BuildError(
            "build_outcome_indeterminate",
            "The Docker daemon build outcome is unknown.",
        )
        fields["error"] = {
            "code": safe_error.code,
            "message": safe_error.message[:500],
            "occurred_at": now,
        }

    try:
        updated = collection.find_one_and_update(
            {
                "_id": reservation["_id"],
                "attempt_id": reservation["attempt_id"],
                "status": "building",
            },
            {"$set": fields},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        raise ImageBuildStoreError from error
    if updated is None:
        raise ImageBuildConflict("The image-build attempt no longer owns its reservation.")
    return updated


def run_image_build(database, project_id: ObjectId) -> tuple[dict, bool]:
    """Validate, reserve, run one build, and persist its outcome.

    The boolean return value is true when a previously successful matching
    build was returned without running Docker again.
    """
    _, workspace, detection, workspace_path, dockerfile_path = _load_build_inputs(
        database, project_id
    )
    collection = database.image_builds
    reservation, cached = _reserve_build(
        collection,
        project_id,
        workspace["_id"],
        detection,
        dockerfile_path,
    )
    if cached:
        return reservation, True

    try:
        building = collection.find_one_and_update(
            {
                "_id": reservation["_id"],
                "attempt_id": reservation["attempt_id"],
                "status": "queued",
            },
            {"$set": {"status": "building", "updated_at": _now()}},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        # Do not start Docker if MongoDB could not confirm the transition.
        raise ImageBuildStoreError from error
    if building is None:
        raise ImageBuildConflict("The queued image-build attempt was superseded.")

    try:
        result = build_docker_image(
            str(workspace_path),
            reservation["build_context"],
            dockerfile_path,
            reservation["image_tag"],
            DetectionResult(
                application_type=detection["application_type"],
                build_method=detection["build_method"],
                detected_application_types=detection.get("detected_application_types", []),
                detected_build_methods=detection.get("detected_build_methods", []),
                detected_files=detection.get("detected_files", []),
                confidence=detection.get("confidence", 0.0),
                warnings=detection.get("warnings", []),
            ),
        )
    except Exception:
        # If the engine unexpectedly escapes, assume Docker may have started.
        result = BuildResult(
            success=False,
            image_tag=reservation["image_tag"],
            image_id=None,
            duration_seconds=0.0,
            log_bytes=0,
            log_line_count=0,
            log_truncated=False,
            outcome_known=False,
            error=BuildError(
                "build_outcome_indeterminate",
                "The Docker build ended unexpectedly; its outcome is unknown.",
            ),
        )
    return _save_result(collection, building, result), False


def list_image_builds(database, project_id: ObjectId, skip: int, limit: int) -> list[dict]:
    """Return a bounded, newest-first page of build history."""
    try:
        return list(
            database.image_builds.find({"project_id": project_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
    except PyMongoError as error:
        raise ImageBuildStoreError from error


def get_image_build(database, project_id: ObjectId, build_id: ObjectId) -> dict | None:
    """Load a build only when it belongs to the requested project."""
    try:
        return database.image_builds.find_one(
            {"_id": build_id, "project_id": project_id}
        )
    except PyMongoError as error:
        raise ImageBuildStoreError from error
