"""Coordinate container deployment reservations, Docker execution, and MongoDB persistence."""

from datetime import datetime, timezone
import re
from typing import Literal
from uuid import uuid4

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.database.initialize import ACTIVE_DEPLOYMENT_STATUSES
from backend.services import docker_runtime

IMAGE_TAG_PATTERN = re.compile(r"^deploysphere/build:[a-f0-9]{32}$")
ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
ATTEMPT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
RECONCILIATION_ERROR_CODE = "deployment_reconciled_failed"
RECONCILIATION_ERROR_MESSAGE = (
    "Manually reconciled as failed after the deployment container was verified as not running."
)


class DeploymentRequestError(Exception):
    """A safe validation or state error suitable for an HTTP response."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class DeploymentConflictError(Exception):
    """Another deployment owns the project reservation or an idempotency key conflict occurred."""

    def __init__(self, code: str, message: str, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class DeploymentStoreError(Exception):
    """MongoDB could not reserve or persist a deployment record."""


def _now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _normalize_object_id(value: object, field_name: str, status_code: int = 400) -> ObjectId:
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    raise DeploymentRequestError(
        f"invalid_{field_name}",
        f"{field_name} must be a valid ObjectId.",
        status_code,
    )


def _validate_port(port: object) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise DeploymentRequestError(
            "invalid_container_port",
            "container_port must be an integer from 1 to 65535.",
            422,
        )
    return port


def _validate_idempotency_key(key: object) -> str:
    if not isinstance(key, str) or not (1 <= len(key) <= 128):
        raise DeploymentRequestError(
            "invalid_idempotency_key",
            "idempotency_key must be a non-empty string up to 128 characters.",
            422,
        )
    return key


def _validate_timeout(timeout: object) -> int:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not (1 <= timeout <= 300)
    ):
        raise DeploymentRequestError(
            "invalid_timeout",
            "timeout_seconds must be an integer from 1 to 300.",
            422,
        )
    return timeout


def _safe_error(
    error: docker_runtime.RuntimeError | None,
    *,
    default_code: str,
    default_message: str,
    occurred_at: datetime,
) -> dict:
    if error is not None:
        raw_code = getattr(error, "code", default_code)
        code = raw_code if ERROR_CODE_PATTERN.fullmatch(str(raw_code)) and len(str(raw_code)) <= 64 else default_code
        message = str(getattr(error, "message", default_message))[:500]
    else:
        code = default_code
        message = default_message[:500]
    return {
        "code": code,
        "message": message,
        "occurred_at": occurred_at,
    }


def _emergency_mark_unknown(
    collection,
    *,
    deployment_id: ObjectId,
    attempt_id: str,
    expected_statuses: tuple[str, ...],
    container_id: str | None,
    code: str,
    message: str,
) -> None:
    """Best-effort update to record uncertainty without masking primary exceptions."""
    try:
        now = _now()
        fields: dict = {
            "status": "unknown",
            "error": {
                "code": code,
                "message": message[:500],
                "occurred_at": now,
            },
            "updated_at": now,
        }
        if container_id:
            fields["container_id"] = container_id
        collection.update_one(
            {
                "_id": deployment_id,
                "attempt_id": attempt_id,
                "status": {"$in": expected_statuses},
            },
            {"$set": fields},
        )
    except Exception:
        # If the emergency write fails, leave existing active state intact
        pass


def _update_status(
    collection,
    *,
    deployment_id: ObjectId,
    attempt_id: str,
    expected_status: str,
    target_status: str,
    extra_fields: dict | None = None,
    unset_fields: tuple[str, ...] = (),
    now: datetime,
) -> dict:
    fields = {"status": target_status, "updated_at": now}
    if extra_fields:
        fields.update(extra_fields)
    try:
        update = {"$set": fields}
        if unset_fields:
            update["$unset"] = {field: "" for field in unset_fields}
        updated = collection.find_one_and_update(
            {
                "_id": deployment_id,
                "attempt_id": attempt_id,
                "status": expected_status,
            },
            update,
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        raise DeploymentStoreError(
            f"MongoDB could not transition deployment to {target_status}."
        ) from error
    if updated is None:
        raise DeploymentConflictError(
            "deployment_superseded",
            f"The deployment attempt was superseded before reaching {target_status}.",
            409,
        )
    return updated


def _load_deployment_inputs(
    database: Database,
    project_id: ObjectId,
    image_build_id: ObjectId,
) -> tuple[dict, dict]:
    """Validate project and image build existence, ownership, and metadata."""
    try:
        project = database.projects.find_one({"_id": project_id})
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB could not load the project.") from error

    if project is None:
        raise DeploymentRequestError("project_not_found", "Project not found.", 404)

    try:
        build = database.image_builds.find_one({
            "_id": image_build_id,
            "project_id": project_id,
        })
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB could not load the image build.") from error

    if build is None:
        try:
            other_build = database.image_builds.find_one({"_id": image_build_id})
        except PyMongoError as error:
            raise DeploymentStoreError("MongoDB could not load the image build.") from error

        if other_build is not None:
            raise DeploymentRequestError(
                "image_build_mismatch",
                "The image build belongs to a different project.",
                409,
            )
        raise DeploymentRequestError("image_build_not_found", "Image build not found.", 404)

    if build.get("status") != "succeeded":
        raise DeploymentRequestError(
            "image_build_not_succeeded",
            "Only successfully built images can be deployed.",
            409,
        )

    image_tag = build.get("image_tag")
    if not isinstance(image_tag, str) or not IMAGE_TAG_PATTERN.fullmatch(image_tag):
        raise DeploymentRequestError(
            "invalid_image_tag",
            "The image build tag is invalid or corrupted.",
            409,
        )

    workspace_id = build.get("workspace_id")
    detection_id = build.get("detection_id")
    if not isinstance(workspace_id, ObjectId) or not isinstance(detection_id, ObjectId):
        raise DeploymentRequestError(
            "missing_build_metadata",
            "The image build lacks required workspace or detection metadata.",
            409,
        )

    return project, build


def _reserve_deployment(
    database: Database,
    project_id: ObjectId,
    build: dict,
    container_port: int,
    idempotency_key: str,
) -> tuple[dict, bool]:
    """Reserve a new deployment record or return an idempotent replay."""
    try:
        existing = database.deployments.find_one({
            "project_id": project_id,
            "idempotency_key": idempotency_key,
        })
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB could not check idempotency key.") from error

    if existing is not None:
        if (
            existing.get("image_build_id") == build["_id"]
            and existing.get("container_port") == container_port
        ):
            return existing, True
        raise DeploymentConflictError(
            "idempotency_conflict",
            "The idempotency key was already used with different deployment parameters.",
            409,
        )

    deployment_id = ObjectId()
    attempt_id = uuid4().hex
    container_name = f"deploysphere-{deployment_id}"
    now = _now()
    record = {
        "_id": deployment_id,
        "project_id": project_id,
        "image_build_id": build["_id"],
        "workspace_id": build["workspace_id"],
        "detection_id": build["detection_id"],
        "attempt_id": attempt_id,
        "idempotency_key": idempotency_key,
        "status": "queued",
        "commit_hash": build.get("commit_hash"),
        "docker_image": build["image_tag"],
        "container_name": container_name,
        "container_port": container_port,
        "created_at": now,
        "updated_at": now,
    }

    try:
        database.deployments.insert_one(record)
        return record, False
    except DuplicateKeyError as error:
        try:
            existing = database.deployments.find_one({
                "project_id": project_id,
                "idempotency_key": idempotency_key,
            })
        except PyMongoError as store_error:
            raise DeploymentStoreError("MongoDB could not re-check idempotency key.") from store_error

        if existing is not None:
            if (
                existing.get("image_build_id") == build["_id"]
                and existing.get("container_port") == container_port
            ):
                return existing, True
            raise DeploymentConflictError(
                "idempotency_conflict",
                "The idempotency key was already used with different deployment parameters.",
                409,
            ) from error

        raise DeploymentConflictError(
            "active_deployment_exists",
            "An active deployment already exists for this project.",
            409,
        ) from error
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB could not insert deployment reservation.") from error


def deploy_project(
    database: Database,
    project_id: ObjectId | str,
    image_build_id: ObjectId | str,
    container_port: int,
    idempotency_key: str,
    *,
    timeout_seconds: int = docker_runtime.DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> tuple[dict, bool]:
    """Validate, reserve, create, start, and persist one container deployment."""
    p_id = _normalize_object_id(project_id, "project_id", 400)
    b_id = _normalize_object_id(
        image_build_id, "image_build_id", 422 if image_build_id is None else 400
    )
    port = _validate_port(container_port)
    key = _validate_idempotency_key(idempotency_key)
    timeout = _validate_timeout(timeout_seconds)

    _, build = _load_deployment_inputs(database, p_id, b_id)
    reservation, is_replay = _reserve_deployment(database, p_id, build, port, key)
    if is_replay:
        return reservation, True

    deployment_id = reservation["_id"]
    attempt_id = reservation["attempt_id"]
    image_tag = build["image_tag"]

    # Stage A: Begin creation (queued -> creating)
    now = _now()
    try:
        creating_doc = database.deployments.find_one_and_update(
            {
                "_id": deployment_id,
                "attempt_id": attempt_id,
                "status": "queued",
            },
            {"$set": {"status": "creating", "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB could not transition deployment to creating state.") from error

    if creating_doc is None:
        raise DeploymentConflictError(
            "deployment_superseded",
            "The queued deployment attempt was superseded.",
            409,
        )

    # Stage B: Create container
    try:
        create_result = docker_runtime.create_container(
            image_tag=image_tag,
            project_id=p_id,
            deployment_id=deployment_id,
            container_port=port,
            timeout_seconds=timeout,
        )
    except Exception:
        create_result = docker_runtime.RuntimeResult(
            operation="create",
            outcome="indeterminate",
            container_id=None,
            container_name=reservation["container_name"],
            container_port=port,
            error=docker_runtime.RuntimeError(
                "docker_outcome_indeterminate",
                "The Docker container creation ended unexpectedly; its daemon outcome is unknown.",
            ),
        )

    if create_result.outcome == "failed":
        now = _now()
        error_info = _safe_error(
            create_result.error,
            default_code="docker_command_failed",
            default_message="Docker could not create the container.",
            occurred_at=now,
        )
        return _update_status(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_status="creating",
            target_status="failed",
            extra_fields={"error": error_info},
            now=now,
        ), False

    if create_result.outcome == "indeterminate":
        now = _now()
        error_info = _safe_error(
            create_result.error,
            default_code="docker_outcome_indeterminate",
            default_message="The Docker container creation outcome is unknown.",
            occurred_at=now,
        )
        extra = {"error": error_info}
        if create_result.container_id:
            extra["container_id"] = create_result.container_id
        return _update_status(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_status="creating",
            target_status="unknown",
            extra_fields=extra,
            now=now,
        ), False

    # Stage C: Persist container identity before start (creating -> starting)
    container_id = create_result.container_id
    if not container_id:
        now = _now()
        error_info = {
            "code": "invalid_docker_response",
            "message": "Docker reported creation success but returned no container ID.",
            "occurred_at": now,
        }
        return _update_status(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_status="creating",
            target_status="unknown",
            extra_fields={"error": error_info},
            now=now,
        ), False

    now = _now()
    try:
        starting_doc = database.deployments.find_one_and_update(
            {
                "_id": deployment_id,
                "attempt_id": attempt_id,
                "status": "creating",
            },
            {"$set": {
                "status": "starting",
                "container_id": container_id,
                "updated_at": now,
            }},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        _emergency_mark_unknown(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_statuses=("creating", "starting"),
            container_id=container_id,
            code="mongodb_persistence_failed",
            message="Failed to persist container ID before starting.",
        )
        raise DeploymentStoreError("MongoDB could not persist container ID before start.") from error

    if starting_doc is None:
        raise DeploymentConflictError(
            "deployment_superseded",
            "The deployment attempt was superseded before start.",
            409,
        )

    # Stage D: Start container
    try:
        start_result = docker_runtime.start_container(
            container_id=container_id,
            project_id=p_id,
            deployment_id=deployment_id,
            timeout_seconds=timeout,
        )
    except Exception:
        start_result = docker_runtime.RuntimeResult(
            operation="start",
            outcome="indeterminate",
            container_id=container_id,
            container_name=reservation["container_name"],
            container_port=port,
            error=docker_runtime.RuntimeError(
                "docker_outcome_indeterminate",
                "The Docker start operation ended unexpectedly; its daemon outcome is unknown.",
            ),
        )

    if start_result.outcome == "failed":
        now = _now()
        error_info = _safe_error(
            start_result.error,
            default_code="docker_command_failed",
            default_message="Docker could not start the container.",
            occurred_at=now,
        )
        return _update_status(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_status="starting",
            target_status="failed",
            extra_fields={"error": error_info},
            now=now,
        ), False

    if start_result.outcome == "indeterminate":
        now = _now()
        error_info = _safe_error(
            start_result.error,
            default_code="docker_outcome_indeterminate",
            default_message="The Docker container start outcome is unknown.",
            occurred_at=now,
        )
        return _update_status(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_status="starting",
            target_status="unknown",
            extra_fields={"error": error_info},
            now=now,
        ), False

    # Stage E: Verify running state & allocated host port
    allocated_port = start_result.allocated_port
    if (
        allocated_port is None
        or isinstance(allocated_port, bool)
        or not isinstance(allocated_port, int)
        or not (1 <= allocated_port <= 65535)
        or start_result.state != "running"
    ):
        now = _now()
        error_info = {
            "code": "port_binding_unavailable",
            "message": "The container is running but its allocated loopback port could not be verified.",
            "occurred_at": now,
        }
        return _update_status(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_status="starting",
            target_status="unknown",
            extra_fields={"error": error_info},
            now=now,
        ), False

    # Persist running state
    deployment_url = f"http://127.0.0.1:{allocated_port}"
    now = _now()
    try:
        running_doc = database.deployments.find_one_and_update(
            {
                "_id": deployment_id,
                "attempt_id": attempt_id,
                "status": "starting",
            },
            {"$set": {
                "status": "running",
                "allocated_port": allocated_port,
                "deployment_url": deployment_url,
                "started_at": now,
                "updated_at": now,
            }},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        _emergency_mark_unknown(
            database.deployments,
            deployment_id=deployment_id,
            attempt_id=attempt_id,
            expected_statuses=("starting",),
            container_id=container_id,
            code="mongodb_persistence_failed",
            message="Failed to persist running state after starting container.",
        )
        raise DeploymentStoreError(
            "MongoDB could not persist running state after container start."
        ) from error

    if running_doc is None:
        raise DeploymentConflictError(
            "deployment_superseded",
            "The deployment attempt was superseded before running state could be persisted.",
            409,
        )

    return running_doc, False


def get_deployment(
    database: Database,
    project_id: ObjectId | str,
    deployment_id: ObjectId | str,
) -> dict | None:
    """Load a deployment only when it belongs to the requested project."""
    try:
        p_id = _normalize_object_id(project_id, "project_id")
        d_id = _normalize_object_id(deployment_id, "deployment_id")
    except DeploymentRequestError:
        return None

    try:
        return database.deployments.find_one({"_id": d_id, "project_id": p_id})
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB query failed.") from error


def list_deployments(
    database: Database,
    project_id: ObjectId | str,
    skip: int = 0,
    limit: int = 50,
) -> list[dict]:
    """Return a bounded, newest-first page of deployment history."""
    p_id = _normalize_object_id(project_id, "project_id")
    if not isinstance(skip, int) or isinstance(skip, bool) or not (0 <= skip <= 10_000):
        raise DeploymentRequestError(
            "invalid_pagination", "skip must be an integer between 0 and 10000.", 422
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 100):
        raise DeploymentRequestError(
            "invalid_pagination", "limit must be an integer between 1 and 100.", 422
        )

    try:
        return list(
            database.deployments.find({"project_id": p_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB query failed.") from error


def get_active_deployment(
    database: Database,
    project_id: ObjectId | str,
) -> dict | None:
    """Return the active deployment for a project, if one exists."""
    try:
        p_id = _normalize_object_id(project_id, "project_id")
    except DeploymentRequestError:
        return None

    try:
        return database.deployments.find_one({
            "project_id": p_id,
            "status": {"$in": list(ACTIVE_DEPLOYMENT_STATUSES)},
        })
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB query failed.") from error


def reconcile_unknown_deployment_as_failed(
    database: Database,
    project_id: ObjectId | str,
    deployment_id: ObjectId | str,
    container_id: str,
    attempt_id: str,
) -> dict:
    """Release one verified, non-running unknown deployment without touching Docker."""
    p_id = _normalize_object_id(project_id, "project_id")
    d_id = _normalize_object_id(deployment_id, "deployment_id")
    if not isinstance(container_id, str) or not docker_runtime.CONTAINER_ID_PATTERN.fullmatch(
        container_id
    ):
        raise DeploymentRequestError(
            "invalid_container_id",
            "container_id must be a full Docker container ID.",
            422,
        )
    if not isinstance(attempt_id, str) or not ATTEMPT_ID_PATTERN.fullmatch(attempt_id):
        raise DeploymentRequestError(
            "invalid_attempt_id",
            "attempt_id must be a valid deployment attempt ID.",
            422,
        )

    filter_document = {
        "_id": d_id,
        "project_id": p_id,
        "status": "unknown",
        "container_id": container_id,
        "attempt_id": attempt_id,
    }

    now = _now()
    try:
        reconciled = database.deployments.find_one_and_update(
            filter_document,
            {
                "$set": {
                    "status": "failed",
                    "updated_at": now,
                    "error": {
                        "code": RECONCILIATION_ERROR_CODE,
                        "message": RECONCILIATION_ERROR_MESSAGE,
                        "occurred_at": now,
                    },
                }
            },
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        raise DeploymentStoreError("MongoDB could not reconcile the deployment.") from error

    if reconciled is None:
        raise DeploymentConflictError(
            "deployment_reconciliation_conflict",
            "The deployment no longer matches the verified unknown deployment identity.",
        )
    return reconciled


def _operate_deployment(
    database: Database,
    project_id: ObjectId | str,
    deployment_id: ObjectId | str,
    *,
    operation: Literal["stop", "restart"],
    timeout_seconds: int = docker_runtime.DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> dict:
    """Run one safe lifecycle operation on a project-owned deployment."""
    deployment = get_deployment(database, project_id, deployment_id)
    if deployment is None:
        raise DeploymentRequestError("deployment_not_found", "Deployment not found.", 404)

    original_status = deployment.get("status")
    allowed_statuses = {
        "stop": ("running",),
        "restart": ("running", "stopped"),
    }
    if operation not in allowed_statuses:
        raise ValueError("Unsupported deployment operation.")
    if original_status not in allowed_statuses[operation]:
        action = "stopped" if operation == "stop" else "restarted"
        raise DeploymentConflictError(
            "invalid_deployment_state",
            f"A deployment in {original_status!r} state cannot be {action}.",
        )

    attempt_id = deployment.get("attempt_id")
    container_id = deployment.get("container_id")
    if not isinstance(attempt_id, str) or not attempt_id or not isinstance(container_id, str) or not container_id:
        raise DeploymentConflictError(
            "deployment_identity_unavailable",
            "The deployment does not have a persisted container identity.",
        )

    transitional_status = "stopping" if operation == "stop" else "restarting"
    deployment_id_value = deployment["_id"]
    project_id_value = deployment["project_id"]
    now = _now()
    try:
        _update_status(
            database.deployments,
            deployment_id=deployment_id_value,
            attempt_id=attempt_id,
            expected_status=original_status,
            target_status=transitional_status,
            now=now,
        )
    except DeploymentStoreError as error:
        if isinstance(error.__cause__, DuplicateKeyError):
            raise DeploymentConflictError(
                "active_deployment_exists",
                "Another active deployment already exists for this project.",
            ) from error
        raise

    try:
        runtime_operation = (
            docker_runtime.stop_container if operation == "stop" else docker_runtime.restart_container
        )
        result = runtime_operation(
            container_id=container_id,
            project_id=project_id_value,
            deployment_id=deployment_id_value,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        result = docker_runtime.RuntimeResult(
            operation=operation,
            outcome="indeterminate",
            container_id=container_id,
            container_name=deployment.get("container_name"),
            error=docker_runtime.RuntimeError(
                "docker_outcome_indeterminate",
                f"The Docker {operation} outcome could not be confirmed.",
            ),
        )

    succeeded = (
        result.outcome == "succeeded"
        and result.owned is True
        and result.state == ("exited" if operation == "stop" else "running")
        and (
            operation == "stop"
            or (
                isinstance(result.allocated_port, int)
                and not isinstance(result.allocated_port, bool)
                and 1 <= result.allocated_port <= 65535
            )
        )
    )
    outcome_known_failed = result.outcome == "failed" and result.owned is not False
    if succeeded:
        target_status = "stopped" if operation == "stop" else "running"
        extra_fields = (
            {"stopped_at": _now()}
            if operation == "stop"
            else {
                "allocated_port": result.allocated_port,
                "deployment_url": f"http://127.0.0.1:{result.allocated_port}",
                "started_at": _now(),
            }
        )
        unset_fields = (
            ("allocated_port", "deployment_url", "error")
            if operation == "stop"
            else ("stopped_at", "error")
        )
    elif outcome_known_failed:
        target_status = original_status
        extra_fields = {
            "error": _safe_error(
                result.error,
                default_code="docker_command_failed",
                default_message=f"Docker could not {operation} the container.",
                occurred_at=_now(),
            )
        }
        unset_fields = ()
    else:
        target_status = "unknown"
        extra_fields = {
            "error": _safe_error(
                result.error,
                default_code="docker_outcome_indeterminate",
                default_message=f"The Docker {operation} outcome is unknown.",
                occurred_at=_now(),
            )
        }
        unset_fields = ("allocated_port", "deployment_url")

    try:
        return _update_status(
            database.deployments,
            deployment_id=deployment_id_value,
            attempt_id=attempt_id,
            expected_status=transitional_status,
            target_status=target_status,
            extra_fields=extra_fields,
            unset_fields=unset_fields,
            now=_now(),
        )
    except DeploymentStoreError:
        _emergency_mark_unknown(
            database.deployments,
            deployment_id=deployment_id_value,
            attempt_id=attempt_id,
            expected_statuses=(transitional_status,),
            container_id=container_id,
            code="mongodb_persistence_failed",
            message=f"Failed to persist deployment state after Docker {operation}.",
        )
        raise


def stop_deployment(
    database: Database,
    project_id: ObjectId | str,
    deployment_id: ObjectId | str,
    *,
    timeout_seconds: int = docker_runtime.DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> dict:
    """Stop a running deployment without removing its container."""
    return _operate_deployment(
        database,
        project_id,
        deployment_id,
        operation="stop",
        timeout_seconds=timeout_seconds,
    )


def restart_deployment(
    database: Database,
    project_id: ObjectId | str,
    deployment_id: ObjectId | str,
    *,
    timeout_seconds: int = docker_runtime.DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> dict:
    """Restart a running or stopped deployment without recreating its container."""
    return _operate_deployment(
        database,
        project_id,
        deployment_id,
        operation="restart",
        timeout_seconds=timeout_seconds,
    )
