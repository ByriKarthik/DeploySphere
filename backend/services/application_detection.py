"""Coordinate application scans and persist their results in MongoDB."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.services.application_detector import (
    DetectionError,
    ScanLimits,
    detect_application,
)


# The detector's normal scan limit is ten seconds. This lease leaves ample
# room for that bounded scan; a crashed request can be recovered after expiry.
RESERVATION_LEASE = timedelta(minutes=2)


def _reservation_lease() -> timedelta:
    """Keep a lease longer than the configured bounded detector scan."""
    try:
        limits = ScanLimits.from_environment()
        configured_lease = timedelta(seconds=limits.max_scan_seconds + 30)
    except (DetectionError, OverflowError):
        # Invalid detector configuration fails safely during the scan itself.
        return RESERVATION_LEASE
    return max(RESERVATION_LEASE, configured_lease)


class DetectionConflict(Exception):
    """The scan is already running or a newer attempt replaced this one."""


class DetectionStoreError(Exception):
    """MongoDB could not reserve or save a detection."""


class DetectionRunError(Exception):
    """A detection failed with safe, client-facing error details."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _blank_result() -> dict:
    """Provide valid placeholder fields while a scan is in progress."""
    return {
        "application_type": "unknown",
        "build_method": "unknown",
        "detected_application_types": [],
        "detected_build_methods": [],
        "detected_files": [],
        "confidence": 0.0,
        "warnings": [],
    }


def _claim_existing(
    collection,
    existing: dict,
    commit_hash: str,
    now: datetime,
    lease: timedelta,
) -> dict | None:
    """Compare-and-set an old or failed record into a fresh reservation."""
    attempt_id = uuid4().hex
    query = {"_id": existing["_id"], "status": existing["status"]}
    if "attempt_id" in existing:
        query["attempt_id"] = existing["attempt_id"]
    else:
        query["attempt_id"] = {"$exists": False}

    if existing["status"] == "detecting":
        expiry = existing.get("reservation_expires_at")
        if expiry is not None:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if expiry > now:
                raise DetectionConflict("A detection is already in progress for this workspace.")
            query["reservation_expires_at"] = existing["reservation_expires_at"]
        else:
            updated_at = existing.get("updated_at")
            if updated_at is None:
                raise DetectionConflict("A detection is already in progress for this workspace.")
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            if updated_at > now - lease:
                raise DetectionConflict("A detection is already in progress for this workspace.")
            query["reservation_expires_at"] = {"$exists": False}
            query["updated_at"] = existing["updated_at"]

    try:
        return collection.find_one_and_update(
            query,
            {
                "$set": {
                    **_blank_result(),
                    "status": "detecting",
                    "commit_hash": commit_hash,
                    "attempt_id": attempt_id,
                    "reservation_expires_at": now + lease,
                    "updated_at": now,
                },
                "$unset": {"error": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        raise DetectionStoreError from error


def _reserve_detection(
    collection,
    project_id: ObjectId,
    workspace_id: ObjectId,
    commit_hash: str,
) -> tuple[dict, bool]:
    """Atomically create or claim one project/workspace detection record.

    Returns the reserved record and whether it was created for the first time.
    """
    now = _now()
    lease = _reservation_lease()
    key = {"project_id": project_id, "workspace_id": workspace_id}
    try:
        existing = collection.find_one(key)
        if existing is not None:
            claimed = _claim_existing(collection, existing, commit_hash, now, lease)
            if claimed is None:
                raise DetectionConflict("Another detection attempt has already claimed this workspace.")
            return claimed, False

        attempt_id = uuid4().hex
        document = {
            **key,
            **_blank_result(),
            "commit_hash": commit_hash,
            "attempt_id": attempt_id,
            "status": "detecting",
            "reservation_expires_at": now + lease,
            "created_at": now,
            "updated_at": now,
        }
        collection.insert_one(document)
        return document, True
    except (DetectionConflict, DetectionStoreError):
        raise
    except DuplicateKeyError:
        # A competing first request inserted the unique key. It owns the scan.
        raise DetectionConflict("Another detection attempt has already claimed this workspace.")
    except PyMongoError as error:
        raise DetectionStoreError from error


def run_detection(
    database,
    project_id: ObjectId,
    workspace_id: ObjectId,
    commit_hash: str,
    workspace_path: str,
) -> tuple[dict, int]:
    """Reserve a scan, run the safe detector, and save only the current attempt."""
    collection = database.application_detections
    reservation, first_record = _reserve_detection(
        collection, project_id, workspace_id, commit_hash
    )

    attempt_id = reservation["attempt_id"]
    try:
        result = detect_application(workspace_path)
        output = {
            "application_type": result.application_type,
            "build_method": result.build_method,
            "detected_application_types": result.detected_application_types,
            "detected_build_methods": result.detected_build_methods,
            "detected_files": result.detected_files,
            "confidence": result.confidence,
            "warnings": result.warnings,
        }
    except DetectionError as error:
        failure = DetectionRunError(error.code, error.message, 422)
        _save_failure(collection, reservation, failure)
        raise failure from error
    except Exception as error:
        failure = DetectionRunError(
            "detection_failed", "Application detection failed unexpectedly.", 500
        )
        _save_failure(collection, reservation, failure)
        raise failure from error

    now = _now()
    try:
        updated = collection.find_one_and_update(
            {
                "_id": reservation["_id"],
                "attempt_id": attempt_id,
                "status": "detecting",
                "commit_hash": commit_hash,
            },
            {
                "$set": {**output, "status": "completed", "updated_at": now},
                "$unset": {"error": "", "reservation_expires_at": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as error:
        raise DetectionStoreError from error
    if updated is None:
        raise DetectionConflict("This detection attempt was superseded by a newer attempt.")
    return updated, 201 if first_record else 200


def _save_failure(collection, reservation: dict, error: DetectionRunError) -> None:
    """Save safe failure details only if this attempt still owns the record."""
    now = _now()
    try:
        updated = collection.find_one_and_update(
            {
                "_id": reservation["_id"],
                "attempt_id": reservation["attempt_id"],
                "status": "detecting",
            },
            {
                "$set": {
                    "status": "failed",
                    "error": {"code": error.code, "message": error.message, "occurred_at": now},
                    "updated_at": now,
                },
                "$unset": {"reservation_expires_at": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as database_error:
        raise DetectionStoreError from database_error
    if updated is None:
        raise DetectionConflict("This detection attempt was superseded by a newer attempt.")
