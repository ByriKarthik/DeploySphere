"""MongoDB schema and index tests for Docker image build records."""

from datetime import datetime, timezone
from uuid import uuid4

from bson import ObjectId
import pytest
from pymongo.errors import OperationFailure

from backend.database.connection import client
from backend.database.initialize import (
    COLLECTION_VALIDATORS,
    MAX_BUILD_LOG_BYTES,
    MAX_BUILD_LOG_EXCERPT_BYTES,
    MAX_BUILD_LOG_LINE_COUNT,
    initialize_database,
)


@pytest.fixture
def disposable_database():
    """Create a uniquely named database and clean up only that test database."""
    name = f"deploysphere_image_build_test_{uuid4().hex}"
    database = client[name]
    try:
        yield database
    finally:
        client.drop_database(name)


def _valid_image_build(**overrides):
    """Return a representative valid image-build document."""
    now = datetime.now(timezone.utc)
    document = {
        "project_id": ObjectId(),
        "workspace_id": ObjectId(),
        "detection_id": ObjectId(),
        "commit_hash": "a" * 40,
        "attempt_id": uuid4().hex,
        "status": "queued",
        "image_tag": "deploysphere/project:commit-build",
        "build_context": ".",
        "dockerfile_path": "services/api/Dockerfile",
        "build_duration": 0.0,
        "created_at": now,
        "updated_at": now,
        "reservation_expires_at": now,
        "log_bytes": 0,
        "log_line_count": 0,
        "log_truncated": False,
    }
    document.update(overrides)
    return document


def test_image_build_collection_has_strict_validator_and_required_indexes(
    disposable_database,
):
    initialize_database(disposable_database)

    assert "image_builds" in disposable_database.list_collection_names()
    info = next(disposable_database.list_collections(filter={"name": "image_builds"}))
    assert info["options"]["validationLevel"] == "strict"
    assert info["options"]["validationAction"] == "error"
    assert info["options"]["validator"] == COLLECTION_VALIDATORS["image_builds"]

    indexes = disposable_database.image_builds.index_information()
    assert indexes["attempt_id_1"]["unique"] is True
    active_index = indexes["one_active_build_per_workspace"]
    assert active_index["unique"] is True
    assert active_index["partialFilterExpression"] == {
        "status": {"$in": ["queued", "building"]}
    }
    assert "project_id_1_created_at_-1" in indexes
    assert "workspace_id_1_created_at_-1" in indexes
    assert "status_1_updated_at_-1" in indexes
    assert "project_id_1_status_1_updated_at_-1" in indexes

    inserted = disposable_database.image_builds.insert_one(_valid_image_build())
    assert disposable_database.image_builds.find_one({"_id": inserted.inserted_id})[
        "status"
    ] == "queued"


@pytest.mark.parametrize("status", ["queued", "building", "succeeded", "failed", "cancelled"])
def test_image_build_status_values_are_accepted(disposable_database, status):
    initialize_database(disposable_database)

    inserted = disposable_database.image_builds.insert_one(
        _valid_image_build(status=status)
    )

    assert disposable_database.image_builds.find_one({"_id": inserted.inserted_id})[
        "status"
    ] == status


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "running"},
        {"build_duration": -0.1},
        {"build_duration": "1.5"},
        {"build_context": "/var/private/workspace"},
        {"build_context": "C:/private/workspace"},
        {"build_context": "../outside"},
        {"build_context": "api/../../outside"},
        {"build_context": "api\\..\\outside"},
        {"dockerfile_path": "/etc/Dockerfile"},
        {"dockerfile_path": "C:\\private\\Dockerfile"},
        {"dockerfile_path": "../Dockerfile"},
        {"dockerfile_path": "api/../Dockerfile"},
    ],
)
def test_image_build_rejects_invalid_status_duration_or_paths(
    disposable_database, overrides
):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(_valid_image_build(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": "not-an-object-id"},
        {"workspace_id": "not-an-object-id"},
        {"detection_id": "not-an-object-id"},
        {"attempt_id": 15},
        {"image_id": 123},
        {"reservation_expires_at": "2026-01-01"},
        {"created_at": "2026-01-01"},
        {"updated_at": "2026-01-01"},
        {"log_bytes": "large"},
        {"log_line_count": -1},
        {"log_truncated": "false"},
    ],
)
def test_image_build_bson_types_and_optional_metadata_are_validated(
    disposable_database, overrides
):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(_valid_image_build(**overrides))


def test_build_log_metadata_accepts_maximum_values(disposable_database):
    initialize_database(disposable_database)

    inserted = disposable_database.image_builds.insert_one(
        _valid_image_build(
            log_bytes=MAX_BUILD_LOG_BYTES,
            log_line_count=MAX_BUILD_LOG_LINE_COUNT,
        )
    )

    saved = disposable_database.image_builds.find_one({"_id": inserted.inserted_id})
    assert saved["log_bytes"] == MAX_BUILD_LOG_BYTES
    assert saved["log_line_count"] == MAX_BUILD_LOG_LINE_COUNT


def test_build_log_excerpt_accepts_maximum_length(disposable_database):
    initialize_database(disposable_database)
    excerpt = "x" * MAX_BUILD_LOG_EXCERPT_BYTES

    inserted = disposable_database.image_builds.insert_one(
        _valid_image_build(log_excerpt=excerpt)
    )

    saved = disposable_database.image_builds.find_one({"_id": inserted.inserted_id})
    assert saved["log_excerpt"] == excerpt


def test_build_log_excerpt_rejects_values_above_maximum(disposable_database):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(
            _valid_image_build(log_excerpt="x" * (MAX_BUILD_LOG_EXCERPT_BYTES + 1))
        )


@pytest.mark.parametrize(
    "field, maximum",
    [
        ("log_bytes", MAX_BUILD_LOG_BYTES),
        ("log_line_count", MAX_BUILD_LOG_LINE_COUNT),
    ],
)
def test_build_log_metadata_rejects_values_above_maximum(
    disposable_database, field, maximum
):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(
            _valid_image_build(**{field: maximum + 1})
        )


@pytest.mark.parametrize(
    "field",
    [
        "project_id",
        "workspace_id",
        "detection_id",
        "commit_hash",
        "attempt_id",
        "status",
        "image_tag",
        "build_context",
        "dockerfile_path",
        "build_duration",
        "created_at",
        "updated_at",
    ],
)
def test_image_build_required_fields_are_enforced(disposable_database, field):
    initialize_database(disposable_database)
    document = _valid_image_build()
    document.pop(field)

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(document)


def test_image_build_error_information_is_safe_and_bounded(disposable_database):
    initialize_database(disposable_database)
    document = _valid_image_build(
        status="failed",
        error={
            "code": "image_build_failed",
            "message": "Docker could not complete this build.",
            "occurred_at": datetime.now(timezone.utc),
        },
    )
    inserted = disposable_database.image_builds.insert_one(document)
    assert disposable_database.image_builds.find_one({"_id": inserted.inserted_id})[
        "error"
    ]["code"] == "image_build_failed"

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(
            _valid_image_build(
                error={
                    "code": "contains spaces",
                    "message": "x" * 501,
                    "occurred_at": datetime.now(timezone.utc),
                }
            )
        )


def test_unique_attempt_id_is_enforced(disposable_database):
    initialize_database(disposable_database)
    attempt_id = uuid4().hex
    disposable_database.image_builds.insert_one(
        _valid_image_build(attempt_id=attempt_id)
    )

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(
            _valid_image_build(attempt_id=attempt_id)
        )


def test_only_one_queued_or_building_record_is_allowed_per_workspace(disposable_database):
    initialize_database(disposable_database)
    project_id = ObjectId()
    workspace_id = ObjectId()
    disposable_database.image_builds.insert_one(
        _valid_image_build(project_id=project_id, workspace_id=workspace_id)
    )

    with pytest.raises(OperationFailure):
        disposable_database.image_builds.insert_one(
            _valid_image_build(project_id=project_id, workspace_id=workspace_id)
        )

    # A terminal historical record does not reserve the workspace.
    terminal = disposable_database.image_builds.insert_one(
        _valid_image_build(
            project_id=project_id,
            workspace_id=workspace_id,
            status="failed",
        )
    )
    assert disposable_database.image_builds.find_one({"_id": terminal.inserted_id})[
        "status"
    ] == "failed"


def test_active_duplicate_reservations_stop_index_creation_without_data_changes(
    disposable_database,
):
    collection = disposable_database.create_collection("image_builds")
    project_id = ObjectId()
    workspace_id = ObjectId()
    collection.insert_many(
        [
            _valid_image_build(project_id=project_id, workspace_id=workspace_id),
            _valid_image_build(project_id=project_id, workspace_id=workspace_id),
        ]
    )

    with pytest.raises(ValueError, match="unique active image-build reservation index.*reconcile.*manually"):
        initialize_database(disposable_database)

    assert collection.count_documents({"project_id": project_id, "workspace_id": workspace_id}) == 2


def test_image_build_initialization_is_repeatable_and_preserves_all_records(
    disposable_database,
):
    initialize_database(disposable_database)
    now = datetime.now(timezone.utc)
    project_id = ObjectId()
    workspace_id = ObjectId()
    detection = {
        "project_id": project_id,
        "workspace_id": workspace_id,
        "attempt_id": uuid4().hex,
        "commit_hash": "a" * 40,
        "status": "completed",
        "application_type": "python",
        "build_method": "native",
        "detected_application_types": ["python"],
        "detected_build_methods": ["native"],
        "detected_files": ["requirements.txt"],
        "confidence": 0.8,
        "warnings": [],
        "created_at": now,
        "updated_at": now,
    }
    records = {
        "projects": {
            "name": "Existing project",
            "repository_url": "https://example.test/image-build",
            "branch": "main",
            "project_type": "python",
            "created_at": now,
            "updated_at": now,
        },
        "deployments": {"project_id": project_id, "status": "succeeded", "created_at": now},
        "deployment_logs": {
            "deployment_id": ObjectId(), "level": "info", "message": "kept", "timestamp": now
        },
        "repository_workspaces": {
            "project_id": project_id,
            "status": "cloned",
            "repository_url": "https://example.test/image-build",
            "branch": "main",
            "workspace_path": f"/private/workspaces/{project_id}",
            "commit_hash": "a" * 40,
            "created_at": now,
            "updated_at": now,
        },
        "application_detections": detection,
        "image_builds": _valid_image_build(project_id=project_id, workspace_id=workspace_id),
    }
    inserted_ids = {
        name: disposable_database[name].insert_one(document).inserted_id
        for name, document in records.items()
    }
    before = {
        name: disposable_database[name].find_one({"_id": inserted_id})
        for name, inserted_id in inserted_ids.items()
    }

    initialize_database(disposable_database)
    initialize_database(disposable_database)

    after = {
        name: disposable_database[name].find_one({"_id": inserted_id})
        for name, inserted_id in inserted_ids.items()
    }
    assert after == before
    assert disposable_database.image_builds.count_documents({}) == 1


def test_existing_image_build_validator_and_index_are_preserved(disposable_database):
    legacy_validator = {"legacy_marker": {"$type": "string"}}
    collection = disposable_database.create_collection(
        "image_builds",
        validator=legacy_validator,
        validationLevel="moderate",
        validationAction="warn",
    )
    collection.create_index("legacy_marker")
    legacy_id = collection.insert_one({"legacy_marker": "retain this document"}).inserted_id
    legacy_document = collection.find_one({"_id": legacy_id})

    initialize_database(disposable_database)

    info = next(disposable_database.list_collections(filter={"name": "image_builds"}))
    options = info["options"]
    assert options["validationLevel"] == "moderate"
    assert options["validationAction"] == "warn"
    assert legacy_validator in options["validator"]["$and"]
    assert any("$jsonSchema" in item for item in options["validator"]["$and"])
    assert "legacy_marker_1" in disposable_database.image_builds.index_information()
    assert disposable_database.image_builds.find_one({"_id": legacy_id}) == legacy_document


def test_conflicting_existing_attempt_index_has_actionable_error(disposable_database):
    collection = disposable_database.create_collection("image_builds")
    collection.create_index("attempt_id")

    with pytest.raises(
        RuntimeError,
        match="could not ensure index attempt_id_1.*conflicting keys or options",
    ):
        initialize_database(disposable_database)


def test_duplicate_attempt_ids_abort_initialization_without_deleting_documents(
    disposable_database,
):
    collection = disposable_database.create_collection("image_builds")
    attempt_id = uuid4().hex
    collection.insert_many(
        [
            _valid_image_build(attempt_id=attempt_id),
            _valid_image_build(attempt_id=attempt_id),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Cannot create unique index attempt_id_1.*duplicate indexed values",
    ):
        initialize_database(disposable_database)

    assert disposable_database.image_builds.count_documents({}) == 2
