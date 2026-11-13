"""Disposable MongoDB tests for the Milestone 8.1 deployment schema."""

from datetime import datetime, timezone
from uuid import uuid4

from bson import BSON, ObjectId
import pytest
from pymongo.errors import OperationFailure

from backend.database.connection import client
from backend.database.initialize import (
    ACTIVE_DEPLOYMENT_STATUSES,
    COLLECTION_VALIDATORS,
    initialize_database,
)


@pytest.fixture
def disposable_database():
    """Create and clean up only a uniquely named test database."""
    name = f"deploysphere_deployments_test_{uuid4().hex}"
    database = client[name]
    try:
        yield database
    finally:
        client.drop_database(name)


def _deployment(project_id=None, **overrides):
    """Return a valid new-format deployment document."""
    now = datetime.now(timezone.utc)
    document = {
        "project_id": project_id or ObjectId(),
        "image_build_id": ObjectId(),
        "workspace_id": ObjectId(),
        "detection_id": ObjectId(),
        "attempt_id": uuid4().hex,
        "idempotency_key": uuid4().hex,
        "status": "queued",
        "commit_hash": "a" * 40,
        "docker_image": "deploysphere/test:build-1",
        "container_id": "container-id",
        "container_name": f"deploysphere-{ObjectId()}",
        "container_port": 8080,
        "allocated_port": 49152,
        "deployment_url": "http://localhost:49152",
        "build_duration": 1.5,
        "health_path": "/health",
        "health_status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    document.update(overrides)
    return document


def _mongo_normalized(document):
    """Match MongoDB's millisecond datetime precision and naive client reads."""
    document_without_id = {
        key: value for key, value in document.items() if key != "_id"
    }
    return BSON.encode(document_without_id).decode()


def test_new_deployment_document_validates_and_keeps_ports_separate(
    disposable_database,
):
    initialize_database(disposable_database)

    deployment = _deployment()
    result = disposable_database.deployments.insert_one(deployment)

    saved = disposable_database.deployments.find_one({"_id": result.inserted_id})
    assert saved["container_port"] == 8080
    assert saved["allocated_port"] == 49152
    assert saved["status"] == "queued"


@pytest.mark.parametrize("legacy_status", ["complete", "succeeded"])
def test_legacy_statuses_and_missing_new_fields_remain_compatible(
    disposable_database, legacy_status
):
    initialize_database(disposable_database)
    legacy = {
        "project_id": ObjectId(),
        "status": legacy_status,
        "created_at": datetime.now(timezone.utc),
    }

    inserted_id = disposable_database.deployments.insert_one(legacy).inserted_id

    expected = _mongo_normalized(legacy)
    expected["_id"] = inserted_id
    assert disposable_database.deployments.find_one({"_id": inserted_id}) == expected


@pytest.mark.parametrize("legacy_status", ["complete", "succeeded"])
def test_legacy_deployment_allows_ordinary_field_update(
    disposable_database, legacy_status
):
    initialize_database(disposable_database)
    legacy = {
        "project_id": ObjectId(),
        "status": legacy_status,
        "created_at": datetime.now(timezone.utc),
    }
    inserted_id = disposable_database.deployments.insert_one(legacy).inserted_id

    result = disposable_database.deployments.update_one(
        {"_id": inserted_id},
        {"$set": {"deployment_url": "http://localhost:49152"}},
    )

    assert result.matched_count == 1
    assert result.modified_count == 1
    saved = disposable_database.deployments.find_one({"_id": inserted_id})
    assert saved["status"] == legacy_status
    assert saved["deployment_url"] == "http://localhost:49152"
    assert saved["project_id"] == legacy["project_id"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": "not-an-object-id"},
        {"container_port": "8080"},
        {"container_port": 0},
        {"container_port": 65536},
        {"allocated_port": 0},
        {"allocated_port": 65536},
        {"image_build_id": "not-an-object-id"},
        {"created_at": "not-a-date"},
        {"status": "not-a-deployment-status"},
        {"idempotency_key": ""},
        {"container_name": "../container"},
    ],
)
def test_invalid_deployment_documents_are_rejected(disposable_database, overrides):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.deployments.insert_one(_deployment(**overrides))


def test_deployment_required_legacy_fields_remain_required(disposable_database):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.deployments.insert_one(
            {"project_id": ObjectId(), "status": "queued"}
        )


def test_new_lifecycle_status_requires_new_deployment_identity_fields(
    disposable_database,
):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.deployments.insert_one(
            {
                "project_id": ObjectId(),
                "status": "queued",
                "created_at": datetime.now(timezone.utc),
            }
        )


def test_idempotency_key_is_unique_per_project_but_legacy_missing_keys_are_excluded(
    disposable_database,
):
    initialize_database(disposable_database)
    project_id = ObjectId()
    idempotency_key = "request-123"

    disposable_database.deployments.insert_one(
        _deployment(project_id, idempotency_key=idempotency_key)
    )
    with pytest.raises(OperationFailure):
        disposable_database.deployments.insert_one(
            _deployment(project_id, idempotency_key=idempotency_key)
        )

    # The same key is scoped to a project, and old documents without a key are
    # outside the partial index so multiple legacy records remain untouched.
    disposable_database.deployments.insert_one(
        _deployment(ObjectId(), idempotency_key=idempotency_key)
    )
    disposable_database.deployments.insert_one(
        {"project_id": project_id, "status": "complete", "created_at": datetime.now(timezone.utc)}
    )
    disposable_database.deployments.insert_one(
        {"project_id": project_id, "status": "succeeded", "created_at": datetime.now(timezone.utc)}
    )


@pytest.mark.parametrize("active_status", ACTIVE_DEPLOYMENT_STATUSES)
def test_each_defined_active_status_is_unique_per_project(
    disposable_database, active_status
):
    initialize_database(disposable_database)
    project_id = ObjectId()
    disposable_database.deployments.insert_one(
        _deployment(project_id, status=active_status)
    )

    with pytest.raises(OperationFailure):
        disposable_database.deployments.insert_one(
            _deployment(project_id, status="unknown")
        )


@pytest.mark.parametrize("terminal_status", ["failed", "stopped"])
def test_failed_and_stopped_are_excluded_from_active_index(
    disposable_database, terminal_status
):
    initialize_database(disposable_database)
    project_id = ObjectId()
    disposable_database.deployments.insert_one(
        _deployment(project_id, status=terminal_status)
    )
    disposable_database.deployments.insert_one(
        _deployment(project_id, status=terminal_status)
    )

    assert disposable_database.deployments.count_documents(
        {"project_id": project_id}
    ) == 2


def test_initialization_reports_duplicate_legacy_active_records_without_repairing(
    disposable_database,
):
    project_id = ObjectId()
    now = datetime.now(timezone.utc)
    original = [
        {"project_id": project_id, "status": "queued", "created_at": now},
        {"project_id": project_id, "status": "unknown", "created_at": now},
    ]
    disposable_database.create_collection("deployments")
    disposable_database.deployments.insert_many(original)

    with pytest.raises(ValueError, match="Cannot create the unique active deployment index") as error:
        initialize_database(disposable_database)

    assert str(project_id) in str(error.value)
    saved = list(disposable_database.deployments.find({"project_id": project_id}))
    saved_without_ids = [
        {key: value for key, value in item.items() if key != "_id"}
        for item in saved
    ]
    assert saved_without_ids == [_mongo_normalized(item) for item in original]


def test_repeat_initialization_preserves_existing_deployment_record(disposable_database):
    initialize_database(disposable_database)
    original = _deployment(status="running")
    inserted_id = disposable_database.deployments.insert_one(original).inserted_id

    initialize_database(disposable_database)
    initialize_database(disposable_database)

    saved = disposable_database.deployments.find_one({"_id": inserted_id})
    saved.pop("_id")
    assert saved == _mongo_normalized(original)


def test_validator_declares_additive_deployment_fields_and_legacy_statuses():
    schema = COLLECTION_VALIDATORS["deployments"]["$jsonSchema"]

    assert schema["required"] == ["project_id", "status", "created_at"]
    legacy_branch, new_lifecycle_branch = schema["oneOf"]
    assert legacy_branch["properties"]["status"]["enum"] == [
        "complete", "succeeded"
    ]
    assert set(new_lifecycle_branch["required"]) >= {
        "image_build_id",
        "workspace_id",
        "detection_id",
        "attempt_id",
        "idempotency_key",
        "container_name",
        "container_port",
        "updated_at",
    }
    assert set(schema["properties"]["status"]["enum"]) >= {
        *ACTIVE_DEPLOYMENT_STATUSES,
        "failed",
        "stopped",
        "complete",
        "succeeded",
    }
    assert schema["properties"]["container_port"]["bsonType"] == "int"
    assert schema["properties"]["allocated_port"]["bsonType"] == "int"
