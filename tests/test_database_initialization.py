"""Integration tests for the DeploySphere MongoDB initializer."""

from datetime import datetime, timezone
from uuid import uuid4

from bson import ObjectId
import pytest
from pymongo.errors import OperationFailure

from backend.database import connection
from backend.database.connection import client
from backend.database.initialize import COLLECTION_VALIDATORS, initialize_database


@pytest.fixture
def disposable_database():
    """Provide an isolated database and remove only that test database."""
    name = f"deploysphere_test_{uuid4().hex}"
    db = client[name]
    try:
        yield db
    finally:
        client.drop_database(name)


def test_initialize_creates_collections_and_indexes(disposable_database):
    initialize_database(disposable_database)

    assert set(disposable_database.list_collection_names()) == {
        "projects", "deployments", "deployment_logs", "repository_workspaces",
        "application_detections", "image_builds",
    }

    project_indexes = disposable_database.projects.index_information()
    assert project_indexes["repository_url_1_branch_1"]["unique"] is True

    deployment_indexes = disposable_database.deployments.index_information()
    assert "project_id_1_created_at_-1" in deployment_indexes
    assert "status_1_created_at_-1" in deployment_indexes
    active_index = deployment_indexes["one_active_deployment_per_project"]
    assert active_index["unique"] is True
    assert active_index["partialFilterExpression"] == {
        "status": {
            "$in": [
                "queued", "creating", "starting", "checking_ready",
                "running", "stopping", "restarting", "unknown",
            ]
        }
    }
    idempotency_index = deployment_indexes["project_idempotency_key_unique"]
    assert idempotency_index["unique"] is True
    assert idempotency_index["partialFilterExpression"] == {
        "idempotency_key": {"$type": "string"}
    }

    log_indexes = disposable_database.deployment_logs.index_information()
    assert "deployment_id_1_timestamp_1" in log_indexes

    workspace_indexes = disposable_database.repository_workspaces.index_information()
    assert workspace_indexes["project_id_1"]["unique"] is True
    assert workspace_indexes["workspace_path_1"]["unique"] is True
    assert "status_1_updated_at_-1" in workspace_indexes

    detection_indexes = disposable_database.application_detections.index_information()
    assert detection_indexes["project_id_1_workspace_id_1"]["unique"] is True
    assert "status_1_updated_at_-1" in detection_indexes


def test_workspace_collection_uses_strict_schema_validation(disposable_database):
    initialize_database(disposable_database)

    info = next(
        disposable_database.list_collections(filter={"name": "repository_workspaces"})
    )
    assert info["options"]["validationLevel"] == "strict"
    assert info["options"]["validationAction"] == "error"

    with pytest.raises(OperationFailure):
        disposable_database.repository_workspaces.insert_one(
            {
                "project_id": ObjectId(),
                "status": "complete",
                "repository_url": "https://github.com/example/repo",
                "branch": "main",
                "workspace_path": "C:/work/repo",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )


def _valid_detection(**overrides):
    """Return a valid detection document for collection validation tests."""
    now = datetime.now(timezone.utc)
    document = {
        "project_id": ObjectId(),
        "workspace_id": ObjectId(),
        "attempt_id": uuid4().hex,
        "commit_hash": "a" * 40,
        "status": "completed",
        "application_type": "mixed",
        "build_method": "mixed",
        "detected_application_types": ["python", "nodejs"],
        "detected_build_methods": ["dockerfile", "compose"],
        "detected_files": ["api/requirements.txt", "web/package.json"],
        "confidence": 0.75,
        "warnings": ["Filename detection does not prove deployability."],
        "created_at": now,
        "updated_at": now,
    }
    document.update(overrides)
    return document


def test_application_detection_collection_has_strict_schema(disposable_database):
    initialize_database(disposable_database)

    info = next(
        disposable_database.list_collections(filter={"name": "application_detections"})
    )
    assert info["options"]["validationLevel"] == "strict"
    assert info["options"]["validationAction"] == "error"
    assert info["options"]["validator"] == COLLECTION_VALIDATORS[
        "application_detections"
    ]

    inserted = disposable_database.application_detections.insert_one(_valid_detection())
    assert disposable_database.application_detections.find_one(
        {"_id": inserted.inserted_id}
    )["application_type"] == "mixed"


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0])
def test_detection_confidence_accepts_inclusive_boundaries(disposable_database, confidence):
    initialize_database(disposable_database)

    inserted = disposable_database.application_detections.insert_one(
        _valid_detection(confidence=confidence)
    )
    assert disposable_database.application_detections.find_one(
        {"_id": inserted.inserted_id}
    )["confidence"] == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection_confidence_rejects_values_outside_boundaries(
    disposable_database, confidence
):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.application_detections.insert_one(
            _valid_detection(confidence=confidence)
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "queued"},
        {"application_type": "ruby"},
        {"build_method": "shell"},
        {"detected_application_types": ["python", "ruby"]},
        {"detected_build_methods": ["compose", "shell"]},
        {"detected_files": "package.json"},
        {"warnings": ["warning", 4]},
    ],
)
def test_detection_enum_and_array_types_are_validated(disposable_database, overrides):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.application_detections.insert_one(
            _valid_detection(**overrides)
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": "not-an-object-id"},
        {"workspace_id": "not-an-object-id"},
        {"commit_hash": 123},
        {"created_at": "2026-01-01"},
        {"updated_at": "2026-01-01"},
        {"confidence": "0.5"},
    ],
)
def test_detection_bson_field_types_are_validated(disposable_database, overrides):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.application_detections.insert_one(
            _valid_detection(**overrides)
        )


def test_detection_required_fields_are_enforced(disposable_database):
    initialize_database(disposable_database)
    document = _valid_detection()
    document.pop("workspace_id")

    with pytest.raises(OperationFailure):
        disposable_database.application_detections.insert_one(document)


def test_failed_detection_can_store_optional_safe_error_information(disposable_database):
    initialize_database(disposable_database)
    detection = _valid_detection(
        status="failed",
        error={
            "code": "scan_timeout",
            "message": "Application detection exceeded its time limit.",
            "occurred_at": datetime.now(timezone.utc),
        },
    )

    inserted = disposable_database.application_detections.insert_one(detection)

    saved = disposable_database.application_detections.find_one({"_id": inserted.inserted_id})
    assert saved["status"] == "failed"
    assert saved["error"]["code"] == "scan_timeout"


def test_detection_error_information_validates_nested_bson_types(disposable_database):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.application_detections.insert_one(
            _valid_detection(
                status="failed",
                error={
                    "code": "scan_timeout",
                    "message": "Timed out.",
                    "occurred_at": "not-a-bson-date",
                },
            )
        )


def test_detection_unique_project_workspace_pair_is_enforced(disposable_database):
    initialize_database(disposable_database)
    project_id = ObjectId()
    workspace_id = ObjectId()
    disposable_database.application_detections.insert_one(
        _valid_detection(project_id=project_id, workspace_id=workspace_id)
    )

    with pytest.raises(OperationFailure):
        disposable_database.application_detections.insert_one(
            _valid_detection(project_id=project_id, workspace_id=workspace_id)
        )

    # A distinct workspace for the same project is a different compound key.
    disposable_database.application_detections.insert_one(
        _valid_detection(project_id=project_id, workspace_id=ObjectId())
    )


def test_existing_detection_validator_and_policy_are_preserved(disposable_database):
    existing_validator = {"legacy_marker": {"$type": "string"}}
    disposable_database.create_collection(
        "application_detections",
        validator=existing_validator,
        validationLevel="moderate",
        validationAction="warn",
    )

    initialize_database(disposable_database)

    info = next(
        disposable_database.list_collections(filter={"name": "application_detections"})
    )
    options = info["options"]
    assert options["validationLevel"] == "moderate"
    assert options["validationAction"] == "warn"
    assert existing_validator in options["validator"]["$and"]
    assert _contains_detection_schema(options["validator"])


def _contains_detection_schema(validator):
    """Return whether the new detection schema remains in a combined validator."""
    return any(
        "$jsonSchema" in condition
        for condition in validator.get("$and", [])
        if isinstance(condition, dict)
    )


def test_detection_index_conflict_reports_duplicates_without_deleting_documents(
    disposable_database,
):
    disposable_database.create_collection("application_detections")
    pair = {"project_id": ObjectId(), "workspace_id": ObjectId()}
    disposable_database.application_detections.insert_many(
        [_valid_detection(**pair), _valid_detection(**pair)]
    )

    with pytest.raises(
        ValueError,
        match="Cannot create unique index project_id_1_workspace_id_1.*duplicate indexed values",
    ):
        initialize_database(disposable_database)

    assert disposable_database.application_detections.count_documents({}) == 2


def test_conflicting_existing_detection_index_has_actionable_error(disposable_database):
    collection = disposable_database.create_collection("application_detections")
    collection.create_index([("project_id", 1), ("workspace_id", 1)])

    with pytest.raises(
        RuntimeError,
        match="could not ensure index project_id_1_workspace_id_1.*conflicting keys or options",
    ):
        initialize_database(disposable_database)

    index = collection.index_information()["project_id_1_workspace_id_1"]
    assert index.get("unique", False) is False


def test_initialize_is_repeatable_and_preserves_documents(disposable_database):
    initialize_database(disposable_database)
    project = {
        "name": "Example",
        "repository_url": "https://example.com/repo.git",
        "branch": "main",
        "project_type": "python",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    project_id = disposable_database.projects.insert_one(project).inserted_id
    workspace = {
        "project_id": project_id,
        "status": "cloned",
        "repository_url": project["repository_url"],
        "branch": project["branch"],
        "workspace_path": f"/test-workspaces/{project_id}",
        "commit_hash": "a" * 40,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    workspace_id = disposable_database.repository_workspaces.insert_one(workspace).inserted_id
    detection_id = disposable_database.application_detections.insert_one(
        _valid_detection(project_id=project_id)
    ).inserted_id

    initialize_database(disposable_database)

    assert disposable_database.projects.count_documents({}) == 1
    assert disposable_database.projects.find_one({"_id": project_id})["name"] == "Example"
    assert disposable_database.repository_workspaces.count_documents({}) == 1
    assert disposable_database.repository_workspaces.find_one({"_id": workspace_id})[
        "status"
    ] == "cloned"
    assert disposable_database.application_detections.count_documents({}) == 1
    assert disposable_database.application_detections.find_one({"_id": detection_id})[
        "status"
    ] == "completed"


def test_project_validator_rejects_missing_required_fields(disposable_database):
    initialize_database(disposable_database)

    with pytest.raises(OperationFailure):
        disposable_database.projects.insert_one({"name": "Incomplete"})


def test_existing_collection_without_validator_gets_strict_error_policy(disposable_database):
    disposable_database.create_collection("projects")

    initialize_database(disposable_database)

    info = next(disposable_database.list_collections(filter={"name": "projects"}))
    assert info["options"]["validationLevel"] == "strict"
    assert info["options"]["validationAction"] == "error"


def test_existing_validation_policy_is_preserved(disposable_database):
    disposable_database.create_collection(
        "projects",
        validator={"name": {"$type": "string"}},
        validationLevel="moderate",
        validationAction="warn",
    )

    initialize_database(disposable_database)

    info = next(disposable_database.list_collections(filter={"name": "projects"}))
    assert info["options"]["validationLevel"] == "moderate"
    assert info["options"]["validationAction"] == "warn"
    assert "$and" in info["options"]["validator"]


def test_strict_validator_retains_legacy_documents_but_rejects_updates(disposable_database):
    disposable_database.create_collection("projects")
    project_id = disposable_database.projects.insert_one(
        {"repository_url": "https://example.test/repo", "branch": "main"}
    ).inserted_id

    initialize_database(disposable_database)

    assert disposable_database.projects.find_one({"_id": project_id}) is not None
    with pytest.raises(OperationFailure):
        disposable_database.projects.update_one(
            {"_id": project_id}, {"$set": {"touched": True}}
        )


def test_duplicate_project_pairs_raise_actionable_error(disposable_database):
    disposable_database.create_collection("projects")
    project = {"repository_url": "https://example.test/repo", "branch": "main"}
    disposable_database.projects.insert_many([project.copy(), project.copy()])

    with pytest.raises(ValueError, match="resolve these duplicate combinations manually"):
        initialize_database(disposable_database)

    assert disposable_database.projects.count_documents({}) == 2


def test_deployment_config_validates_supported_nested_fields(disposable_database):
    initialize_database(disposable_database)
    project = {
        "name": "Example",
        "repository_url": "https://example.test/repo",
        "branch": "main",
        "project_type": "python",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    project["deployment_config"] = {
        "port": 8080,
        "environment": "production",
        "environment_variables": {"MODE": "production"},
    }
    disposable_database.projects.insert_one(project)

    project["repository_url"] = "https://example.test/invalid"
    project["deployment_config"]["environment_variables"] = {"PORT": 8080}
    with pytest.raises(OperationFailure):
        disposable_database.projects.insert_one(project)


def test_application_detection_initialization_does_not_mutate_existing_documents(
    disposable_database,
):
    initialize_database(disposable_database)
    now = datetime.now(timezone.utc)
    project_id = ObjectId()
    deployment_id = ObjectId()
    workspace_id = ObjectId()
    documents = {
        "projects": {
            "name": "Existing project",
            "repository_url": "https://example.test/project",
            "branch": "main",
            "project_type": "python",
            "created_at": now,
            "updated_at": now,
        },
        "deployments": {
            "project_id": project_id,
            "status": "complete",
            "created_at": now,
        },
        "deployment_logs": {
            "deployment_id": deployment_id,
            "level": "info",
            "message": "Existing log",
            "timestamp": now,
        },
        "repository_workspaces": {
            "project_id": project_id,
            "status": "cloned",
            "repository_url": "https://example.test/project",
            "branch": "main",
            "workspace_path": f"/workspaces/{project_id}",
            "commit_hash": "a" * 40,
            "created_at": now,
            "updated_at": now,
        },
    }
    inserted_ids = {
        name: disposable_database[name].insert_one(document).inserted_id
        for name, document in documents.items()
    }
    detection_id = disposable_database.application_detections.insert_one(
        _valid_detection(project_id=project_id, workspace_id=workspace_id)
    ).inserted_id
    before = {
        name: disposable_database[name].find_one({"_id": inserted_id})
        for name, inserted_id in inserted_ids.items()
    }
    detection_before = disposable_database.application_detections.find_one(
        {"_id": detection_id}
    )

    initialize_database(disposable_database)

    after = {
        name: disposable_database[name].find_one({"_id": inserted_id})
        for name, inserted_id in inserted_ids.items()
    }
    assert after == before
    assert disposable_database.application_detections.find_one(
        {"_id": detection_id}
    ) == detection_before


def test_default_database_is_resolved_when_initializer_is_called(
    disposable_database, monkeypatch
):
    monkeypatch.setattr(connection, "database", disposable_database)

    initialize_database()

    assert set(disposable_database.list_collection_names()) == {
        "projects", "deployments", "deployment_logs", "repository_workspaces",
        "application_detections", "image_builds",
    }
