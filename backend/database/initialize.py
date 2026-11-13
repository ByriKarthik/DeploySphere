"""Create and configure DeploySphere's MongoDB collections."""

from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, OperationFailure

from backend.database import connection


# Build logs store bounded metadata and a redacted output excerpt.
MAX_BUILD_LOG_BYTES = 100 * 1024 * 1024
MAX_BUILD_LOG_LINE_COUNT = 1_000_000
MAX_BUILD_LOG_EXCERPT_BYTES = 16 * 1024

# New lifecycle: queued -> creating -> starting -> checking_ready -> running.
# A running deployment may move through stopping -> stopped or
# restarting -> starting. Known terminal errors use failed; uncertain Docker
# outcomes use unknown until manual reconciliation. ``complete`` and
# ``succeeded`` remain accepted only so legacy records continue to validate.
# New deployment code should write only the statuses in DEPLOYMENT_STATUSES.
DEPLOYMENT_STATUSES = (
    "queued",
    "creating",
    "starting",
    "checking_ready",
    "running",
    "stopping",
    "stopped",
    "restarting",
    "failed",
    "unknown",
)
LEGACY_DEPLOYMENT_STATUSES = ("complete", "succeeded")
NEW_DEPLOYMENT_REQUIRED_FIELDS = (
    "project_id",
    "image_build_id",
    "workspace_id",
    "detection_id",
    "attempt_id",
    "idempotency_key",
    "container_name",
    "container_port",
    "updated_at",
)
ACTIVE_DEPLOYMENT_STATUSES = (
    "queued",
    "creating",
    "starting",
    "checking_ready",
    "running",
    "stopping",
    "restarting",
    "unknown",
)


COLLECTION_VALIDATORS = {
    "projects": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "name", "repository_url", "branch", "project_type",
                "created_at", "updated_at",
            ],
            "properties": {
                "name": {"bsonType": "string"},
                "repository_url": {"bsonType": "string"},
                "branch": {"bsonType": "string"},
                "project_type": {"bsonType": "string"},
                "dockerfile_path": {"bsonType": "string"},
                "deployment_config": {
                    "bsonType": "object",
                    "properties": {
                        "port": {"bsonType": "int"},
                        "environment": {"bsonType": "string"},
                        "environment_variables": {
                            "bsonType": "object",
                            "additionalProperties": {"bsonType": "string"},
                        },
                    },
                },
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    },
    "deployments": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["project_id", "status", "created_at"],
            # Legacy statuses retain their original minimal shape. Every new
            # lifecycle status requires the stable build/workspace identity,
            # idempotency data, container identity, and request port.
            "oneOf": [
                {
                    "properties": {
                        "status": {"enum": list(LEGACY_DEPLOYMENT_STATUSES)}
                    }
                },
                {
                    "required": list(NEW_DEPLOYMENT_REQUIRED_FIELDS),
                    "properties": {
                        "status": {"enum": list(DEPLOYMENT_STATUSES)}
                    },
                },
            ],
            "properties": {
                "project_id": {"bsonType": "objectId"},
                "image_build_id": {"bsonType": "objectId"},
                "workspace_id": {"bsonType": "objectId"},
                "detection_id": {"bsonType": "objectId"},
                "attempt_id": {"bsonType": "string", "minLength": 1, "maxLength": 128},
                "idempotency_key": {
                    "bsonType": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "status": {
                    "bsonType": "string",
                    "enum": [*DEPLOYMENT_STATUSES, *LEGACY_DEPLOYMENT_STATUSES],
                },
                "commit_hash": {"bsonType": "string"},
                "docker_image": {"bsonType": "string"},
                "container_id": {"bsonType": "string"},
                "container_name": {
                    "bsonType": "string",
                    "pattern": "^deploysphere-[a-f0-9]{24}$",
                },
                "container_port": {
                    "bsonType": "int",
                    "minimum": 1,
                    "maximum": 65535,
                },
                "allocated_port": {
                    "bsonType": "int",
                    "minimum": 1,
                    "maximum": 65535,
                },
                "deployment_url": {"bsonType": "string"},
                "build_duration": {"bsonType": ["int", "long", "double", "decimal"]},
                "updated_at": {"bsonType": "date"},
                "started_at": {"bsonType": "date"},
                "stopped_at": {"bsonType": "date"},
                "health_path": {
                    "bsonType": "string",
                    "maxLength": 255,
                    "pattern": "^/[^\\r\\n]*$",
                },
                "health_status": {
                    "bsonType": "string",
                    "enum": ["pending", "ready", "failed", "unknown"],
                },
                "health_checked_at": {"bsonType": "date"},
                "error": {
                    "bsonType": "object",
                    "required": ["code", "message", "occurred_at"],
                    "properties": {
                        "code": {
                            "bsonType": "string",
                            "minLength": 1,
                            "maxLength": 64,
                            "pattern": "^[a-z][a-z0-9_]*$",
                        },
                        "message": {"bsonType": "string", "maxLength": 500},
                        "occurred_at": {"bsonType": "date"},
                    },
                },
                "created_at": {"bsonType": "date"},
            },
        }
    },
    "deployment_logs": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["deployment_id", "level", "message", "timestamp"],
            "properties": {
                "deployment_id": {"bsonType": "objectId"},
                "level": {"bsonType": "string"},
                "message": {"bsonType": "string"},
                "timestamp": {"bsonType": "date"},
            },
        }
    },
    "repository_workspaces": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "project_id", "status", "repository_url", "branch",
                "workspace_path", "created_at", "updated_at",
            ],
            "properties": {
                "project_id": {"bsonType": "objectId"},
                "status": {
                    "bsonType": "string",
                    "enum": ["cloning", "cloned", "failed"],
                },
                "repository_url": {"bsonType": "string"},
                "branch": {"bsonType": "string"},
                "workspace_path": {"bsonType": "string"},
                "commit_hash": {"bsonType": "string"},
                "error": {
                    "bsonType": "object",
                    "required": ["code", "message", "occurred_at"],
                    "properties": {
                        "code": {"bsonType": "string"},
                        "message": {"bsonType": "string"},
                        "occurred_at": {"bsonType": "date"},
                    },
                },
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    },
    "application_detections": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "project_id", "workspace_id", "attempt_id", "commit_hash", "status",
                "application_type", "build_method", "detected_application_types",
                "detected_build_methods", "detected_files", "confidence",
                "warnings", "created_at", "updated_at",
            ],
            "properties": {
                "project_id": {"bsonType": "objectId"},
                "workspace_id": {"bsonType": "objectId"},
                # attempt_id changes whenever a reservation is claimed. Final
                # updates compare it so an older scan cannot overwrite a retry.
                "attempt_id": {"bsonType": "string"},
                "reservation_expires_at": {"bsonType": "date"},
                "commit_hash": {"bsonType": "string"},
                "status": {
                    "bsonType": "string",
                    "enum": ["detecting", "completed", "failed"],
                },
                "application_type": {
                    "bsonType": "string",
                    "enum": ["python", "nodejs", "mixed", "unknown"],
                },
                "build_method": {
                    "bsonType": "string",
                    "enum": ["native", "dockerfile", "compose", "mixed", "unknown"],
                },
                "detected_application_types": {
                    "bsonType": "array",
                    "uniqueItems": True,
                    "items": {
                        "bsonType": "string",
                        "enum": ["python", "nodejs"],
                    },
                },
                "detected_build_methods": {
                    "bsonType": "array",
                    "uniqueItems": True,
                    "items": {
                        "bsonType": "string",
                        "enum": ["native", "dockerfile", "compose"],
                    },
                },
                "detected_files": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                },
                "confidence": {
                    "bsonType": ["int", "long", "double", "decimal"],
                    "minimum": 0,
                    "maximum": 1,
                },
                "warnings": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                },
                "error": {
                    "bsonType": "object",
                    "required": ["code", "message", "occurred_at"],
                    "properties": {
                        "code": {"bsonType": "string"},
                        "message": {"bsonType": "string"},
                        "occurred_at": {"bsonType": "date"},
                    },
                },
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    },
    "image_builds": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "project_id", "workspace_id", "detection_id", "commit_hash",
                "attempt_id", "status", "image_tag", "build_context",
                "dockerfile_path", "build_duration", "created_at", "updated_at",
            ],
            "properties": {
                "project_id": {"bsonType": "objectId"},
                "workspace_id": {"bsonType": "objectId"},
                "detection_id": {"bsonType": "objectId"},
                "commit_hash": {"bsonType": "string", "minLength": 1, "maxLength": 64},
                "attempt_id": {"bsonType": "string", "minLength": 1, "maxLength": 128},
                "status": {
                    "bsonType": "string",
                    "enum": ["queued", "building", "succeeded", "failed", "cancelled"],
                },
                "image_tag": {"bsonType": "string", "minLength": 1, "maxLength": 255},
                "image_id": {"bsonType": "string", "maxLength": 255},
                # These values must be POSIX-style relative paths or a safe
                # staging identifier. Filesystem containment is checked by
                # the future build service before it opens or copies a path.
                "build_context": {
                    "bsonType": "string",
                    "minLength": 1,
                    "maxLength": 512,
                    "pattern": "^(?!/)(?![A-Za-z]:)(?!.*\\\\)(?!.*(?:^|/)\\.\\.(?:/|$)).+$",
                },
                "dockerfile_path": {
                    "bsonType": "string",
                    "minLength": 1,
                    "maxLength": 512,
                    "pattern": "^(?!/)(?![A-Za-z]:)(?!.*\\\\)(?!.*(?:^|/)\\.\\.(?:/|$)).+$",
                },
                "build_duration": {
                    "bsonType": ["int", "long", "double", "decimal"],
                    "minimum": 0,
                },
                "reservation_expires_at": {"bsonType": "date"},
                "error": {
                    "bsonType": "object",
                    "required": ["code", "message", "occurred_at"],
                    "properties": {
                        "code": {
                            "bsonType": "string",
                            "minLength": 1,
                            "maxLength": 64,
                            "pattern": "^[a-z][a-z0-9_]*$",
                        },
                        "message": {"bsonType": "string", "maxLength": 500},
                        "occurred_at": {"bsonType": "date"},
                    },
                },
                # Store bounded metadata rather than raw Docker output.
                "log_bytes": {
                    "bsonType": ["int", "long"],
                    "minimum": 0,
                    "maximum": MAX_BUILD_LOG_BYTES,
                },
                "log_line_count": {
                    "bsonType": ["int", "long"],
                    "minimum": 0,
                    "maximum": MAX_BUILD_LOG_LINE_COUNT,
                },
                "log_truncated": {"bsonType": "bool"},
                "log_excerpt": {
                    "bsonType": "string",
                    "maxLength": MAX_BUILD_LOG_EXCERPT_BYTES,
                },
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    },
}

COLLECTION_INDEXES = {
    "projects": [
        ([ ("repository_url", 1), ("branch", 1) ], {"unique": True}),
    ],
    "deployments": [
        ([ ("project_id", 1), ("created_at", -1) ], {}),
        ([ ("status", 1), ("created_at", -1) ], {}),
        (
            [("project_id", 1)],
            {
                "unique": True,
                "partialFilterExpression": {
                    "status": {"$in": list(ACTIVE_DEPLOYMENT_STATUSES)}
                },
                "name": "one_active_deployment_per_project",
            },
        ),
        (
            [("project_id", 1), ("idempotency_key", 1)],
            {
                "unique": True,
                "partialFilterExpression": {"idempotency_key": {"$type": "string"}},
                "name": "project_idempotency_key_unique",
            },
        ),
    ],
    "deployment_logs": [
        ([ ("deployment_id", 1), ("timestamp", 1) ], {}),
    ],
    "repository_workspaces": [
        ([ ("project_id", 1) ], {"unique": True}),
        ([ ("workspace_path", 1) ], {"unique": True}),
        ([ ("status", 1), ("updated_at", -1) ], {}),
    ],
    "application_detections": [
        ([ ("project_id", 1), ("workspace_id", 1) ], {"unique": True}),
        ([ ("status", 1), ("updated_at", -1) ], {}),
    ],
    "image_builds": [
        ([ ("attempt_id", 1) ], {"unique": True}),
        (
            [("project_id", 1), ("workspace_id", 1)],
            {
                "unique": True,
                "partialFilterExpression": {
                    "status": {"$in": ["queued", "building"]}
                },
                "name": "one_active_build_per_workspace",
            },
        ),
        ([ ("project_id", 1), ("created_at", -1) ], {}),
        ([ ("workspace_id", 1), ("created_at", -1) ], {}),
        ([ ("status", 1), ("updated_at", -1) ], {}),
        ([ ("project_id", 1), ("status", 1), ("updated_at", -1) ], {}),
    ],
}


def _contains_validator(existing: dict, target: dict) -> bool:
    """Return whether a validator already includes the requested schema."""
    if existing == target:
        return True
    return any(
        _contains_validator(item, target)
        for item in existing.get("$and", [])
        if isinstance(item, dict)
    )


def _ensure_validator(db: Database, name: str, validator: dict) -> None:
    """Add a schema validator while retaining any validator already present."""
    info = next(db.list_collections(filter={"name": name}), None)
    existing = (info or {}).get("options", {}).get("validator", {})
    if _contains_validator(existing, validator):
        return

    combined = validator if not existing else {"$and": [existing, validator]}
    command = {"collMod": name, "validator": combined}
    if existing:
        # MongoDB defaults these options to strict/error. Supplying the
        # current effective values prevents collMod from changing the policy.
        options = (info or {}).get("options", {})
        command["validationLevel"] = options.get("validationLevel", "strict")
        command["validationAction"] = options.get("validationAction", "error")
    else:
        command["validationLevel"] = "strict"
        command["validationAction"] = "error"
    db.command(command)


def _check_unique_project_pairs(db: Database) -> None:
    """Raise an actionable error if repository/branch pairs are duplicated."""
    duplicates = list(
        db.projects.aggregate([
            {
                "$group": {
                    "_id": {"repository_url": "$repository_url", "branch": "$branch"},
                    "count": {"$sum": 1},
                }
            },
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"_id.repository_url": 1, "_id.branch": 1}},
        ])
    )
    if duplicates:
        details = "; ".join(
            f"repository_url={item['_id'].get('repository_url')!r}, "
            f"branch={item['_id'].get('branch')!r} ({item['count']} documents)"
            for item in duplicates
        )
        raise ValueError(
            "Cannot create the unique projects repository_url + branch index; "
            f"resolve these duplicate combinations manually: {details}"
        )


def _check_active_image_build_reservations(db: Database) -> None:
    """Report duplicate active project/workspace builds before adding the index."""
    duplicates = list(
        db.image_builds.aggregate(
            [
                {"$match": {"status": {"$in": ["queued", "building"]}}},
                {
                    "$group": {
                        "_id": {"project_id": "$project_id", "workspace_id": "$workspace_id"},
                        "count": {"$sum": 1},
                    }
                },
                {"$match": {"count": {"$gt": 1}}},
            ]
        )
    )
    if duplicates:
        details = "; ".join(
            f"project_id={item['_id'].get('project_id')}, "
            f"workspace_id={item['_id'].get('workspace_id')} "
            f"({item['count']} active records)"
            for item in duplicates
        )
        raise ValueError(
            "Cannot create the unique active image-build reservation index; "
            "reconcile these queued/building records manually without deleting "
            f"or rewriting them: {details}"
        )


def _check_active_deployments(db: Database) -> None:
    """Report projects with multiple active records before adding the index."""
    duplicates = list(
        db.deployments.aggregate(
            [
                {"$match": {"status": {"$in": list(ACTIVE_DEPLOYMENT_STATUSES)}}},
                {"$group": {"_id": "$project_id", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"_id": 1}},
            ]
        )
    )
    if duplicates:
        details = "; ".join(
            f"project_id={item['_id']} ({item['count']} active records)"
            for item in duplicates
        )
        raise ValueError(
            "Cannot create the unique active deployment index; reconcile these "
            "records manually without deleting or rewriting them: "
            f"{details}"
        )


def _create_index(collection, keys: list[tuple[str, int]], options: dict) -> None:
    """Create an index and report duplicate-data or index-option conflicts clearly."""
    index_name = "_".join(f"{field}_{direction}" for field, direction in keys)
    try:
        collection.create_index(keys, **options)
    except DuplicateKeyError as error:
        raise ValueError(
            f"Cannot create unique index {index_name} on {collection.name}: "
            "existing documents contain duplicate indexed values. Resolve the "
            "duplicates manually; initialization does not change documents."
        ) from error
    except OperationFailure as error:
        raise RuntimeError(
            f"MongoDB could not ensure index {index_name} on {collection.name}. "
            "Check for an existing index with conflicting keys or options."
        ) from error


def _migration_collection_info(db: Database, name: str) -> dict | None:
    return next(db.list_collections(filter={"name": name}), None)


def _migration_validation_policy(
    name: str, info: dict, *, require_strict_error: bool
) -> tuple[str, str]:
    """Resolve and validate the policy before the migration performs any DDL."""
    options = (info or {}).get("options", {})
    level = options.get("validationLevel", "strict")
    action = options.get("validationAction", "error")
    if level not in {"strict", "moderate"} or action not in {"error", "warn"}:
        raise RuntimeError(
            f"Unexpected validation policy for {name}: "
            f"validationLevel={level!r}, validationAction={action!r}."
        )
    if require_strict_error and (level != "strict" or action != "error"):
        raise RuntimeError(
            f"Unexpected validation policy for {name}; expected strict/error, "
            f"found {level}/{action}."
        )
    return level, action


def _migration_index_name(keys: list[tuple[str, int]]) -> str:
    return "_".join(f"{field}_{direction}" for field, direction in keys)


def _plan_migration_indexes(collection, specifications: list[tuple]) -> list[tuple]:
    """Return missing indexes, refusing to reinterpret conflicting definitions."""
    existing_indexes = collection.index_information()
    missing = []

    for keys, options in specifications:
        expected_name = options.get("name", _migration_index_name(keys))
        existing = existing_indexes.get(expected_name)
        expected_keys = [tuple(key) for key in keys]
        expected_options = {
            key: value for key, value in options.items() if key != "name"
        }

        if existing is None:
            same_keys = [
                name
                for name, index in existing_indexes.items()
                if [tuple(key) for key in index["key"]] == expected_keys
            ]
            if same_keys:
                raise RuntimeError(
                    f"Cannot ensure index {expected_name} on {collection.name}; "
                    f"the same key pattern exists as {', '.join(same_keys)}."
                )
            missing.append((keys, options))
            continue

        actual_keys = [tuple(key) for key in existing["key"]]
        actual_options = {
            key: value
            for key, value in existing.items()
            if key not in {"key", "name", "v", "ns"}
        }
        if actual_keys != expected_keys or actual_options != expected_options:
            raise RuntimeError(
                f"Existing index {expected_name} on {collection.name} does not "
                "match the declared key order and options; no index was changed."
            )

    return missing


def _check_migration_unique_index_data(
    collection, keys: list[tuple[str, int]], options: dict
) -> None:
    """Check a missing unique index for duplicates without changing documents."""
    pipeline = []
    partial_filter = options.get("partialFilterExpression")
    if partial_filter:
        pipeline.append({"$match": partial_filter})
    pipeline.extend(
        [
            {
                "$group": {
                    "_id": {field: f"${field}" for field, _ in keys},
                    "count": {"$sum": 1},
                }
            },
            {"$match": {"count": {"$gt": 1}}},
            {"$limit": 1},
        ]
    )
    if next(collection.aggregate(pipeline), None):
        index_name = options.get("name", _migration_index_name(keys))
        raise ValueError(
            f"Cannot create unique index {index_name} on {collection.name}; "
            "duplicate indexed values exist. Resolve them manually without "
            "deleting or rewriting documents."
        )


def migrate_audited_schema(
    db: Database,
    *,
    expected_deployment_validator: dict,
) -> None:
    """Migrate only audited schema gaps, refusing unexpected live definitions.

    The caller must pass the exact deployments validator captured during the
    read-only audit and quiesce application/schema writes for the migration.
    This function does not modify documents or unrelated collections and is
    safe to rerun after an interrupted migration.
    """
    target_names = ("application_detections", "deployments", "image_builds")
    existing_names = set(db.list_collection_names())
    missing_required = {"application_detections", "deployments"} - existing_names
    if missing_required:
        raise RuntimeError(
            "Cannot migrate audited schema; expected collections are missing: "
            + ", ".join(sorted(missing_required))
        )

    collection_infos = {
        name: _migration_collection_info(db, name)
        for name in target_names
        if name in existing_names
    }
    _migration_validation_policy(
        "application_detections",
        collection_infos["application_detections"],
        require_strict_error=True,
    )
    deployment_validation_policy = _migration_validation_policy(
        "deployments",
        collection_infos["deployments"],
        require_strict_error=False,
    )
    if "image_builds" in existing_names:
        _migration_validation_policy(
            "image_builds",
            collection_infos["image_builds"],
            require_strict_error=True,
        )

    detected_validator = COLLECTION_VALIDATORS["application_detections"]
    deployment_validator = COLLECTION_VALIDATORS["deployments"]
    image_build_validator = COLLECTION_VALIDATORS["image_builds"]

    current_detection_validator = (
        (collection_infos["application_detections"] or {})
        .get("options", {})
        .get("validator", {})
    )
    if current_detection_validator not in ({}, detected_validator):
        raise RuntimeError(
            "Unexpected application_detections validator; no schema was changed."
        )

    current_deployment_validator = (
        (collection_infos["deployments"] or {})
        .get("options", {})
        .get("validator", {})
    )
    replacing_deployment_validator = current_deployment_validator != deployment_validator
    if replacing_deployment_validator and (
        current_deployment_validator != expected_deployment_validator
    ):
        raise RuntimeError(
            "The deployments validator differs from both the audited legacy "
            "definition and the current source definition; no schema was changed."
        )
    if replacing_deployment_validator and db.deployments.count_documents({}) != 0:
        raise RuntimeError(
            "The deployments validator can be replaced only while the audited "
            "collection remains empty."
        )

    if "image_builds" in existing_names:
        current_image_validator = (
            (collection_infos["image_builds"] or {})
            .get("options", {})
            .get("validator", {})
        )
        if current_image_validator != image_build_validator:
            raise RuntimeError(
                "Unexpected image_builds validator; no schema was changed."
            )

    planned_indexes = {}
    for name in target_names:
        if name not in existing_names:
            planned_indexes[name] = list(COLLECTION_INDEXES[name])
            continue

        collection = db[name]
        missing_indexes = _plan_migration_indexes(
            collection, COLLECTION_INDEXES[name]
        )
        for keys, options in missing_indexes:
            if options.get("unique"):
                _check_migration_unique_index_data(collection, keys, options)
        planned_indexes[name] = missing_indexes

    if current_detection_validator == {}:
        db.command(
            {
                "collMod": "application_detections",
                "validator": detected_validator,
                "validationLevel": "strict",
                "validationAction": "error",
            }
        )

    if replacing_deployment_validator:
        if db.deployments.count_documents({}) != 0:
            raise RuntimeError(
                "The deployments collection changed during migration preflight; "
                "its validator was not replaced."
            )
        db.command(
            {
                "collMod": "deployments",
                "validator": deployment_validator,
                "validationLevel": deployment_validation_policy[0],
                "validationAction": deployment_validation_policy[1],
            }
        )

    if "image_builds" not in existing_names:
        db.create_collection(
            "image_builds",
            validator=image_build_validator,
            validationLevel="strict",
            validationAction="error",
        )

    for name, indexes in planned_indexes.items():
        collection = db[name]
        for keys, options in indexes:
            _create_index(collection, keys, options)


def initialize_database(db: Database | None = None) -> None:
    """Create missing collections and ensure validators and indexes exist.

    Existing collections and documents are retained. Legacy documents that do
    not satisfy a newly added validator remain stored; with strict validation,
    future inserts and updates that leave or make documents invalid are rejected.
    """
    if db is None:
        db = connection.database

    existing_names = set(db.list_collection_names())
    for name, validator in COLLECTION_VALIDATORS.items():
        if name not in existing_names:
            db.create_collection(
                name,
                validator=validator,
                validationLevel="strict",
                validationAction="error",
            )
        else:
            _ensure_validator(db, name, validator)

        collection = db[name]
        for keys, options in COLLECTION_INDEXES[name]:
            if name == "projects" and options.get("unique"):
                _check_unique_project_pairs(db)
            if name == "deployments" and options.get("name") == "one_active_deployment_per_project":
                _check_active_deployments(db)
            if name == "image_builds" and options.get("partialFilterExpression"):
                _check_active_image_build_reservations(db)
            _create_index(collection, keys, options)
