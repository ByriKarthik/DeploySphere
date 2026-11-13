"""Read-only preflight for the audited DeploySphere schema migration."""

from pathlib import Path
from pprint import pformat
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymongo.errors import PyMongoError

from backend.database import connection
from backend.database.initialize import (
    COLLECTION_INDEXES,
    COLLECTION_VALIDATORS,
    _migration_index_name,
    _migration_validation_policy,
)


TARGET_DATABASE = "deploysphere"
TARGET_COLLECTIONS = (
    "application_detections",
    "deployments",
    "image_builds",
)
_MISSING_VALIDATOR = object()


def _index_match(indexes, keys, options):
    expected_name = options.get("name", _migration_index_name(keys))
    expected_keys = [tuple(key) for key in keys]
    expected_options = {
        key: value for key, value in options.items() if key != "name"
    }
    existing = indexes.get(expected_name)

    if existing is not None:
        actual_keys = [tuple(key) for key in existing["key"]]
        actual_options = {
            key: value
            for key, value in existing.items()
            if key not in {"key", "name", "v", "ns"}
        }
        if actual_keys == expected_keys and actual_options == expected_options:
            return "MATCH", expected_name
        return "CONFLICT", expected_name

    same_key_names = [
        name
        for name, index in indexes.items()
        if [tuple(key) for key in index["key"]] == expected_keys
    ]
    if same_key_names:
        return "CONFLICT", f"{expected_name} (same keys as {same_key_names})"
    return "MISSING", expected_name


def _duplicate_group_count(collection, keys, options):
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
            {"$count": "duplicate_groups"},
        ]
    )
    result = next(collection.aggregate(pipeline), None)
    return result["duplicate_groups"] if result else 0


def _inspect_collection(db, name, info, blockers):
    collection = db[name]
    options = (info or {}).get("options", {})
    validator_present = "validator" in options
    validator = options.get("validator", _MISSING_VALIDATOR)
    count = collection.count_documents({})
    indexes = collection.index_information()

    print(f"\n[{name}]")
    print("exists: yes")
    print(f"document_count: {count}")
    if validator_present:
        print(f"validator:\n{pformat(validator, sort_dicts=False)}")
    else:
        print("validator: <absent>")
    print(
        "validationLevel: "
        f"{options.get('validationLevel', 'strict')} "
        f"(effective; explicitly set: {'validationLevel' in options})"
    )
    print(
        "validationAction: "
        f"{options.get('validationAction', 'error')} "
        f"(effective; explicitly set: {'validationAction' in options})"
    )

    print("existing_indexes:")
    for index_name, index_spec in indexes.items():
        print(f"  {index_name}: {pformat(index_spec, sort_dicts=False)}")

    expected_validator = COLLECTION_VALIDATORS[name]
    if name == "application_detections":
        if not validator_present:
            print("validator_comparison: MISSING (expected migration gap)")
        elif validator == expected_validator:
            print("validator_comparison: MATCH")
        else:
            print("validator_comparison: CONFLICT")
            blockers.append(
                "application_detections validator is neither absent nor the "
                "current source validator."
            )
    elif name == "image_builds":
        if validator != expected_validator:
            print("validator_comparison: CONFLICT")
            blockers.append(
                "Existing image_builds validator differs from the current "
                "source validator."
            )
        else:
            print("validator_comparison: MATCH")
    elif validator_present and validator == expected_validator:
        print("validator_comparison: MATCH")
    elif not validator_present:
        print("validator_comparison: MISSING")
    else:
        print(
            "validator_comparison: differs; capture this exact validator "
            "for the migration"
        )

    try:
        require_strict_error = name != "deployments"
        policy = _migration_validation_policy(
            name, info, require_strict_error=require_strict_error
        )
        print(f"migration_validation_policy: accepted {policy[0]}/{policy[1]}")
    except RuntimeError as error:
        blockers.append(str(error))
        print(f"migration_validation_policy: BLOCKED ({error})")

    if name == "deployments" and validator != expected_validator and count != 0:
        blockers.append(
            "deployments has a non-source validator and existing documents; "
            "the migration requires it to remain empty before validator replacement."
        )

    missing_indexes = []
    for keys, index_options in COLLECTION_INDEXES[name]:
        state, detail = _index_match(indexes, keys, index_options)
        print(f"expected_index: {detail}: {state}")
        if state == "CONFLICT":
            blockers.append(f"{name} has a conflicting index definition: {detail}.")
        if state in {"MISSING", "CONFLICT"}:
            missing_indexes.append((keys, index_options, detail))

    for keys, index_options, index_name in missing_indexes:
        if not index_options.get("unique"):
            continue
        duplicate_groups = _duplicate_group_count(collection, keys, index_options)
        print(
            "duplicate_groups_for_missing_unique_index "
            f"{index_name}: {duplicate_groups}"
        )
        if duplicate_groups:
            blockers.append(
                f"{name} has {duplicate_groups} duplicate group(s) for missing "
                f"unique index {index_name}."
            )

    return validator


def run_preflight(db):
    blockers = []
    collection_infos = {
        info["name"]: info
        for info in db.list_collections(
            filter={"name": {"$in": list(TARGET_COLLECTIONS)}}
        )
    }

    print(f"selected_database: {db.name}")
    print("read_only: yes")
    print(f"inspected_collections: {', '.join(TARGET_COLLECTIONS)}")
    print(
        "not_inspected_or_modified: projects, repository_workspaces, "
        "deployment_logs"
    )

    deployment_validator = _MISSING_VALIDATOR
    for name in TARGET_COLLECTIONS:
        info = collection_infos.get(name)
        if info is None:
            print(f"\n[{name}]")
            print("exists: no")
            print("document_count: 0 (collection does not exist)")
            print("validator: N/A")
            print("validationLevel: N/A")
            print("validationAction: N/A")
            print("existing_indexes: none")
            if name in {"application_detections", "deployments"}:
                blockers.append(f"Required collection {name} is missing.")
            else:
                print(
                    "migration_expectation: create with source validator "
                    "and indexes"
                )
                for keys, index_options in COLLECTION_INDEXES[name]:
                    index_name = index_options.get(
                        "name", _migration_index_name(keys)
                    )
                    print(f"expected_index: {index_name}: MISSING")
                    if index_options.get("unique"):
                        print(
                            f"duplicate_groups_for_missing_unique_index "
                            f"{index_name}: 0 (collection does not exist)"
                        )
            continue

        validator = _inspect_collection(db, name, info, blockers)
        if name == "deployments":
            deployment_validator = validator

    print("\nobserved_deployment_validator:")
    if deployment_validator is _MISSING_VALIDATOR:
        print("<absent>")
    else:
        print(pformat(deployment_validator, sort_dicts=False))

    print("\npreflight_result:")
    if blockers:
        print("BLOCKED")
        for blocker in blockers:
            print(f"- {blocker}")
        return 1

    print("READY, subject to quiescing application/schema writes before migration")
    return 0


def main():
    client = connection.client
    try:
        db = connection.database
        if (
            connection.DATABASE_NAME != TARGET_DATABASE
            or db.name != TARGET_DATABASE
        ):
            print(
                "ERROR: configured database is not exactly 'deploysphere'; "
                "aborting without database reads.",
                file=sys.stderr,
            )
            return 2
        return run_preflight(db)
    except PyMongoError as error:
        print(
            "ERROR: MongoDB read failed "
            f"({type(error).__name__}); no changes were attempted.",
            file=sys.stderr,
        )
        return 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
