"""Unit tests for the narrowly scoped, resumable schema migration."""

import pytest

from backend.database.initialize import (
    COLLECTION_INDEXES,
    COLLECTION_VALIDATORS,
    migrate_audited_schema,
)


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.indexes = {
            "_id_": {"key": [("_id", 1)], "name": "_id_", "v": 2}
        }
        self.created_indexes = []
        self.aggregate_pipelines = []
        self.duplicate_result = None
        self.fail_once_on = None

    def index_information(self):
        return self.indexes.copy()

    def aggregate(self, pipeline):
        self.aggregate_pipelines.append(pipeline)
        if self.duplicate_result is not None:
            return iter([self.duplicate_result])
        return iter(())

    def count_documents(self, query):
        assert query == {}
        return 0

    def create_index(self, keys, **options):
        name = options.get(
            "name", "_".join(f"{field}_{direction}" for field, direction in keys)
        )
        if self.fail_once_on == name:
            self.fail_once_on = None
            raise RuntimeError("simulated interruption during index creation")
        self.indexes[name] = {
            "key": list(keys),
            "name": name,
            "v": 2,
            **{key: value for key, value in options.items() if key != "name"},
        }
        self.created_indexes.append(name)
        return name


class FakeDatabase:
    def __init__(
        self,
        deployment_validator,
        deployment_validation_level="strict",
        deployment_validation_action="error",
        detection_validation_level="strict",
        detection_validation_action="error",
        image_validation_level=None,
        image_validation_action="error",
    ):
        self.collections = {
            "application_detections": FakeCollection("application_detections"),
            "deployments": FakeCollection("deployments"),
        }
        for keys, options in COLLECTION_INDEXES["deployments"][:2]:
            self.collections["deployments"].create_index(keys, **options)
        self.collections["deployments"].created_indexes.clear()
        self.infos = {
            "application_detections": {
                "name": "application_detections",
                "options": {
                    "validationLevel": detection_validation_level,
                    "validationAction": detection_validation_action,
                },
            },
            "deployments": {
                "name": "deployments",
                "options": {
                    "validator": deployment_validator,
                    "validationLevel": deployment_validation_level,
                    "validationAction": deployment_validation_action,
                },
            },
        }
        if image_validation_level is not None:
            self.collections["image_builds"] = FakeCollection("image_builds")
            self.infos["image_builds"] = {
                "name": "image_builds",
                "options": {
                    "validator": COLLECTION_VALIDATORS["image_builds"],
                    "validationLevel": image_validation_level,
                    "validationAction": image_validation_action,
                },
            }
        self.commands = []
        self.created_collections = []

    def list_collection_names(self):
        return list(self.collections)

    def list_collections(self, filter):
        info = self.infos.get(filter["name"])
        return iter([info] if info else [])

    def __getitem__(self, name):
        return self.collections[name]

    def __getattr__(self, name):
        try:
            return self.collections[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def command(self, command):
        self.commands.append(command)
        info = self.infos[command["collMod"]]
        info["options"]["validator"] = command["validator"]
        info["options"]["validationLevel"] = command["validationLevel"]
        info["options"]["validationAction"] = command["validationAction"]

    def create_collection(self, name, **options):
        self.created_collections.append(name)
        self.collections[name] = FakeCollection(name)
        self.infos[name] = {"name": name, "options": options}
        return self.collections[name]


def _insert_existing_index(collection, name, keys, **options):
    collection.indexes[name] = {
        "key": list(keys),
        "name": name,
        "v": 2,
        **options,
    }


def _assert_no_schema_changes(database):
    assert database.commands == []
    assert database.created_collections == []
    assert all(
        not collection.created_indexes
        for collection in database.collections.values()
    )


def test_migration_only_changes_audited_gaps_and_can_resume():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(legacy_validator)

    migrate_audited_schema(
        database, expected_deployment_validator=legacy_validator
    )

    assert [command["collMod"] for command in database.commands] == [
        "application_detections",
        "deployments",
    ]
    assert database.created_collections == ["image_builds"]
    assert database.collections["application_detections"].created_indexes == [
        "project_id_1_workspace_id_1",
        "status_1_updated_at_-1",
    ]
    assert database.collections["deployments"].created_indexes == [
        "one_active_deployment_per_project",
        "project_idempotency_key_unique",
    ]
    assert set(database.collections["image_builds"].created_indexes) == {
        "_".join(f"{field}_{direction}" for field, direction in keys)
        if "name" not in options
        else options["name"]
        for keys, options in COLLECTION_INDEXES["image_builds"]
    }

    prior_commands = len(database.commands)
    prior_created_collections = len(database.created_collections)
    prior_index_creations = sum(
        len(collection.created_indexes) for collection in database.collections.values()
    )
    migrate_audited_schema(
        database, expected_deployment_validator=legacy_validator
    )

    assert len(database.commands) == prior_commands
    assert len(database.created_collections) == prior_created_collections
    assert sum(
        len(collection.created_indexes) for collection in database.collections.values()
    ) == prior_index_creations
    assert database.infos["application_detections"]["options"]["validator"] == (
        COLLECTION_VALIDATORS["application_detections"]
    )
    assert database.infos["deployments"]["options"]["validator"] == (
        COLLECTION_VALIDATORS["deployments"]
    )


@pytest.mark.parametrize(
    "collection_name,index_name,keys,options",
    [
        (
            "application_detections",
            "project_id_1_workspace_id_1",
            COLLECTION_INDEXES["application_detections"][0][0],
            {"unique": False},
        ),
        (
            "deployments",
            "one_active_deployment_per_project",
            COLLECTION_INDEXES["deployments"][2][0],
            {
                "unique": True,
                "partialFilterExpression": {"status": {"$in": ["running"]}},
            },
        ),
    ],
)
def test_same_named_index_with_wrong_options_fails_preflight(
    collection_name, index_name, keys, options
):
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(legacy_validator)
    _insert_existing_index(
        database.collections[collection_name], index_name, keys, **options
    )

    with pytest.raises(RuntimeError, match="does not match"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    _assert_no_schema_changes(database)


def test_same_key_pattern_under_a_different_name_fails_preflight():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(legacy_validator)
    keys, _ = COLLECTION_INDEXES["application_detections"][0]
    _insert_existing_index(
        database.collections["application_detections"],
        "legacy_detection_pair",
        keys,
        unique=True,
    )

    with pytest.raises(RuntimeError, match="same key pattern exists"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    _assert_no_schema_changes(database)


def test_duplicates_for_partial_unique_index_abort_before_schema_changes():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(legacy_validator)
    database.collections["deployments"].duplicate_result = {
        "_id": {"project_id": "duplicate"},
        "count": 2,
    }

    with pytest.raises(ValueError, match="duplicate indexed values"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    deployment_pipeline = database.collections["deployments"].aggregate_pipelines[0]
    assert deployment_pipeline[0] == {
        "$match": COLLECTION_INDEXES["deployments"][2][1][
            "partialFilterExpression"
        ]
    }
    _assert_no_schema_changes(database)


def test_interrupted_migration_resumes_without_repeating_completed_operations():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(legacy_validator)
    failing_index = "workspace_id_1_created_at_-1"
    # The collection is created by the migration; configure its fake behavior
    # when create_collection is called.
    original_create_collection = database.create_collection

    def create_collection_with_interruption(name, **options):
        collection = original_create_collection(name, **options)
        collection.fail_once_on = failing_index
        return collection

    database.create_collection = create_collection_with_interruption

    with pytest.raises(RuntimeError, match="simulated interruption"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    image_builds = database.collections["image_builds"]
    completed_before_retry = list(image_builds.created_indexes)
    assert completed_before_retry == [
        "attempt_id_1",
        "one_active_build_per_workspace",
        "project_id_1_created_at_-1",
    ]
    commands_before_retry = list(database.commands)

    migrate_audited_schema(
        database, expected_deployment_validator=legacy_validator
    )

    assert database.commands == commands_before_retry
    assert database.created_collections == ["image_builds"]
    assert image_builds.created_indexes[: len(completed_before_retry)] == (
        completed_before_retry
    )
    expected_names = {
        options.get(
            "name", "_".join(f"{field}_{direction}" for field, direction in keys)
        )
        for keys, options in COLLECTION_INDEXES["image_builds"]
    }
    assert set(image_builds.created_indexes) == expected_names


def test_deployment_validator_update_preserves_validation_policy():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(
        legacy_validator,
        deployment_validation_level="moderate",
        deployment_validation_action="warn",
    )

    migrate_audited_schema(
        database, expected_deployment_validator=legacy_validator
    )

    deployment_command = next(
        command for command in database.commands
        if command["collMod"] == "deployments"
    )
    assert deployment_command["validationLevel"] == "moderate"
    assert deployment_command["validationAction"] == "warn"
    assert database.infos["deployments"]["options"]["validationLevel"] == "moderate"
    assert database.infos["deployments"]["options"]["validationAction"] == "warn"


def test_unexpected_detection_validation_policy_fails_before_schema_changes():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(
        legacy_validator,
        detection_validation_level="moderate",
        detection_validation_action="warn",
    )

    with pytest.raises(RuntimeError, match="application_detections.*strict/error"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    _assert_no_schema_changes(database)


def test_disabled_deployment_validation_policy_fails_before_schema_changes():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(
        legacy_validator,
        deployment_validation_level="off",
        deployment_validation_action="error",
    )

    with pytest.raises(RuntimeError, match="Unexpected validation policy for deployments"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    _assert_no_schema_changes(database)


def test_unexpected_image_build_validation_policy_fails_before_schema_changes():
    legacy_validator = {"$jsonSchema": {"bsonType": "object"}}
    database = FakeDatabase(
        legacy_validator,
        image_validation_level="strict",
        image_validation_action="warn",
    )

    with pytest.raises(RuntimeError, match="image_builds.*strict/error"):
        migrate_audited_schema(
            database, expected_deployment_validator=legacy_validator
        )

    _assert_no_schema_changes(database)


def test_unexpected_deployment_validator_fails_before_any_schema_changes():
    database = FakeDatabase({"unexpected": "validator"})

    with pytest.raises(RuntimeError, match="differs from both"):
        migrate_audited_schema(
            database,
            expected_deployment_validator={"$jsonSchema": {"bsonType": "object"}},
        )

    assert database.commands == []
    assert database.created_collections == []
    assert all(
        not collection.created_indexes
        for collection in database.collections.values()
    )
