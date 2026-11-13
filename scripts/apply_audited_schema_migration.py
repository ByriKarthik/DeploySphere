"""Apply the explicitly audited DeploySphere schema migration once."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymongo.errors import PyMongoError

from backend.database import connection
from backend.database.initialize import migrate_audited_schema


legacy_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["project_id", "status", "created_at"],
        "properties": {
            "project_id": {"bsonType": "objectId"},
            "status": {"bsonType": "string"},
            "commit_hash": {"bsonType": "string"},
            "docker_image": {"bsonType": "string"},
            "container_id": {"bsonType": "string"},
            "allocated_port": {"bsonType": "int"},
            "deployment_url": {"bsonType": "string"},
            "build_duration": {
                "bsonType": ["int", "long", "double", "decimal"]
            },
            "created_at": {"bsonType": "date"},
        },
    }
}


def main():
    client = connection.client
    try:
        db = connection.database
        print(f"selected_database: {db.name}")

        if (
            connection.DATABASE_NAME != "deploysphere"
            or db.name != "deploysphere"
        ):
            print(
                "Refusing migration: both configured and selected database "
                "names must equal 'deploysphere'.",
                file=sys.stderr,
            )
            return 2

        print("Both database names verified. Starting audited schema migration.")
        migrate_audited_schema(
            connection.database,
            expected_deployment_validator=legacy_validator,
        )
        print("Audited schema migration completed successfully.")
        return 0
    except PyMongoError as error:
        print(
            "Audited schema migration failed "
            f"({type(error).__name__}); MongoDB error details suppressed.",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(
            f"Audited schema migration failed ({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
