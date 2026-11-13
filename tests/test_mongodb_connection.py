
from backend.database.connection import check_mongodb_connection


def test_mongodb_connection():
    result = check_mongodb_connection()

    assert result["connected"] is True
    assert result["database"] == "deploysphere"
