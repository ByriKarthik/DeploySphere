
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Load configuration from the project's root .env file
load_dotenv()

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017"
)

DATABASE_NAME = os.getenv(
    "MONGODB_DATABASE",
    "deploysphere"
)

# Create the MongoDB client
client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

# Select the database
database = client[DATABASE_NAME]


def check_mongodb_connection():
    """Check whether the MongoDB server is reachable."""
    try:
        client.admin.command("ping")
        return {
            "connected": True,
            "database": DATABASE_NAME,
            "message": "MongoDB connection successful"
        }
    except PyMongoError as error:
        return {
            "connected": False,
            "database": DATABASE_NAME,
            "message": str(error)
        }
