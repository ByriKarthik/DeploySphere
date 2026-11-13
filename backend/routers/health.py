"""Health check endpoints for the DeploySphere API."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.database.connection import DATABASE_NAME, check_mongodb_connection


router = APIRouter()


class HealthResponse(BaseModel):
    """Status information returned by the health endpoint."""

    api_status: str
    mongodb_status: str
    database: str


@router.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse | JSONResponse:
    """Check API availability and verify that MongoDB responds to a ping."""
    mongo_result = check_mongodb_connection()
    is_connected = mongo_result["connected"]
    response = HealthResponse(
        api_status="ok" if is_connected else "degraded",
        mongodb_status="connected" if is_connected else "unavailable",
        database=DATABASE_NAME,
    )

    if not is_connected:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response
