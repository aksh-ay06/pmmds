"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from apps.api.db import async_engine
from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])
settings = get_settings()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    database: str


@router.get(
    "/healthz",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(response: Response) -> HealthResponse:
    """Check service health.

    Verifies:
    - API is responding
    - Database connection is alive
    """
    db_status = "healthy"

    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("health_check_db_failed", error=str(e))
        db_status = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    health_status = "healthy" if db_status == "healthy" else "degraded"

    return HealthResponse(
        status=health_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.api_version,
        database=db_status,
    )


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> dict[str, str]:
    """Kubernetes readiness probe.

    Simple check that the API can serve requests.
    """
    return {"status": "ready"}
