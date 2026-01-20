"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from apps.api.db import async_engine
from shared.config import get_settings
from shared.utils import get_logger, get_metrics

logger = get_logger(__name__)
router = APIRouter(tags=["health"])
settings = get_settings()
metrics = get_metrics()


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


@router.get("/model", status_code=status.HTTP_200_OK)
async def model_info() -> dict:
    """Get information about the currently loaded model.

    Returns model name, version, and source.
    """
    from apps.api.models import get_model_loader

    loader = get_model_loader()
    return loader.get_model_info()


@router.post("/model/reload", status_code=status.HTTP_200_OK)
async def reload_model() -> dict:
    """Reload model from MLflow registry.

    Forces the API to fetch the latest production model.
    Useful after a model promotion.

    Returns:
        New model information.
    """
    from apps.api.models import get_model_loader

    loader = get_model_loader()
    old_info = loader.get_model_info()
    old_version = old_info.get("version", "unknown")

    # Force reload
    model = loader.reload()

    new_info = loader.get_model_info()

    # Record metrics
    metrics.record_model_reload(model_name=new_info.get("name", "unknown"))
    metrics.set_current_model(
        model_name=new_info.get("name", "unknown"),
        model_version=new_info.get("version", "unknown"),
    )

    logger.info(
        "model_reloaded",
        old_version=old_version,
        new_version=new_info.get("version", "unknown"),
        model_name=new_info.get("name"),
    )

    return {
        "status": "reloaded",
        "previous_version": old_version,
        "current_version": new_info.get("version", "unknown"),
        "model": new_info,
    }
