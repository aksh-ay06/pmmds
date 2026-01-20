"""API routes."""

from apps.api.routes.health import router as health_router
from apps.api.routes.predict import router as predict_router

__all__ = ["health_router", "predict_router"]
