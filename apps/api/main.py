"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.db import init_db
from apps.api.middleware import MetricsMiddleware, RequestLoggingMiddleware
from apps.api.routes import health_router, metrics_router, predict_router
from shared.config import get_settings
from shared.utils import get_logger, get_metrics, setup_logging

settings = get_settings()

# Setup structured logging
setup_logging(log_level=settings.log_level, json_format=settings.log_json)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Startup:
    - Initialize database tables
    - Warm up model cache
    - Initialize metrics

    Shutdown:
    - Cleanup resources
    """
    logger.info("application_starting", version=settings.api_version)

    # Initialize database
    await init_db()
    logger.info("database_ready")

    # Warm up model loader
    from apps.api.models import get_model_loader

    model_loader = get_model_loader()
    model = model_loader.get_current()
    logger.info("model_loaded", name=model.name, version=model.version)

    # Initialize metrics with current model info
    metrics = get_metrics()
    metrics.set_current_model(model.name, model.version)

    yield

    # Cleanup database connection pool
    from apps.api.db.connection import async_engine

    await async_engine.dispose()
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Production ML Monitoring & Drift Detection System - Inference API",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Metrics middleware (must be added first to capture all requests)
    app.add_middleware(
        MetricsMiddleware,
        exclude_paths=["/healthz", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )

    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths=["/healthz", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(predict_router, prefix="/api/v1")

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
