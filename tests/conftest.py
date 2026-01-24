"""Pytest configuration and fixtures."""

import os

# Set test environment variables before any app imports
os.environ["PMMDS_DB_HOST"] = "localhost"
os.environ["PMMDS_MODEL_FALLBACK_TO_DUMMY"] = "true"

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from apps.api.db.models import Base


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database and model overrides."""
    from apps.api.db import get_db_session
    from apps.api.models.loader import DummyModel, ModelLoader

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db

    # Create a test app with a no-op lifespan to avoid connecting to real services
    @asynccontextmanager
    async def test_lifespan(app):
        yield

    from fastapi import FastAPI
    from apps.api.routes import health_router, metrics_router, predict_router
    from apps.api.middleware import MetricsMiddleware, RequestLoggingMiddleware

    test_app = FastAPI(
        title="PMMDS Inference API",
        version="0.1.0",
        openapi_url="/openapi.json",
        lifespan=test_lifespan,
    )

    test_app.add_middleware(
        MetricsMiddleware,
        exclude_paths=["/healthz", "/ready", "/metrics"],
    )
    test_app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths=["/healthz", "/ready", "/metrics"],
    )

    test_app.include_router(health_router)
    test_app.include_router(metrics_router)
    test_app.include_router(predict_router, prefix="/api/v1")

    test_app.dependency_overrides[get_db_session] = override_get_db

    # Create dummy loader that returns DummyModel without MLflow
    dummy_loader = ModelLoader.__new__(ModelLoader)
    dummy_loader._cached_model = DummyModel()
    dummy_loader._model_load_time = 0.01
    dummy_loader._tracking_uri = "http://test"
    dummy_loader._model_name = "nyc-taxi-fare"
    dummy_loader._model_alias = "production"
    dummy_loader._fallback_to_dummy = True

    # Create a test engine for the health endpoint
    test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    with (
        patch("apps.api.routes.predict.get_model_loader", return_value=dummy_loader),
        patch("apps.api.routes.health.async_engine", test_engine),
        patch("apps.api.models.loader._model_loader", dummy_loader),
        patch("apps.api.models.loader.get_model_loader", return_value=dummy_loader),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app),
            base_url="http://test",
        ) as ac:
            yield ac

    await test_engine.dispose()
    test_app.dependency_overrides.clear()


@pytest.fixture
def sample_features() -> dict[str, Any]:
    """Sample taxi trip feature payload for testing."""
    return {
        "trip_distance": 3.5,
        "passenger_count": 2,
        "pickup_hour": 14,
        "pickup_day_of_week": 3,
        "pickup_month": 1,
        "trip_duration_minutes": 15.0,
        "is_weekend": 0,
        "is_rush_hour": 0,
        "RatecodeID": 1,
        "payment_type": 1,
        "pickup_borough": "Manhattan",
        "dropoff_borough": "Brooklyn",
    }
