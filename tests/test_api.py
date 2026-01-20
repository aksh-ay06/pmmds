"""Tests for API endpoints."""

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Test health endpoint returns expected fields."""
    response = await client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_readiness_check(client: AsyncClient) -> None:
    """Test readiness endpoint."""
    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_predict_success(
    client: AsyncClient, sample_features: dict[str, Any]
) -> None:
    """Test successful prediction request."""
    payload = {"features": sample_features}

    response = await client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_name" in data
    assert "model_version" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_predict_with_custom_request_id(
    client: AsyncClient, sample_features: dict[str, Any]
) -> None:
    """Test prediction with custom request ID."""
    request_id = "550e8400-e29b-41d4-a716-446655440000"
    payload = {"request_id": request_id, "features": sample_features}

    response = await client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == request_id


@pytest.mark.asyncio
async def test_predict_invalid_payload(client: AsyncClient) -> None:
    """Test prediction with invalid payload."""
    payload = {"features": {"invalid_field": "value"}}

    response = await client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_predict_missing_features(client: AsyncClient) -> None:
    """Test prediction with missing features."""
    payload = {}

    response = await client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_openapi_docs(client: AsyncClient) -> None:
    """Test OpenAPI documentation is available."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/api/v1/predict" in data["paths"]
