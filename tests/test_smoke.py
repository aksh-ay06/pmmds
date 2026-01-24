"""Smoke tests for API endpoints.

These tests verify basic functionality without mocking.
Designed to run against a live server with dummy model.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_smoke_health_endpoint(client: AsyncClient) -> None:
    """Smoke test: Health endpoint responds."""
    response = await client.get("/healthz")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_smoke_ready_endpoint(client: AsyncClient) -> None:
    """Smoke test: Ready endpoint responds."""
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_smoke_metrics_prometheus(client: AsyncClient) -> None:
    """Smoke test: Prometheus metrics endpoint responds."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "pmmds_" in response.text


@pytest.mark.asyncio
async def test_smoke_metrics_json(client: AsyncClient) -> None:
    """Smoke test: JSON metrics endpoint responds."""
    response = await client.get("/metrics/json")
    assert response.status_code == 200
    data = response.json()
    assert "requests" in data
    assert "predictions" in data
    assert "system" in data


@pytest.mark.asyncio
async def test_smoke_predict_valid_payload(client: AsyncClient) -> None:
    """Smoke test: Prediction with valid taxi trip payload."""
    payload = {
        "features": {
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
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "request_id" in data
    assert "predicted_fare" in data
    assert data["predicted_fare"] > 0
    assert "model_name" in data
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_smoke_predict_short_trip(client: AsyncClient) -> None:
    """Smoke test: Prediction for a short trip."""
    payload = {
        "features": {
            "trip_distance": 0.5,
            "passenger_count": 1,
            "pickup_hour": 10,
            "pickup_day_of_week": 2,
            "pickup_month": 1,
            "trip_duration_minutes": 3.0,
            "is_weekend": 0,
            "is_rush_hour": 0,
            "RatecodeID": 1,
            "payment_type": 1,
            "pickup_borough": "Manhattan",
            "dropoff_borough": "Manhattan",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_fare"] > 0


@pytest.mark.asyncio
async def test_smoke_predict_long_trip(client: AsyncClient) -> None:
    """Smoke test: Prediction for a long trip (JFK rate)."""
    payload = {
        "features": {
            "trip_distance": 20.0,
            "passenger_count": 3,
            "pickup_hour": 18,
            "pickup_day_of_week": 5,
            "pickup_month": 2,
            "trip_duration_minutes": 45.0,
            "is_weekend": 0,
            "is_rush_hour": 1,
            "RatecodeID": 2,
            "payment_type": 1,
            "pickup_borough": "Queens",
            "dropoff_borough": "Manhattan",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_fare"] > 0


@pytest.mark.asyncio
async def test_smoke_predict_invalid_borough(client: AsyncClient) -> None:
    """Smoke test: Prediction with invalid borough."""
    payload = {
        "features": {
            "trip_distance": 3.0,
            "passenger_count": 1,
            "pickup_hour": 12,
            "pickup_day_of_week": 4,
            "pickup_month": 1,
            "trip_duration_minutes": 10.0,
            "is_weekend": 0,
            "is_rush_hour": 0,
            "RatecodeID": 1,
            "payment_type": 1,
            "pickup_borough": "InvalidBorough",
            "dropoff_borough": "Manhattan",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_smoke_openapi_available(client: AsyncClient) -> None:
    """Smoke test: OpenAPI spec is available."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert data["info"]["title"] == "PMMDS Inference API"
    assert "/api/v1/predict" in data["paths"]
    assert "/healthz" in data["paths"]
    assert "/metrics" in data["paths"]


@pytest.mark.asyncio
async def test_smoke_model_info(client: AsyncClient) -> None:
    """Smoke test: Model info endpoint."""
    response = await client.get("/model")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "tracking_uri" in data
