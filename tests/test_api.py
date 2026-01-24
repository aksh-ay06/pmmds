"""Tests for the prediction API endpoint."""

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_predict_valid_request(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with valid features."""
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 200

    data = response.json()
    assert "predicted_fare" in data
    assert data["predicted_fare"] > 0
    assert "model_name" in data
    assert "model_version" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_predict_missing_feature(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with missing required feature."""
    del sample_features["trip_distance"]
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_trip_distance(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with trip_distance out of range."""
    sample_features["trip_distance"] = 200.0  # Max is 100
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_passenger_count(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with invalid passenger count."""
    sample_features["passenger_count"] = 0  # Min is 1
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_borough(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with invalid borough name."""
    sample_features["pickup_borough"] = "Atlantis"
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_rate_code(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with invalid rate code."""
    sample_features["RatecodeID"] = 10  # Max is 6
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_extra_field_rejected(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test that extra fields are rejected (strict schema)."""
    sample_features["extra_field"] = "value"
    response = await client.post("/api/v1/predict", json={"features": sample_features})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_custom_request_id(client: AsyncClient, sample_features: dict[str, Any]) -> None:
    """Test prediction with custom request ID."""
    import uuid

    req_id = str(uuid.uuid4())
    response = await client.post(
        "/api/v1/predict",
        json={"request_id": req_id, "features": sample_features},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == req_id


@pytest.mark.asyncio
async def test_predict_rush_hour_trip(client: AsyncClient) -> None:
    """Test prediction for rush hour trip."""
    features = {
        "trip_distance": 5.0,
        "passenger_count": 1,
        "pickup_hour": 17,
        "pickup_day_of_week": 4,
        "pickup_month": 1,
        "trip_duration_minutes": 25.0,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "RatecodeID": 1,
        "payment_type": 1,
        "pickup_borough": "Manhattan",
        "dropoff_borough": "Manhattan",
    }
    response = await client.post("/api/v1/predict", json={"features": features})
    assert response.status_code == 200
    assert response.json()["predicted_fare"] > 0


@pytest.mark.asyncio
async def test_predict_weekend_trip(client: AsyncClient) -> None:
    """Test prediction for weekend trip."""
    features = {
        "trip_distance": 8.0,
        "passenger_count": 4,
        "pickup_hour": 22,
        "pickup_day_of_week": 7,
        "pickup_month": 2,
        "trip_duration_minutes": 30.0,
        "is_weekend": 1,
        "is_rush_hour": 0,
        "RatecodeID": 1,
        "payment_type": 2,
        "pickup_borough": "Brooklyn",
        "dropoff_borough": "Queens",
    }
    response = await client.post("/api/v1/predict", json={"features": features})
    assert response.status_code == 200
    assert response.json()["predicted_fare"] > 0
