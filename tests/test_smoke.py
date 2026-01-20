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
    assert response.status_code in [200, 503]  # May be degraded without real DB
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
    """Smoke test: Prediction with valid payload."""
    payload = {
        "features": {
            "gender": "Male",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 70.35,
            "total_charges": 840.20,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "request_id" in data
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_name" in data
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_smoke_predict_high_risk_customer(client: AsyncClient) -> None:
    """Smoke test: Prediction for high churn risk customer."""
    # High risk profile: Month-to-month, short tenure, high charges
    payload = {
        "features": {
            "gender": "Female",
            "senior_citizen": 1,
            "partner": "No",
            "dependents": "No",
            "tenure": 1,
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 95.00,
            "total_charges": 95.00,
            "phone_service": "Yes",
            "multiple_lines": "Yes",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # High risk customers typically have higher probability
    assert data["prediction"] in [0, 1]


@pytest.mark.asyncio
async def test_smoke_predict_low_risk_customer(client: AsyncClient) -> None:
    """Smoke test: Prediction for low churn risk customer."""
    # Low risk profile: Two year contract, long tenure, low charges
    payload = {
        "features": {
            "gender": "Male",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "Yes",
            "tenure": 72,
            "contract": "Two year",
            "paperless_billing": "No",
            "payment_method": "Bank transfer (automatic)",
            "monthly_charges": 25.00,
            "total_charges": 1800.00,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "Yes",
            "online_backup": "Yes",
            "device_protection": "Yes",
            "tech_support": "Yes",
            "streaming_tv": "No",
            "streaming_movies": "No",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]


@pytest.mark.asyncio
async def test_smoke_predict_invalid_contract(client: AsyncClient) -> None:
    """Smoke test: Prediction with invalid contract value."""
    payload = {
        "features": {
            "gender": "Male",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "contract": "Invalid Contract",  # Invalid value
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 70.35,
            "total_charges": 840.20,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
        }
    }

    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Validation error


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
    assert "source" in data
