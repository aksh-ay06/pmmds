"""Tests for validation module."""

import pytest

from shared.validation import validate_inference_payload, validate_dataframe
from shared.validation.expectations import FEATURE_EXPECTATIONS, NUMERIC_CONSTRAINTS


def test_validate_inference_payload_valid() -> None:
    """Test validation with valid taxi features."""
    features = {
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
    result = validate_inference_payload(features)
    assert result.success


def test_validate_inference_payload_invalid_distance() -> None:
    """Test validation with trip_distance out of range."""
    features = {
        "trip_distance": 150.0,  # Max is 100
        "passenger_count": 1,
        "pickup_hour": 12,
        "pickup_day_of_week": 3,
        "pickup_month": 1,
        "trip_duration_minutes": 10.0,
        "is_weekend": 0,
        "is_rush_hour": 0,
        "RatecodeID": 1,
        "payment_type": 1,
        "pickup_borough": "Manhattan",
        "dropoff_borough": "Manhattan",
    }
    result = validate_inference_payload(features)
    assert not result.success
    assert any(e.get("feature") == "trip_distance" for e in result.errors)


def test_validate_inference_payload_invalid_borough() -> None:
    """Test validation with invalid borough."""
    features = {
        "trip_distance": 3.0,
        "passenger_count": 1,
        "pickup_hour": 12,
        "pickup_day_of_week": 3,
        "pickup_month": 1,
        "trip_duration_minutes": 10.0,
        "is_weekend": 0,
        "is_rush_hour": 0,
        "RatecodeID": 1,
        "payment_type": 1,
        "pickup_borough": "InvalidPlace",
        "dropoff_borough": "Manhattan",
    }
    result = validate_inference_payload(features)
    assert not result.success


def test_validate_inference_payload_missing_feature() -> None:
    """Test validation with missing feature."""
    features = {
        "trip_distance": 3.0,
        # Missing passenger_count and others
    }
    result = validate_inference_payload(features)
    assert not result.success


def test_validate_dataframe_valid() -> None:
    """Test DataFrame validation with valid data."""
    import pandas as pd

    data = {
        "trip_distance": [3.5, 5.0, 1.2],
        "passenger_count": [1, 2, 3],
        "pickup_hour": [14, 8, 22],
        "pickup_day_of_week": [3, 5, 1],
        "pickup_month": [1, 2, 1],
        "trip_duration_minutes": [15.0, 25.0, 5.0],
        "is_weekend": [0, 0, 1],
        "is_rush_hour": [0, 1, 0],
        "RatecodeID": [1, 1, 2],
        "payment_type": [1, 2, 1],
        "pickup_borough": ["Manhattan", "Brooklyn", "Queens"],
        "dropoff_borough": ["Brooklyn", "Manhattan", "Manhattan"],
        "fare_amount": [12.0, 20.0, 8.0],
    }
    df = pd.DataFrame(data)
    result = validate_dataframe(df, include_target=True)
    assert result.success


def test_validate_dataframe_out_of_range() -> None:
    """Test DataFrame validation with out-of-range values."""
    import pandas as pd

    data = {
        "trip_distance": [200.0],  # Out of range
        "passenger_count": [1],
        "pickup_hour": [14],
        "pickup_day_of_week": [3],
        "pickup_month": [1],
        "trip_duration_minutes": [15.0],
        "is_weekend": [0],
        "is_rush_hour": [0],
        "RatecodeID": [1],
        "payment_type": [1],
        "pickup_borough": ["Manhattan"],
        "dropoff_borough": ["Manhattan"],
        "fare_amount": [50.0],
    }
    df = pd.DataFrame(data)
    result = validate_dataframe(df, include_target=True, use_ge=False)
    assert not result.success


def test_numeric_constraints_defined() -> None:
    """Test that all expected numeric constraints are defined."""
    expected = [
        "trip_distance", "passenger_count", "pickup_hour",
        "pickup_day_of_week", "pickup_month", "trip_duration_minutes",
        "fare_amount",
    ]
    for feature in expected:
        assert feature in NUMERIC_CONSTRAINTS, f"Missing constraint for {feature}"
        assert "min" in NUMERIC_CONSTRAINTS[feature]
        assert "max" in NUMERIC_CONSTRAINTS[feature]


def test_feature_expectations_defined() -> None:
    """Test that all expected features have expectations."""
    expected_features = [
        "trip_distance", "passenger_count", "pickup_hour",
        "pickup_day_of_week", "pickup_month", "trip_duration_minutes",
        "is_weekend", "is_rush_hour", "RatecodeID", "payment_type",
        "pickup_borough", "dropoff_borough", "fare_amount",
    ]
    for feature in expected_features:
        assert feature in FEATURE_EXPECTATIONS, f"Missing expectation for {feature}"
