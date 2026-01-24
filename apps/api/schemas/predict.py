"""Prediction request/response schemas for NYC Yellow Taxi fare prediction."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.data.locations import VALID_BOROUGHS


class FeaturePayload(BaseModel):
    """Input features for fare prediction.

    NYC Yellow Taxi trip features covering distance, time, location,
    and trip characteristics.
    """

    model_config = ConfigDict(extra="forbid")

    # Numeric features
    trip_distance: float = Field(..., ge=0.1, le=100.0, description="Trip distance in miles")
    passenger_count: int = Field(..., ge=1, le=6, description="Number of passengers")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_day_of_week: int = Field(..., ge=1, le=7, description="Day of week (1=Sun, 7=Sat)")
    pickup_month: int = Field(..., ge=1, le=12, description="Month of pickup (1-12)")
    trip_duration_minutes: float = Field(..., ge=1.0, le=180.0, description="Trip duration in minutes")

    # Binary features
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend indicator (0/1)")
    is_rush_hour: int = Field(..., ge=0, le=1, description="Rush hour indicator (0/1)")

    # Categorical features
    RatecodeID: int = Field(..., ge=1, le=6, description="Rate code (1=Standard, 2=JFK, etc.)")
    payment_type: int = Field(..., ge=1, le=4, description="Payment type (1=Card, 2=Cash, 3=No charge, 4=Dispute)")
    pickup_borough: str = Field(..., description="Pickup borough")
    dropoff_borough: str = Field(..., description="Dropoff borough")

    @field_validator("pickup_borough", "dropoff_borough")
    @classmethod
    def validate_borough(cls, v: str) -> str:
        """Validate borough is one of the known NYC boroughs."""
        if v not in VALID_BOROUGHS:
            raise ValueError(f"Invalid borough: {v}. Must be one of {VALID_BOROUGHS}")
        return v

    def to_feature_dict(self) -> dict[str, Any]:
        """Convert to flat feature dictionary for model input."""
        return self.model_dump()


class PredictionRequest(BaseModel):
    """Inference request payload."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    features: FeaturePayload = Field(..., description="Trip features for fare prediction")


class PredictionResponse(BaseModel):
    """Inference response payload."""

    request_id: UUID = Field(..., description="Request ID for tracing")
    predicted_fare: float = Field(..., description="Predicted fare amount in USD")
    model_name: str = Field(..., description="Model name used for prediction")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., ge=0, description="Inference latency in ms")


class PredictionLog(BaseModel):
    """Schema for logging predictions to database."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(default=None, description="Auto-generated ID")
    request_id: UUID = Field(..., description="Request ID for tracing")
    timestamp: datetime = Field(..., description="Prediction timestamp (UTC)")

    # Model metadata
    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")

    # Prediction output
    predicted_fare: float = Field(..., description="Predicted fare amount")
    latency_ms: float = Field(..., description="Inference latency")

    # Feature statistics (no raw values)
    feature_hash: str = Field(..., description="SHA256 hash of feature vector")
    numeric_feature_stats: dict[str, Any] = Field(
        default_factory=dict, description="Aggregated numeric feature stats"
    )
