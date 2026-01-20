"""Prediction request/response schemas.

These schemas define the contract for the inference API.
Feature names follow the Telco Churn dataset convention.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class FeaturePayload(BaseModel):
    """Input features for prediction.

    Based on Telco Customer Churn dataset.
    Numeric and categorical features for churn prediction.
    """

    model_config = ConfigDict(extra="forbid")

    # Customer demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    senior_citizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0/1)")
    partner: str = Field(..., description="Has partner (Yes/No)")
    dependents: str = Field(..., description="Has dependents (Yes/No)")

    # Account information
    tenure: int = Field(..., ge=0, description="Months with company")
    contract: str = Field(..., description="Contract type")
    paperless_billing: str = Field(..., description="Paperless billing (Yes/No)")
    payment_method: str = Field(..., description="Payment method")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges ($)")
    total_charges: float = Field(..., ge=0, description="Total charges ($)")

    # Services
    phone_service: str = Field(..., description="Has phone service (Yes/No)")
    multiple_lines: str = Field(..., description="Has multiple lines")
    internet_service: str = Field(..., description="Internet service type")
    online_security: str = Field(..., description="Has online security")
    online_backup: str = Field(..., description="Has online backup")
    device_protection: str = Field(..., description="Has device protection")
    tech_support: str = Field(..., description="Has tech support")
    streaming_tv: str = Field(..., description="Has streaming TV")
    streaming_movies: str = Field(..., description="Has streaming movies")


class PredictionRequest(BaseModel):
    """Inference request payload."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    features: FeaturePayload = Field(..., description="Input features")


class PredictionResponse(BaseModel):
    """Inference response payload."""

    request_id: UUID = Field(..., description="Request ID for tracing")
    prediction: int = Field(..., description="Predicted class (0=No Churn, 1=Churn)")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of churn"
    )
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    latency_ms: float = Field(..., ge=0, description="Inference latency in ms")


class PredictionLog(BaseModel):
    """Schema for logging predictions to database.

    Stores metadata without raw PII. Features are hashed/aggregated.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(default=None, description="Auto-generated ID")
    request_id: UUID = Field(..., description="Request ID for tracing")
    timestamp: datetime = Field(..., description="Prediction timestamp (UTC)")

    # Model metadata
    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")

    # Prediction output
    prediction: int = Field(..., description="Predicted class")
    probability: float = Field(..., description="Prediction probability")
    latency_ms: float = Field(..., description="Inference latency")

    # Feature statistics (no raw values)
    feature_hash: str = Field(..., description="SHA256 hash of feature vector")
    numeric_feature_stats: dict[str, Any] = Field(
        default_factory=dict, description="Aggregated numeric feature stats"
    )
