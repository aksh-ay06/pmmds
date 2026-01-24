"""Prediction route for NYC Yellow Taxi fare prediction."""

import hashlib
import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db_session
from apps.api.db.models import PredictionLogDB
from apps.api.models.loader import get_model_loader
from apps.api.schemas.predict import (
    PredictionRequest,
    PredictionResponse,
)
from shared.utils import get_logger, get_metrics
from shared.validation import validate_inference_payload

logger = get_logger(__name__)
metrics = get_metrics()

router = APIRouter(tags=["predictions"])


def _compute_feature_hash(features: dict) -> str:
    """Compute deterministic hash of feature values."""
    serialized = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _compute_numeric_stats(features: dict) -> dict:
    """Compute binned stats for numeric features for drift monitoring."""
    stats = {}

    # Trip distance bins
    trip_distance = features.get("trip_distance", 0)
    if trip_distance <= 1:
        stats["trip_distance_bin"] = "0-1"
    elif trip_distance <= 3:
        stats["trip_distance_bin"] = "1-3"
    elif trip_distance <= 5:
        stats["trip_distance_bin"] = "3-5"
    elif trip_distance <= 10:
        stats["trip_distance_bin"] = "5-10"
    else:
        stats["trip_distance_bin"] = "10+"

    # Trip duration bins
    duration = features.get("trip_duration_minutes", 0)
    if duration <= 5:
        stats["trip_duration_bin"] = "0-5"
    elif duration <= 15:
        stats["trip_duration_bin"] = "5-15"
    elif duration <= 30:
        stats["trip_duration_bin"] = "15-30"
    elif duration <= 60:
        stats["trip_duration_bin"] = "30-60"
    else:
        stats["trip_duration_bin"] = "60+"

    # Hour bins
    hour = features.get("pickup_hour", 0)
    if hour < 6:
        stats["hour_bin"] = "night"
    elif hour < 10:
        stats["hour_bin"] = "morning"
    elif hour < 16:
        stats["hour_bin"] = "midday"
    elif hour < 20:
        stats["hour_bin"] = "evening"
    else:
        stats["hour_bin"] = "night"

    # Passenger count
    stats["passenger_count"] = features.get("passenger_count", 1)

    # Store categorical and numeric features for drift monitoring
    stats["pickup_borough"] = features.get("pickup_borough", "Unknown")
    stats["dropoff_borough"] = features.get("dropoff_borough", "Unknown")
    stats["RatecodeID"] = features.get("RatecodeID", 1)
    stats["payment_type"] = features.get("payment_type", 1)
    stats["is_weekend"] = features.get("is_weekend", 0)
    stats["is_rush_hour"] = features.get("is_rush_hour", 0)
    stats["trip_distance"] = features.get("trip_distance", 0)
    stats["trip_duration_minutes"] = features.get("trip_duration_minutes", 0)

    return stats


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db_session),
) -> PredictionResponse:
    """Generate fare prediction for a taxi trip.

    Args:
        request: Prediction request with trip features.
        db: Database session.

    Returns:
        PredictionResponse with predicted fare.
    """
    start_time = time.perf_counter()
    request_id = request.request_id

    try:
        # Get feature dict
        features = request.features.to_feature_dict()

        # Validate features
        validation_result = validate_inference_payload(features)
        if not validation_result.success:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Feature validation failed",
                    "errors": validation_result.errors,
                },
            )

        # Get model and predict
        loader = get_model_loader()
        model = loader.get_current()
        predicted_fare = model.predict(features)

        # Compute latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Build response
        response = PredictionResponse(
            request_id=request_id,
            predicted_fare=predicted_fare,
            model_name=model.name,
            model_version=model.version,
            latency_ms=round(latency_ms, 2),
        )

        # Log to database (non-blocking best-effort)
        try:
            feature_hash = _compute_feature_hash(features)
            numeric_stats = _compute_numeric_stats(features)

            log_entry = PredictionLogDB(
                request_id=str(request_id),
                timestamp=datetime.now(timezone.utc),
                model_name=model.name,
                model_version=model.version,
                predicted_fare=predicted_fare,
                latency_ms=latency_ms,
                feature_hash=feature_hash,
                numeric_feature_stats=numeric_stats,
            )

            db.add(log_entry)
            await db.commit()
        except Exception as log_error:
            logger.warning(
                "prediction_log_failed",
                error=str(log_error),
                request_id=str(request_id),
            )

        # Record metrics
        metrics.record_prediction(
            model_name=model.name,
            model_version=model.version,
            prediction=round(predicted_fare),
            latency_seconds=latency_ms / 1000,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "prediction_failed",
            error=str(e),
            error_type=type(e).__name__,
            request_id=str(request_id),
            latency_ms=latency_ms,
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "Prediction failed", "error": str(e)},
        )
