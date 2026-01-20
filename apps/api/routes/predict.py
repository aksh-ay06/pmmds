"""Prediction endpoint."""

import hashlib
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import PredictionLogDB, get_db_session
from apps.api.models import get_model_loader
from apps.api.schemas import PredictionLog, PredictionRequest, PredictionResponse
from shared.utils import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["inference"])


def compute_feature_hash(features: dict) -> str:
    """Compute SHA256 hash of feature vector.

    Args:
        features: Feature dictionary.

    Returns:
        Hex digest of feature hash.
    """
    # Sort keys for consistent hashing
    sorted_items = sorted(features.items())
    feature_str = str(sorted_items).encode("utf-8")
    return hashlib.sha256(feature_str).hexdigest()


def extract_numeric_stats(features: dict) -> dict:
    """Extract aggregated statistics from numeric features.

    No raw values stored - only aggregates for drift detection.

    Args:
        features: Feature dictionary.

    Returns:
        Dictionary with numeric feature statistics.
    """
    numeric_keys = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
    stats = {}

    for key in numeric_keys:
        if key in features:
            value = features[key]
            # Store binned ranges instead of exact values
            if key == "tenure":
                stats[f"{key}_bin"] = (
                    "0-12" if value <= 12 else "13-36" if value <= 36 else "37+"
                )
            elif key == "monthly_charges":
                stats[f"{key}_bin"] = (
                    "low" if value < 35 else "medium" if value < 70 else "high"
                )
            elif key == "total_charges":
                stats[f"{key}_bin"] = (
                    "low" if value < 500 else "medium" if value < 2000 else "high"
                )
            else:
                stats[key] = value

    return stats


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request payload"},
        500: {"description": "Model inference failed"},
    },
)
async def predict(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db_session),
) -> PredictionResponse:
    """Generate prediction for input features.

    Workflow:
    1. Validate request payload
    2. Load current model
    3. Run inference
    4. Log prediction metadata (no raw PII)
    5. Return prediction response

    Args:
        request: Prediction request with features.
        db: Database session.

    Returns:
        Prediction response with class and probability.
    """
    start_time = time.perf_counter()

    try:
        # Get model
        model_loader = get_model_loader()
        model = model_loader.get_current()

        # Extract features as dict
        features_dict = request.features.model_dump()

        # Run inference
        prediction, probability = model.predict(features_dict)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Prepare response
        response = PredictionResponse(
            request_id=request.request_id,
            prediction=prediction,
            probability=probability,
            model_name=model.name,
            model_version=model.version,
            latency_ms=round(latency_ms, 2),
        )

        # Log to database (async, no raw PII)
        prediction_log = PredictionLogDB(
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc),
            model_name=model.name,
            model_version=model.version,
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
            feature_hash=compute_feature_hash(features_dict),
            numeric_feature_stats=extract_numeric_stats(features_dict),
        )
        db.add(prediction_log)

        logger.info(
            "prediction_completed",
            request_id=str(request.request_id),
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
            model_name=model.name,
            model_version=model.version,
        )

        return response

    except Exception as e:
        logger.error(
            "prediction_failed",
            request_id=str(request.request_id),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {type(e).__name__}",
        ) from e
