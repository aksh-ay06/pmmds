"""Drift detection metrics implementation.

Implements PSI (Population Stability Index), KL divergence,
and JS divergence for numeric and categorical features.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from shared.utils import get_logger

logger = get_logger(__name__)


class DriftSeverity(str, Enum):
    """Drift severity levels based on PSI thresholds."""

    NONE = "none"  # PSI < 0.1
    LOW = "low"  # 0.1 <= PSI < 0.2
    MEDIUM = "medium"  # 0.2 <= PSI < 0.25
    HIGH = "high"  # PSI >= 0.25


@dataclass
class FeatureDrift:
    """Drift metrics for a single feature."""

    feature_name: str
    psi: float
    kl_divergence: float | None = None
    js_divergence: float | None = None
    severity: DriftSeverity = DriftSeverity.NONE
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "psi": self.psi,
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
            "severity": self.severity.value,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "reference_std": self.reference_std,
            "current_std": self.current_std,
        }


@dataclass
class DriftResult:
    """Overall drift detection result."""

    timestamp: str
    reference_window_start: str
    reference_window_end: str
    current_window_start: str
    current_window_end: str
    reference_sample_count: int
    current_sample_count: int
    feature_drifts: list[FeatureDrift] = field(default_factory=list)
    prediction_drift: FeatureDrift | None = None
    drift_detected: bool = False
    features_with_drift: list[str] = field(default_factory=list)
    max_psi: float = 0.0
    avg_psi: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp,
            "reference_window_start": self.reference_window_start,
            "reference_window_end": self.reference_window_end,
            "current_window_start": self.current_window_start,
            "current_window_end": self.current_window_end,
            "reference_sample_count": self.reference_sample_count,
            "current_sample_count": self.current_sample_count,
            "feature_drifts": [fd.to_dict() for fd in self.feature_drifts],
            "prediction_drift": (
                self.prediction_drift.to_dict() if self.prediction_drift else None
            ),
            "drift_detected": self.drift_detected,
            "features_with_drift": self.features_with_drift,
            "max_psi": self.max_psi,
            "avg_psi": self.avg_psi,
        }


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) for numeric features.

    PSI measures the shift in distribution between reference and current data.

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change, monitor
    - 0.2 <= PSI < 0.25: Significant change, investigation needed
    - PSI >= 0.25: Major change, action required

    Args:
        reference: Reference distribution (training data).
        current: Current distribution (inference data).
        n_bins: Number of bins for histogram.
        eps: Small value to avoid log(0).

    Returns:
        PSI value.
    """
    # Handle edge cases
    if len(reference) == 0 or len(current) == 0:
        logger.warning("psi_empty_data", reference_len=len(reference), current_len=len(current))
        return 0.0

    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Define bin edges based on reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    # Handle constant values
    if max_val == min_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms (proportions)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions
    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    # Add epsilon to avoid division by zero and log(0)
    ref_props = np.clip(ref_props, eps, 1.0)
    cur_props = np.clip(cur_props, eps, 1.0)

    # Compute PSI
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

    return float(psi)


def compute_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    eps: float = 1e-6,
) -> float:
    """Compute PSI for categorical features.

    Args:
        reference: Reference series (training data).
        current: Current series (inference data).
        eps: Small value to avoid log(0).

    Returns:
        PSI value.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Get all unique categories from both distributions
    all_categories = set(reference.unique()) | set(current.unique())

    # Compute proportions
    ref_counts = reference.value_counts(normalize=True)
    cur_counts = current.value_counts(normalize=True)

    psi = 0.0
    for cat in all_categories:
        ref_prop = ref_counts.get(cat, eps)
        cur_prop = cur_counts.get(cat, eps)

        # Clip to avoid issues
        ref_prop = max(ref_prop, eps)
        cur_prop = max(cur_prop, eps)

        psi += (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)

    return float(psi)


def compute_kl_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute KL divergence between reference and current distributions.

    KL(P||Q) = sum(P * log(P/Q))

    Args:
        reference: Reference distribution (P).
        current: Current distribution (Q).
        n_bins: Number of bins for histogram.
        eps: Small value to avoid log(0).

    Returns:
        KL divergence value.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    if max_val == min_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    ref_props = np.clip(ref_props, eps, 1.0)
    cur_props = np.clip(cur_props, eps, 1.0)

    kl = np.sum(ref_props * np.log(ref_props / cur_props))

    return float(kl)


def compute_js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Jensen-Shannon divergence.

    JS divergence is symmetric and bounded [0, 1] when using log2.

    Args:
        reference: Reference distribution.
        current: Current distribution.
        n_bins: Number of bins for histogram.
        eps: Small value to avoid log(0).

    Returns:
        JS divergence value.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    if max_val == min_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    ref_props = np.clip(ref_props, eps, 1.0)
    cur_props = np.clip(cur_props, eps, 1.0)

    # Midpoint distribution
    m = 0.5 * (ref_props + cur_props)

    # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = np.sum(ref_props * np.log(ref_props / m))
    kl_qm = np.sum(cur_props * np.log(cur_props / m))

    js = 0.5 * kl_pm + 0.5 * kl_qm

    return float(js)


def get_severity(psi: float) -> DriftSeverity:
    """Determine drift severity from PSI value.

    Args:
        psi: PSI value.

    Returns:
        DriftSeverity level.
    """
    if psi < 0.1:
        return DriftSeverity.NONE
    elif psi < 0.2:
        return DriftSeverity.LOW
    elif psi < 0.25:
        return DriftSeverity.MEDIUM
    else:
        return DriftSeverity.HIGH


class DriftMetrics:
    """Compute drift metrics between reference and current data."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str] | None = None,
        psi_threshold: float = 0.2,
        n_bins: int = 10,
    ):
        """Initialize drift metrics calculator.

        Args:
            numeric_features: List of numeric feature names.
            categorical_features: List of categorical feature names.
            psi_threshold: PSI threshold for drift detection.
            n_bins: Number of bins for PSI calculation.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features or []
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins

    def compute_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> list[FeatureDrift]:
        """Compute drift for all features.

        Args:
            reference_df: Reference DataFrame (training data).
            current_df: Current DataFrame (inference data).

        Returns:
            List of FeatureDrift results.
        """
        feature_drifts = []

        # Numeric features - compute PSI, KL, JS
        for feature in self.numeric_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                logger.warning("feature_missing", feature=feature)
                continue

            ref_values = reference_df[feature].values
            cur_values = current_df[feature].values

            psi = compute_psi(ref_values, cur_values, n_bins=self.n_bins)
            kl = compute_kl_divergence(ref_values, cur_values, n_bins=self.n_bins)
            js = compute_js_divergence(ref_values, cur_values, n_bins=self.n_bins)

            feature_drift = FeatureDrift(
                feature_name=feature,
                psi=psi,
                kl_divergence=kl,
                js_divergence=js,
                severity=get_severity(psi),
                reference_mean=float(np.nanmean(ref_values)),
                current_mean=float(np.nanmean(cur_values)),
                reference_std=float(np.nanstd(ref_values)),
                current_std=float(np.nanstd(cur_values)),
            )
            feature_drifts.append(feature_drift)

        # Categorical features - compute PSI only
        for feature in self.categorical_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                logger.warning("feature_missing", feature=feature)
                continue

            psi = compute_categorical_psi(reference_df[feature], current_df[feature])

            feature_drift = FeatureDrift(
                feature_name=feature,
                psi=psi,
                severity=get_severity(psi),
            )
            feature_drifts.append(feature_drift)

        return feature_drifts

    def compute_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> FeatureDrift:
        """Compute drift in prediction distribution.

        Args:
            reference_predictions: Reference predictions.
            current_predictions: Current predictions.

        Returns:
            FeatureDrift for predictions.
        """
        psi = compute_psi(reference_predictions, current_predictions, n_bins=self.n_bins)

        return FeatureDrift(
            feature_name="prediction",
            psi=psi,
            severity=get_severity(psi),
            reference_mean=float(np.nanmean(reference_predictions)),
            current_mean=float(np.nanmean(current_predictions)),
            reference_std=float(np.nanstd(reference_predictions)),
            current_std=float(np.nanstd(current_predictions)),
        )

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_predictions: np.ndarray | None = None,
        current_predictions: np.ndarray | None = None,
        timestamp: str | None = None,
        reference_window: tuple[str, str] | None = None,
        current_window: tuple[str, str] | None = None,
    ) -> DriftResult:
        """Run full drift detection.

        Args:
            reference_df: Reference DataFrame.
            current_df: Current DataFrame.
            reference_predictions: Optional reference predictions.
            current_predictions: Optional current predictions.
            timestamp: Detection timestamp.
            reference_window: (start, end) for reference window.
            current_window: (start, end) for current window.

        Returns:
            DriftResult with all metrics.
        """
        from datetime import datetime, timezone

        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        ref_start, ref_end = reference_window or ("", "")
        cur_start, cur_end = current_window or ("", "")

        # Compute feature drift
        feature_drifts = self.compute_feature_drift(reference_df, current_df)

        # Compute prediction drift if provided
        prediction_drift = None
        if reference_predictions is not None and current_predictions is not None:
            prediction_drift = self.compute_prediction_drift(
                reference_predictions, current_predictions
            )

        # Determine which features have significant drift
        features_with_drift = [
            fd.feature_name
            for fd in feature_drifts
            if fd.psi >= self.psi_threshold
        ]

        # Check prediction drift too
        if prediction_drift and prediction_drift.psi >= self.psi_threshold:
            features_with_drift.append("prediction")

        # Compute aggregate metrics
        psi_values = [fd.psi for fd in feature_drifts]
        max_psi = max(psi_values) if psi_values else 0.0
        avg_psi = float(np.mean(psi_values)) if psi_values else 0.0

        # Drift detected if 3+ features exceed threshold (per CLAUDE.md)
        drift_detected = len(features_with_drift) >= 3

        result = DriftResult(
            timestamp=timestamp,
            reference_window_start=ref_start,
            reference_window_end=ref_end,
            current_window_start=cur_start,
            current_window_end=cur_end,
            reference_sample_count=len(reference_df),
            current_sample_count=len(current_df),
            feature_drifts=feature_drifts,
            prediction_drift=prediction_drift,
            drift_detected=drift_detected,
            features_with_drift=features_with_drift,
            max_psi=max_psi,
            avg_psi=avg_psi,
        )

        logger.info(
            "drift_detection_complete",
            drift_detected=drift_detected,
            features_with_drift=features_with_drift,
            max_psi=max_psi,
            avg_psi=avg_psi,
            reference_samples=len(reference_df),
            current_samples=len(current_df),
        )

        return result
