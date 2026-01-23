"""Database models for drift monitoring."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from apps.api.db.models import Base

# Use the same Base as the API for unified schema management
MonitorBase = Base


class DriftMetricDB(MonitorBase):
    """Database model for drift metrics.

    Stores drift detection results for historical tracking
    and trend analysis.
    """

    __tablename__ = "drift_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Model context
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Window information
    reference_window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reference_window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    current_window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    current_window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    reference_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    current_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Aggregate metrics
    max_psi: Mapped[float] = mapped_column(Float, nullable=False)
    avg_psi: Mapped[float] = mapped_column(Float, nullable=False)
    drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    features_with_drift: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    drift_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Detailed feature drift (JSON blob)
    feature_drift_details: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Prediction drift
    prediction_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_drift_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Configuration used
    psi_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    min_drift_features: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<DriftMetric(id={self.id}, run_id={self.run_id}, "
            f"drift_detected={self.drift_detected}, max_psi={self.max_psi:.3f})>"
        )


class ReferenceDatasetDB(MonitorBase):
    """Database model for reference dataset metadata.

    Tracks which datasets are used as reference for drift detection.
    """

    __tablename__ = "reference_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Dataset info
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Statistics for reference
    dataset_stats: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Path or storage info
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ReferenceDataset(id={self.id}, name={self.name}, "
            f"samples={self.sample_count}, active={self.is_active})>"
        )


class DriftAlertDB(MonitorBase):
    """Database model for drift alerts.

    Records when drift thresholds are exceeded and actions taken.
    """

    __tablename__ = "drift_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drift_metric_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Alert details
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # low, medium, high
    message: Mapped[str] = mapped_column(Text, nullable=False)
    features_affected: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    # Action tracking
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    action_taken: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrain_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<DriftAlert(id={self.id}, severity={self.severity}, "
            f"acknowledged={self.acknowledged})>"
        )
