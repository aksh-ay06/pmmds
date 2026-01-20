"""Model promotion decision tracking."""

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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class PromotionBase(DeclarativeBase):
    """Base class for promotion database models."""

    pass


class PromotionDecisionDB(PromotionBase):
    """Database model for model promotion decisions.

    Records the outcome of champion vs challenger comparisons
    and promotion decisions for audit trail.
    """

    __tablename__ = "promotion_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Trigger context
    trigger_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # drift, scheduled, manual
    drift_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Champion model (current production)
    champion_model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    champion_model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    champion_mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    champion_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Challenger model (newly trained)
    challenger_model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    challenger_model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    challenger_mlflow_run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    challenger_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Comparison results
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metric_improvement: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_acceptable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    primary_metric_name: Mapped[str] = mapped_column(
        String(50), nullable=False, default="roc_auc"
    )
    primary_metric_improvement: Mapped[float] = mapped_column(Float, nullable=False)

    # Decision outcome
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    promotion_reason: Mapped[str] = mapped_column(Text, nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Timing
    comparison_duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<PromotionDecision(id={self.id}, "
            f"challenger={self.challenger_model_version}, "
            f"promoted={self.promoted})>"
        )


class RetrainingRunDB(PromotionBase):
    """Database model for retraining runs.

    Tracks all retraining attempts, whether triggered by drift
    or scheduled.
    """

    __tablename__ = "retraining_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Trigger information
    trigger_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # drift, scheduled, manual
    drift_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    drift_features: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    drift_max_psi: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Training details
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    training_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Results
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, running, completed, failed
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Promotion outcome
    promotion_decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Timing
    training_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<RetrainingRun(id={self.id}, run_id={self.run_id}, "
            f"status={self.status}, promoted={self.promoted})>"
        )
