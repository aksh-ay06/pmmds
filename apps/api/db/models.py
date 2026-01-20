"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class PredictionLogDB(Base):
    """Database model for prediction logs.

    Stores inference metadata without raw PII.
    Used for monitoring and drift detection.
    """

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), unique=True, index=True, nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Model metadata
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Prediction output
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Feature metadata (no raw values)
    feature_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    numeric_feature_stats: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<PredictionLog(id={self.id}, request_id={self.request_id}, "
            f"prediction={self.prediction}, model={self.model_name}:{self.model_version})>"
        )
