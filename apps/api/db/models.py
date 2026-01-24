"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class PredictionLogDB(Base):
    """Prediction log database model.

    Stores prediction metadata for drift monitoring and audit trail.
    """

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Model metadata
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Prediction output
    predicted_fare: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Feature statistics
    feature_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    numeric_feature_stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
