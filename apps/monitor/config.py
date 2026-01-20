"""Drift monitoring configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DriftConfig(BaseSettings):
    """Configuration for drift monitoring.

    Loads from environment variables with PMMDS_DRIFT_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PMMDS_DRIFT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PSI thresholds
    psi_threshold: float = Field(
        default=0.2,
        description="PSI threshold for individual feature drift detection",
    )
    prediction_psi_threshold: float = Field(
        default=0.2,
        description="PSI threshold for prediction drift",
    )

    # Drift detection trigger
    min_drift_features: int = Field(
        default=3,
        description="Minimum number of features with drift to trigger alert",
    )

    # Window configuration
    reference_window_days: int = Field(
        default=30,
        description="Number of days for reference window (training data)",
    )
    current_window_hours: int = Field(
        default=24,
        description="Number of hours for current/recent inference window",
    )
    min_samples_required: int = Field(
        default=100,
        description="Minimum samples required for drift detection",
    )

    # Monitoring schedule
    monitor_interval_hours: int = Field(
        default=1,
        description="Hours between drift monitoring runs",
    )
    enable_scheduled_monitoring: bool = Field(
        default=True,
        description="Enable scheduled drift monitoring",
    )

    # Alert settings
    enable_alerts: bool = Field(
        default=True,
        description="Enable drift alerts",
    )
    alert_on_prediction_drift: bool = Field(
        default=True,
        description="Alert when prediction distribution drifts",
    )

    # Feature configuration
    numeric_features: list[str] = Field(
        default=["tenure", "monthly_charges", "total_charges"],
        description="Numeric features to monitor for drift",
    )
    categorical_features: list[str] = Field(
        default=[
            "gender",
            "partner",
            "dependents",
            "phone_service",
            "multiple_lines",
            "internet_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
            "contract",
            "paperless_billing",
            "payment_method",
        ],
        description="Categorical features to monitor for drift",
    )

    # Reference dataset
    reference_dataset_path: str = Field(
        default="data/processed/telco_churn.csv",
        description="Path to reference dataset",
    )
    use_training_data_as_reference: bool = Field(
        default=True,
        description="Use training data as reference distribution",
    )


def get_drift_config() -> DriftConfig:
    """Get drift configuration instance."""
    return DriftConfig()
