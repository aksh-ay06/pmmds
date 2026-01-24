"""Drift monitoring configuration for NYC Yellow Taxi."""

from dataclasses import dataclass, field


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Reference data
    reference_data_path: str = "data/processed/reference.csv"

    # Monitoring window
    window_hours: int = 24
    min_samples: int = 100

    # PSI thresholds
    psi_threshold_warning: float = 0.1
    psi_threshold_critical: float = 0.2

    # Drift trigger for retraining (from CLAUDE.md)
    min_features_for_retrain: int = 3

    # Numeric features to monitor (continuous values stored in stats)
    numeric_features: list[str] = field(default_factory=lambda: [
        "trip_distance",
        "trip_duration_minutes",
        "passenger_count",
        "pickup_hour",
        "pickup_day_of_week",
        "is_weekend",
        "is_rush_hour",
    ])

    # Categorical features to monitor
    categorical_features: list[str] = field(default_factory=lambda: [
        "pickup_borough",
        "dropoff_borough",
        "RatecodeID",
        "payment_type",
    ])

    # Prediction drift monitoring
    monitor_prediction_drift: bool = True
    prediction_feature_name: str = "predicted_fare"
    prediction_psi_threshold: float = 0.15

    # Database
    db_url: str | None = None

    @property
    def all_features(self) -> list[str]:
        """Get all monitored features."""
        return self.numeric_features + self.categorical_features


def get_drift_config() -> DriftConfig:
    """Get drift configuration instance."""
    return DriftConfig()
