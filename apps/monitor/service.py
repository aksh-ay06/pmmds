"""Drift monitoring service.

Compares reference data distribution against recent inference data
to detect feature and prediction drift for NYC Yellow Taxi fare prediction.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from apps.monitor.config import DriftConfig, get_drift_config
from apps.monitor.db import DriftAlertDB, DriftMetricDB, MonitorBase
from shared.config import get_settings
from shared.data.dataset import (
    TARGET_COLUMN,
    load_dataset,
)
from shared.drift.metrics import DriftMetrics, DriftResult, DriftSeverity
from shared.utils import get_logger, get_metrics

logger = get_logger(__name__)
app_metrics = get_metrics()


class DriftMonitorService:
    """Service for monitoring feature and prediction drift.

    Compares reference data (training) against recent inference data
    from the prediction logs table.
    """

    def __init__(
        self,
        config: DriftConfig | None = None,
        db_url: str | None = None,
    ):
        """Initialize drift monitor service.

        Args:
            config: Drift monitoring configuration.
            db_url: Database URL. Defaults to settings.
        """
        self.config = config or get_drift_config()
        settings = get_settings()

        # Use sync URL for the monitoring service
        self.db_url = db_url or settings.database_url_sync
        self.engine = create_engine(self.db_url, echo=False)

        # Initialize drift metrics calculator
        self.drift_calculator = DriftMetrics(
            numeric_features=self.config.numeric_features,
            categorical_features=self.config.categorical_features,
            psi_threshold=self.config.psi_threshold_critical,
        )

        # Ensure monitoring tables exist
        self._init_db()

        logger.info(
            "drift_monitor_initialized",
            psi_threshold=self.config.psi_threshold_critical,
            min_drift_features=self.config.min_features_for_retrain,
            numeric_features=len(self.config.numeric_features),
            categorical_features=len(self.config.categorical_features),
        )

    def _init_db(self) -> None:
        """Initialize monitoring database tables."""
        MonitorBase.metadata.create_all(self.engine)
        logger.info("monitoring_tables_initialized")

    def _generate_run_id(self) -> str:
        """Generate unique run ID for this monitoring run."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _severity_from_psi(self, max_psi: float) -> DriftSeverity:
        """Derive overall severity from max PSI."""
        if max_psi >= 0.25:
            return DriftSeverity.HIGH
        elif max_psi >= 0.2:
            return DriftSeverity.MEDIUM
        return DriftSeverity.LOW

    def load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset for comparison.

        Returns:
            Reference DataFrame.

        Raises:
            FileNotFoundError: If reference data not found.
        """
        ref_path = Path(self.config.reference_data_path)

        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference dataset not found: {ref_path}. "
                "Run 'make download-data' first."
            )

        df = load_dataset(ref_path)

        # Select only monitored features
        all_features = self.config.numeric_features + self.config.categorical_features
        available_features = [f for f in all_features if f in df.columns]

        logger.info(
            "reference_data_loaded",
            path=str(ref_path),
            total_rows=len(df),
            features_available=len(available_features),
        )

        return df[available_features]

    def fetch_recent_inference_data(
        self,
        window_hours: int | None = None,
        model_name: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch recent inference data from prediction logs.

        Reconstructs feature distributions from the logged numeric_feature_stats.

        Args:
            window_hours: Hours of data to fetch.
            model_name: Filter by model name.

        Returns:
            Tuple of (DataFrame with reconstructed features, window metadata).
        """
        window_hours = window_hours or self.config.window_hours
        settings = get_settings()
        model_name = model_name or settings.model_name

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=window_hours)

        query = text("""
            SELECT
                timestamp,
                model_name,
                model_version,
                predicted_fare,
                numeric_feature_stats
            FROM prediction_logs
            WHERE timestamp >= :start_time
              AND timestamp <= :end_time
              AND model_name = :model_name
            ORDER BY timestamp DESC
        """)

        with Session(self.engine) as session:
            result = session.execute(
                query,
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "model_name": model_name,
                },
            )
            rows = result.fetchall()

        if not rows:
            logger.warning(
                "no_inference_data",
                window_hours=window_hours,
                model_name=model_name,
            )
            return pd.DataFrame(), {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "sample_count": 0,
            }

        # Reconstruct data from logged stats
        data = []
        predictions = []

        for row in rows:
            stats = row.numeric_feature_stats or {}

            # Reconstruct numeric features from stored values
            record = {}
            record["trip_distance"] = stats.get("trip_distance", 3.0)
            record["trip_duration_minutes"] = stats.get("trip_duration_minutes", 15.0)
            record["passenger_count"] = stats.get("passenger_count", 1)
            record["pickup_hour"] = stats.get("pickup_hour", 12)

            # Reconstruct pickup_day_of_week from is_weekend
            is_weekend = stats.get("is_weekend", 0)
            record["pickup_day_of_week"] = 1 if is_weekend else 3  # Sun or Wed
            record["is_weekend"] = is_weekend
            record["is_rush_hour"] = stats.get("is_rush_hour", 0)

            # Categorical features
            record["pickup_borough"] = stats.get("pickup_borough", "Manhattan")
            record["dropoff_borough"] = stats.get("dropoff_borough", "Manhattan")
            record["RatecodeID"] = stats.get("RatecodeID", 1)
            record["payment_type"] = stats.get("payment_type", 1)

            data.append(record)
            predictions.append(row.predicted_fare)

        df = pd.DataFrame(data)

        # Add predictions for prediction drift
        df["_predicted_fare"] = predictions

        window_metadata = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "sample_count": len(df),
            "model_name": model_name,
        }

        logger.info(
            "inference_data_fetched",
            window_hours=window_hours,
            sample_count=len(df),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

        return df, window_metadata

    def run_drift_detection(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> DriftResult | None:
        """Run drift detection comparing reference vs recent data.

        Args:
            model_name: Model name for filtering.
            model_version: Model version for logging.

        Returns:
            DriftResult or None if insufficient data.
        """
        settings = get_settings()
        model_name = model_name or settings.model_name
        model_version = model_version or "unknown"

        run_id = self._generate_run_id()
        timestamp = datetime.now(timezone.utc)

        logger.info(
            "drift_detection_started",
            run_id=run_id,
            model_name=model_name,
        )

        try:
            # Load reference data
            reference_df = self.load_reference_data()

            # Fetch recent inference data
            current_df, window_meta = self.fetch_recent_inference_data(
                model_name=model_name
            )

            # Check minimum sample requirement
            if len(current_df) < self.config.min_samples:
                logger.warning(
                    "insufficient_samples",
                    current_samples=len(current_df),
                    required=self.config.min_samples,
                )
                return None

            # Extract predictions for prediction drift
            current_predictions = current_df["_predicted_fare"].values
            current_df = current_df.drop(columns=["_predicted_fare"])

            # Find common features between reference and current
            numeric_features = [
                f for f in self.config.numeric_features
                if f in reference_df.columns and f in current_df.columns
            ]
            categorical_features = [
                f for f in self.config.categorical_features
                if f in reference_df.columns and f in current_df.columns
            ]

            all_features = numeric_features + categorical_features
            reference_subset = reference_df[all_features]
            current_subset = current_df[all_features]

            # Use fare_amount from reference as reference predictions
            ref_full = load_dataset(Path(self.config.reference_data_path))
            if TARGET_COLUMN in ref_full.columns:
                reference_predictions = ref_full[TARGET_COLUMN].values.astype(float)
            else:
                # Fallback: generate synthetic fare distribution
                reference_predictions = np.random.default_rng(42).normal(15.0, 10.0, len(reference_df))
                reference_predictions = np.clip(reference_predictions, 2.5, 200.0)

            # Run drift detection
            drift_result = self.drift_calculator.detect_drift(
                reference_df=reference_subset,
                current_df=current_subset,
                reference_predictions=reference_predictions,
                current_predictions=current_predictions,
                timestamp=timestamp.isoformat(),
                reference_window=("training_data", "training_data"),
                current_window=(window_meta["start_time"], window_meta["end_time"]),
            )

            # Store results in database
            self._store_drift_result(
                run_id=run_id,
                timestamp=timestamp,
                model_name=model_name,
                model_version=model_version,
                drift_result=drift_result,
                window_meta=window_meta,
            )

            # Create alert if drift detected
            if drift_result.drift_detected:
                self._create_drift_alert(
                    run_id=run_id,
                    timestamp=timestamp,
                    drift_result=drift_result,
                )

            # Record drift metrics for observability
            app_metrics.record_drift_check(model_name=model_name)
            if drift_result.drift_detected:
                sev = self._severity_from_psi(drift_result.max_psi)
                app_metrics.record_drift_event(
                    drift_type="feature",
                    severity=sev.value,
                )

            logger.info(
                "drift_detection_completed",
                run_id=run_id,
                drift_detected=drift_result.drift_detected,
                max_psi=drift_result.max_psi,
                features_with_drift=drift_result.features_with_drift,
            )

            return drift_result

        except Exception as e:
            logger.error(
                "drift_detection_failed",
                run_id=run_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _store_drift_result(
        self,
        run_id: str,
        timestamp: datetime,
        model_name: str,
        model_version: str,
        drift_result: DriftResult,
        window_meta: dict[str, Any],
    ) -> None:
        """Store drift result in database."""
        current_start = datetime.fromisoformat(window_meta["start_time"])
        current_end = datetime.fromisoformat(window_meta["end_time"])

        feature_details = {
            fd.feature_name: fd.to_dict()
            for fd in drift_result.feature_drifts
        }

        db_record = DriftMetricDB(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            model_version=model_version,
            reference_window_start=None,
            reference_window_end=None,
            current_window_start=current_start,
            current_window_end=current_end,
            reference_sample_count=drift_result.reference_sample_count,
            current_sample_count=drift_result.current_sample_count,
            max_psi=drift_result.max_psi,
            avg_psi=drift_result.avg_psi,
            drift_detected=drift_result.drift_detected,
            features_with_drift=drift_result.features_with_drift,
            drift_count=len(drift_result.features_with_drift),
            feature_drift_details=feature_details,
            prediction_psi=(
                drift_result.prediction_drift.psi
                if drift_result.prediction_drift
                else None
            ),
            prediction_drift_detected=(
                drift_result.prediction_drift.psi >= self.config.prediction_psi_threshold
                if drift_result.prediction_drift
                else False
            ),
            psi_threshold=self.config.psi_threshold_critical,
            min_drift_features=self.config.min_features_for_retrain,
        )

        with Session(self.engine) as session:
            session.add(db_record)
            session.commit()

        logger.info(
            "drift_result_stored",
            run_id=run_id,
            drift_detected=drift_result.drift_detected,
        )

    def _create_drift_alert(
        self,
        run_id: str,
        timestamp: datetime,
        drift_result: DriftResult,
    ) -> None:
        """Create drift alert in database."""
        if drift_result.max_psi >= 0.25:
            severity = "high"
        elif drift_result.max_psi >= 0.2:
            severity = "medium"
        else:
            severity = "low"

        message = (
            f"Drift detected in {len(drift_result.features_with_drift)} features: "
            f"{', '.join(drift_result.features_with_drift)}. "
            f"Max PSI: {drift_result.max_psi:.3f}, Avg PSI: {drift_result.avg_psi:.3f}"
        )

        with Session(self.engine) as session:
            result = session.execute(
                select(DriftMetricDB.id).where(DriftMetricDB.run_id == run_id)
            )
            metric_row = result.first()
            drift_metric_id = metric_row[0] if metric_row else 0

            alert = DriftAlertDB(
                drift_metric_id=drift_metric_id,
                created_at=timestamp,
                severity=severity,
                message=message,
                features_affected=drift_result.features_with_drift,
                acknowledged=False,
                retrain_triggered=False,
            )
            session.add(alert)
            session.commit()

        logger.warning(
            "drift_alert_created",
            severity=severity,
            features_affected=drift_result.features_with_drift,
            max_psi=drift_result.max_psi,
        )

    def get_drift_summary(
        self,
        hours: int = 24,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Get drift summary for recent period."""
        settings = get_settings()
        model_name = model_name or settings.model_name

        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        with Session(self.engine) as session:
            query = select(DriftMetricDB).where(
                DriftMetricDB.timestamp >= since,
                DriftMetricDB.model_name == model_name,
            ).order_by(DriftMetricDB.timestamp.desc())

            results = session.execute(query).scalars().all()

        if not results:
            return {
                "period_hours": hours,
                "model_name": model_name,
                "runs": 0,
                "drift_detected_count": 0,
                "latest_drift": None,
            }

        drift_detected_count = sum(1 for r in results if r.drift_detected)
        latest = results[0]

        return {
            "period_hours": hours,
            "model_name": model_name,
            "runs": len(results),
            "drift_detected_count": drift_detected_count,
            "latest_run": {
                "run_id": latest.run_id,
                "timestamp": latest.timestamp.isoformat(),
                "drift_detected": latest.drift_detected,
                "max_psi": latest.max_psi,
                "avg_psi": latest.avg_psi,
                "features_with_drift": latest.features_with_drift,
                "current_sample_count": latest.current_sample_count,
            },
            "avg_max_psi": float(np.mean([r.max_psi for r in results])),
            "avg_avg_psi": float(np.mean([r.avg_psi for r in results])),
        }

    def get_unacknowledged_alerts(self) -> list[dict[str, Any]]:
        """Get all unacknowledged drift alerts."""
        with Session(self.engine) as session:
            query = select(DriftAlertDB).where(
                DriftAlertDB.acknowledged == False  # noqa: E712
            ).order_by(DriftAlertDB.created_at.desc())

            results = session.execute(query).scalars().all()

        return [
            {
                "id": alert.id,
                "created_at": alert.created_at.isoformat(),
                "severity": alert.severity,
                "message": alert.message,
                "features_affected": alert.features_affected,
            }
            for alert in results
        ]
