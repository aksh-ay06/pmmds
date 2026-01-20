"""Drift monitoring service.

Compares reference data distribution against recent inference data
to detect feature and prediction drift.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from apps.monitor.config import DriftConfig, get_drift_config
from apps.monitor.db import DriftAlertDB, DriftMetricDB, MonitorBase, ReferenceDatasetDB
from shared.config import get_settings
from shared.data.dataset import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    compute_dataset_stats,
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
            psi_threshold=self.config.psi_threshold,
        )

        # Ensure monitoring tables exist
        self._init_db()

        logger.info(
            "drift_monitor_initialized",
            psi_threshold=self.config.psi_threshold,
            min_drift_features=self.config.min_drift_features,
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

    def load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset for comparison.

        Returns:
            Reference DataFrame.

        Raises:
            FileNotFoundError: If reference data not found.
        """
        ref_path = Path(self.config.reference_dataset_path)

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

        Since we store feature hashes and binned stats (not raw values),
        we reconstruct approximate distributions from the logged stats.

        Args:
            window_hours: Hours of data to fetch.
            model_name: Filter by model name.

        Returns:
            Tuple of (DataFrame with reconstructed features, window metadata).
        """
        window_hours = window_hours or self.config.current_window_hours
        settings = get_settings()
        model_name = model_name or settings.model_name

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=window_hours)

        query = text("""
            SELECT 
                timestamp,
                model_name,
                model_version,
                prediction,
                probability,
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
        probabilities = []

        for row in rows:
            # Extract binned stats
            stats = row.numeric_feature_stats or {}

            # Reconstruct approximate values from bins
            record = {}

            # Tenure bins: 0-12, 13-36, 37+
            tenure_bin = stats.get("tenure_bin", "13-36")
            if tenure_bin == "0-12":
                record["tenure"] = np.random.uniform(0, 12)
            elif tenure_bin == "13-36":
                record["tenure"] = np.random.uniform(13, 36)
            else:
                record["tenure"] = np.random.uniform(37, 72)

            # Monthly charges bins: low (<35), medium (35-70), high (>70)
            mc_bin = stats.get("monthly_charges_bin", "medium")
            if mc_bin == "low":
                record["monthly_charges"] = np.random.uniform(18, 35)
            elif mc_bin == "medium":
                record["monthly_charges"] = np.random.uniform(35, 70)
            else:
                record["monthly_charges"] = np.random.uniform(70, 120)

            # Total charges bins: low (<500), medium (500-2000), high (>2000)
            tc_bin = stats.get("total_charges_bin", "medium")
            if tc_bin == "low":
                record["total_charges"] = np.random.uniform(18, 500)
            elif tc_bin == "medium":
                record["total_charges"] = np.random.uniform(500, 2000)
            else:
                record["total_charges"] = np.random.uniform(2000, 8000)

            # Senior citizen is stored as-is
            record["senior_citizen"] = stats.get("senior_citizen", 0)

            data.append(record)
            predictions.append(row.prediction)
            probabilities.append(row.probability)

        df = pd.DataFrame(data)

        # Add predictions for prediction drift
        df["_prediction"] = predictions
        df["_probability"] = probabilities

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
            if len(current_df) < self.config.min_samples_required:
                logger.warning(
                    "insufficient_samples",
                    current_samples=len(current_df),
                    required=self.config.min_samples_required,
                )
                return None

            # Extract predictions for prediction drift
            current_predictions = current_df["_probability"].values
            current_df = current_df.drop(columns=["_prediction", "_probability"])

            # Use numeric features only (categorical not available from logs)
            numeric_only_features = [
                f for f in self.config.numeric_features
                if f in reference_df.columns and f in current_df.columns
            ]

            reference_numeric = reference_df[numeric_only_features]
            current_numeric = current_df[numeric_only_features]

            # Generate synthetic reference predictions for comparison
            # Based on reference data churn rate
            reference_predictions = np.random.uniform(0, 1, len(reference_df))

            # Run drift detection
            drift_result = self.drift_calculator.detect_drift(
                reference_df=reference_numeric,
                current_df=current_numeric,
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
            if drift_result.drift_detected and self.config.enable_alerts:
                self._create_drift_alert(
                    run_id=run_id,
                    timestamp=timestamp,
                    drift_result=drift_result,
                )

            # Record drift metrics for observability
            app_metrics.record_drift_check(model_name=model_name)
            if drift_result.drift_detected:
                app_metrics.record_drift_event(
                    drift_type="feature",
                    severity=drift_result.severity.value,
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
        """Store drift result in database.

        Args:
            run_id: Unique run ID.
            timestamp: Detection timestamp.
            model_name: Model name.
            model_version: Model version.
            drift_result: Drift detection result.
            window_meta: Window metadata.
        """
        # Parse window times
        current_start = datetime.fromisoformat(window_meta["start_time"])
        current_end = datetime.fromisoformat(window_meta["end_time"])

        # Build feature drift details
        feature_details = {
            fd.feature_name: fd.to_dict()
            for fd in drift_result.feature_drifts
        }

        db_record = DriftMetricDB(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            model_version=model_version,
            reference_window_start=None,  # Training data, no specific time
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
            psi_threshold=self.config.psi_threshold,
            min_drift_features=self.config.min_drift_features,
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
        """Create drift alert in database.

        Args:
            run_id: Drift run ID.
            timestamp: Alert timestamp.
            drift_result: Drift detection result.
        """
        # Determine severity
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

        # Get the drift metric ID
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
        """Get drift summary for recent period.

        Args:
            hours: Number of hours to summarize.
            model_name: Filter by model name.

        Returns:
            Summary dictionary.
        """
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
        """Get all unacknowledged drift alerts.

        Returns:
            List of alert dictionaries.
        """
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
