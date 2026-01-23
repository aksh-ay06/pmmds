"""Automated retraining service.

Coordinates the full retraining workflow:
1. Check if retraining is needed (drift threshold exceeded)
2. Train a challenger model
3. Compare against current champion
4. Promote if improvement criteria met
5. Record all decisions
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from apps.monitor.config import DriftConfig, get_drift_config
from apps.monitor.db import DriftMetricDB, MonitorBase
from pipelines.retrain.comparator import ComparisonResult, ModelComparator
from pipelines.retrain.db import PromotionBase, PromotionDecisionDB, RetrainingRunDB
from pipelines.train.trainer import ChurnModelTrainer, TrainingConfig, TrainingMetrics
from shared.config import get_settings
from shared.data.dataset import TARGET_COLUMN, get_feature_target_split, load_dataset
from shared.utils import get_logger, get_metrics
from shared.validation import validate_dataframe

logger = get_logger(__name__)
settings = get_settings()
app_metrics = get_metrics()


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""

    # Paths
    train_data_path: str = "data/processed/train.csv"
    test_data_path: str = "data/processed/test.csv"

    # Thresholds (from CLAUDE.md)
    min_drift_features: int = 3
    psi_threshold: float = 0.2

    # Promotion criteria
    primary_metric: str = "roc_auc"
    min_improvement: float = 0.001  # 0.1%
    max_latency_regression: float = 1.2  # 20% slower OK

    # MLflow settings
    model_name: str = "churn-classifier"
    production_alias: str = "production"


class RetrainingService:
    """Service for automated retraining and model promotion."""

    def __init__(
        self,
        config: RetrainingConfig | None = None,
        drift_config: DriftConfig | None = None,
        db_url: str | None = None,
    ):
        """Initialize retraining service.

        Args:
            config: Retraining configuration.
            drift_config: Drift monitoring configuration.
            db_url: Database URL.
        """
        self.config = config or RetrainingConfig()
        self.drift_config = drift_config or get_drift_config()

        # Database setup
        self.db_url = db_url or settings.database_url_sync
        self.engine = create_engine(self.db_url, echo=False)

        # Ensure tables exist
        self._init_db()

        # Initialize comparator
        self.comparator = ModelComparator(
            primary_metric=self.config.primary_metric,
            min_improvement=self.config.min_improvement,
            max_latency_regression=self.config.max_latency_regression,
        )

        logger.info(
            "retraining_service_initialized",
            model_name=self.config.model_name,
            primary_metric=self.config.primary_metric,
        )

    def _init_db(self) -> None:
        """Initialize database tables."""
        PromotionBase.metadata.create_all(self.engine)
        MonitorBase.metadata.create_all(self.engine)
        logger.info("retraining_tables_initialized")

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.sha256(f"decision-{timestamp}".encode()).hexdigest()[:16]

    def check_drift_trigger(self) -> tuple[bool, dict[str, Any] | None]:
        """Check if drift threshold has been exceeded.

        Returns:
            Tuple of (should_retrain, drift_info).
        """
        with Session(self.engine) as session:
            # Get most recent drift detection
            query = select(DriftMetricDB).order_by(
                DriftMetricDB.timestamp.desc()
            ).limit(1)
            result = session.execute(query).scalar_one_or_none()

            if result is None:
                logger.info("no_drift_data_available")
                return False, None

            drift_info = {
                "run_id": result.run_id,
                "timestamp": result.timestamp.isoformat(),
                "drift_detected": result.drift_detected,
                "features_with_drift": result.features_with_drift,
                "drift_count": result.drift_count,
                "max_psi": result.max_psi,
            }

            # Check threshold: ≥3 features with PSI > 0.2
            should_retrain = (
                result.drift_detected
                and result.drift_count >= self.config.min_drift_features
            )

            if should_retrain:
                logger.warning(
                    "drift_threshold_exceeded",
                    drift_count=result.drift_count,
                    features=result.features_with_drift,
                    max_psi=result.max_psi,
                )
            else:
                logger.info(
                    "drift_below_threshold",
                    drift_detected=result.drift_detected,
                    drift_count=result.drift_count,
                )

            return should_retrain, drift_info

    def train_challenger(
        self,
        run_id: str,
        trigger_type: str = "drift",
        drift_info: dict[str, Any] | None = None,
    ) -> tuple[TrainingMetrics, str, str]:
        """Train a new challenger model.

        Args:
            run_id: Retraining run ID.
            trigger_type: What triggered retraining (drift, scheduled, manual).
            drift_info: Drift detection info if triggered by drift.

        Returns:
            Tuple of (metrics, mlflow_run_id, model_version).
        """
        logger.info(
            "training_challenger",
            run_id=run_id,
            trigger_type=trigger_type,
        )

        # Initialize trainer
        training_config = TrainingConfig(
            experiment_name=f"{self.config.model_name}-retrain",
            model_name=self.config.model_name,
            registered_model_name=self.config.model_name,
        )
        trainer = ChurnModelTrainer(
            config=training_config,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
        )

        # Train model
        _, metrics, mlflow_run_id = trainer.train(
            train_path=self.config.train_data_path,
            test_path=self.config.test_data_path,
            validate_data=True,
        )

        # Get the model version that was just registered
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{self.config.model_name}'"
        )
        model_version = max(v.version for v in versions)

        logger.info(
            "challenger_trained",
            run_id=run_id,
            mlflow_run_id=mlflow_run_id,
            model_version=model_version,
            roc_auc=metrics.roc_auc,
        )

        return metrics, mlflow_run_id, str(model_version)

    def compare_and_promote(
        self,
        challenger_version: str,
        challenger_metrics: TrainingMetrics,
    ) -> ComparisonResult:
        """Compare challenger against champion and promote if better.

        Args:
            challenger_version: Challenger model version.
            challenger_metrics: Training metrics from challenger.

        Returns:
            ComparisonResult with decision.
        """
        # Load test data for evaluation
        test_df = load_dataset(self.config.test_data_path)
        X_test, y_test = get_feature_target_split(test_df, TARGET_COLUMN)

        # Load champion model
        try:
            champion_model, champion_version = self.comparator.load_model_by_alias(
                model_name=self.config.model_name,
                alias=self.config.production_alias,
            )
            logger.info(
                "champion_loaded",
                version=champion_version,
            )
        except Exception as e:
            # No champion yet - auto-promote challenger
            logger.warning(
                "no_champion_found",
                error=str(e),
            )
            self.comparator.promote_challenger(
                model_name=self.config.model_name,
                challenger_version=challenger_version,
                alias=self.config.production_alias,
            )
            # Return a synthetic result
            from pipelines.retrain.comparator import ModelMetrics

            return ComparisonResult(
                champion_metrics=ModelMetrics(),
                challenger_metrics=ModelMetrics(
                    roc_auc=challenger_metrics.roc_auc,
                    accuracy=challenger_metrics.accuracy,
                    f1=challenger_metrics.f1,
                    model_version=challenger_version,
                ),
                validation_passed=True,
                metric_improvement=True,
                latency_acceptable=True,
                should_promote=True,
                primary_metric_name=self.config.primary_metric,
                primary_metric_improvement=challenger_metrics.roc_auc,
                promotion_reason="First model - auto-promoted as initial champion",
            )

        # Load challenger model
        challenger_model = self.comparator.load_model_by_version(
            model_name=self.config.model_name,
            version=challenger_version,
        )

        # Compare models
        comparison = self.comparator.compare(
            champion_model=champion_model,
            champion_version=champion_version,
            challenger_model=challenger_model,
            challenger_version=challenger_version,
            X_test=X_test,
            y_test=y_test,
            challenger_validation_passed=True,  # Already validated during training
        )

        # Promote if appropriate
        if comparison.should_promote:
            self.comparator.promote_challenger(
                model_name=self.config.model_name,
                challenger_version=challenger_version,
                alias=self.config.production_alias,
            )

        return comparison

    def record_retraining_run(
        self,
        run_id: str,
        trigger_type: str,
        drift_info: dict[str, Any] | None,
        mlflow_run_id: str | None,
        model_version: str | None,
        metrics: TrainingMetrics | None,
        status: str,
        error_message: str | None = None,
        comparison: ComparisonResult | None = None,
        decision_id: str | None = None,
        training_duration: float | None = None,
    ) -> None:
        """Record retraining run in database.

        Args:
            run_id: Retraining run ID.
            trigger_type: Trigger type (drift, scheduled, manual).
            drift_info: Drift detection info.
            mlflow_run_id: MLflow run ID.
            model_version: Model version.
            metrics: Training metrics.
            status: Run status.
            error_message: Error message if failed.
            comparison: Comparison result.
            decision_id: Promotion decision ID.
            training_duration: Training duration in seconds.
        """
        record = RetrainingRunDB(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            trigger_type=trigger_type,
            drift_run_id=drift_info.get("run_id") if drift_info else None,
            drift_features=drift_info.get("features_with_drift", []) if drift_info else [],
            drift_max_psi=drift_info.get("max_psi") if drift_info else None,
            mlflow_run_id=mlflow_run_id,
            model_version=model_version,
            training_config={},
            status=status,
            metrics=metrics.to_dict() if metrics else {},
            error_message=error_message,
            promotion_decision_id=decision_id,
            promoted=comparison.should_promote if comparison else False,
            training_duration_seconds=training_duration,
        )

        with Session(self.engine) as session:
            session.add(record)
            session.commit()

        # Record metrics for observability
        outcome = "success" if status == "completed" else status
        app_metrics.record_retraining(trigger_type=trigger_type, outcome=outcome)

        if comparison and comparison.should_promote:
            # Record promotion
            from_version = str(int(comparison.challenger_metrics.model_version) - 1) if int(comparison.challenger_metrics.model_version) > 1 else "0"
            app_metrics.record_promotion(
                from_version=from_version,
                to_version=comparison.challenger_metrics.model_version,
            )

        logger.info(
            "retraining_run_recorded",
            run_id=run_id,
            status=status,
            promoted=comparison.should_promote if comparison else False,
        )

    def record_promotion_decision(
        self,
        decision_id: str,
        trigger_type: str,
        drift_run_id: str | None,
        champion_version: str,
        champion_metrics: dict[str, Any],
        challenger_version: str,
        challenger_mlflow_run_id: str,
        challenger_metrics: dict[str, Any],
        comparison: ComparisonResult,
        comparison_duration: float,
    ) -> None:
        """Record promotion decision in database.

        Args:
            decision_id: Unique decision ID.
            trigger_type: Trigger type.
            drift_run_id: Drift run ID if drift-triggered.
            champion_version: Champion model version.
            champion_metrics: Champion metrics dict.
            challenger_version: Challenger model version.
            challenger_mlflow_run_id: Challenger MLflow run ID.
            challenger_metrics: Challenger metrics dict.
            comparison: Comparison result.
            comparison_duration: Comparison duration in seconds.
        """
        record = PromotionDecisionDB(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            trigger_type=trigger_type,
            drift_run_id=drift_run_id,
            champion_model_name=self.config.model_name,
            champion_model_version=champion_version,
            champion_mlflow_run_id=None,
            champion_metrics=champion_metrics,
            challenger_model_name=self.config.model_name,
            challenger_model_version=challenger_version,
            challenger_mlflow_run_id=challenger_mlflow_run_id,
            challenger_metrics=challenger_metrics,
            validation_passed=comparison.validation_passed,
            metric_improvement=comparison.metric_improvement,
            latency_acceptable=comparison.latency_acceptable,
            primary_metric_name=comparison.primary_metric_name,
            primary_metric_improvement=comparison.primary_metric_improvement,
            promoted=comparison.should_promote,
            promotion_reason=comparison.promotion_reason,
            rejection_reasons=comparison.rejection_reasons,
            comparison_duration_seconds=comparison_duration,
        )

        with Session(self.engine) as session:
            session.add(record)
            session.commit()

        logger.info(
            "promotion_decision_recorded",
            decision_id=decision_id,
            promoted=comparison.should_promote,
            reason=comparison.promotion_reason[:100],
        )

    def run_retraining(
        self,
        trigger_type: str = "drift",
        force: bool = False,
    ) -> dict[str, Any]:
        """Run the full retraining workflow.

        Args:
            trigger_type: Trigger type (drift, scheduled, manual).
            force: Force retraining even if drift threshold not met.

        Returns:
            Dictionary with retraining results.
        """
        run_id = self._generate_run_id()
        start_time = time.time()

        logger.info(
            "retraining_started",
            run_id=run_id,
            trigger_type=trigger_type,
            force=force,
        )

        result: dict[str, Any] = {
            "run_id": run_id,
            "trigger_type": trigger_type,
            "status": "pending",
        }

        try:
            # Check drift trigger (unless forced)
            drift_info = None
            if trigger_type == "drift" and not force:
                should_retrain, drift_info = self.check_drift_trigger()
                if not should_retrain:
                    result["status"] = "skipped"
                    result["reason"] = "Drift threshold not exceeded"
                    self.record_retraining_run(
                        run_id=run_id,
                        trigger_type=trigger_type,
                        drift_info=drift_info,
                        mlflow_run_id=None,
                        model_version=None,
                        metrics=None,
                        status="skipped",
                    )
                    return result

            result["drift_info"] = drift_info

            # Train challenger
            result["status"] = "training"
            metrics, mlflow_run_id, challenger_version = self.train_challenger(
                run_id=run_id,
                trigger_type=trigger_type,
                drift_info=drift_info,
            )
            result["challenger_version"] = challenger_version
            result["challenger_metrics"] = metrics.to_dict()
            result["mlflow_run_id"] = mlflow_run_id

            # Compare and potentially promote
            result["status"] = "comparing"
            comparison_start = time.time()
            comparison = self.compare_and_promote(
                challenger_version=challenger_version,
                challenger_metrics=metrics,
            )
            comparison_duration = time.time() - comparison_start

            result["comparison"] = comparison.to_dict()
            result["promoted"] = comparison.should_promote

            # Record promotion decision
            decision_id = self._generate_decision_id()
            result["decision_id"] = decision_id

            # Get champion version for recording
            try:
                client = mlflow.MlflowClient()
                champion_info = client.get_model_version_by_alias(
                    name=self.config.model_name,
                    alias=self.config.production_alias,
                )
                # If we promoted, this is now the challenger; otherwise it's still champion
                if comparison.should_promote:
                    champion_version = str(int(challenger_version) - 1) if int(challenger_version) > 1 else "1"
                else:
                    champion_version = champion_info.version
            except Exception:
                champion_version = "0"

            self.record_promotion_decision(
                decision_id=decision_id,
                trigger_type=trigger_type,
                drift_run_id=drift_info.get("run_id") if drift_info else None,
                champion_version=champion_version,
                champion_metrics=comparison.champion_metrics.to_dict(),
                challenger_version=challenger_version,
                challenger_mlflow_run_id=mlflow_run_id,
                challenger_metrics=comparison.challenger_metrics.to_dict(),
                comparison=comparison,
                comparison_duration=comparison_duration,
            )

            # Record retraining run
            training_duration = time.time() - start_time
            self.record_retraining_run(
                run_id=run_id,
                trigger_type=trigger_type,
                drift_info=drift_info,
                mlflow_run_id=mlflow_run_id,
                model_version=challenger_version,
                metrics=metrics,
                status="completed",
                comparison=comparison,
                decision_id=decision_id,
                training_duration=training_duration,
            )

            result["status"] = "completed"
            result["training_duration_seconds"] = training_duration

            logger.info(
                "retraining_completed",
                run_id=run_id,
                promoted=comparison.should_promote,
                duration=training_duration,
            )

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

            self.record_retraining_run(
                run_id=run_id,
                trigger_type=trigger_type,
                drift_info=drift_info if "drift_info" in result else None,
                mlflow_run_id=result.get("mlflow_run_id"),
                model_version=result.get("challenger_version"),
                metrics=None,
                status="failed",
                error_message=str(e),
            )

            logger.error(
                "retraining_failed",
                run_id=run_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

        return result

    def get_recent_decisions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent promotion decisions.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            List of decision dictionaries.
        """
        with Session(self.engine) as session:
            query = select(PromotionDecisionDB).order_by(
                PromotionDecisionDB.timestamp.desc()
            ).limit(limit)
            results = session.execute(query).scalars().all()

        return [
            {
                "decision_id": r.decision_id,
                "timestamp": r.timestamp.isoformat(),
                "trigger_type": r.trigger_type,
                "challenger_version": r.challenger_model_version,
                "promoted": r.promoted,
                "primary_metric_improvement": r.primary_metric_improvement,
                "promotion_reason": r.promotion_reason,
            }
            for r in results
        ]
