"""Model training and evaluation."""

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from pipelines.train.preprocessing import create_preprocessor
from shared.config import get_settings
from shared.data.dataset import (
    TARGET_COLUMN,
    compute_dataset_stats,
    get_feature_target_split,
    load_dataset,
)
from shared.utils import get_logger
from shared.validation import validate_dataframe

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model parameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    subsample: float = 0.8
    random_state: int = 42

    # Training settings
    cv_folds: int = 5
    test_size: float = 0.2

    # MLflow settings
    experiment_name: str = "churn-prediction"
    model_name: str = "churn-classifier"
    registered_model_name: str = "churn-classifier"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MLflow logging."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample,
            "random_state": self.random_state,
            "cv_folds": self.cv_folds,
        }


@dataclass
class TrainingMetrics:
    """Training metrics container."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    training_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for MLflow logging."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1,
            "roc_auc": self.roc_auc,
            "cv_mean_accuracy": self.cv_mean,
            "cv_std_accuracy": self.cv_std,
            "train_samples": float(self.train_samples),
            "test_samples": float(self.test_samples),
            "training_time_seconds": self.training_time_seconds,
        }


@dataclass
class ChurnModelTrainer:
    """Trainer for churn prediction model."""

    config: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow_tracking_uri: str = "http://localhost:5000"

    def __post_init__(self) -> None:
        """Initialize MLflow."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logger.info("mlflow_configured", tracking_uri=self.mlflow_tracking_uri)

    def _create_model(self) -> GradientBoostingClassifier:
        """Create the model with configured parameters."""
        return GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
            verbose=0,
        )

    def _evaluate(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingMetrics:
        """Evaluate model on test set.

        Args:
            model: Trained sklearn pipeline.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            TrainingMetrics with evaluation results.
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = TrainingMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_prob),
            test_samples=len(y_test),
        )

        logger.info(
            "model_evaluated",
            accuracy=round(metrics.accuracy, 4),
            roc_auc=round(metrics.roc_auc, 4),
            f1=round(metrics.f1, 4),
        )

        return metrics

    def train(
        self,
        train_path: str | Path,
        test_path: str | Path,
        validate_data: bool = True,
    ) -> tuple[Pipeline, TrainingMetrics, str]:
        """Train model and log to MLflow.

        Args:
            train_path: Path to training data CSV.
            test_path: Path to test data CSV.
            validate_data: Whether to validate data before training.

        Returns:
            Tuple of (trained_pipeline, metrics, run_id).
        """
        import time

        start_time = time.time()

        # Load data
        train_df = load_dataset(train_path)
        test_df = load_dataset(test_path)

        # Validate data if enabled
        if validate_data:
            logger.info("validating_training_data")
            train_validation = validate_dataframe(train_df, include_target=True)
            if not train_validation.success:
                logger.error(
                    "training_data_validation_failed",
                    errors=train_validation.errors,
                )
                raise ValueError(
                    f"Training data validation failed: {train_validation.errors}"
                )
            logger.info(
                "training_data_validated",
                warnings=len(train_validation.warnings),
            )

            test_validation = validate_dataframe(test_df, include_target=True)
            if not test_validation.success:
                logger.error(
                    "test_data_validation_failed",
                    errors=test_validation.errors,
                )
                raise ValueError(
                    f"Test data validation failed: {test_validation.errors}"
                )
            logger.info(
                "test_data_validated",
                warnings=len(test_validation.warnings),
            )

        X_train, y_train = get_feature_target_split(train_df, TARGET_COLUMN)
        X_test, y_test = get_feature_target_split(test_df, TARGET_COLUMN)

        # Compute dataset stats for logging
        train_stats = compute_dataset_stats(train_df)

        # Set up MLflow experiment
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("mlflow_run_started", run_id=run_id)

            # Log parameters
            mlflow.log_params(self.config.to_dict())
            mlflow.log_param("train_path", str(train_path))
            mlflow.log_param("test_path", str(test_path))

            # Create pipeline
            preprocessor = create_preprocessor()
            classifier = self._create_model()

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", classifier),
                ]
            )

            # Train
            logger.info("training_started", train_samples=len(X_train))
            pipeline.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=self.config.cv_folds,
                scoring="accuracy",
            )

            # Evaluate
            metrics = self._evaluate(pipeline, X_test, y_test)
            metrics.cv_mean = float(cv_scores.mean())
            metrics.cv_std = float(cv_scores.std())
            metrics.train_samples = len(X_train)
            metrics.training_time_seconds = time.time() - start_time

            # Log metrics
            mlflow.log_metrics(metrics.to_dict())

            # Log dataset stats as artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_dataset_stats.json", delete=False
            ) as f:
                json.dump(train_stats, f, indent=2, default=str)
                stats_path = f.name
            mlflow.log_artifact(stats_path, "data")
            Path(stats_path).unlink(missing_ok=True)

            # Log validation results if validation was performed
            if validate_data:
                validation_results = {
                    "train_validation": train_validation.to_dict(),
                    "test_validation": test_validation.to_dict(),
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_validation_results.json", delete=False
                ) as f:
                    json.dump(validation_results, f, indent=2, default=str)
                    validation_path = f.name
                mlflow.log_artifact(validation_path, "validation")
                Path(validation_path).unlink(missing_ok=True)
                mlflow.log_param("data_validated", True)

            # Log model with signature
            from mlflow.models import infer_signature

            signature = infer_signature(X_train, pipeline.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                registered_model_name=self.config.registered_model_name,
            )

            logger.info(
                "training_completed",
                run_id=run_id,
                accuracy=round(metrics.accuracy, 4),
                roc_auc=round(metrics.roc_auc, 4),
                training_time=round(metrics.training_time_seconds, 2),
            )

        return pipeline, metrics, run_id

    def promote_to_production(self, model_name: str, version: int | None = None) -> None:
        """Promote model version to production stage.

        Args:
            model_name: Registered model name.
            version: Specific version to promote (latest if None).
        """
        client = mlflow.MlflowClient()

        if version is None:
            # Get latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            version = max(int(v.version) for v in versions)

        # Transition to production using aliases (MLflow 2.x)
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=str(version),
        )

        logger.info(
            "model_promoted",
            model_name=model_name,
            version=version,
            alias="production",
        )


def run_training_pipeline(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    mlflow_uri: str = "http://localhost:5000",
    promote: bool = True,
) -> tuple[Pipeline, TrainingMetrics, str]:
    """Run the complete training pipeline.

    Args:
        train_path: Path to training data.
        test_path: Path to test data.
        mlflow_uri: MLflow tracking server URI.
        promote: Whether to promote model to production.

    Returns:
        Tuple of (trained_pipeline, metrics, run_id).
    """
    config = TrainingConfig()
    trainer = ChurnModelTrainer(config=config, mlflow_tracking_uri=mlflow_uri)

    # Train
    pipeline, metrics, run_id = trainer.train(train_path, test_path)

    # Promote to production
    if promote:
        trainer.promote_to_production(config.registered_model_name)

    return pipeline, metrics, run_id
