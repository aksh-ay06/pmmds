"""Model training and evaluation using PySpark MLlib."""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.spark
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession

from pipelines.train.preprocessing import create_spark_preprocessing_pipeline
from shared.config import get_settings
from shared.data.dataset import (
    TARGET_COLUMN,
    compute_dataset_stats,
    load_dataset,
)
from shared.utils import get_logger
from shared.validation import validate_dataframe

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # GBTRegressor parameters
    max_iter: int = 100
    max_depth: int = 5
    step_size: float = 0.1
    subsample_rate: float = 0.8
    seed: int = 42

    # Training settings
    test_size: float = 0.2

    # MLflow settings
    experiment_name: str = "nyc-taxi-fare-prediction"
    model_name: str = "nyc-taxi-fare"
    registered_model_name: str = "nyc-taxi-fare"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MLflow logging."""
        return {
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "step_size": self.step_size,
            "subsample_rate": self.subsample_rate,
            "seed": self.seed,
        }


@dataclass
class TrainingMetrics:
    """Training metrics container."""

    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    mape: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    training_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for MLflow logging."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "train_samples": float(self.train_samples),
            "test_samples": float(self.test_samples),
            "training_time_seconds": self.training_time_seconds,
        }


@dataclass
class TaxiFareTrainer:
    """Trainer for NYC taxi fare prediction model using PySpark MLlib."""

    config: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow_tracking_uri: str = "http://localhost:5000"

    def __post_init__(self) -> None:
        """Initialize MLflow and Spark."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logger.info("mlflow_configured", tracking_uri=self.mlflow_tracking_uri)

    def _get_or_create_spark(self) -> SparkSession:
        """Get or create SparkSession."""
        import os
        import sys

        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        return (
            SparkSession.builder
            .master(settings.spark_master)
            .appName(settings.spark_app_name)
            .config("spark.driver.memory", settings.spark_driver_memory)
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .getOrCreate()
        )

    def _evaluate(
        self,
        model: PipelineModel,
        test_df: Any,
    ) -> TrainingMetrics:
        """Evaluate model on test set.

        Args:
            model: Trained PySpark PipelineModel.
            test_df: Spark DataFrame with test data.

        Returns:
            TrainingMetrics with evaluation results.
        """
        predictions = model.transform(test_df)

        # RMSE
        rmse_eval = RegressionEvaluator(
            labelCol=TARGET_COLUMN, predictionCol="prediction", metricName="rmse"
        )
        rmse = rmse_eval.evaluate(predictions)

        # MAE
        mae_eval = RegressionEvaluator(
            labelCol=TARGET_COLUMN, predictionCol="prediction", metricName="mae"
        )
        mae = mae_eval.evaluate(predictions)

        # R2
        r2_eval = RegressionEvaluator(
            labelCol=TARGET_COLUMN, predictionCol="prediction", metricName="r2"
        )
        r2 = r2_eval.evaluate(predictions)

        # MAPE (computed manually from predictions)
        pred_pd = predictions.select(TARGET_COLUMN, "prediction").toPandas()
        actual = pred_pd[TARGET_COLUMN].values
        predicted = pred_pd["prediction"].values
        mask = actual > 0
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

        metrics = TrainingMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            test_samples=test_df.count(),
        )

        logger.info(
            "model_evaluated",
            rmse=round(metrics.rmse, 4),
            mae=round(metrics.mae, 4),
            r2=round(metrics.r2, 4),
            mape=round(metrics.mape, 2),
        )

        return metrics

    def train(
        self,
        train_path: str | Path,
        test_path: str | Path,
        validate_data: bool = True,
    ) -> tuple[PipelineModel, TrainingMetrics, str]:
        """Train model and log to MLflow.

        Args:
            train_path: Path to training data (Parquet).
            test_path: Path to test data (Parquet).
            validate_data: Whether to validate data before training.

        Returns:
            Tuple of (trained_pipeline_model, metrics, run_id).
        """
        import time

        start_time = time.time()

        spark = self._get_or_create_spark()

        # Load data
        train_pd = load_dataset(train_path)
        test_pd = load_dataset(test_path)

        # Validate data if enabled
        if validate_data:
            logger.info("validating_training_data")
            train_validation = validate_dataframe(train_pd, include_target=True)
            if not train_validation.success:
                logger.error(
                    "training_data_validation_failed",
                    errors=train_validation.errors,
                )
                raise ValueError(
                    f"Training data validation failed: {train_validation.errors}"
                )
            logger.info("training_data_validated", warnings=len(train_validation.warnings))

            test_validation = validate_dataframe(test_pd, include_target=True)
            if not test_validation.success:
                logger.error(
                    "test_data_validation_failed",
                    errors=test_validation.errors,
                )
                raise ValueError(
                    f"Test data validation failed: {test_validation.errors}"
                )
            logger.info("test_data_validated", warnings=len(test_validation.warnings))

        # Convert to Spark DataFrames
        # Cast categorical columns to string for StringIndexer
        train_pd["RatecodeID"] = train_pd["RatecodeID"].astype(str)
        train_pd["payment_type"] = train_pd["payment_type"].astype(str)
        test_pd["RatecodeID"] = test_pd["RatecodeID"].astype(str)
        test_pd["payment_type"] = test_pd["payment_type"].astype(str)

        train_spark = spark.createDataFrame(train_pd)
        test_spark = spark.createDataFrame(test_pd)

        # Compute dataset stats for logging
        train_stats = compute_dataset_stats(train_pd)

        # Set up MLflow experiment
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("mlflow_run_started", run_id=run_id)

            # Log parameters
            mlflow.log_params(self.config.to_dict())
            mlflow.log_param("train_path", str(train_path))
            mlflow.log_param("test_path", str(test_path))

            # Create preprocessing + model pipeline
            preprocessing_pipeline = create_spark_preprocessing_pipeline()

            gbt = GBTRegressor(
                featuresCol="features",
                labelCol=TARGET_COLUMN,
                maxIter=self.config.max_iter,
                maxDepth=self.config.max_depth,
                stepSize=self.config.step_size,
                subsamplingRate=self.config.subsample_rate,
                seed=self.config.seed,
            )

            full_pipeline = Pipeline(
                stages=preprocessing_pipeline.getStages() + [gbt]
            )

            # Train
            logger.info("training_started", train_samples=train_spark.count())
            pipeline_model = full_pipeline.fit(train_spark)

            # Evaluate
            metrics = self._evaluate(pipeline_model, test_spark)
            metrics.train_samples = train_spark.count()
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

            # Log Spark model
            mlflow.spark.log_model(
                spark_model=pipeline_model,
                artifact_path="model",
                registered_model_name=self.config.registered_model_name,
            )

            logger.info(
                "training_completed",
                run_id=run_id,
                rmse=round(metrics.rmse, 4),
                r2=round(metrics.r2, 4),
                training_time=round(metrics.training_time_seconds, 2),
            )

        return pipeline_model, metrics, run_id

    def promote_to_production(self, model_name: str, version: int | None = None) -> None:
        """Promote model version to production stage.

        Args:
            model_name: Registered model name.
            version: Specific version to promote (latest if None).
        """
        client = mlflow.MlflowClient()

        if version is None:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            version = max(int(v.version) for v in versions)

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
    train_path: str = "data/processed/train.parquet",
    test_path: str = "data/processed/test.parquet",
    mlflow_uri: str = "http://localhost:5000",
    promote: bool = True,
) -> tuple[PipelineModel, TrainingMetrics, str]:
    """Run the complete training pipeline.

    Args:
        train_path: Path to training data.
        test_path: Path to test data.
        mlflow_uri: MLflow tracking server URI.
        promote: Whether to promote model to production.

    Returns:
        Tuple of (trained_pipeline_model, metrics, run_id).
    """
    config = TrainingConfig()
    trainer = TaxiFareTrainer(config=config, mlflow_tracking_uri=mlflow_uri)

    # Train
    pipeline_model, metrics, run_id = trainer.train(train_path, test_path)

    # Promote to production
    if promote:
        trainer.promote_to_production(config.registered_model_name)

    return pipeline_model, metrics, run_id
