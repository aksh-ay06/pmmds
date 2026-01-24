#!/usr/bin/env python3
"""Run the initial model training pipeline.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --mlflow-uri http://localhost:5000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.utils import get_logger, setup_logging

setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)


def main(
    train_path: str = "data/processed/train.parquet",
    test_path: str = "data/processed/test.parquet",
    mlflow_uri: str = "http://localhost:5000",
    promote: bool = True,
) -> None:
    """Run the training pipeline.

    Args:
        train_path: Path to training data.
        test_path: Path to test data.
        mlflow_uri: MLflow tracking URI.
        promote: Whether to promote model to production.
    """
    from pipelines.train.trainer import run_training_pipeline

    logger.info(
        "starting_training",
        train_path=train_path,
        test_path=test_path,
        mlflow_uri=mlflow_uri,
    )

    # Check data exists
    if not Path(train_path).exists():
        logger.error("training_data_not_found", path=train_path)
        print(f"Training data not found: {train_path}")
        print("Run 'make download-data' first.")
        sys.exit(1)

    if not Path(test_path).exists():
        logger.error("test_data_not_found", path=test_path)
        print(f"Test data not found: {test_path}")
        print("Run 'make download-data' first.")
        sys.exit(1)

    # Run training
    pipeline_model, metrics, run_id = run_training_pipeline(
        train_path=train_path,
        test_path=test_path,
        mlflow_uri=mlflow_uri,
        promote=promote,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  MLflow Run ID: {run_id}")
    print(f"  RMSE:          {metrics.rmse:.4f}")
    print(f"  MAE:           {metrics.mae:.4f}")
    print(f"  R2:            {metrics.r2:.4f}")
    print(f"  MAPE:          {metrics.mape:.2f}%")
    print(f"  Train Samples: {metrics.train_samples}")
    print(f"  Test Samples:  {metrics.test_samples}")
    print(f"  Duration:      {metrics.training_time_seconds:.1f}s")
    print(f"  Promoted:      {promote}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NYC taxi fare prediction model")
    parser.add_argument("--train-path", type=str, default="data/processed/train.parquet")
    parser.add_argument("--test-path", type=str, default="data/processed/test.parquet")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--no-promote", action="store_true", help="Skip promotion to production")
    args = parser.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        mlflow_uri=args.mlflow_uri,
        promote=not args.no_promote,
    )
