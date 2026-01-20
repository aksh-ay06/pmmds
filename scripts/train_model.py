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

from pipelines.train import run_training_pipeline
from shared.utils import get_logger, setup_logging

setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/train.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/processed/test.csv",
        help="Path to test data",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip promoting model to production",
    )
    args = parser.parse_args()

    # Check data exists
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    if not train_path.exists():
        logger.error("training_data_not_found", path=str(train_path))
        logger.info("hint", message="Run 'python scripts/download_data.py' first")
        sys.exit(1)

    if not test_path.exists():
        logger.error("test_data_not_found", path=str(test_path))
        sys.exit(1)

    logger.info(
        "training_pipeline_starting",
        train_path=str(train_path),
        test_path=str(test_path),
        mlflow_uri=args.mlflow_uri,
    )

    try:
        pipeline, metrics, run_id = run_training_pipeline(
            train_path=str(train_path),
            test_path=str(test_path),
            mlflow_uri=args.mlflow_uri,
            promote=not args.no_promote,
        )

        logger.info(
            "training_pipeline_completed",
            run_id=run_id,
            accuracy=round(metrics.accuracy, 4),
            roc_auc=round(metrics.roc_auc, 4),
            f1_score=round(metrics.f1, 4),
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Training Pipeline Complete")
        print("=" * 60)
        print(f"MLflow Run ID: {run_id}")
        print(f"Accuracy:      {metrics.accuracy:.4f}")
        print(f"ROC AUC:       {metrics.roc_auc:.4f}")
        print(f"F1 Score:      {metrics.f1:.4f}")
        print(f"Precision:     {metrics.precision:.4f}")
        print(f"Recall:        {metrics.recall:.4f}")
        print(f"CV Accuracy:   {metrics.cv_mean:.4f} (+/- {metrics.cv_std:.4f})")
        print(f"Training Time: {metrics.training_time_seconds:.2f}s")
        print("=" * 60)

    except Exception as e:
        logger.error("training_pipeline_failed", error=str(e), error_type=type(e).__name__)
        raise


if __name__ == "__main__":
    main()
