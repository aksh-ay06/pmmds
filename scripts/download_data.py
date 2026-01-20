#!/usr/bin/env python3
"""Download and prepare the Telco Customer Churn dataset.

Dataset: IBM Telco Customer Churn
Source: Kaggle / IBM Sample Data
Features: 21 columns including customer demographics, services, and churn label

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output-dir data/raw
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from shared.utils import get_logger, setup_logging

setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)

# Telco Churn dataset URL (public mirror)
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
EXPECTED_SHA256 = None  # Set after first download for reproducibility
EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = 21


def download_dataset(url: str, output_path: Path) -> pd.DataFrame:
    """Download dataset from URL.

    Args:
        url: Dataset URL.
        output_path: Path to save raw CSV.

    Returns:
        Downloaded DataFrame.
    """
    logger.info("downloading_dataset", url=url)

    df = pd.read_csv(url)

    # Save raw data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "dataset_downloaded",
        rows=len(df),
        columns=len(df.columns),
        path=str(output_path),
    )

    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset structure and quality.

    Args:
        df: Dataset DataFrame.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    # Check dimensions
    if len(df) < EXPECTED_ROWS * 0.9:  # Allow 10% tolerance
        raise ValueError(f"Dataset too small: {len(df)} rows, expected ~{EXPECTED_ROWS}")

    if len(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Column count mismatch: {len(df.columns)}, expected {EXPECTED_COLUMNS}"
        )

    # Check required columns
    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check target distribution
    churn_rate = df["Churn"].value_counts(normalize=True).get("Yes", 0)
    if not 0.2 < churn_rate < 0.35:
        logger.warning("unexpected_churn_rate", churn_rate=churn_rate)

    logger.info("dataset_validated", churn_rate=round(churn_rate, 3))
    return True


def prepare_dataset(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Clean and prepare dataset for training.

    Args:
        df: Raw dataset DataFrame.
        output_dir: Directory for processed files.

    Returns:
        Tuple of (train_path, test_path).
    """
    logger.info("preparing_dataset", rows=len(df))

    # Make a copy
    df = df.copy()

    # Drop customerID (not a feature)
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (has some spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0 (new customers with tenure=0)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert target to binary
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Rename columns to snake_case (match API schema)
    column_mapping = {
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
    }
    df = df.rename(columns=column_mapping)

    # Ensure lowercase for remaining columns
    df.columns = df.columns.str.lower()

    # Shuffle and split (80/20)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Save processed files
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        "dataset_prepared",
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_path=str(train_path),
        test_path=str(test_path),
    )

    return train_path, test_path


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main(output_dir: str = "data") -> None:
    """Main download and preparation workflow.

    Args:
        output_dir: Base output directory.
    """
    base_dir = Path(output_dir)
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    # Download
    raw_path = raw_dir / "telco_churn.csv"
    if raw_path.exists():
        logger.info("using_cached_dataset", path=str(raw_path))
        df = pd.read_csv(raw_path)
    else:
        df = download_dataset(DATASET_URL, raw_path)

    # Validate
    validate_dataset(df)

    # Log file hash for reproducibility
    file_hash = compute_file_hash(raw_path)
    logger.info("dataset_hash", sha256=file_hash[:16] + "...")

    # Prepare
    train_path, test_path = prepare_dataset(df, processed_dir)

    logger.info(
        "data_preparation_complete",
        raw_path=str(raw_path),
        train_path=str(train_path),
        test_path=str(test_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Telco Churn dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for data files",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir)
