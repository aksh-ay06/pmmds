"""Dataset loading utilities."""

from pathlib import Path
from typing import Any

import pandas as pd

from shared.utils import get_logger

logger = get_logger(__name__)

# Feature definitions for Telco Churn dataset
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges"]
BINARY_FEATURES = ["senior_citizen"]
CATEGORICAL_FEATURES = [
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
]
TARGET_COLUMN = "churn"
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        path: Path to CSV file.

    Returns:
        Loaded DataFrame.
    """
    df = pd.read_csv(path)
    logger.info("dataset_loaded", path=str(path), rows=len(df), columns=len(df.columns))
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target.

    Args:
        df: Input DataFrame.
        target_column: Name of target column.

    Returns:
        Tuple of (features_df, target_series).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def get_column_types() -> dict[str, list[str]]:
    """Get column type definitions.

    Returns:
        Dictionary mapping column types to column names.
    """
    return {
        "numeric": NUMERIC_FEATURES,
        "binary": BINARY_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "target": [TARGET_COLUMN],
    }


def compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary of statistics.
    """
    stats: dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": {},
        "categorical_stats": {},
    }

    # Numeric feature stats
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            stats["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }

    # Categorical feature stats
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in df.columns:
            value_counts = df[col].value_counts().to_dict()
            stats["categorical_stats"][col] = {
                "n_unique": df[col].nunique(),
                "value_counts": value_counts,
            }

    # Target stats
    if TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts().to_dict()
        stats["target_distribution"] = target_counts
        stats["target_rate"] = float(df[TARGET_COLUMN].mean())

    return stats
