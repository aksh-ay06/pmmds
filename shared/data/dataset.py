"""Dataset loading utilities for NYC Yellow Taxi fare prediction."""

from pathlib import Path
from typing import Any

import pandas as pd

from shared.utils import get_logger

logger = get_logger(__name__)

# Feature definitions for NYC Yellow Taxi dataset
NUMERIC_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "trip_duration_minutes",
]
BINARY_FEATURES = ["is_weekend", "is_rush_hour"]
CATEGORICAL_FEATURES = [
    "RatecodeID",
    "payment_type",
    "pickup_borough",
    "dropoff_borough",
]
TARGET_COLUMN = "fare_amount"
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load dataset from Parquet or CSV file.

    Args:
        path: Path to data file.

    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
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
                "value_counts": {str(k): v for k, v in value_counts.items()},
            }

    # Target stats
    if TARGET_COLUMN in df.columns:
        stats["target_stats"] = {
            "mean": float(df[TARGET_COLUMN].mean()),
            "std": float(df[TARGET_COLUMN].std()),
            "min": float(df[TARGET_COLUMN].min()),
            "max": float(df[TARGET_COLUMN].max()),
            "median": float(df[TARGET_COLUMN].median()),
        }

    return stats
