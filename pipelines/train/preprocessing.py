"""Feature preprocessing pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.data.dataset import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from shared.utils import get_logger

logger = get_logger(__name__)


def create_preprocessor() -> ColumnTransformer:
    """Create sklearn preprocessing pipeline.

    Returns:
        ColumnTransformer for feature preprocessing.
    """
    # Numeric features: impute + scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Binary features: pass through (already 0/1)
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # Categorical features: impute + one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("binary", binary_transformer, BINARY_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "preprocessor_created",
        numeric_features=len(NUMERIC_FEATURES),
        binary_features=len(BINARY_FEATURES),
        categorical_features=len(CATEGORICAL_FEATURES),
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    """Get feature names after preprocessing.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Sample input DataFrame (for getting categories).

    Returns:
        List of feature names after transformation.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions
        feature_names = []

        # Numeric features (unchanged names)
        feature_names.extend([f"numeric__{col}" for col in NUMERIC_FEATURES])

        # Binary features
        feature_names.extend([f"binary__{col}" for col in BINARY_FEATURES])

        # Categorical features (one-hot encoded)
        cat_transformer = preprocessor.named_transformers_["categorical"]
        encoder = cat_transformer.named_steps["encoder"]
        for i, col in enumerate(CATEGORICAL_FEATURES):
            categories = encoder.categories_[i]
            for cat in categories:
                feature_names.append(f"categorical__{col}_{cat}")

        return feature_names


def preprocess_single_sample(
    features: dict[str, Any],
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Preprocess a single prediction sample.

    Args:
        features: Feature dictionary from API request.
        preprocessor: Fitted ColumnTransformer.

    Returns:
        Preprocessed feature array.
    """
    # Convert to DataFrame with correct column order
    df = pd.DataFrame([features])

    # Ensure all expected columns exist
    all_features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    for col in all_features:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns
    df = df[all_features]

    # Transform
    X_transformed = preprocessor.transform(df)

    return X_transformed
