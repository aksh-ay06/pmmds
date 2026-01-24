"""Great Expectations expectation definitions for NYC Yellow Taxi dataset.

Defines data quality expectations for:
- Schema validation (required columns, types)
- Value constraints (ranges, categories)
- Distribution checks (null rates, uniqueness)
"""

from typing import Any

from shared.data.dataset import TARGET_COLUMN

# Valid categorical values for each feature
CATEGORICAL_VALUES: dict[str, list[str | int]] = {
    "RatecodeID": [1, 2, 3, 4, 5, 6],
    "payment_type": [1, 2, 3, 4],
    "pickup_borough": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"],
    "dropoff_borough": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"],
}

# Numeric feature constraints
NUMERIC_CONSTRAINTS = {
    "trip_distance": {"min": 0.1, "max": 100.0},
    "passenger_count": {"min": 1, "max": 6},
    "pickup_hour": {"min": 0, "max": 23},
    "pickup_day_of_week": {"min": 1, "max": 7},
    "pickup_month": {"min": 1, "max": 12},
    "trip_duration_minutes": {"min": 1.0, "max": 180.0},
    "is_weekend": {"min": 0, "max": 1},
    "is_rush_hour": {"min": 0, "max": 1},
    "fare_amount": {"min": 2.5, "max": 200.0},
}

# Feature expectations registry
FEATURE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    # Numeric features
    "trip_distance": {
        "type": "float",
        "nullable": False,
        "min_value": 0.1,
        "max_value": 100.0,
    },
    "passenger_count": {
        "type": "int",
        "nullable": False,
        "min_value": 1,
        "max_value": 6,
    },
    "pickup_hour": {
        "type": "int",
        "nullable": False,
        "min_value": 0,
        "max_value": 23,
    },
    "pickup_day_of_week": {
        "type": "int",
        "nullable": False,
        "min_value": 1,
        "max_value": 7,
    },
    "pickup_month": {
        "type": "int",
        "nullable": False,
        "min_value": 1,
        "max_value": 12,
    },
    "trip_duration_minutes": {
        "type": "float",
        "nullable": False,
        "min_value": 1.0,
        "max_value": 180.0,
    },
    # Binary features
    "is_weekend": {
        "type": "int",
        "nullable": False,
        "min_value": 0,
        "max_value": 1,
    },
    "is_rush_hour": {
        "type": "int",
        "nullable": False,
        "min_value": 0,
        "max_value": 1,
    },
    # Categorical features
    "RatecodeID": {
        "type": "int",
        "nullable": False,
        "min_value": 1,
        "max_value": 6,
    },
    "payment_type": {
        "type": "int",
        "nullable": False,
        "min_value": 1,
        "max_value": 4,
    },
    "pickup_borough": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["pickup_borough"],
    },
    "dropoff_borough": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["dropoff_borough"],
    },
    # Target
    "fare_amount": {
        "type": "float",
        "nullable": False,
        "min_value": 2.5,
        "max_value": 200.0,
    },
}


def get_feature_expectation(feature_name: str) -> dict[str, Any] | None:
    """Get expectation config for a feature.

    Args:
        feature_name: Name of the feature.

    Returns:
        Expectation configuration or None if not found.
    """
    return FEATURE_EXPECTATIONS.get(feature_name)


def create_training_data_expectations() -> list[dict[str, Any]]:
    """Create list of GE expectations for training data validation.

    Returns:
        List of expectation configurations compatible with GE.
    """
    expectations = []

    # Required columns expectation
    required_columns = list(FEATURE_EXPECTATIONS.keys())
    expectations.append(
        {
            "expectation_type": "expect_table_columns_to_match_set",
            "kwargs": {
                "column_set": required_columns,
                "exact_match": False,
            },
        }
    )

    # Per-column expectations
    for col_name, config in FEATURE_EXPECTATIONS.items():
        # Null check
        if not config.get("nullable", True):
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": col_name},
                }
            )

        # Numeric range checks
        if "min_value" in config:
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": col_name,
                        "min_value": config["min_value"],
                        "max_value": config.get("max_value"),
                    },
                }
            )

        # Categorical value checks
        if "allowed_values" in config:
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": col_name,
                        "value_set": config["allowed_values"],
                    },
                }
            )

    # Table-level expectations
    expectations.append(
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {"min_value": 100},
        }
    )

    return expectations


def get_inference_feature_names() -> list[str]:
    """Get list of feature names required for inference (excludes target).

    Returns:
        List of feature names.
    """
    return [name for name in FEATURE_EXPECTATIONS.keys() if name != TARGET_COLUMN]
