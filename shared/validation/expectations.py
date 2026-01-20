"""Great Expectations expectation definitions for Telco Churn dataset.

Defines data quality expectations for:
- Schema validation (required columns, types)
- Value constraints (ranges, categories)
- Distribution checks (null rates, uniqueness)
"""

from typing import Any

# Valid categorical values for each feature
CATEGORICAL_VALUES = {
    "gender": ["Male", "Female"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["Yes", "No", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "online_backup": ["Yes", "No", "No internet service"],
    "device_protection": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Numeric feature constraints
NUMERIC_CONSTRAINTS = {
    "senior_citizen": {"min": 0, "max": 1},
    "tenure": {"min": 0, "max": 100},  # months
    "monthly_charges": {"min": 0, "max": 500},  # dollars
    "total_charges": {"min": 0, "max": 10000},  # dollars
}

# Feature expectations registry
FEATURE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    # Numeric features
    "senior_citizen": {
        "type": "int",
        "nullable": False,
        "min_value": 0,
        "max_value": 1,
    },
    "tenure": {
        "type": "int",
        "nullable": False,
        "min_value": 0,
        "max_value": 100,
    },
    "monthly_charges": {
        "type": "float",
        "nullable": False,
        "min_value": 0,
        "max_value": 500,
    },
    "total_charges": {
        "type": "float",
        "nullable": True,  # Can be empty for new customers
        "min_value": 0,
        "max_value": 10000,
    },
    # Categorical features
    "gender": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["gender"],
    },
    "partner": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["partner"],
    },
    "dependents": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["dependents"],
    },
    "phone_service": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["phone_service"],
    },
    "multiple_lines": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["multiple_lines"],
    },
    "internet_service": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["internet_service"],
    },
    "online_security": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["online_security"],
    },
    "online_backup": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["online_backup"],
    },
    "device_protection": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["device_protection"],
    },
    "tech_support": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["tech_support"],
    },
    "streaming_tv": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["streaming_tv"],
    },
    "streaming_movies": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["streaming_movies"],
    },
    "contract": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["contract"],
    },
    "paperless_billing": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["paperless_billing"],
    },
    "payment_method": {
        "type": "str",
        "nullable": False,
        "allowed_values": CATEGORICAL_VALUES["payment_method"],
    },
    # Target
    "churn": {
        "type": "int",
        "nullable": False,
        "min_value": 0,
        "max_value": 1,
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
                "exact_match": False,  # Allow extra columns
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
            "kwargs": {"min_value": 100},  # Minimum rows for training
        }
    )

    return expectations


def get_inference_feature_names() -> list[str]:
    """Get list of feature names required for inference (excludes target).

    Returns:
        List of feature names.
    """
    return [name for name in FEATURE_EXPECTATIONS.keys() if name != "churn"]
