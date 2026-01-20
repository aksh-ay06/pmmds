"""Data validation using Great Expectations.

Provides validators for:
- Training/test datasets (batch validation)
- Inference payloads (single-record validation)
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from shared.utils import get_logger
from shared.validation.expectations import (
    CATEGORICAL_VALUES,
    FEATURE_EXPECTATIONS,
    NUMERIC_CONSTRAINTS,
    create_training_data_expectations,
)

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    success: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
        }

    def raise_if_failed(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.success:
            error_messages = [e.get("message", str(e)) for e in self.errors]
            raise ValueError(f"Validation failed: {'; '.join(error_messages)}")


class DataValidator:
    """Validates data using Great Expectations."""

    def __init__(self, use_ge: bool = True) -> None:
        """Initialize validator.

        Args:
            use_ge: Whether to use Great Expectations (falls back to pandas if False).
        """
        self._use_ge = use_ge
        self._ge_context = None

        if use_ge:
            self._init_ge()

    def _init_ge(self) -> None:
        """Initialize Great Expectations context."""
        try:
            import great_expectations as gx

            self._ge_context = gx.get_context()
            logger.info("great_expectations_initialized")
        except Exception as e:
            logger.warning(
                "great_expectations_init_failed",
                error=str(e),
                fallback="pandas",
            )
            self._use_ge = False

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        expectation_suite_name: str = "training_data",
        include_target: bool = True,
    ) -> ValidationResult:
        """Validate a DataFrame against expectations.

        Args:
            df: DataFrame to validate.
            expectation_suite_name: Name of the expectation suite.
            include_target: Whether to validate target column.

        Returns:
            ValidationResult with success status and any errors.
        """
        if self._use_ge:
            return self._validate_with_ge(df, expectation_suite_name, include_target)
        else:
            return self._validate_with_pandas(df, include_target)

    def _validate_with_ge(
        self,
        df: pd.DataFrame,
        suite_name: str,
        include_target: bool,
    ) -> ValidationResult:
        """Validate using Great Expectations."""
        import great_expectations as gx
        from great_expectations.expectations.expectation_configuration import (
            ExpectationConfiguration,
        )

        errors = []
        warnings = []
        statistics = {"rows": len(df), "columns": len(df.columns)}

        try:
            # Create in-memory batch
            context = gx.get_context()

            # Create expectation suite
            expectations = create_training_data_expectations()

            # Filter out target expectations if not included
            if not include_target:
                expectations = [
                    e
                    for e in expectations
                    if e.get("kwargs", {}).get("column") != "churn"
                ]

            # Create a validator
            data_source = context.data_sources.add_pandas("pandas_source")
            data_asset = data_source.add_dataframe_asset("validation_asset")
            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "batch_def"
            )
            batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

            # Run expectations
            failed_expectations = []
            for exp_config in expectations:
                try:
                    exp_type = exp_config["expectation_type"]
                    kwargs = exp_config.get("kwargs", {})

                    # Skip if column doesn't exist
                    if "column" in kwargs and kwargs["column"] not in df.columns:
                        warnings.append(
                            {
                                "type": "missing_column",
                                "column": kwargs["column"],
                                "message": f"Column '{kwargs['column']}' not found",
                            }
                        )
                        continue

                    # Create and validate expectation
                    expectation = ExpectationConfiguration(
                        type=exp_type,
                        kwargs=kwargs,
                    )
                    result = batch.validate(expectation)

                    if not result.success:
                        failed_expectations.append(
                            {
                                "expectation": exp_type,
                                "column": kwargs.get("column"),
                                "message": f"Failed: {exp_type}",
                                "details": result.result,
                            }
                        )

                except Exception as e:
                    warnings.append(
                        {
                            "type": "expectation_error",
                            "expectation": exp_config.get("expectation_type"),
                            "message": str(e),
                        }
                    )

            if failed_expectations:
                errors = failed_expectations

            success = len(errors) == 0

            logger.info(
                "ge_validation_complete",
                success=success,
                errors=len(errors),
                warnings=len(warnings),
            )

            return ValidationResult(
                success=success,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )

        except Exception as e:
            logger.error("ge_validation_failed", error=str(e))
            # Fallback to pandas validation
            return self._validate_with_pandas(df, include_target)

    def _validate_with_pandas(
        self,
        df: pd.DataFrame,
        include_target: bool,
    ) -> ValidationResult:
        """Validate using pandas (fallback when GE unavailable)."""
        errors = []
        warnings = []
        statistics = {"rows": len(df), "columns": len(df.columns)}

        features_to_check = FEATURE_EXPECTATIONS.copy()
        if not include_target:
            features_to_check.pop("churn", None)

        for col_name, config in features_to_check.items():
            if col_name not in df.columns:
                errors.append(
                    {
                        "type": "missing_column",
                        "column": col_name,
                        "message": f"Required column '{col_name}' not found",
                    }
                )
                continue

            col = df[col_name]

            # Null check
            if not config.get("nullable", True):
                null_count = col.isnull().sum()
                if null_count > 0:
                    errors.append(
                        {
                            "type": "null_values",
                            "column": col_name,
                            "message": f"Column '{col_name}' has {null_count} null values",
                            "count": int(null_count),
                        }
                    )

            # Range check for numeric
            if "min_value" in config:
                min_val = config["min_value"]
                max_val = config.get("max_value")
                non_null = col.dropna()

                if len(non_null) > 0:
                    actual_min = non_null.min()
                    actual_max = non_null.max()

                    if actual_min < min_val:
                        errors.append(
                            {
                                "type": "value_below_minimum",
                                "column": col_name,
                                "message": f"Column '{col_name}' has values below {min_val}",
                                "min_found": float(actual_min),
                            }
                        )

                    if max_val is not None and actual_max > max_val:
                        errors.append(
                            {
                                "type": "value_above_maximum",
                                "column": col_name,
                                "message": f"Column '{col_name}' has values above {max_val}",
                                "max_found": float(actual_max),
                            }
                        )

            # Categorical value check
            if "allowed_values" in config:
                allowed = set(config["allowed_values"])
                actual = set(col.dropna().unique())
                invalid = actual - allowed

                if invalid:
                    errors.append(
                        {
                            "type": "invalid_category",
                            "column": col_name,
                            "message": f"Column '{col_name}' has invalid values: {invalid}",
                            "invalid_values": list(invalid),
                        }
                    )

        # Row count check
        if len(df) < 100 and include_target:
            warnings.append(
                {
                    "type": "low_row_count",
                    "message": f"Dataset has only {len(df)} rows (recommended: >100)",
                    "count": len(df),
                }
            )

        success = len(errors) == 0

        logger.info(
            "pandas_validation_complete",
            success=success,
            errors=len(errors),
            warnings=len(warnings),
        )

        return ValidationResult(
            success=success,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )


def validate_dataframe(
    df: pd.DataFrame,
    include_target: bool = True,
    use_ge: bool = True,
) -> ValidationResult:
    """Convenience function to validate a DataFrame.

    Args:
        df: DataFrame to validate.
        include_target: Whether to validate target column.
        use_ge: Whether to use Great Expectations.

    Returns:
        ValidationResult with success status and errors.
    """
    validator = DataValidator(use_ge=use_ge)
    return validator.validate_dataframe(df, include_target=include_target)


def validate_inference_payload(
    features: dict[str, Any],
    strict: bool = True,
) -> ValidationResult:
    """Validate a single inference payload.

    This is a lightweight validation for real-time inference.
    Does not require Great Expectations for speed.

    Args:
        features: Feature dictionary from API request.
        strict: If True, fail on any error. If False, allow warnings.

    Returns:
        ValidationResult with success status and errors.
    """
    errors = []
    warnings = []

    # Get expected features (exclude target)
    expected_features = {
        k: v for k, v in FEATURE_EXPECTATIONS.items() if k != "churn"
    }

    # Check for missing features
    missing = set(expected_features.keys()) - set(features.keys())
    if missing:
        errors.append(
            {
                "type": "missing_features",
                "message": f"Missing required features: {missing}",
                "features": list(missing),
            }
        )

    # Check for extra features
    extra = set(features.keys()) - set(expected_features.keys())
    if extra:
        warnings.append(
            {
                "type": "extra_features",
                "message": f"Unexpected features (will be ignored): {extra}",
                "features": list(extra),
            }
        )

    # Validate each feature
    for feature_name, config in expected_features.items():
        if feature_name not in features:
            continue

        value = features[feature_name]

        # Null check
        if value is None:
            if not config.get("nullable", True):
                errors.append(
                    {
                        "type": "null_value",
                        "feature": feature_name,
                        "message": f"Feature '{feature_name}' cannot be null",
                    }
                )
            continue

        # Type check
        expected_type = config.get("type")
        if expected_type == "int":
            if not isinstance(value, (int, float)) or (
                isinstance(value, float) and not value.is_integer()
            ):
                # Allow float that is actually integer
                if isinstance(value, float) and value.is_integer():
                    pass  # OK
                elif not isinstance(value, int):
                    errors.append(
                        {
                            "type": "type_error",
                            "feature": feature_name,
                            "message": f"Feature '{feature_name}' must be integer",
                            "received": type(value).__name__,
                        }
                    )
        elif expected_type == "float":
            if not isinstance(value, (int, float)):
                errors.append(
                    {
                        "type": "type_error",
                        "feature": feature_name,
                        "message": f"Feature '{feature_name}' must be numeric",
                        "received": type(value).__name__,
                    }
                )
        elif expected_type == "str":
            if not isinstance(value, str):
                errors.append(
                    {
                        "type": "type_error",
                        "feature": feature_name,
                        "message": f"Feature '{feature_name}' must be string",
                        "received": type(value).__name__,
                    }
                )

        # Range check
        if "min_value" in config and isinstance(value, (int, float)):
            if value < config["min_value"]:
                errors.append(
                    {
                        "type": "value_below_minimum",
                        "feature": feature_name,
                        "message": f"Feature '{feature_name}' must be >= {config['min_value']}",
                        "received": value,
                    }
                )
            if "max_value" in config and value > config["max_value"]:
                errors.append(
                    {
                        "type": "value_above_maximum",
                        "feature": feature_name,
                        "message": f"Feature '{feature_name}' must be <= {config['max_value']}",
                        "received": value,
                    }
                )

        # Categorical check
        if "allowed_values" in config and isinstance(value, str):
            if value not in config["allowed_values"]:
                errors.append(
                    {
                        "type": "invalid_category",
                        "feature": feature_name,
                        "message": f"Feature '{feature_name}' must be one of {config['allowed_values']}",
                        "received": value,
                    }
                )

    success = len(errors) == 0

    logger.debug(
        "inference_validation_complete",
        success=success,
        errors=len(errors),
        warnings=len(warnings),
    )

    return ValidationResult(
        success=success,
        errors=errors,
        warnings=warnings,
        statistics={"features_validated": len(expected_features)},
    )
