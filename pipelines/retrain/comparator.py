"""Model comparison and promotion logic for regression models.

Implements champion vs challenger comparison following CLAUDE.md rules:
- Validation must pass
- RMSE must decrease (lower is better)
- No latency regression
"""

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ModelMetrics:
    """Container for regression model evaluation metrics."""

    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    mape: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    validation_passed: bool = True
    validation_errors: list[str] = field(default_factory=list)
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "model_version": self.model_version,
        }


@dataclass
class ComparisonResult:
    """Result of champion vs challenger comparison."""

    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    validation_passed: bool
    metric_improvement: bool
    latency_acceptable: bool
    should_promote: bool
    primary_metric_name: str
    primary_metric_improvement: float
    promotion_reason: str
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "validation_passed": self.validation_passed,
            "metric_improvement": self.metric_improvement,
            "latency_acceptable": self.latency_acceptable,
            "should_promote": self.should_promote,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_improvement": self.primary_metric_improvement,
            "promotion_reason": self.promotion_reason,
            "rejection_reasons": self.rejection_reasons,
        }


class ModelComparator:
    """Compare champion and challenger regression models for promotion decisions.

    For regression, lower RMSE is better, so improvement is measured
    as champion_rmse - challenger_rmse (positive = challenger is better).
    """

    def __init__(
        self,
        primary_metric: str = "rmse",
        min_improvement: float = 0.5,  # RMSE reduction of at least 0.5
        max_latency_regression: float = 1.2,  # Allow up to 20% latency increase
        latency_samples: int = 100,
        tracking_uri: str | None = None,
    ):
        """Initialize model comparator.

        Args:
            primary_metric: Primary metric for comparison (default: rmse).
            min_improvement: Minimum RMSE reduction required.
            max_latency_regression: Max acceptable latency ratio.
            latency_samples: Number of samples for latency testing.
            tracking_uri: MLflow tracking URI.
        """
        self.primary_metric = primary_metric
        self.min_improvement = min_improvement
        self.max_latency_regression = max_latency_regression
        self.latency_samples = latency_samples
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = mlflow.MlflowClient()

    def load_model_by_version(
        self,
        model_name: str,
        version: str,
    ) -> Any:
        """Load a specific model version from MLflow as pyfunc.

        Args:
            model_name: Registered model name.
            version: Model version.

        Returns:
            Loaded pyfunc model.
        """
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)

    def load_model_by_alias(
        self,
        model_name: str,
        alias: str = "production",
    ) -> tuple[Any, str]:
        """Load model by alias from MLflow.

        Args:
            model_name: Registered model name.
            alias: Model alias (e.g., 'production').

        Returns:
            Tuple of (model, version_string).
        """
        model_uri = f"models:/{model_name}@{alias}"
        model_version = self._client.get_model_version_by_alias(
            name=model_name, alias=alias
        )
        model = mlflow.pyfunc.load_model(model_uri)
        return model, model_version.version

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelMetrics:
        """Evaluate a regression model on test data.

        Args:
            model: MLflow pyfunc model.
            X_test: Test features.
            y_test: Test labels (fare_amount).

        Returns:
            ModelMetrics with evaluation results.
        """
        # Cast categoricals to string as the Spark model expects
        X_eval = X_test.copy()
        if "RatecodeID" in X_eval.columns:
            X_eval["RatecodeID"] = X_eval["RatecodeID"].astype(str)
        if "payment_type" in X_eval.columns:
            X_eval["payment_type"] = X_eval["payment_type"].astype(str)

        # Get predictions
        y_pred = model.predict(X_eval)

        actual = y_test.values.astype(float)
        predicted = np.array(y_pred, dtype=float)

        # RMSE
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        # MAE
        mae = float(np.mean(np.abs(actual - predicted)))

        # R2
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-10))

        # MAPE
        mask = actual > 0
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

        metrics = ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
        )

        return metrics

    def measure_latency(
        self,
        model: Any,
        X_sample: pd.DataFrame,
    ) -> tuple[float, float]:
        """Measure inference latency for a model.

        Args:
            model: MLflow pyfunc model.
            X_sample: Sample data for inference.

        Returns:
            Tuple of (avg_latency_ms, p95_latency_ms).
        """
        n_samples = min(self.latency_samples, len(X_sample))
        sample_indices = np.random.choice(len(X_sample), n_samples, replace=False)

        # Cast categoricals
        X_eval = X_sample.copy()
        if "RatecodeID" in X_eval.columns:
            X_eval["RatecodeID"] = X_eval["RatecodeID"].astype(str)
        if "payment_type" in X_eval.columns:
            X_eval["payment_type"] = X_eval["payment_type"].astype(str)

        latencies = []
        for idx in sample_indices:
            row = X_eval.iloc[[idx]]
            start = time.perf_counter()
            _ = model.predict(row)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        return avg_latency, p95_latency

    def compare(
        self,
        champion_model: Any,
        champion_version: str,
        challenger_model: Any,
        challenger_version: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        challenger_validation_passed: bool = True,
        challenger_validation_errors: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare champion and challenger regression models.

        For RMSE, lower is better. Improvement = champion_rmse - challenger_rmse.

        Args:
            champion_model: Current production model.
            champion_version: Champion model version.
            challenger_model: Newly trained model.
            challenger_version: Challenger model version.
            X_test: Test features for evaluation.
            y_test: Test target values.
            challenger_validation_passed: Whether challenger passed validation.
            challenger_validation_errors: Validation error messages.

        Returns:
            ComparisonResult with comparison outcome.
        """
        logger.info(
            "comparing_models",
            champion_version=champion_version,
            challenger_version=challenger_version,
            test_samples=len(X_test),
        )

        rejection_reasons = []

        # Evaluate champion
        champion_metrics = self.evaluate_model(champion_model, X_test, y_test)
        champ_avg_lat, champ_p95_lat = self.measure_latency(champion_model, X_test)
        champion_metrics.avg_latency_ms = champ_avg_lat
        champion_metrics.p95_latency_ms = champ_p95_lat
        champion_metrics.model_version = champion_version

        # Evaluate challenger
        challenger_metrics = self.evaluate_model(challenger_model, X_test, y_test)
        chal_avg_lat, chal_p95_lat = self.measure_latency(challenger_model, X_test)
        challenger_metrics.avg_latency_ms = chal_avg_lat
        challenger_metrics.p95_latency_ms = chal_p95_lat
        challenger_metrics.validation_passed = challenger_validation_passed
        challenger_metrics.validation_errors = challenger_validation_errors or []
        challenger_metrics.model_version = challenger_version

        # Check validation
        if not challenger_validation_passed:
            rejection_reasons.append(
                f"Validation failed: {challenger_validation_errors}"
            )

        # Check metric improvement (for RMSE, lower is better)
        champion_primary = getattr(champion_metrics, self.primary_metric)
        challenger_primary = getattr(challenger_metrics, self.primary_metric)

        # Improvement = champion_rmse - challenger_rmse (positive means challenger is better)
        improvement = champion_primary - challenger_primary
        metric_improved = improvement >= self.min_improvement

        if not metric_improved:
            rejection_reasons.append(
                f"Insufficient {self.primary_metric} reduction: "
                f"{improvement:.4f} < {self.min_improvement:.4f} required "
                f"(champion={champion_primary:.4f}, challenger={challenger_primary:.4f})"
            )

        # Check latency
        latency_ratio = challenger_metrics.avg_latency_ms / max(
            champion_metrics.avg_latency_ms, 0.001
        )
        latency_acceptable = latency_ratio <= self.max_latency_regression

        if not latency_acceptable:
            rejection_reasons.append(
                f"Latency regression: {latency_ratio:.2f}x > "
                f"{self.max_latency_regression:.2f}x max allowed"
            )

        # Final decision
        should_promote = (
            challenger_validation_passed
            and metric_improved
            and latency_acceptable
        )

        if should_promote:
            promotion_reason = (
                f"Challenger v{challenger_version} outperforms champion v{champion_version}. "
                f"{self.primary_metric}: {challenger_primary:.4f} vs {champion_primary:.4f} "
                f"(reduction: {improvement:.4f}). Validation passed. Latency OK ({latency_ratio:.2f}x)."
            )
        else:
            promotion_reason = f"Challenger v{challenger_version} rejected: {'; '.join(rejection_reasons)}"

        result = ComparisonResult(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            validation_passed=challenger_validation_passed,
            metric_improvement=metric_improved,
            latency_acceptable=latency_acceptable,
            should_promote=should_promote,
            primary_metric_name=self.primary_metric,
            primary_metric_improvement=improvement,
            promotion_reason=promotion_reason,
            rejection_reasons=rejection_reasons,
        )

        logger.info(
            "comparison_complete",
            should_promote=should_promote,
            champion_rmse=champion_primary,
            challenger_rmse=challenger_primary,
            improvement=improvement,
            latency_ratio=latency_ratio,
        )

        return result

    def promote_challenger(
        self,
        model_name: str,
        challenger_version: str,
        alias: str = "production",
    ) -> None:
        """Promote challenger to production.

        Args:
            model_name: Registered model name.
            challenger_version: Version to promote.
            alias: Alias to set (default: 'production').
        """
        self._client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=challenger_version,
        )

        logger.info(
            "model_promoted",
            model_name=model_name,
            version=challenger_version,
            alias=alias,
        )
