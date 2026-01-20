"""Prometheus-compatible metrics tracking.

Thread-safe metrics collection for observability:
- Request counts by endpoint and status
- Latency histograms
- Error counts
- Drift events
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    """Thread-safe counter metric."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, value: float = 1.0, **label_values: str) -> None:
        """Increment counter by value.

        Args:
            value: Amount to increment (default 1).
            **label_values: Label key-value pairs.
        """
        key = self._make_key(label_values)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def _make_key(self, label_values: dict[str, str]) -> tuple:
        """Create a hashable key from label values."""
        return tuple(label_values.get(label, "") for label in self.labels)

    def get(self, **label_values: str) -> float:
        """Get current counter value."""
        key = self._make_key(label_values)
        with self._lock:
            return self._values.get(key, 0.0)

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key, value in self._values.items():
                if self.labels:
                    label_str = ",".join(
                        f'{label}="{key[i]}"' for i, label in enumerate(self.labels)
                    )
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        with self._lock:
            if not self.labels:
                return {"value": sum(self._values.values())}
            return {
                "values": [
                    {
                        "labels": dict(zip(self.labels, key)),
                        "value": value,
                    }
                    for key, value in self._values.items()
                ]
            }


@dataclass
class Histogram:
    """Thread-safe histogram metric with buckets."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    _counts: dict[tuple, dict[float, int]] = field(default_factory=dict)
    _sums: dict[tuple, float] = field(default_factory=dict)
    _totals: dict[tuple, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **label_values: str) -> None:
        """Record an observation.

        Args:
            value: Observed value.
            **label_values: Label key-value pairs.
        """
        key = self._make_key(label_values)
        with self._lock:
            if key not in self._counts:
                self._counts[key] = {b: 0 for b in self.buckets}
                self._sums[key] = 0.0
                self._totals[key] = 0

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1

            self._sums[key] += value
            self._totals[key] += 1

    def _make_key(self, label_values: dict[str, str]) -> tuple:
        """Create a hashable key from label values."""
        return tuple(label_values.get(label, "") for label in self.labels)

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key, bucket_counts in self._counts.items():
                label_prefix = ""
                if self.labels:
                    label_prefix = ",".join(
                        f'{label}="{key[i]}"' for i, label in enumerate(self.labels)
                    )

                # Bucket lines
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += bucket_counts[bucket] - (
                        bucket_counts.get(self.buckets[self.buckets.index(bucket) - 1], 0)
                        if self.buckets.index(bucket) > 0
                        else 0
                    )
                for bucket in self.buckets:
                    if label_prefix:
                        lines.append(
                            f'{self.name}_bucket{{{label_prefix},le="{bucket}"}} '
                            f"{bucket_counts[bucket]}"
                        )
                    else:
                        lines.append(f'{self.name}_bucket{{le="{bucket}"}} {bucket_counts[bucket]}')

                # +Inf bucket
                if label_prefix:
                    lines.append(f'{self.name}_bucket{{{label_prefix},le="+Inf"}} {self._totals[key]}')
                    lines.append(f"{self.name}_sum{{{label_prefix}}} {self._sums[key]}")
                    lines.append(f"{self.name}_count{{{label_prefix}}} {self._totals[key]}")
                else:
                    lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._totals[key]}')
                    lines.append(f"{self.name}_sum {self._sums[key]}")
                    lines.append(f"{self.name}_count {self._totals[key]}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        with self._lock:
            result: dict[str, Any] = {}
            for key in self._counts.keys():
                entry = {
                    "count": self._totals.get(key, 0),
                    "sum": self._sums.get(key, 0.0),
                    "avg": (
                        self._sums.get(key, 0.0) / self._totals[key]
                        if self._totals.get(key, 0) > 0
                        else 0.0
                    ),
                }
                if self.labels:
                    if "values" not in result:
                        result["values"] = []
                    result["values"].append({"labels": dict(zip(self.labels, key)), **entry})
                else:
                    result.update(entry)
            return result


@dataclass
class Gauge:
    """Thread-safe gauge metric."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **label_values: str) -> None:
        """Set gauge value.

        Args:
            value: New value.
            **label_values: Label key-value pairs.
        """
        key = self._make_key(label_values)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **label_values: str) -> None:
        """Increment gauge by value."""
        key = self._make_key(label_values)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1.0, **label_values: str) -> None:
        """Decrement gauge by value."""
        key = self._make_key(label_values)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - value

    def _make_key(self, label_values: dict[str, str]) -> tuple:
        """Create a hashable key from label values."""
        return tuple(label_values.get(label, "") for label in self.labels)

    def get(self, **label_values: str) -> float:
        """Get current gauge value."""
        key = self._make_key(label_values)
        with self._lock:
            return self._values.get(key, 0.0)

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for key, value in self._values.items():
                if self.labels:
                    label_str = ",".join(
                        f'{label}="{key[i]}"' for i, label in enumerate(self.labels)
                    )
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        with self._lock:
            if not self.labels:
                total = sum(self._values.values()) if self._values else 0.0
                return {"value": total}
            return {
                "values": [
                    {
                        "labels": dict(zip(self.labels, key)),
                        "value": value,
                    }
                    for key, value in self._values.items()
                ]
            }


class MetricsRegistry:
    """Central registry for all application metrics.

    Singleton pattern ensures consistent metrics across the application.
    """

    _instance: "MetricsRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        """Ensure single instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics registry."""
        if self._initialized:
            return

        self._start_time = time.time()

        # Request metrics
        self.requests_total = Counter(
            name="pmmds_requests_total",
            description="Total number of requests",
            labels=["method", "endpoint", "status"],
        )

        self.request_latency = Histogram(
            name="pmmds_request_latency_seconds",
            description="Request latency in seconds",
            labels=["method", "endpoint"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # Prediction metrics
        self.predictions_total = Counter(
            name="pmmds_predictions_total",
            description="Total number of predictions",
            labels=["model_name", "model_version", "prediction"],
        )

        self.prediction_latency = Histogram(
            name="pmmds_prediction_latency_seconds",
            description="Model inference latency in seconds",
            labels=["model_name", "model_version"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        # Error metrics
        self.errors_total = Counter(
            name="pmmds_errors_total",
            description="Total number of errors",
            labels=["endpoint", "error_type"],
        )

        # Validation metrics
        self.validation_failures = Counter(
            name="pmmds_validation_failures_total",
            description="Total number of validation failures",
            labels=["endpoint", "failure_type"],
        )

        # Drift metrics
        self.drift_events_total = Counter(
            name="pmmds_drift_events_total",
            description="Total number of drift detection events",
            labels=["drift_type", "severity"],
        )

        self.drift_checks_total = Counter(
            name="pmmds_drift_checks_total",
            description="Total number of drift checks performed",
            labels=["model_name"],
        )

        # Model metrics
        self.model_reloads = Counter(
            name="pmmds_model_reloads_total",
            description="Total number of model reloads",
            labels=["model_name"],
        )

        self.current_model_info = Gauge(
            name="pmmds_model_info",
            description="Current model information (value is always 1)",
            labels=["model_name", "model_version"],
        )

        # Retraining metrics
        self.retraining_runs_total = Counter(
            name="pmmds_retraining_runs_total",
            description="Total number of retraining runs",
            labels=["trigger_type", "outcome"],
        )

        self.promotions_total = Counter(
            name="pmmds_promotions_total",
            description="Total number of model promotions",
            labels=["from_version", "to_version"],
        )

        # System metrics
        self.uptime = Gauge(
            name="pmmds_uptime_seconds",
            description="Time since application start in seconds",
        )

        self._initialized = True

    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        latency_seconds: float,
    ) -> None:
        """Record an HTTP request.

        Args:
            method: HTTP method.
            endpoint: Request endpoint path.
            status: HTTP status code.
            latency_seconds: Request latency in seconds.
        """
        self.requests_total.inc(method=method, endpoint=endpoint, status=str(status))
        self.request_latency.observe(latency_seconds, method=method, endpoint=endpoint)

    def record_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction: int,
        latency_seconds: float,
    ) -> None:
        """Record a prediction.

        Args:
            model_name: Name of the model.
            model_version: Version of the model.
            prediction: Prediction class.
            latency_seconds: Inference latency in seconds.
        """
        self.predictions_total.inc(
            model_name=model_name,
            model_version=model_version,
            prediction=str(prediction),
        )
        self.prediction_latency.observe(
            latency_seconds,
            model_name=model_name,
            model_version=model_version,
        )

    def record_error(self, endpoint: str, error_type: str) -> None:
        """Record an error.

        Args:
            endpoint: Request endpoint.
            error_type: Type of error.
        """
        self.errors_total.inc(endpoint=endpoint, error_type=error_type)

    def record_validation_failure(self, endpoint: str, failure_type: str) -> None:
        """Record a validation failure.

        Args:
            endpoint: Request endpoint.
            failure_type: Type of validation failure.
        """
        self.validation_failures.inc(endpoint=endpoint, failure_type=failure_type)

    def record_drift_event(self, drift_type: str, severity: str) -> None:
        """Record a drift detection event.

        Args:
            drift_type: Type of drift (feature, prediction).
            severity: Drift severity level.
        """
        self.drift_events_total.inc(drift_type=drift_type, severity=severity)

    def record_drift_check(self, model_name: str) -> None:
        """Record a drift check.

        Args:
            model_name: Name of the model checked.
        """
        self.drift_checks_total.inc(model_name=model_name)

    def record_model_reload(self, model_name: str) -> None:
        """Record a model reload.

        Args:
            model_name: Name of the model.
        """
        self.model_reloads.inc(model_name=model_name)

    def set_current_model(self, model_name: str, model_version: str) -> None:
        """Set current model information.

        Args:
            model_name: Name of the model.
            model_version: Version of the model.
        """
        self.current_model_info.set(1.0, model_name=model_name, model_version=model_version)

    def record_retraining(self, trigger_type: str, outcome: str) -> None:
        """Record a retraining run.

        Args:
            trigger_type: What triggered the retraining (drift, scheduled, manual).
            outcome: Outcome of the run (success, failure, skipped).
        """
        self.retraining_runs_total.inc(trigger_type=trigger_type, outcome=outcome)

    def record_promotion(self, from_version: str, to_version: str) -> None:
        """Record a model promotion.

        Args:
            from_version: Previous champion version.
            to_version: New champion version.
        """
        self.promotions_total.inc(from_version=from_version, to_version=to_version)

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string.
        """
        # Update uptime
        self.uptime.set(time.time() - self._start_time)

        metrics = [
            self.requests_total,
            self.request_latency,
            self.predictions_total,
            self.prediction_latency,
            self.errors_total,
            self.validation_failures,
            self.drift_events_total,
            self.drift_checks_total,
            self.model_reloads,
            self.current_model_info,
            self.retraining_runs_total,
            self.promotions_total,
            self.uptime,
        ]

        return "\n\n".join(m.to_prometheus() for m in metrics) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as dictionary.

        Returns:
            Dictionary with all metrics.
        """
        # Update uptime
        self.uptime.set(time.time() - self._start_time)

        return {
            "requests": {
                "total": self.requests_total.to_dict(),
                "latency": self.request_latency.to_dict(),
            },
            "predictions": {
                "total": self.predictions_total.to_dict(),
                "latency": self.prediction_latency.to_dict(),
            },
            "errors": self.errors_total.to_dict(),
            "validation_failures": self.validation_failures.to_dict(),
            "drift": {
                "events": self.drift_events_total.to_dict(),
                "checks": self.drift_checks_total.to_dict(),
            },
            "model": {
                "reloads": self.model_reloads.to_dict(),
                "current": self.current_model_info.to_dict(),
            },
            "retraining": {
                "runs": self.retraining_runs_total.to_dict(),
                "promotions": self.promotions_total.to_dict(),
            },
            "system": {
                "uptime_seconds": self.uptime.get(),
            },
        }

    def reset(self) -> None:
        """Reset all metrics. Mainly for testing."""
        self._initialized = False
        self.__init__()


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry instance.

    Returns:
        MetricsRegistry singleton.
    """
    return MetricsRegistry()
