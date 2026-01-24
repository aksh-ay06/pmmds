"""Tests for metrics module."""

import pytest

from shared.utils.metrics import Counter, Gauge, Histogram, MetricsRegistry, get_metrics


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_increment(self) -> None:
        """Test counter increments correctly."""
        counter = Counter("test_counter", "test")
        counter.inc()
        assert counter.get() == 1
        counter.inc(5)
        assert counter.get() == 6

    def test_counter_labels(self) -> None:
        """Test counter with labels."""
        counter = Counter("test_counter", "test", labels=["method"])
        counter.inc(method="POST")
        counter.inc(method="GET")
        assert counter.get(method="POST") == 1
        assert counter.get(method="GET") == 1


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set(self) -> None:
        """Test gauge set and get."""
        gauge = Gauge("test_gauge", "test")
        gauge.set(42.0)
        assert gauge.get() == 42.0

    def test_gauge_inc_dec(self) -> None:
        """Test gauge increment and decrement."""
        gauge = Gauge("test_gauge", "test")
        gauge.set(10.0)
        gauge.inc()
        assert gauge.get() == 11.0
        gauge.dec(3)
        assert gauge.get() == 8.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_observe(self) -> None:
        """Test histogram observes values."""
        hist = Histogram("test_histogram", "test")
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(1.0)
        # Use the public dict to check totals
        key = ()
        assert hist._totals[key] == 3
        assert hist._sums[key] == pytest.approx(1.6)

    def test_histogram_buckets(self) -> None:
        """Test histogram bucket counting."""
        buckets = (0.1, 0.5, 1.0, 5.0)
        hist = Histogram("test_histogram", "test", buckets=buckets)
        hist.observe(0.05)  # <= 0.1
        hist.observe(0.3)   # <= 0.5
        hist.observe(2.0)   # <= 5.0
        hist.observe(10.0)  # > all buckets
        key = ()
        assert hist._totals[key] == 4


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_record_prediction(self) -> None:
        """Test recording a prediction metric."""
        registry = MetricsRegistry()
        registry.record_prediction(
            model_name="nyc-taxi-fare",
            model_version="1",
            prediction=15,
            latency_seconds=0.012,
        )
        # Should not raise

    def test_record_drift_check(self) -> None:
        """Test recording drift check metrics."""
        registry = MetricsRegistry()
        registry.record_drift_check(model_name="nyc-taxi-fare")

    def test_record_drift_event(self) -> None:
        """Test recording drift event."""
        registry = MetricsRegistry()
        registry.record_drift_event(drift_type="feature", severity="medium")

    def test_record_retraining(self) -> None:
        """Test recording retraining event."""
        registry = MetricsRegistry()
        registry.record_retraining(trigger_type="drift", outcome="success")

    def test_prometheus_format(self) -> None:
        """Test Prometheus exposition format output."""
        registry = MetricsRegistry()
        registry.record_prediction(
            model_name="nyc-taxi-fare",
            model_version="1",
            prediction=12,
            latency_seconds=0.01,
        )
        output = registry.to_prometheus()
        assert "pmmds_" in output

    def test_json_format(self) -> None:
        """Test JSON metrics output."""
        registry = MetricsRegistry()
        data = registry.to_dict()
        assert "requests" in data
        assert "predictions" in data
        assert "system" in data


class TestGetMetrics:
    """Tests for get_metrics singleton."""

    def test_get_metrics_returns_registry(self) -> None:
        """Test get_metrics returns a MetricsRegistry."""
        metrics = get_metrics()
        assert isinstance(metrics, MetricsRegistry)

    def test_get_metrics_singleton(self) -> None:
        """Test get_metrics returns same instance."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2
