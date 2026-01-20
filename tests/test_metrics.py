"""Tests for metrics module."""

import pytest

from shared.utils.metrics import Counter, Gauge, Histogram, MetricsRegistry, get_metrics


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_increment(self) -> None:
        """Test counter increment."""
        counter = Counter(name="test_counter", description="Test counter")
        counter.inc()
        assert counter.get() == 1.0
        counter.inc(5)
        assert counter.get() == 6.0

    def test_counter_with_labels(self) -> None:
        """Test counter with labels."""
        counter = Counter(
            name="test_counter_labels",
            description="Test counter with labels",
            labels=["method", "status"],
        )
        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="200")
        counter.inc(method="GET", status="200")

        assert counter.get(method="GET", status="200") == 2.0
        assert counter.get(method="POST", status="200") == 1.0
        assert counter.get(method="GET", status="404") == 0.0

    def test_counter_to_prometheus(self) -> None:
        """Test Prometheus format export."""
        counter = Counter(
            name="http_requests_total",
            description="Total HTTP requests",
            labels=["method"],
        )
        counter.inc(method="GET")
        counter.inc(method="GET")
        counter.inc(method="POST")

        output = counter.to_prometheus()
        assert "# HELP http_requests_total" in output
        assert "# TYPE http_requests_total counter" in output
        assert 'http_requests_total{method="GET"} 2' in output
        assert 'http_requests_total{method="POST"} 1' in output


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set(self) -> None:
        """Test gauge set value."""
        gauge = Gauge(name="test_gauge", description="Test gauge")
        gauge.set(42.0)
        assert gauge.get() == 42.0

    def test_gauge_inc_dec(self) -> None:
        """Test gauge increment and decrement."""
        gauge = Gauge(name="test_gauge", description="Test gauge")
        gauge.set(10.0)
        gauge.inc(5.0)
        assert gauge.get() == 15.0
        gauge.dec(3.0)
        assert gauge.get() == 12.0

    def test_gauge_with_labels(self) -> None:
        """Test gauge with labels."""
        gauge = Gauge(
            name="active_connections",
            description="Active connections",
            labels=["service"],
        )
        gauge.set(10.0, service="api")
        gauge.set(5.0, service="worker")

        assert gauge.get(service="api") == 10.0
        assert gauge.get(service="worker") == 5.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_observe(self) -> None:
        """Test histogram observation."""
        histogram = Histogram(
            name="request_latency",
            description="Request latency",
            buckets=(0.1, 0.5, 1.0, 5.0),
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.7)
        histogram.observe(2.0)

        result = histogram.to_dict()
        assert result["count"] == 4
        assert result["sum"] == pytest.approx(3.05)

    def test_histogram_with_labels(self) -> None:
        """Test histogram with labels."""
        histogram = Histogram(
            name="request_latency",
            description="Request latency",
            labels=["endpoint"],
            buckets=(0.1, 0.5, 1.0),
        )
        histogram.observe(0.05, endpoint="/predict")
        histogram.observe(0.2, endpoint="/predict")
        histogram.observe(0.01, endpoint="/healthz")

        result = histogram.to_dict()
        assert "values" in result
        assert len(result["values"]) == 2


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_singleton(self) -> None:
        """Test registry is singleton."""
        registry1 = get_metrics()
        registry2 = get_metrics()
        assert registry1 is registry2

    def test_record_request(self) -> None:
        """Test recording request metrics."""
        registry = get_metrics()
        registry.reset()

        registry.record_request(
            method="POST",
            endpoint="/predict",
            status=200,
            latency_seconds=0.05,
        )

        assert registry.requests_total.get(
            method="POST", endpoint="/predict", status="200"
        ) == 1.0

    def test_record_prediction(self) -> None:
        """Test recording prediction metrics."""
        registry = get_metrics()
        registry.reset()

        registry.record_prediction(
            model_name="churn-classifier",
            model_version="1",
            prediction=1,
            latency_seconds=0.02,
        )

        assert registry.predictions_total.get(
            model_name="churn-classifier", model_version="1", prediction="1"
        ) == 1.0

    def test_record_drift_event(self) -> None:
        """Test recording drift events."""
        registry = get_metrics()
        registry.reset()

        registry.record_drift_event(drift_type="feature", severity="warning")
        registry.record_drift_event(drift_type="feature", severity="critical")

        assert registry.drift_events_total.get(
            drift_type="feature", severity="warning"
        ) == 1.0
        assert registry.drift_events_total.get(
            drift_type="feature", severity="critical"
        ) == 1.0

    def test_to_prometheus_format(self) -> None:
        """Test Prometheus format export."""
        registry = get_metrics()
        registry.reset()

        registry.record_request(
            method="GET", endpoint="/healthz", status=200, latency_seconds=0.001
        )

        output = registry.to_prometheus()
        assert "# HELP pmmds_requests_total" in output
        assert "# TYPE pmmds_requests_total counter" in output
        assert "pmmds_uptime_seconds" in output

    def test_to_dict_format(self) -> None:
        """Test dictionary format export."""
        registry = get_metrics()
        registry.reset()

        result = registry.to_dict()
        assert "requests" in result
        assert "predictions" in result
        assert "errors" in result
        assert "drift" in result
        assert "model" in result
        assert "system" in result
        assert "uptime_seconds" in result["system"]
