"""Prefect flows for drift monitoring."""

from typing import Any

from prefect import flow, get_run_logger, task

from apps.monitor.config import DriftConfig, get_drift_config
from apps.monitor.service import DriftMonitorService
from shared.config import get_settings
from shared.drift.metrics import DriftResult


@task(
    name="initialize-drift-monitor",
    retries=2,
    retry_delay_seconds=30,
)
def initialize_monitor(config: DriftConfig | None = None) -> DriftMonitorService:
    """Initialize drift monitoring service.

    Args:
        config: Optional drift configuration.

    Returns:
        Initialized DriftMonitorService.
    """
    logger = get_run_logger()
    logger.info("Initializing drift monitor service")

    service = DriftMonitorService(config=config)
    return service


@task(
    name="run-drift-detection",
    retries=1,
    retry_delay_seconds=60,
)
def detect_drift(
    service: DriftMonitorService,
    model_name: str | None = None,
    model_version: str | None = None,
) -> DriftResult | None:
    """Run drift detection task.

    Args:
        service: DriftMonitorService instance.
        model_name: Model name to monitor.
        model_version: Model version.

    Returns:
        DriftResult or None if insufficient data.
    """
    logger = get_run_logger()
    logger.info(f"Running drift detection for model: {model_name or 'default'}")

    result = service.run_drift_detection(
        model_name=model_name,
        model_version=model_version,
    )

    if result is None:
        logger.warning("Insufficient data for drift detection")
    elif result.drift_detected:
        logger.warning(
            f"DRIFT DETECTED - Features: {result.features_with_drift}, "
            f"Max PSI: {result.max_psi:.3f}"
        )
    else:
        logger.info(f"No significant drift detected. Max PSI: {result.max_psi:.3f}")

    return result


@task(name="get-drift-summary")
def get_summary(
    service: DriftMonitorService,
    hours: int = 24,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Get drift summary for reporting.

    Args:
        service: DriftMonitorService instance.
        hours: Hours to summarize.
        model_name: Model name filter.

    Returns:
        Summary dictionary.
    """
    logger = get_run_logger()
    summary = service.get_drift_summary(hours=hours, model_name=model_name)
    logger.info(
        f"Drift summary: {summary['runs']} runs, "
        f"{summary['drift_detected_count']} with drift detected"
    )
    return summary


@task(name="check-alerts")
def check_unacknowledged_alerts(
    service: DriftMonitorService,
) -> list[dict[str, Any]]:
    """Check for unacknowledged drift alerts.

    Args:
        service: DriftMonitorService instance.

    Returns:
        List of unacknowledged alerts.
    """
    logger = get_run_logger()
    alerts = service.get_unacknowledged_alerts()

    if alerts:
        logger.warning(f"Found {len(alerts)} unacknowledged drift alerts")
        for alert in alerts:
            logger.warning(
                f"  [{alert['severity'].upper()}] {alert['created_at']}: "
                f"{alert['message']}"
            )
    else:
        logger.info("No unacknowledged drift alerts")

    return alerts


@flow(
    name="monitor-drift",
    description="Run drift detection comparing reference vs recent inference data",
    retries=1,
    retry_delay_seconds=300,
)
def monitor_drift_flow(
    model_name: str | None = None,
    model_version: str | None = None,
    include_summary: bool = True,
    summary_hours: int = 24,
) -> dict[str, Any]:
    """Main drift monitoring flow.

    Workflow:
    1. Initialize monitoring service
    2. Run drift detection
    3. Optionally get summary
    4. Check for unacknowledged alerts

    Args:
        model_name: Model name to monitor.
        model_version: Model version.
        include_summary: Include drift summary in result.
        summary_hours: Hours for summary.

    Returns:
        Flow result with drift detection and summary.
    """
    logger = get_run_logger()
    logger.info("Starting drift monitoring flow")

    settings = get_settings()
    config = get_drift_config()

    model_name = model_name or settings.model_name

    # Initialize service
    service = initialize_monitor(config=config)

    # Run drift detection
    drift_result = detect_drift(
        service=service,
        model_name=model_name,
        model_version=model_version,
    )

    result: dict[str, Any] = {
        "model_name": model_name,
        "drift_result": drift_result.to_dict() if drift_result else None,
        "drift_detected": drift_result.drift_detected if drift_result else False,
    }

    # Get summary if requested
    if include_summary:
        summary = get_summary(
            service=service,
            hours=summary_hours,
            model_name=model_name,
        )
        result["summary"] = summary

    # Check alerts
    alerts = check_unacknowledged_alerts(service=service)
    result["unacknowledged_alerts"] = len(alerts)

    logger.info(
        f"Drift monitoring complete. "
        f"Drift detected: {result['drift_detected']}, "
        f"Unacknowledged alerts: {result['unacknowledged_alerts']}"
    )

    return result


@flow(
    name="scheduled-drift-monitor",
    description="Scheduled drift monitoring - runs on interval",
)
def scheduled_drift_monitor_flow() -> dict[str, Any]:
    """Scheduled flow for continuous drift monitoring.

    This flow is designed to be deployed with a schedule (e.g., hourly).
    It runs drift detection and logs results.

    Returns:
        Flow result.
    """
    logger = get_run_logger()
    logger.info("Running scheduled drift monitoring")

    # Run the main monitoring flow
    result = monitor_drift_flow(
        include_summary=True,
        summary_hours=24,
    )

    # Log key metrics for observability
    if result["drift_detected"]:
        logger.warning(
            "SCHEDULED CHECK: Drift detected! "
            f"Model: {result['model_name']}, "
            f"Alerts: {result['unacknowledged_alerts']}"
        )
    else:
        logger.info(
            f"SCHEDULED CHECK: No drift. "
            f"Model: {result['model_name']}"
        )

    return result


if __name__ == "__main__":
    # Run directly for testing
    import json

    result = monitor_drift_flow()
    print(json.dumps(result, indent=2, default=str))
