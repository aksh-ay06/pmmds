"""Prefect flows for automated retraining and promotion."""

from typing import Any

from prefect import flow, get_run_logger, task

from apps.monitor.flows import monitor_drift_flow
from pipelines.retrain.service import RetrainingConfig, RetrainingService
from shared.config import get_settings


@task(
    name="check-drift-threshold",
    retries=2,
    retry_delay_seconds=30,
)
def check_drift_threshold(service: RetrainingService) -> tuple[bool, dict[str, Any] | None]:
    """Check if drift threshold exceeded.

    Args:
        service: Retraining service instance.

    Returns:
        Tuple of (should_retrain, drift_info).
    """
    logger = get_run_logger()
    should_retrain, drift_info = service.check_drift_trigger()

    if should_retrain:
        logger.warning(
            f"Drift threshold exceeded! "
            f"Features: {drift_info['features_with_drift']}, "
            f"Max PSI: {drift_info['max_psi']:.4f}"
        )
    else:
        logger.info("Drift below retraining threshold")

    return should_retrain, drift_info


@task(
    name="train-challenger-model",
    retries=1,
    retry_delay_seconds=60,
)
def train_challenger(
    service: RetrainingService,
    run_id: str,
    trigger_type: str,
    drift_info: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str]:
    """Train a new challenger model.

    Args:
        service: Retraining service.
        run_id: Retraining run ID.
        trigger_type: Trigger type.
        drift_info: Drift detection info.

    Returns:
        Tuple of (metrics_dict, mlflow_run_id, model_version).
    """
    logger = get_run_logger()
    logger.info(f"Training challenger model (run_id={run_id})")

    metrics, mlflow_run_id, version = service.train_challenger(
        run_id=run_id,
        trigger_type=trigger_type,
        drift_info=drift_info,
    )

    logger.info(
        f"Challenger trained: v{version}, "
        f"RMSE={metrics.rmse:.4f}, "
        f"R2={metrics.r2:.4f}"
    )

    return metrics.to_dict(), mlflow_run_id, version


@task(name="compare-and-promote")
def compare_and_promote(
    service: RetrainingService,
    challenger_version: str,
    challenger_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare challenger against champion and promote if better.

    Args:
        service: Retraining service.
        challenger_version: Challenger model version.
        challenger_metrics: Challenger metrics dict.

    Returns:
        Comparison result dictionary.
    """
    logger = get_run_logger()
    logger.info(f"Comparing challenger v{challenger_version} against champion")

    # Reconstruct metrics object for comparison
    from pipelines.train.trainer import TrainingMetrics

    metrics = TrainingMetrics(**{
        k: v for k, v in challenger_metrics.items()
        if k in TrainingMetrics.__dataclass_fields__
    })

    comparison = service.compare_and_promote(
        challenger_version=challenger_version,
        challenger_metrics=metrics,
    )

    if comparison.should_promote:
        logger.info(
            f"✅ PROMOTED: Challenger v{challenger_version} is now production! "
            f"{comparison.promotion_reason}"
        )
    else:
        logger.warning(
            f"❌ NOT PROMOTED: Challenger v{challenger_version} rejected. "
            f"Reasons: {comparison.rejection_reasons}"
        )

    return comparison.to_dict()


@task(name="record-results")
def record_results(
    service: RetrainingService,
    run_id: str,
    trigger_type: str,
    drift_info: dict[str, Any] | None,
    mlflow_run_id: str,
    challenger_version: str,
    challenger_metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    """Record retraining results in database.

    Args:
        service: Retraining service.
        run_id: Retraining run ID.
        trigger_type: Trigger type.
        drift_info: Drift info.
        mlflow_run_id: MLflow run ID.
        challenger_version: Challenger version.
        challenger_metrics: Metrics dict.
        comparison: Comparison result dict.
    """
    logger = get_run_logger()

    # This is handled by the service.run_retraining method
    # but we log for visibility
    logger.info(
        f"Results recorded: run_id={run_id}, "
        f"promoted={comparison['should_promote']}"
    )


@flow(
    name="retrain-and-promote",
    description="Train a challenger model and promote if it outperforms champion",
    retries=0,
)
def retrain_and_promote_flow(
    trigger_type: str = "manual",
    force: bool = False,
) -> dict[str, Any]:
    """Main retraining and promotion flow.

    Workflow:
    1. Check if drift threshold exceeded (unless forced)
    2. Train challenger model
    3. Compare against champion
    4. Promote if improvement criteria met
    5. Record all decisions

    Args:
        trigger_type: What triggered retraining (drift, scheduled, manual).
        force: Force retraining even if drift threshold not met.

    Returns:
        Flow result dictionary.
    """
    logger = get_run_logger()
    logger.info(f"Starting retrain-and-promote flow (trigger={trigger_type}, force={force})")

    # Initialize service
    service = RetrainingService()

    # Run full retraining workflow
    result = service.run_retraining(
        trigger_type=trigger_type,
        force=force,
    )

    if result["status"] == "skipped":
        logger.info(f"Retraining skipped: {result.get('reason', 'Unknown')}")
    elif result["status"] == "completed":
        logger.info(
            f"Retraining completed! "
            f"Promoted: {result.get('promoted', False)}, "
            f"Duration: {result.get('training_duration_seconds', 0):.1f}s"
        )

    return result


@flow(
    name="drift-triggered-retrain",
    description="Check drift and trigger retraining if threshold exceeded",
)
def drift_triggered_retrain_flow() -> dict[str, Any]:
    """Flow that checks drift and triggers retraining if needed.

    This is the main scheduled flow for automated retraining.

    Returns:
        Flow result with drift check and retraining results.
    """
    logger = get_run_logger()
    logger.info("Starting drift-triggered retraining check")

    # First run drift monitoring
    drift_result = monitor_drift_flow(include_summary=False)

    result: dict[str, Any] = {
        "drift_check": drift_result,
        "retraining": None,
    }

    # Check if drift was detected
    if drift_result.get("drift_detected", False):
        logger.warning("Drift detected - triggering retraining")

        # Run retraining
        retrain_result = retrain_and_promote_flow(
            trigger_type="drift",
            force=False,
        )
        result["retraining"] = retrain_result
    else:
        logger.info("No drift detected - skipping retraining")
        result["retraining"] = {"status": "skipped", "reason": "No drift detected"}

    return result


@flow(
    name="scheduled-retrain",
    description="Scheduled retraining flow - runs periodically",
)
def scheduled_retrain_flow() -> dict[str, Any]:
    """Scheduled retraining flow.

    This flow is designed to be deployed with a schedule.
    It always runs drift detection first, then retrains only if needed.

    Returns:
        Flow result.
    """
    return drift_triggered_retrain_flow()


@flow(
    name="force-retrain",
    description="Force retraining regardless of drift",
)
def force_retrain_flow() -> dict[str, Any]:
    """Force retraining flow.

    Bypasses drift check and always trains a new model.

    Returns:
        Flow result.
    """
    logger = get_run_logger()
    logger.info("Force retraining triggered")

    return retrain_and_promote_flow(
        trigger_type="manual",
        force=True,
    )


if __name__ == "__main__":
    # Run directly for testing
    import json

    result = retrain_and_promote_flow(trigger_type="manual", force=True)
    print(json.dumps(result, indent=2, default=str))
