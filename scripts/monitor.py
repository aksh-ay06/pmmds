#!/usr/bin/env python3
"""Run drift monitoring.

Usage:
    python scripts/monitor.py              # Run once
    python scripts/monitor.py --summary    # Show summary only
    python scripts/monitor.py --deploy     # Deploy as scheduled flow
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_monitoring(
    model_name: str | None = None,
    summary_only: bool = False,
    summary_hours: int = 24,
) -> None:
    """Run drift monitoring.

    Args:
        model_name: Model name to monitor.
        summary_only: Only show summary, don't run detection.
        summary_hours: Hours for summary period.
    """
    from apps.monitor.service import DriftMonitorService
    from shared.utils import get_logger

    logger = get_logger(__name__)

    print("=" * 60)
    print("PMMDS Drift Monitoring")
    print("=" * 60)

    service = DriftMonitorService()

    if summary_only:
        print(f"\nDrift Summary (last {summary_hours} hours):")
        print("-" * 40)
        summary = service.get_drift_summary(hours=summary_hours, model_name=model_name)
        print(json.dumps(summary, indent=2, default=str))

        print(f"\nUnacknowledged Alerts:")
        print("-" * 40)
        alerts = service.get_unacknowledged_alerts()
        if alerts:
            for alert in alerts:
                print(f"  [{alert['severity'].upper()}] {alert['created_at']}")
                print(f"    {alert['message']}")
        else:
            print("  No unacknowledged alerts")
    else:
        print("\nRunning drift detection...")
        print("-" * 40)

        result = service.run_drift_detection(model_name=model_name)

        if result is None:
            print("⚠️  Insufficient data for drift detection")
            print("   Need at least 100 inference samples in the current window.")
            print("   Run 'make seed-traffic' to generate sample predictions.")
        elif result.drift_detected:
            print("🚨 DRIFT DETECTED!")
            print(f"   Features with drift: {', '.join(result.features_with_drift)}")
            print(f"   Max PSI: {result.max_psi:.4f}")
            print(f"   Avg PSI: {result.avg_psi:.4f}")

            print("\n   Feature Details:")
            for fd in result.feature_drifts:
                status = "⚠️" if fd.psi >= 0.2 else "✓"
                print(f"     {status} {fd.feature_name}: PSI={fd.psi:.4f} ({fd.severity.value})")
        else:
            print("✅ No significant drift detected")
            print(f"   Max PSI: {result.max_psi:.4f}")
            print(f"   Avg PSI: {result.avg_psi:.4f}")

            print("\n   Feature PSI Values:")
            for fd in result.feature_drifts:
                print(f"     ✓ {fd.feature_name}: {fd.psi:.4f}")

        print("\n" + "=" * 60)


def deploy_scheduled_flow(interval_hours: int = 1) -> None:
    """Deploy scheduled monitoring flow.

    Args:
        interval_hours: Hours between runs.
    """
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import IntervalSchedule

    from apps.monitor.flows import scheduled_drift_monitor_flow

    print(f"Deploying scheduled drift monitoring (every {interval_hours}h)...")

    deployment = Deployment.build_from_flow(
        flow=scheduled_drift_monitor_flow,
        name="scheduled-drift-monitor",
        schedule=IntervalSchedule(interval=interval_hours * 3600),
        work_queue_name="default",
        tags=["monitoring", "drift"],
    )

    deployment.apply()
    print("✅ Deployment created successfully!")
    print("   Run 'prefect agent start -q default' to start processing.")


def run_prefect_flow() -> None:
    """Run monitoring via Prefect flow."""
    from apps.monitor.flows import monitor_drift_flow

    print("Running drift monitoring via Prefect flow...")
    result = monitor_drift_flow()
    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run PMMDS drift monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor.py                  # Run drift detection
  python scripts/monitor.py --summary        # Show summary only
  python scripts/monitor.py --prefect        # Run via Prefect flow
  python scripts/monitor.py --deploy         # Deploy scheduled flow
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to monitor (default: from settings)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show drift summary only (don't run detection)",
    )
    parser.add_argument(
        "--summary-hours",
        type=int,
        default=24,
        help="Hours for summary period (default: 24)",
    )
    parser.add_argument(
        "--prefect",
        action="store_true",
        help="Run via Prefect flow",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy scheduled monitoring flow",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval in hours for scheduled deployment (default: 1)",
    )

    args = parser.parse_args()

    if args.deploy:
        deploy_scheduled_flow(interval_hours=args.interval)
    elif args.prefect:
        run_prefect_flow()
    else:
        run_monitoring(
            model_name=args.model,
            summary_only=args.summary,
            summary_hours=args.summary_hours,
        )


if __name__ == "__main__":
    main()
