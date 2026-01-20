#!/usr/bin/env python3
"""Run automated retraining and model promotion.

Usage:
    python scripts/retrain.py              # Run drift-triggered retraining
    python scripts/retrain.py --force      # Force retraining
    python scripts/retrain.py --decisions  # Show recent decisions
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_retraining(
    trigger_type: str = "manual",
    force: bool = False,
) -> None:
    """Run retraining workflow.

    Args:
        trigger_type: Trigger type (drift, scheduled, manual).
        force: Force retraining regardless of drift.
    """
    from pipelines.retrain.service import RetrainingService
    from shared.utils import get_logger

    logger = get_logger(__name__)

    print("=" * 60)
    print("PMMDS Automated Retraining")
    print("=" * 60)
    print(f"Trigger: {trigger_type}")
    print(f"Force: {force}")
    print("-" * 60)

    service = RetrainingService()

    # Check drift first (unless forcing)
    if not force:
        should_retrain, drift_info = service.check_drift_trigger()
        if drift_info:
            print(f"\nDrift Info:")
            print(f"  Drift detected: {drift_info['drift_detected']}")
            print(f"  Features with drift: {drift_info['features_with_drift']}")
            print(f"  Max PSI: {drift_info['max_psi']:.4f}")

        if not should_retrain:
            print("\n⚠️  Drift threshold not exceeded - skipping retraining")
            print("   Use --force to retrain anyway")
            return

    print("\n🚀 Starting retraining...")

    try:
        result = service.run_retraining(
            trigger_type=trigger_type,
            force=force,
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Status: {result['status']}")
        print(f"Run ID: {result['run_id']}")

        if result["status"] == "completed":
            print(f"\nChallenger Model:")
            print(f"  Version: v{result.get('challenger_version', 'N/A')}")
            print(f"  MLflow Run: {result.get('mlflow_run_id', 'N/A')}")

            metrics = result.get("challenger_metrics", {})
            print(f"\nChallenger Metrics:")
            print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")

            comparison = result.get("comparison", {})
            print(f"\nPromotion Decision:")
            print(f"  Promoted: {'✅ YES' if result.get('promoted') else '❌ NO'}")
            print(f"  Validation: {'✓' if comparison.get('validation_passed') else '✗'}")
            print(f"  Metric Improvement: {'✓' if comparison.get('metric_improvement') else '✗'}")
            print(f"  Latency OK: {'✓' if comparison.get('latency_acceptable') else '✗'}")
            print(f"  Reason: {comparison.get('promotion_reason', 'N/A')}")

            print(f"\nDuration: {result.get('training_duration_seconds', 0):.1f}s")

            if result.get("promoted"):
                print("\n🎉 New model promoted to production!")
                print("   API will use new model on next request (or call /model/reload)")

        elif result["status"] == "failed":
            print(f"\n❌ Retraining failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def show_decisions(limit: int = 10) -> None:
    """Show recent promotion decisions.

    Args:
        limit: Maximum number of decisions to show.
    """
    from pipelines.retrain.service import RetrainingService

    print("=" * 60)
    print("Recent Promotion Decisions")
    print("=" * 60)

    service = RetrainingService()
    decisions = service.get_recent_decisions(limit=limit)

    if not decisions:
        print("No promotion decisions recorded yet.")
        return

    for d in decisions:
        status = "✅ Promoted" if d["promoted"] else "❌ Rejected"
        print(f"\n{d['timestamp'][:19]}")
        print(f"  Decision: {d['decision_id']}")
        print(f"  Challenger: v{d['challenger_version']}")
        print(f"  Trigger: {d['trigger_type']}")
        print(f"  Result: {status}")
        print(f"  Metric Improvement: {d['primary_metric_improvement']:.4f}")
        print(f"  Reason: {d['promotion_reason'][:80]}...")


def run_prefect_flow(force: bool = False) -> None:
    """Run retraining via Prefect flow.

    Args:
        force: Force retraining.
    """
    if force:
        from pipelines.retrain.flows import force_retrain_flow
        result = force_retrain_flow()
    else:
        from pipelines.retrain.flows import drift_triggered_retrain_flow
        result = drift_triggered_retrain_flow()

    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run PMMDS automated retraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/retrain.py              # Run drift-triggered retraining
  python scripts/retrain.py --force      # Force retraining
  python scripts/retrain.py --decisions  # Show recent decisions
  python scripts/retrain.py --prefect    # Run via Prefect flow
        """,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of drift threshold",
    )
    parser.add_argument(
        "--decisions",
        action="store_true",
        help="Show recent promotion decisions",
    )
    parser.add_argument(
        "--prefect",
        action="store_true",
        help="Run via Prefect flow",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        choices=["drift", "scheduled", "manual"],
        help="Trigger type (default: manual)",
    )

    args = parser.parse_args()

    if args.decisions:
        show_decisions()
    elif args.prefect:
        run_prefect_flow(force=args.force)
    else:
        run_retraining(
            trigger_type=args.trigger,
            force=args.force,
        )


if __name__ == "__main__":
    main()
