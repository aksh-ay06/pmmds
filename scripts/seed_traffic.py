#!/usr/bin/env python3
"""Generate sample prediction traffic for testing.

This script sends sample prediction requests to the API
to populate the prediction_logs table for drift detection testing.

Usage:
    python scripts/seed_traffic.py              # Generate 100 requests
    python scripts/seed_traffic.py --count 500  # Generate 500 requests
    python scripts/seed_traffic.py --drift      # Generate drifted data
"""

import argparse
import random
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Sample data distributions (based on Telco Churn)
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


def generate_normal_sample() -> dict:
    """Generate a sample with normal (training-like) distribution."""
    # Generate realistic tenure (skewed toward lower values)
    tenure = max(0, int(random.gauss(32, 24)))
    tenure = min(tenure, 72)

    # Monthly charges (somewhat correlated with services)
    monthly_charges = round(random.gauss(64.76, 30.09), 2)
    monthly_charges = max(18.0, min(monthly_charges, 118.75))

    # Total charges (correlated with tenure)
    total_charges = round(monthly_charges * tenure + random.gauss(0, 200), 2)
    total_charges = max(18.0, total_charges)

    return {
        "gender": random.choice(CATEGORICAL_VALUES["gender"]),
        "senior_citizen": random.choice([0, 0, 0, 1]),  # ~16% seniors
        "partner": random.choices(CATEGORICAL_VALUES["partner"], weights=[0.48, 0.52])[0],
        "dependents": random.choices(
            CATEGORICAL_VALUES["dependents"], weights=[0.30, 0.70]
        )[0],
        "tenure": tenure,
        "phone_service": random.choices(
            CATEGORICAL_VALUES["phone_service"], weights=[0.90, 0.10]
        )[0],
        "multiple_lines": random.choice(CATEGORICAL_VALUES["multiple_lines"]),
        "internet_service": random.choices(
            CATEGORICAL_VALUES["internet_service"], weights=[0.34, 0.44, 0.22]
        )[0],
        "online_security": random.choice(CATEGORICAL_VALUES["online_security"]),
        "online_backup": random.choice(CATEGORICAL_VALUES["online_backup"]),
        "device_protection": random.choice(CATEGORICAL_VALUES["device_protection"]),
        "tech_support": random.choice(CATEGORICAL_VALUES["tech_support"]),
        "streaming_tv": random.choice(CATEGORICAL_VALUES["streaming_tv"]),
        "streaming_movies": random.choice(CATEGORICAL_VALUES["streaming_movies"]),
        "contract": random.choices(
            CATEGORICAL_VALUES["contract"], weights=[0.55, 0.21, 0.24]
        )[0],
        "paperless_billing": random.choices(
            CATEGORICAL_VALUES["paperless_billing"], weights=[0.59, 0.41]
        )[0],
        "payment_method": random.choices(
            CATEGORICAL_VALUES["payment_method"], weights=[0.34, 0.23, 0.22, 0.21]
        )[0],
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    }


def generate_drifted_sample() -> dict:
    """Generate a sample with drifted distribution.

    Simulates distribution shift in:
    - tenure: shifts higher (more long-term customers)
    - monthly_charges: shifts higher
    - contract: more month-to-month
    """
    # Drifted tenure (shifted higher)
    tenure = max(0, int(random.gauss(45, 20)))  # Higher mean
    tenure = min(tenure, 72)

    # Drifted monthly charges (shifted higher)
    monthly_charges = round(random.gauss(85.0, 25.0), 2)  # Higher mean
    monthly_charges = max(30.0, min(monthly_charges, 130.0))

    # Total charges (correlated with drifted values)
    total_charges = round(monthly_charges * tenure + random.gauss(0, 300), 2)
    total_charges = max(30.0, total_charges)

    return {
        "gender": random.choice(CATEGORICAL_VALUES["gender"]),
        "senior_citizen": random.choice([0, 0, 1, 1]),  # More seniors (drift)
        "partner": random.choices(CATEGORICAL_VALUES["partner"], weights=[0.48, 0.52])[0],
        "dependents": random.choices(
            CATEGORICAL_VALUES["dependents"], weights=[0.30, 0.70]
        )[0],
        "tenure": tenure,
        "phone_service": random.choices(
            CATEGORICAL_VALUES["phone_service"], weights=[0.90, 0.10]
        )[0],
        "multiple_lines": random.choice(CATEGORICAL_VALUES["multiple_lines"]),
        "internet_service": random.choices(
            CATEGORICAL_VALUES["internet_service"], weights=[0.20, 0.60, 0.20]  # More fiber
        )[0],
        "online_security": random.choice(CATEGORICAL_VALUES["online_security"]),
        "online_backup": random.choice(CATEGORICAL_VALUES["online_backup"]),
        "device_protection": random.choice(CATEGORICAL_VALUES["device_protection"]),
        "tech_support": random.choice(CATEGORICAL_VALUES["tech_support"]),
        "streaming_tv": random.choice(CATEGORICAL_VALUES["streaming_tv"]),
        "streaming_movies": random.choice(CATEGORICAL_VALUES["streaming_movies"]),
        "contract": random.choices(
            CATEGORICAL_VALUES["contract"], weights=[0.75, 0.15, 0.10]  # More month-to-month
        )[0],
        "paperless_billing": random.choices(
            CATEGORICAL_VALUES["paperless_billing"], weights=[0.75, 0.25]  # More paperless
        )[0],
        "payment_method": random.choices(
            CATEGORICAL_VALUES["payment_method"], weights=[0.50, 0.15, 0.20, 0.15]  # More electronic
        )[0],
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    }


def send_prediction(
    base_url: str,
    features: dict,
    request_id: str | None = None,
) -> dict | None:
    """Send prediction request to API.

    Args:
        base_url: API base URL.
        features: Feature dictionary.
        request_id: Optional request ID.

    Returns:
        Response dict or None on failure.
    """
    request_id = request_id or str(uuid4())

    payload = {
        "request_id": request_id,
        "features": features,
    }

    try:
        response = httpx.post(
            f"{base_url}/api/v1/predict",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate sample prediction traffic",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of requests to generate (default: 100)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Generate drifted data distribution",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between requests in seconds (default: 0.05)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PMMDS Traffic Generator")
    print("=" * 60)
    print(f"Target: {args.url}")
    print(f"Count: {args.count}")
    print(f"Mode: {'DRIFTED' if args.drift else 'NORMAL'}")
    print("-" * 60)

    success = 0
    failed = 0
    predictions = {"0": 0, "1": 0}

    for i in range(args.count):
        # Generate sample
        if args.drift:
            features = generate_drifted_sample()
        else:
            features = generate_normal_sample()

        # Send request
        result = send_prediction(args.url, features)

        if result:
            success += 1
            pred = str(result.get("prediction", "?"))
            predictions[pred] = predictions.get(pred, 0) + 1
        else:
            failed += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{args.count} (success: {success}, failed: {failed})")

        time.sleep(args.delay)

    print("-" * 60)
    print(f"Complete! Success: {success}, Failed: {failed}")
    print(f"Prediction distribution: {predictions}")

    if failed > 0:
        print("\n⚠️  Some requests failed. Make sure the API is running:")
        print("   make up && sleep 5")


if __name__ == "__main__":
    main()
