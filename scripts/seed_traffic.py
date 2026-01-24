#!/usr/bin/env python3
"""Seed traffic for NYC Yellow Taxi fare prediction API.

Generates realistic taxi trip requests to populate prediction logs.
Supports normal and drifted modes for testing drift detection.

Usage:
    python scripts/seed_traffic.py --count 100
    python scripts/seed_traffic.py --count 200 --drifted
"""

import argparse
import random
import sys
import time
from pathlib import Path

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.utils import get_logger, setup_logging

setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)

# Borough distributions (normal traffic)
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]
PICKUP_WEIGHTS = [0.55, 0.15, 0.12, 0.08, 0.05, 0.05]
DROPOFF_WEIGHTS = [0.50, 0.18, 0.14, 0.08, 0.05, 0.05]

# Drifted borough distributions (more Manhattan)
PICKUP_WEIGHTS_DRIFTED = [0.75, 0.10, 0.07, 0.04, 0.02, 0.02]
DROPOFF_WEIGHTS_DRIFTED = [0.70, 0.12, 0.08, 0.05, 0.03, 0.02]


def generate_normal_request(rng: random.Random) -> dict:
    """Generate a normal taxi trip request."""
    trip_distance = max(0.1, min(100.0, rng.gauss(3.0, 3.0)))

    min_per_mile = rng.uniform(3.0, 5.0)
    base_duration = trip_distance * min_per_mile
    traffic_factor = rng.uniform(0.8, 1.5)
    trip_duration = max(1.0, min(180.0, base_duration * traffic_factor))

    hour_weights = [
        0.02, 0.01, 0.01, 0.01, 0.01, 0.02,
        0.04, 0.06, 0.08, 0.07, 0.06, 0.05,
        0.05, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
    ]
    pickup_hour = rng.choices(range(24), weights=hour_weights, k=1)[0]

    pickup_day = rng.randint(1, 7)
    is_weekend = 1 if pickup_day in [1, 7] else 0
    is_rush = 1 if (7 <= pickup_hour <= 9 or 16 <= pickup_hour <= 19) else 0
    passenger_count = rng.choices([1, 2, 3, 4, 5, 6], weights=[0.55, 0.22, 0.12, 0.06, 0.03, 0.02], k=1)[0]
    pickup_month = rng.choice([1, 2])
    pickup_borough = rng.choices(BOROUGHS, weights=PICKUP_WEIGHTS, k=1)[0]
    dropoff_borough = rng.choices(BOROUGHS, weights=DROPOFF_WEIGHTS, k=1)[0]
    rate_code = rng.choices([1, 2, 3, 4, 5, 6], weights=[0.95, 0.03, 0.005, 0.005, 0.005, 0.005], k=1)[0]
    payment_type = rng.choices([1, 2, 3, 4], weights=[0.60, 0.35, 0.03, 0.02], k=1)[0]

    return {
        "trip_distance": round(trip_distance, 2),
        "passenger_count": passenger_count,
        "pickup_hour": pickup_hour,
        "pickup_day_of_week": pickup_day,
        "pickup_month": pickup_month,
        "trip_duration_minutes": round(trip_duration, 2),
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush,
        "RatecodeID": rate_code,
        "payment_type": payment_type,
        "pickup_borough": pickup_borough,
        "dropoff_borough": dropoff_borough,
    }


def generate_drifted_request(rng: random.Random) -> dict:
    """Generate a drifted taxi trip request (longer trips, evening-heavy, more Manhattan)."""
    trip_distance = max(0.1, min(100.0, rng.gauss(6.0, 4.0)))

    min_per_mile = rng.uniform(4.0, 7.0)
    base_duration = trip_distance * min_per_mile
    traffic_factor = rng.uniform(1.0, 2.0)
    trip_duration = max(1.0, min(180.0, base_duration * traffic_factor))

    hour_weights = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.02, 0.03, 0.03, 0.03, 0.03, 0.04,
        0.04, 0.04, 0.04, 0.05, 0.07, 0.09,
        0.10, 0.10, 0.08, 0.06, 0.04, 0.03,
    ]
    pickup_hour = rng.choices(range(24), weights=hour_weights, k=1)[0]

    pickup_day = rng.randint(1, 7)
    is_weekend = 1 if pickup_day in [1, 7] else 0
    is_rush = 1 if (7 <= pickup_hour <= 9 or 16 <= pickup_hour <= 19) else 0
    passenger_count = rng.choices([1, 2, 3, 4, 5, 6], weights=[0.35, 0.25, 0.18, 0.12, 0.06, 0.04], k=1)[0]
    pickup_month = rng.choice([1, 2])
    pickup_borough = rng.choices(BOROUGHS, weights=PICKUP_WEIGHTS_DRIFTED, k=1)[0]
    dropoff_borough = rng.choices(BOROUGHS, weights=DROPOFF_WEIGHTS_DRIFTED, k=1)[0]
    rate_code = rng.choices([1, 2, 3, 4, 5, 6], weights=[0.85, 0.10, 0.02, 0.01, 0.01, 0.01], k=1)[0]
    payment_type = rng.choices([1, 2, 3, 4], weights=[0.75, 0.20, 0.03, 0.02], k=1)[0]

    return {
        "trip_distance": round(trip_distance, 2),
        "passenger_count": passenger_count,
        "pickup_hour": pickup_hour,
        "pickup_day_of_week": pickup_day,
        "pickup_month": pickup_month,
        "trip_duration_minutes": round(trip_duration, 2),
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush,
        "RatecodeID": rate_code,
        "payment_type": payment_type,
        "pickup_borough": pickup_borough,
        "dropoff_borough": dropoff_borough,
    }


def main(
    count: int = 100,
    api_url: str = "http://localhost:8000",
    drifted: bool = False,
    seed: int = 42,
    delay: float = 0.05,
) -> None:
    """Send synthetic taxi trip prediction requests."""
    rng = random.Random(seed)
    endpoint = f"{api_url}/api/v1/predict"

    mode = "DRIFTED" if drifted else "NORMAL"
    logger.info("seed_traffic_starting", count=count, mode=mode, api_url=api_url)

    successes = 0
    failures = 0
    fares: list[float] = []
    latencies: list[float] = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(count):
            features = generate_drifted_request(rng) if drifted else generate_normal_request(rng)
            payload = {"features": features}

            try:
                response = client.post(endpoint, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    fare = data["predicted_fare"]
                    latency = data["latency_ms"]
                    fares.append(fare)
                    latencies.append(latency)
                    successes += 1

                    if (i + 1) % 20 == 0:
                        logger.info(
                            "progress",
                            completed=i + 1,
                            total=count,
                            avg_fare=round(sum(fares) / len(fares), 2),
                            avg_latency_ms=round(sum(latencies) / len(latencies), 2),
                        )
                else:
                    failures += 1
                    if failures <= 3:
                        logger.warning("request_failed", status=response.status_code, detail=response.text[:200])

            except Exception as e:
                failures += 1
                if failures <= 3:
                    logger.error("request_error", error=str(e))

            time.sleep(delay)

    if fares:
        logger.info(
            "seed_traffic_complete",
            mode=mode,
            total=count,
            successes=successes,
            failures=failures,
            avg_fare=round(sum(fares) / len(fares), 2),
            min_fare=round(min(fares), 2),
            max_fare=round(max(fares), 2),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2),
        )
    else:
        logger.error("seed_traffic_failed", total=count, successes=0, failures=failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed prediction traffic")
    parser.add_argument("--count", type=int, default=100, help="Number of requests")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--drifted", action="store_true", help="Use drifted distribution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between requests")
    args = parser.parse_args()

    main(count=args.count, api_url=args.api_url, drifted=args.drifted, seed=args.seed, delay=args.delay)
