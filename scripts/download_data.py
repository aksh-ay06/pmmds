#!/usr/bin/env python3
"""Download and prepare the NYC Yellow Taxi dataset using PySpark.

Dataset: NYC TLC Yellow Taxi Trip Records (Parquet format)
Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Features: trip distance, duration, time features, location boroughs, fare

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output-dir data
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType

from shared.data.locations import LOCATION_TO_BOROUGH
from shared.utils import get_logger, setup_logging

setup_logging(log_level="INFO", json_format=False)
logger = get_logger(__name__)

# TLC Yellow Taxi Parquet URLs (2023 data for stability)
TAXI_URLS = [
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
]

EXPECTED_MIN_ROWS = 2_000_000


def create_spark_session() -> SparkSession:
    """Create a local Spark session for data processing."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("pmmds-data-prep")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .getOrCreate()
    )


def download_and_process(spark: SparkSession, output_dir: Path) -> tuple[Path, Path, Path]:
    """Download TLC Parquet data and process with PySpark.

    Args:
        spark: SparkSession instance.
        output_dir: Base output directory.

    Returns:
        Tuple of (train_path, test_path, reference_path).
    """
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download raw parquet files
    import urllib.request

    parquet_paths = []
    for url in TAXI_URLS:
        filename = url.split("/")[-1]
        local_path = raw_dir / filename
        if not local_path.exists():
            logger.info("downloading_parquet", url=url)
            urllib.request.urlretrieve(url, str(local_path))
            logger.info("downloaded", path=str(local_path))
        else:
            logger.info("using_cached", path=str(local_path))
        parquet_paths.append(str(local_path))

    # Read each file separately and normalize types before union
    # (TLC parquet files have inconsistent types for passenger_count across months)
    logger.info("reading_parquet_with_spark", files=len(parquet_paths))
    from pyspark.sql.types import DoubleType
    from functools import reduce

    dfs = []
    for p in parquet_paths:
        file_df = spark.read.parquet(p)
        file_df = file_df.withColumn("passenger_count", F.col("passenger_count").cast(DoubleType()))
        dfs.append(file_df)
    df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)
    initial_count = df.count()
    logger.info("raw_rows_loaded", count=initial_count)

    # Create borough mapping UDF
    borough_map = F.create_map(
        *[item for k, v in LOCATION_TO_BOROUGH.items() for item in (F.lit(k), F.lit(v))]
    )

    # Feature engineering
    df = (
        df
        # Compute trip duration in minutes
        .withColumn(
            "trip_duration_minutes",
            (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
        )
        # Extract time features
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime").cast(IntegerType()))
        .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime").cast(IntegerType()))
        .withColumn("pickup_month", F.month("tpep_pickup_datetime").cast(IntegerType()))
        # Binary features
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("tpep_pickup_datetime").isin([1, 7]), 1).otherwise(0).cast(IntegerType()),
        )
        .withColumn(
            "is_rush_hour",
            F.when(
                (F.hour("tpep_pickup_datetime").between(7, 9))
                | (F.hour("tpep_pickup_datetime").between(16, 19)),
                1,
            ).otherwise(0).cast(IntegerType()),
        )
        # Map location IDs to boroughs
        .withColumn("pickup_borough", borough_map[F.col("PULocationID")])
        .withColumn("dropoff_borough", borough_map[F.col("DOLocationID")])
        # Cast types
        .withColumn("passenger_count", F.col("passenger_count").cast(IntegerType()))
        .withColumn("trip_distance", F.col("trip_distance").cast(FloatType()))
        .withColumn("fare_amount", F.col("fare_amount").cast(FloatType()))
        .withColumn("RatecodeID", F.col("RatecodeID").cast(IntegerType()))
        .withColumn("payment_type", F.col("payment_type").cast(IntegerType()))
    )

    # Filter outliers
    df = df.filter(
        (F.col("trip_distance") >= 0.1) & (F.col("trip_distance") <= 100)
        & (F.col("fare_amount") >= 2.5) & (F.col("fare_amount") <= 200)
        & (F.col("trip_duration_minutes") >= 1) & (F.col("trip_duration_minutes") <= 180)
        & (F.col("passenger_count") >= 1) & (F.col("passenger_count") <= 6)
        & (F.col("pickup_borough").isNotNull())
        & (F.col("dropoff_borough").isNotNull())
        & (F.col("RatecodeID").between(1, 6))
        & (F.col("payment_type").between(1, 4))
    )

    filtered_count = df.count()
    logger.info("after_filtering", count=filtered_count, removed=initial_count - filtered_count)

    # Select only the features and target we need
    feature_columns = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "trip_duration_minutes",
        "is_weekend",
        "is_rush_hour",
        "RatecodeID",
        "payment_type",
        "pickup_borough",
        "dropoff_borough",
        "fare_amount",
    ]
    df = df.select(feature_columns)

    # Drop any remaining nulls
    df = df.dropna()

    # Sample down to ~500K for manageable training
    sample_fraction = min(1.0, 500_000 / max(df.count(), 1))
    if sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=42)

    final_count = df.count()
    logger.info("final_dataset_size", count=final_count)

    # Split 80/20
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Write as parquet
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    # Convert to pandas and write (simpler for small-medium datasets)
    train_pd = train_df.toPandas()
    test_pd = test_df.toPandas()

    train_pd.to_parquet(str(train_path), index=False)
    test_pd.to_parquet(str(test_path), index=False)

    logger.info(
        "dataset_prepared",
        train_rows=len(train_pd),
        test_rows=len(test_pd),
        train_path=str(train_path),
        test_path=str(test_path),
    )

    # Create reference CSV for drift monitoring (sample of training data)
    reference_path = processed_dir / "reference.csv"
    reference_sample = train_pd.sample(n=min(10_000, len(train_pd)), random_state=42)
    reference_sample.to_csv(str(reference_path), index=False)
    logger.info("reference_data_saved", path=str(reference_path), rows=len(reference_sample))

    return train_path, test_path, reference_path


def main(output_dir: str = "data") -> None:
    """Main download and preparation workflow.

    Args:
        output_dir: Base output directory.
    """
    base_dir = Path(output_dir)

    # Check if processed data already exists
    processed_dir = base_dir / "processed"
    if (processed_dir / "train.parquet").exists() and (processed_dir / "test.parquet").exists():
        logger.info("processed_data_exists", path=str(processed_dir))
        logger.info("hint", message="Delete data/processed/ to re-download")
        return

    spark = create_spark_session()
    try:
        train_path, test_path, ref_path = download_and_process(spark, base_dir)
        logger.info(
            "data_preparation_complete",
            train_path=str(train_path),
            test_path=str(test_path),
            reference_path=str(ref_path),
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare NYC Yellow Taxi dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for data files",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir)
