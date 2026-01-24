# PMMDS - Production ML Monitoring & Drift Detection System

[![CI](https://github.com/example/pmmds/workflows/CI/badge.svg)](https://github.com/example/pmmds/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade ML monitoring system** demonstrating drift detection, automated retraining, and model lifecycle management. Built to showcase FAANG-level MLE skills.

## Key Features

| Feature | Description |
|---------|-------------|
| **PySpark MLlib Training** | GBTRegressor pipeline on NYC Yellow Taxi data (~500K rows) |
| **Real-time Inference** | FastAPI service with MLflow pyfunc model serving |
| **Data Validation** | Schema + range validation for training & inference |
| **Drift Detection** | PSI-based feature drift with configurable thresholds |
| **Automated Retraining** | Champion/challenger comparison on RMSE/MAE/R2 |
| **Model Registry** | MLflow for versioning and promotion |
| **Observability** | Prometheus metrics + structured JSON logging |
| **Orchestration** | Prefect flows for all pipelines |

## Dataset

**NYC TLC Yellow Taxi Trip Records** (2023-01, 2023-02)

- Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Task: **Fare amount regression** (predict `fare_amount` from trip features)
- Processing: PySpark reads raw Parquet, extracts time features, maps LocationIDs to boroughs, filters outliers
- Size: ~500K rows after sampling and filtering

### Features (12)

| Feature | Type | Description |
|---------|------|-------------|
| `trip_distance` | Numeric | Trip distance in miles |
| `passenger_count` | Numeric | Number of passengers |
| `pickup_hour` | Numeric | Hour of pickup (0-23) |
| `pickup_day_of_week` | Numeric | Day of week (1=Mon, 7=Sun) |
| `pickup_month` | Numeric | Month of pickup (1-12) |
| `trip_duration_minutes` | Numeric | Trip duration in minutes |
| `is_weekend` | Binary | Weekend indicator (0/1) |
| `is_rush_hour` | Binary | Rush hour indicator (0/1) |
| `RatecodeID` | Categorical | Rate code (1=Standard, 2=JFK, etc.) |
| `payment_type` | Categorical | Payment method (1=Credit, 2=Cash, etc.) |
| `pickup_borough` | Categorical | Pickup borough (Manhattan, Brooklyn, etc.) |
| `dropoff_borough` | Categorical | Dropoff borough |

**Target**: `fare_amount` (continuous, USD, range $2.50-$200)

## Quick Start

### 1. Start Infrastructure

```bash
git clone https://github.com/aksh-ay06/pmmds.git
cd pmmds/infra/compose
docker compose up -d
docker compose ps
```

### 2. Download Data & Train Model

```bash
# Download NYC TLC Parquet and process with PySpark
make download-data

# Train GBTRegressor and register in MLflow
make train
```

### 3. Test the API

```bash
# Check health
curl http://localhost:8000/healthz

# Make a fare prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "trip_distance": 3.5,
      "passenger_count": 2,
      "pickup_hour": 14,
      "pickup_day_of_week": 3,
      "pickup_month": 1,
      "trip_duration_minutes": 15.0,
      "is_weekend": 0,
      "is_rush_hour": 0,
      "RatecodeID": 1,
      "payment_type": 1,
      "pickup_borough": "Manhattan",
      "dropoff_borough": "Brooklyn"
    }
  }'
```

Response:
```json
{
  "request_id": "abc123",
  "predicted_fare": 14.50,
  "model_name": "nyc-taxi-fare",
  "model_version": "1",
  "latency_ms": 12.5
}
```

### 4. View MLflow UI

Open http://localhost:5000 to see experiments, runs, and registered models.

## Architecture

```
  +-------------+         +-----------------------------------------------+
  |   Client    |         |          FastAPI Service (:8000)               |
  |  (requests) |-------->|  /predict   /healthz   /metrics   /model      |
  +-------------+         +-------------------+---------------------------+
                                              |
            +---------------------------------+---------------------------------+
            |                                 |                                 |
            v                                 v                                 v
  +-------------------+         +-------------------+         +-------------------+
  |   PostgreSQL      |         |     MLflow        |         |    Prefect        |
  |    (:5432)        |         |    (:5000)        |         |  (Orchestration)  |
  |                   |         |                   |         |                   |
  | - prediction_logs |         | - Experiments     |         | - train_flow      |
  | - drift_metrics   |         | - Model Registry  |         | - monitor_flow    |
  | - promotions      |         | - Spark Artifacts |         | - retrain_flow    |
  +-------------------+         +-------------------+         +-------------------+

                          +-------------------------------------+
                          |        Monitoring Pipeline          |
                          |                                     |
                          |  Reference Data -> Compare with     |
                          |  Recent Inference -> PSI Drift      |
                          |  -> Alert if threshold exceeded     |
                          +-------------------------------------+
                                         |
                                         v
                          +-------------------------------------+
                          |        Retraining Pipeline          |
                          |                                     |
                          |  Drift Triggered -> Train New GBT   |
                          |  -> Compare RMSE vs Champion        |
                          |  -> Promote if improved >= 0.5      |
                          +-------------------------------------+
```

### Training Pipeline (PySpark MLlib)

```
Raw TLC Parquet -> PySpark Processing -> Feature Engineering
                                              |
                                              v
                    StringIndexer -> OneHotEncoder -> VectorAssembler
                                              |
                                              v
                                      GBTRegressor
                                              |
                                              v
                              mlflow.spark.log_model()
                                              |
                                              v
                            MLflow Registry: "nyc-taxi-fare"
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI inference service (MLflow pyfunc) |
| MLflow | 5000 | Model tracking & registry |
| PostgreSQL | 5432 | Metadata store |

## Project Structure

```
pmmds/
├── apps/
│   ├── api/           # FastAPI inference service
│   │   ├── models/    # Model loading (MLflow pyfunc)
│   │   ├── routes/    # /predict endpoint
│   │   ├── schemas/   # Pydantic request/response
│   │   └── db/        # SQLAlchemy models
│   └── monitor/       # Drift monitoring service
├── pipelines/
│   ├── train/         # PySpark MLlib training pipeline
│   │   ├── preprocessing.py  # StringIndexer + OneHotEncoder + VectorAssembler
│   │   └── trainer.py        # GBTRegressor + MLflow logging
│   └── retrain/       # Champion/challenger comparison
├── scripts/           # CLI scripts (download, train, seed, monitor, retrain)
├── shared/
│   ├── config/        # Configuration management
│   ├── data/          # Dataset utilities + location mappings
│   ├── drift/         # PSI/KL/JS drift metrics
│   ├── utils/         # Logging and metrics utilities
│   └── validation/    # Feature validation (ranges, categories)
├── infra/
│   ├── docker/        # Dockerfiles (includes JDK for PySpark)
│   ├── compose/       # Docker Compose configs
│   └── sql/           # Database init + migrations
└── tests/             # Test suite
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check with DB status |
| `/ready` | GET | Readiness probe |
| `/api/v1/predict` | POST | Generate fare prediction |
| `/metrics` | GET | Prometheus metrics |
| `/metrics/json` | GET | JSON metrics |
| `/model` | GET | Current model info |
| `/docs` | GET | OpenAPI documentation |

## Data Validation

PMMDS validates data at both training and inference time.

### Inference Payload Validation

Every prediction request is validated before inference:

| Check | Description | HTTP Response |
|-------|-------------|---------------|
| Missing features | All 12 features must be present | 422 |
| Invalid borough | Must be Manhattan/Brooklyn/Queens/Bronx/Staten Island | 422 |
| Invalid numeric range | trip_distance 0.1-100, duration 1-180, etc. | 422 |
| Invalid categorical | RatecodeID 1-6, payment_type 1-4 | 422 |
| Extra fields | Strict schema rejects unknown fields | 422 |

### Numeric Constraints

| Feature | Min | Max |
|---------|-----|-----|
| trip_distance | 0.1 | 100.0 |
| passenger_count | 1 | 9 |
| pickup_hour | 0 | 23 |
| pickup_day_of_week | 1 | 7 |
| pickup_month | 1 | 12 |
| trip_duration_minutes | 1.0 | 180.0 |
| fare_amount (target) | 2.5 | 200.0 |

## Drift Monitoring

PMMDS monitors feature and prediction drift by comparing inference data against the training (reference) distribution.

### Drift Metrics

**Population Stability Index (PSI)** with n_bins=20 for continuous fare values:

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | None |
| 0.1 - 0.2 | Moderate change | Monitor |
| 0.2 - 0.25 | Significant change | Investigate |
| >= 0.25 | Major change | Action required |

Additional metrics: KL Divergence, JS Divergence

### Running Drift Detection

```bash
make seed-traffic                                    # Generate normal traffic
python scripts/seed_traffic.py --drift --count 200   # Generate drifted traffic
make monitor                                         # Detect drift
```

### Drift Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PMMDS_DRIFT_PSI_THRESHOLD` | 0.2 | PSI threshold per feature |
| `PMMDS_DRIFT_MIN_DRIFT_FEATURES` | 3 | Min features to trigger alert |
| `PMMDS_DRIFT_CURRENT_WINDOW_HOURS` | 24 | Recent data window |
| `PMMDS_DRIFT_MIN_SAMPLES_REQUIRED` | 100 | Minimum samples needed |

## Automated Retraining

When drift exceeds thresholds, PMMDS retrains a challenger GBTRegressor and compares against the champion.

### Promotion Criteria

All three must be met:

| Criterion | Requirement | Description |
|-----------|-------------|-------------|
| **Validation** | Must pass | Training data validation |
| **RMSE Improvement** | >= 0.5 lower | Challenger RMSE must beat champion by at least 0.5 |
| **Latency** | <= 20% slower | No significant regression |

### Metrics Compared

| Metric | Role | Direction |
|--------|------|-----------|
| RMSE | Primary | Lower is better |
| MAE | Secondary | Lower is better |
| R2 | Secondary | Higher is better |
| MAPE | Secondary | Lower is better |

### Running Retraining

```bash
make retrain                           # Check drift and retrain if needed
python scripts/retrain.py --force      # Force retraining
python scripts/retrain.py --decisions  # View recent decisions
```

## Model Lifecycle

```
+----------+     +----------+     +----------+     +----------+
| Training |---->| Registry |---->| Staging  |---->|Production|
| (PySpark)|     | (MLflow) |     | (Compare)|     | (Alias)  |
+----------+     +----------+     +----------+     +----------+
                      |                                  ^
                      |                                  |
                      +----------------------------------+
                              Promotion Decision
```

- **Training**: GBTRegressor trained via PySpark MLlib, logged with `mlflow.spark.log_model()`
- **Registry**: Model version registered as "nyc-taxi-fare" in MLflow
- **Staging**: Challenger compared against champion on RMSE/MAE/R2
- **Production**: Model aliased as "production", served via `mlflow.pyfunc.load_model()`

## Observability

### Metrics Endpoints

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/metrics` | Prometheus | Prometheus scrape endpoint |
| `/metrics/json` | JSON | Human-readable metrics |

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `pmmds_requests_total` | Counter | Total HTTP requests |
| `pmmds_predictions_total` | Counter | Total predictions |
| `pmmds_prediction_latency_seconds` | Histogram | Model inference latency |
| `pmmds_drift_events_total` | Counter | Drift detection events |
| `pmmds_retraining_runs_total` | Counter | Retraining runs |

### Structured Logging

All logs are JSON-formatted:

```json
{
  "event": "prediction_completed",
  "timestamp": "2025-01-20T10:15:30.123456+00:00",
  "level": "info",
  "request_id": "abc123",
  "predicted_fare": 14.50,
  "latency_ms": 12.5,
  "model_name": "nyc-taxi-fare",
  "model_version": "1"
}
```

## Operational Runbook

### Common Commands

```bash
make up              # Start all services
make down            # Stop all services
make download-data   # Download + process TLC data with PySpark
make train           # Train GBTRegressor + register in MLflow
make seed-traffic    # Generate synthetic taxi predictions
make monitor         # Run drift monitoring
make retrain         # Trigger retraining (if drift detected)
make test            # Run tests
make logs            # View logs
```

### Health Checks

```bash
curl http://localhost:8000/healthz          # API health
curl http://localhost:8000/model            # Current model info
docker exec pmmds-postgres pg_isready -U pmmds  # DB health
curl http://localhost:5000/health           # MLflow health
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| API returns 503 | DB connection failed | Check PostgreSQL: `docker compose ps` |
| Model not found | MLflow model not registered | Run `make train` |
| Drift not detected | Insufficient data | Run `make seed-traffic` |
| Spark startup slow | JVM cold start | First request warms SparkSession |
| High latency | Model cold start | Pre-warmed at API startup |

## MLE Skills Demonstrated

| Skill | Implementation |
|-------|----------------|
| **ML Systems Design** | End-to-end pipeline: training -> serving -> monitoring -> retraining |
| **Big Data Processing** | PySpark MLlib for training on ~500K taxi records |
| **Model Serving** | MLflow pyfunc with pre-warmed SparkSession |
| **Feature/Prediction Drift** | PSI-based drift detection with configurable thresholds |
| **Model Registry** | MLflow with versioning, aliases, and promotion workflow |
| **Automated Retraining** | Champion/challenger comparison on RMSE/MAE/R2 |
| **Observability** | Prometheus metrics, structured logging, health endpoints |
| **Infrastructure as Code** | Docker Compose for reproducible local development |

### Resume Bullets

```
- Designed production ML monitoring system with PSI-based drift detection,
  triggering automated retraining when >=3 features exceed threshold (PSI > 0.2)

- Built PySpark MLlib training pipeline (GBTRegressor) on NYC Yellow Taxi data
  (~500K records), achieving competitive RMSE with champion/challenger promotion

- Implemented automated model promotion with objective criteria: RMSE improvement
  >= 0.5, validation pass, and latency constraint (<= 20% regression)

- Deployed end-to-end MLOps pipeline using MLflow model registry, Prefect
  orchestration, and PySpark for both training and feature engineering
```

## Development

```bash
pip install -e ".[dev]"         # Install with dev deps
pytest                          # Run tests
ruff check .                    # Lint
black .                         # Format
mypy apps shared pipelines      # Type check
```

## License

MIT
