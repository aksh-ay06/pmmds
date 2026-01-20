# PMMDS - Production ML Monitoring & Drift Detection System

[![CI](https://github.com/example/pmmds/workflows/CI/badge.svg)](https://github.com/example/pmmds/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade ML monitoring system** demonstrating drift detection, automated retraining, and model lifecycle management. Built to showcase FAANG-level MLE skills.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Real-time Inference** | FastAPI service with <50ms P95 latency |
| **Data Validation** | Great Expectations for training & inference |
| **Drift Detection** | PSI-based feature drift with configurable thresholds |
| **Automated Retraining** | Champion/challenger model comparison |
| **Model Registry** | MLflow for versioning and promotion |
| **Observability** | Prometheus metrics + structured JSON logging |
| **Orchestration** | Prefect flows for all pipelines |

## 🚀 Quick Start

### 1. Start Infrastructure

```bash
# Clone and start all services
git clone https://github.com/example/pmmds.git
cd pmmds/infra/compose
docker compose up -d

# Verify services are running
docker compose ps
```

### 2. Train Initial Model

```bash
# Download and prepare Telco Churn dataset
make download-data

# Train model and register in MLflow
make train
```

### 3. Test the API

```bash
# Check health
curl http://localhost:8000/healthz

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "gender": "Male",
      "senior_citizen": 0,
      "partner": "Yes",
      "dependents": "No",
      "tenure": 12,
      "contract": "Month-to-month",
      "paperless_billing": "Yes",
      "payment_method": "Electronic check",
      "monthly_charges": 70.35,
      "total_charges": 840.20,
      "phone_service": "Yes",
      "multiple_lines": "No",
      "internet_service": "Fiber optic",
      "online_security": "No",
      "online_backup": "No",
      "device_protection": "No",
      "tech_support": "No",
      "streaming_tv": "No",
      "streaming_movies": "No"
    }
  }'
```

### 4. View MLflow UI

Open http://localhost:5000 to see experiments, runs, and registered models.

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              PMMDS Architecture                             │
└────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐         ┌─────────────────────────────────────────────────┐
  │   Client    │         │              FastAPI Service (:8000)             │
  │  (requests) │────────▶│  /predict   /healthz   /metrics   /model        │
  └─────────────┘         └───────────────────┬───────────────────────────────┘
                                              │
            ┌─────────────────────────────────┼─────────────────────────────────┐
            │                                 │                                 │
            ▼                                 ▼                                 ▼
  ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
  │   PostgreSQL    │           │     MLflow      │           │    Prefect      │
  │    (:5432)      │           │    (:5000)      │           │  (Orchestration)│
  │                 │           │                 │           │                 │
  │ • prediction_logs│          │ • Experiments   │           │ • train_flow    │
  │ • drift_metrics │           │ • Model Registry│           │ • monitor_flow  │
  │ • promotions    │           │ • Artifacts     │           │ • retrain_flow  │
  └─────────────────┘           └─────────────────┘           └─────────────────┘

                          ┌─────────────────────────────────┐
                          │        Monitoring Pipeline       │
                          │                                 │
                          │  Reference Data → Compare with  │
                          │  Recent Inference → PSI Drift   │
                          │  → Alert if threshold exceeded  │
                          └─────────────────────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────────┐
                          │        Retraining Pipeline       │
                          │                                 │
                          │  Drift Triggered → Train New    │
                          │  → Compare vs Champion → Promote│
                          │  → Update Production Alias      │
                          └─────────────────────────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI inference service |
| MLflow | 5000 | Model tracking & registry |
| PostgreSQL | 5432 | Metadata store |

## Project Structure

```
pmm-drift-system/
├── apps/
│   ├── api/           # FastAPI inference service
│   └── monitor/       # Drift monitoring service
├── pipelines/
│   └── train/         # Model training pipeline
├── scripts/           # CLI scripts
├── shared/
│   ├── config/        # Configuration management
│   ├── data/          # Dataset utilities
│   ├── drift/         # Drift detection metrics
│   ├── utils/         # Logging and utilities
│   └── validation/    # Data validation
├── infra/
│   ├── docker/        # Dockerfiles
│   ├── compose/       # Docker Compose configs
│   └── sql/           # Database init scripts
├── tests/             # Test suite
└── docs/              # Documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check with DB status |
| `/ready` | GET | Readiness probe |
| `/api/v1/predict` | POST | Generate prediction |
| `/docs` | GET | OpenAPI documentation |

## Data Validation

PMMDS uses Great Expectations for comprehensive data validation at multiple stages.

### Training Data Validation

Before model training, the dataset is validated against predefined expectations:

- **Missing values**: Checks for nulls/NaNs in required columns
- **Type validation**: Ensures correct data types (categorical vs numeric)
- **Value constraints**: Categorical features must be in allowed value sets
- **Range validation**: Numeric features must be within expected ranges

Training validation results are logged as MLflow artifacts for auditability.

```python
# Training with validation (default)
from pipelines.train.trainer import train
train(experiment_name="churn-training")

# Disable validation for debugging
train(experiment_name="churn-training", validate_data=False)
```

### Inference Payload Validation

Every prediction request is validated before inference:

| Check | Description | HTTP Response |
|-------|-------------|---------------|
| Missing features | All required features must be present | 400 Bad Request |
| Invalid categorical | Values must be in allowed sets | 400 Bad Request |
| Invalid numeric | Must be numbers within expected ranges | 400 Bad Request |
| Unknown features | Extra features trigger warnings | 200 (with warnings logged) |

### Validation Error Response

When validation fails, the API returns HTTP 400 with error details:

```json
{
  "detail": {
    "message": "Inference payload validation failed",
    "errors": [
      "Missing required feature: tenure",
      "Invalid value for contract: 'Annual'. Expected one of: Month-to-month, One year, Two year"
    ],
    "warnings": []
  }
}
```

### Feature Expectations

**Categorical Features** (allowed values):
- `gender`: Male, Female
- `contract`: Month-to-month, One year, Two year
- `payment_method`: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
- `internet_service`: DSL, Fiber optic, No
- Binary (Yes/No): partner, dependents, phone_service, paperless_billing, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies
- Binary (Yes/No/No internet service, or Yes/No/No phone service): multiple_lines

**Numeric Features**:
- `tenure`: 0-100 (months)
- `monthly_charges`: 0-200 ($)
- `total_charges`: 0-10000 ($)
- `senior_citizen`: 0 or 1

## Drift Monitoring

PMMDS monitors feature and prediction drift by comparing inference data against the training (reference) distribution.

### Drift Metrics

**Population Stability Index (PSI)** is used to detect distribution shifts:

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | None |
| 0.1 - 0.2 | Moderate change | Monitor |
| 0.2 - 0.25 | Significant change | Investigate |
| ≥ 0.25 | Major change | Action required |

Additional metrics computed:
- **KL Divergence**: Asymmetric divergence measure
- **JS Divergence**: Symmetric, bounded divergence

### Running Drift Detection

```bash
# Run drift monitoring
make monitor

# With drifted traffic for testing
make seed-traffic        # Generate normal traffic
python scripts/seed_traffic.py --drift --count 200  # Generate drifted traffic
make monitor             # Detect drift
```

### Drift Thresholds (Configurable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PMMDS_DRIFT_PSI_THRESHOLD` | 0.2 | PSI threshold per feature |
| `PMMDS_DRIFT_MIN_DRIFT_FEATURES` | 3 | Min features to trigger alert |
| `PMMDS_DRIFT_CURRENT_WINDOW_HOURS` | 24 | Recent data window |
| `PMMDS_DRIFT_MIN_SAMPLES_REQUIRED` | 100 | Minimum samples needed |

### Drift Detection Flow

```
┌─────────────────┐     ┌─────────────────┐
│ Reference Data  │     │ Recent Inference│
│ (Training Set)  │     │ (24h Window)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
         ┌─────────────────────┐
         │  Compute PSI/KL/JS  │
         │  for each feature   │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ Check Thresholds    │
         │ ≥3 features > 0.2?  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ Store in Postgres   │
         │ Create Alert if     │
         │ drift detected      │
         └─────────────────────┘
```

### Database Tables

| Table | Purpose |
|-------|---------|
| `drift_metrics` | Stores drift detection results |
| `drift_alerts` | Tracks alerts and acknowledgments |
| `reference_datasets` | Reference dataset metadata |

### Scheduled Monitoring (Prefect)

```bash
# Install orchestration dependencies
pip install -e ".[orchestration]"

# Run via Prefect flow
python scripts/monitor.py --prefect

# Deploy scheduled flow (hourly)
python scripts/monitor.py --deploy --interval 1

# Start Prefect agent
prefect agent start -q default
```

## Automated Retraining

When drift exceeds thresholds (≥3 features with PSI > 0.2), PMMDS automatically retrains and promotes a new model.

### Model Promotion Workflow

```
┌─────────────────┐
│ Drift Detected  │
│ ≥3 features     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Train Challenger│
│ Model           │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Compare vs      │
│ Champion        │
└────────┬────────┘
         ▼
┌─────────────────┐    No    ┌─────────────────┐
│ Meets Promotion │────────▶ │ Keep Champion   │
│ Criteria?       │          │ Log Decision    │
└────────┬────────┘          └─────────────────┘
         │ Yes
         ▼
┌─────────────────┐
│ Promote to      │
│ Production      │
└─────────────────┘
```

### Promotion Criteria

All three must be met:

| Criterion | Requirement | Description |
|-----------|-------------|-------------|
| **Validation** | Must pass | Training data validation |
| **Metric Improvement** | ≥0.1% better | Primary metric (ROC-AUC) |
| **Latency** | ≤20% slower | No significant regression |

### Running Retraining

```bash
# Check drift and retrain if needed
make retrain

# Force retraining (bypass drift check)
python scripts/retrain.py --force

# View recent promotion decisions
python scripts/retrain.py --decisions

# Run via Prefect flow
python scripts/retrain.py --prefect
```

### API Model Refresh

After promotion, the API serves the new model:

```bash
# Check current model
curl http://localhost:8000/model

# Force model reload (after promotion)
curl -X POST http://localhost:8000/model/reload
```

### Database Tables

| Table | Purpose |
|-------|---------|
| `promotion_decisions` | Records all champion vs challenger comparisons |
| `retraining_runs` | Tracks retraining attempts and outcomes |

### Example Decision Log

```json
{
  "decision_id": "a1b2c3d4e5f6",
  "challenger_version": "3",
  "promoted": true,
  "primary_metric_improvement": 0.0042,
  "promotion_reason": "Challenger v3 outperforms champion v2. roc_auc: 0.8456 vs 0.8414 (+0.0042). Validation passed. Latency OK (0.95x)."
}
```

## Model Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Training │────▶│ Registry │────▶│ Staging  │────▶│Production│
│ Pipeline │     │ (MLflow) │     │ (Testing)│     │ (Alias)  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                      │                                  ▲
                      │                                  │
                      └──────────────────────────────────┘
                              Promotion Decision
```

- **Training**: Model trained and metrics logged to MLflow
- **Registry**: Model version registered in MLflow Model Registry
- **Staging**: Challenger model compared against champion
- **Production**: Model aliased as "production" in MLflow

## Observability

PMMDS provides comprehensive observability through metrics and structured logging.

### Metrics Endpoints

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/metrics` | Prometheus | Prometheus scrape endpoint |
| `/metrics/json` | JSON | Human-readable metrics |

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pmmds_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `pmmds_request_latency_seconds` | Histogram | method, endpoint | Request latency distribution |
| `pmmds_predictions_total` | Counter | model_name, model_version, prediction | Total predictions |
| `pmmds_prediction_latency_seconds` | Histogram | model_name, model_version | Model inference latency |
| `pmmds_errors_total` | Counter | endpoint, error_type | Total errors |
| `pmmds_validation_failures_total` | Counter | endpoint, failure_type | Validation failures |
| `pmmds_drift_events_total` | Counter | drift_type, severity | Drift detection events |
| `pmmds_drift_checks_total` | Counter | model_name | Drift checks performed |
| `pmmds_model_reloads_total` | Counter | model_name | Model reloads |
| `pmmds_retraining_runs_total` | Counter | trigger_type, outcome | Retraining runs |
| `pmmds_promotions_total` | Counter | from_version, to_version | Model promotions |
| `pmmds_uptime_seconds` | Gauge | - | Application uptime |

### Prometheus Integration

The API exposes metrics at `/metrics` in Prometheus format:

```bash
# Scrape metrics
curl http://localhost:8000/metrics

# Example output
# HELP pmmds_requests_total Total number of requests
# TYPE pmmds_requests_total counter
pmmds_requests_total{method="POST",endpoint="/api/v1/predict",status="200"} 1542
pmmds_requests_total{method="GET",endpoint="/healthz",status="200"} 89
```

### JSON Metrics

For debugging and dashboards:

```bash
curl http://localhost:8000/metrics/json | jq
```

### Structured Logging

All logs are JSON-formatted for easy parsing:

```json
{
  "event": "prediction_completed",
  "timestamp": "2025-01-20T10:15:30.123456+00:00",
  "level": "info",
  "logger": "apps.api.routes.predict",
  "request_id": "abc123",
  "prediction": 1,
  "probability": 0.85,
  "latency_ms": 12.5,
  "model_name": "churn-classifier",
  "model_version": "3"
}
```

### Docker Compose Labels

The API service includes Prometheus labels for service discovery:

```yaml
labels:
  - "prometheus.scrape=true"
  - "prometheus.port=8000"
  - "prometheus.path=/metrics"
```

## 📋 Operational Runbook

### Common Commands

```bash
# Start all services
make up

# Stop all services
make down

# Train initial model
make train

# Generate synthetic traffic
make seed-traffic

# Run drift monitoring
make monitor

# Trigger retraining (if needed)
make retrain

# Run tests
make test

# View logs
make logs
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| API returns 503 | DB connection failed | Check PostgreSQL is running: `docker compose ps` |
| Model not found | MLflow model not registered | Run `make train` to register model |
| Drift not detected | Insufficient data | Generate more traffic: `make seed-traffic` |
| Retraining fails | DB schema drift | Run migrations: `make migrate` |
| High latency | Model cold start | First request warms cache; subsequent are faster |

### Health Checks

```bash
# API health
curl http://localhost:8000/healthz

# Database connectivity
docker exec pmmds-postgres pg_isready -U pmmds

# MLflow health
curl http://localhost:5000/health

# View current model
curl http://localhost:8000/model
```

### Scaling Considerations

- **API**: Horizontally scalable (stateless). Deploy multiple replicas behind load balancer.
- **Database**: Consider connection pooling (PgBouncer) for high traffic.
- **MLflow**: Shared artifact storage (S3/GCS) for multi-node deployments.
- **Monitoring**: Batch drift checks during off-peak hours.

## 🎓 MLE Skills Demonstrated

This project demonstrates production ML engineering skills valued at FAANG companies:

| Skill | Implementation |
|-------|----------------|
| **ML Systems Design** | End-to-end pipeline: training → serving → monitoring → retraining |
| **Data Quality** | Great Expectations for schema validation at training & inference |
| **Model Serving** | FastAPI with async inference, <50ms P95 latency |
| **Feature/Prediction Drift** | PSI-based drift detection with configurable thresholds |
| **Model Registry** | MLflow with versioning, aliases, and promotion workflow |
| **Automated Retraining** | Champion/challenger comparison with objective metrics |
| **Observability** | Prometheus metrics, structured logging, health endpoints |
| **Infrastructure as Code** | Docker Compose for reproducible local development |
| **CI/CD** | GitHub Actions for lint, typecheck, test, smoke test |
| **Code Quality** | Type hints, docstrings, ruff, black, mypy |

### Resume Bullets

```
• Designed production ML monitoring system with PSI-based drift detection,
  triggering automated retraining when ≥3 features exceed threshold (PSI > 0.2)

• Implemented champion/challenger model promotion with objective criteria:
  validation pass, metric improvement (≥0.1%), and latency constraint (≤20% slower)

• Built FastAPI inference service achieving <50ms P95 latency with async
  database logging and Prometheus metrics exposition

• Deployed end-to-end MLOps pipeline using MLflow model registry,
  Prefect orchestration, and Great Expectations data validation
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=apps --cov=shared --cov-report=html

# Lint
ruff check .

# Format
black .

# Type check
mypy apps shared pipelines
```

### CI Pipeline

The GitHub Actions workflow runs on every push and PR:

1. **Lint & Format**: ruff + black
2. **Type Check**: mypy strict mode
3. **Tests**: pytest with coverage
4. **Smoke Test**: Live API endpoint tests
5. **Docker Build**: Verify image builds

## License

MIT
