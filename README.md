# PMMDS - Production ML Monitoring & Drift Detection System

A production-grade ML monitoring system demonstrating drift detection, automated retraining, and model lifecycle management.

## Quick Start

### 1. Start Infrastructure

```bash
# Start all services (PostgreSQL, MLflow, API)
cd infra/compose
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

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│  PostgreSQL │
│             │     │  /predict   │     │  (logs)     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   MLflow    │
                    │  Registry   │
                    └─────────────┘
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
│   └── api/           # FastAPI inference service
├── pipelines/
│   └── train/         # Model training pipeline
├── scripts/           # CLI scripts
├── shared/            # Common utilities
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

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy apps shared
```

## License

MIT
