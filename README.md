# PMMDS - Production ML Monitoring & Drift Detection System

A production-grade ML monitoring system demonstrating drift detection, automated retraining, and model lifecycle management.

## Quick Start

```bash
# Start all services
cd infra/compose
docker compose up -d

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

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│  PostgreSQL │
│             │     │  /predict   │     │  (logs)     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Model     │
                    │  (MLflow)   │
                    └─────────────┘
```

## Project Structure

```
pmm-drift-system/
├── apps/
│   └── api/           # FastAPI inference service
├── pipelines/         # Training and retraining flows
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
