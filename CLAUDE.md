CLAUDE.md — Production ML Monitoring & Drift System (PMMDS)

This file tells Claude (and other LLM copilots) how to work in this repo: goals, architecture, conventions, and what “done” means.

0) Project intent (north star)

Build a production-grade ML monitoring + drift detection + automated retraining system that demonstrates FAANG-level MLE signals:

Data validation (schema + quality) before training and before inference

Drift detection (feature + prediction drift) on live/batch traffic

Model registry + promotion workflow (MLflow)

Automated retraining orchestration (Prefect or Airflow)

Observability (latency, error rate, drift metrics) + alerting hooks

Reproducible local dev with Docker Compose

Clean APIs and operational runbooks

Non-goal: a pretty UI. Prefer a minimal dashboard endpoint or lightweight Streamlit later.

1) Target audience

MLE hiring managers / FAANG screeners

Recruiters scanning for: production ML, monitoring, drift, orchestration, model registry, CI/CD

Everything should be designed to produce crisp resume bullets and a strong README.

2) Scope and MVP definition
MVP (must ship)

Train a baseline model from a public tabular dataset

Serve predictions via FastAPI (/predict) with request/response logging

Log features and predictions to Postgres (or Parquet)

Validate data using Great Expectations

Detect drift daily or hourly (PSI, JS/KL, KS-test optional)

Register models in MLflow with versioning and metadata

Retrain pipeline runs when drift exceeds threshold

Observability endpoints (/metrics, /healthz)

Docker Compose brings up all services

V1+ (nice to have)

Canary / shadow deployment

Slack/email alerting

Grafana dashboard

Feature store (Feast)

Backfill + replay jobs

3) Repo structure (required)
.
├── apps/
│   ├── api/
│   └── monitor/
├── pipelines/
│   ├── train/
│   └── retrain/
├── shared/
│   ├── config/
│   ├── schemas/
│   ├── features/
│   ├── data/
│   └── utils/
├── infra/
│   ├── docker/
│   ├── compose/
│   └── sql/
├── tests/
├── scripts/
├── docs/
├── .github/workflows/
├── pyproject.toml
├── README.md
└── CLAUDE.md

4) Architecture (high level)
Services

FastAPI inference service

Drift monitoring job

MLflow tracking + registry

Postgres metadata store

Prefect/Airflow orchestration

Data flow

/predict called

Payload validated

Model loaded from registry

Prediction returned + logged

Monitor computes drift

Retrain triggered if threshold exceeded

Model promotion decision recorded

5) Documentation & Reference Sources

Claude should consult the following documentation when implementing or modifying related components.

ML Monitoring & Drift

MLflow Model Registry
https://mlflow.org/docs/latest/model-registry.html

Use when implementing model versioning, promotion, and rollback logic.

Great Expectations
https://docs.greatexpectations.io/

Use for schema validation, training data checks, and inference payload validation.

Evidently AI (reference only)
https://docs.evidentlyai.com/

Use as conceptual guidance for drift metrics and monitoring patterns; do not depend on it directly unless explicitly instructed.

Orchestration

Prefect
https://docs.prefect.io/

Use for implementing training, monitoring, and retraining flows.

Apache Airflow (alternative)
https://airflow.apache.org/docs/

Use only if Prefect is insufficient or explicitly requested.

Model Serving & APIs

FastAPI
https://fastapi.tiangolo.com/

Use for REST endpoints, request validation, async inference, and health checks.

Pydantic
https://docs.pydantic.dev/latest/

Use for request/response schemas and configuration management.

Metrics & Observability

Prometheus exposition format
https://prometheus.io/docs/instrumenting/exposition_formats/

Use when exposing /metrics endpoints.

OpenTelemetry (conceptual reference)
https://opentelemetry.io/docs/

Reference for tracing and structured observability patterns.

Data & Storage

PostgreSQL
https://www.postgresql.org/docs/

Use for schema design, indexing, and query optimization.

Redis
https://redis.io/docs/

Use for caching, lightweight queues, and ephemeral state.

Containers & DevOps

Docker
https://docs.docker.com/

Use for Dockerfiles, multi-stage builds, and Compose setups.

Docker Compose
https://docs.docker.com/compose/

Use for local orchestration and service dependency management.

Claude should prefer official documentation above blog posts, tutorials, or StackOverflow unless explicitly instructed otherwise.

All implementations should target:

Python 3.11+

FastAPI >= 0.110

MLflow >= 2.x

Prefect >= 2.x

6) Dataset guidelines

Dataset: NYC TLC Yellow Taxi Trip Records (Parquet format from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Task: Fare amount regression (predict fare_amount from trip features)

Features (12 total):
- Numeric (6): trip_distance, passenger_count, pickup_hour, pickup_day_of_week, pickup_month, trip_duration_minutes
- Binary (2): is_weekend, is_rush_hour
- Categorical (4): RatecodeID, payment_type, pickup_borough, dropoff_borough

Target: fare_amount (continuous, USD)

Processing: PySpark reads raw TLC Parquet, extracts time features, computes trip_duration, maps LocationIDs to boroughs, filters outliers. Output: data/processed/{train,test}.parquet + reference.csv

Download script: scripts/download_data.py

7) Engineering standards

Python 3.11+

pyproject.toml, ruff, black, mypy

FastAPI + Pydantic

Structured JSON logging

No notebooks in core pipeline

Docker-first execution

8) Drift metrics

PSI for numeric features (n_bins=20 for continuous fare/distance values)

JS/KL divergence for categoricals (boroughs, payment_type, RatecodeID)

Prediction drift: PSI on predicted_fare distribution (continuous regression output)

Retrain trigger if:

≥3 features PSI > 0.2

Prediction distribution shift (fare predictions diverge from reference)

Validation failure rate exceeds threshold

9) Model promotion logic

Champion vs challenger (regression: GBTRegressor via PySpark MLlib)

Primary metric: RMSE (lower is better)
Secondary metrics: MAE, R2, MAPE

Promotion requires:

Validation pass

RMSE improvement ≥ 0.5 (challenger RMSE must be at least 0.5 lower than champion)

No latency regression

Models served via MLflow pyfunc with pre-warmed SparkSession

Record all decisions with metadata.

10) Orchestration

Prefer Prefect for local-first flows:

train_initial_flow

monitor_drift_flow

retrain_and_promote_flow

11) Implementation milestones

API + logging

Training + MLflow

Validation

Drift monitoring

Automated retraining

CI + documentation

12) README requirements

README must explain:

Architecture

Drift logic

Retraining triggers

Model lifecycle

How this maps to MLE signals

13) Code style expectations

Small, reviewable changes

Type hints + docstrings

Clear naming

Minimal over-engineering

14) Acceptance criteria

Project is complete when:

docker compose up works cleanly

Predictions logged

Drift detected

Retraining triggered

Promotion decisions recorded

CI passes

15) Expected commands

docker compose up

make train

make seed-traffic

make monitor

make retrain

make test


16) Default guidance

When uncertain:

Favor simplicity

Prioritize reproducibility

Produce measurable outputs

Write like a production system, not a class project