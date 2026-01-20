.PHONY: help up down logs test lint format typecheck clean download-data train monitor seed-traffic retrain ci migrate

help:
	@echo "PMMDS - Production ML Monitoring & Drift Detection System"
	@echo ""
	@echo "Usage:"
	@echo "  make up            - Start all services"
	@echo "  make down          - Stop all services"
	@echo "  make logs          - View service logs"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make typecheck     - Run type checker"
	@echo "  make ci            - Run full CI pipeline locally"
	@echo "  make clean         - Remove containers and volumes"
	@echo "  make download-data - Download and prepare dataset"
	@echo "  make train         - Train initial model"
	@echo "  make monitor       - Run drift monitoring"
	@echo "  make seed-traffic  - Generate sample predictions"
	@echo "  make retrain       - Run automated retraining"
	@echo "  make migrate       - Run database migrations"

up:
	cd infra/compose && docker compose up -d

down:
	cd infra/compose && docker compose down

logs:
	cd infra/compose && docker compose logs -f

test:
	pytest tests/ -v --cov=apps --cov=shared --cov-report=term-missing

lint:
	ruff check apps shared pipelines tests

format:
	black apps shared pipelines tests
	ruff check --fix apps shared pipelines tests

typecheck:
	mypy apps shared pipelines --ignore-missing-imports

ci: lint typecheck test
	@echo "✓ All CI checks passed!"

download-data:
	python scripts/download_data.py

train:
	python scripts/train_model.py

monitor:
	python scripts/monitor.py

seed-traffic:
	python scripts/seed_traffic.py

retrain:
	python scripts/retrain.py

migrate:
	@echo "Running database migrations..."
	psql -h localhost -U pmmds -d pmmds -f infra/sql/init.sql
	psql -h localhost -U pmmds -d pmmds -f infra/sql/002_drift_monitoring.sql
	psql -h localhost -U pmmds -d pmmds -f infra/sql/003_retraining.sql

clean:
	cd infra/compose && docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
