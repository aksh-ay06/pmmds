.PHONY: help up down logs test lint clean download-data train monitor seed-traffic

help:
	@echo "PMMDS - Production ML Monitoring & Drift Detection System"
	@echo ""
	@echo "Usage:"
	@echo "  make up            - Start all services"
	@echo "  make down          - Stop all services"
	@echo "  make logs          - View service logs"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linters"
	@echo "  make clean         - Remove containers and volumes"
	@echo "  make download-data - Download and prepare dataset"
	@echo "  make train         - Train initial model"
	@echo "  make monitor       - Run drift monitoring"
	@echo "  make seed-traffic  - Generate sample predictions"

up:
	cd infra/compose && docker compose up -d

down:
	cd infra/compose && docker compose down

logs:
	cd infra/compose && docker compose logs -f

test:
	pytest tests/ -v

lint:
	ruff check apps shared pipelines
	mypy apps shared pipelines

download-data:
	python scripts/download_data.py

train:
	python scripts/train_model.py

monitor:
	python scripts/monitor.py

seed-traffic:
	python scripts/seed_traffic.py

clean:
	cd infra/compose && docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
