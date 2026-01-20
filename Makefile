.PHONY: help up down logs test lint clean

help:
	@echo "PMMDS - Production ML Monitoring & Drift Detection System"
	@echo ""
	@echo "Usage:"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - View service logs"
	@echo "  make test      - Run test suite"
	@echo "  make lint      - Run linters"
	@echo "  make clean     - Remove containers and volumes"

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

clean:
	cd infra/compose && docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
