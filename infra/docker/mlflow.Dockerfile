# MLflow Tracking Server (pinned to match your host)
FROM python:3.12-slim

# minimal deps (curl optional, psycopg2-binary avoids needing libpq-dev)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==3.8.1 psycopg2-binary

RUN mkdir -p /mlflow/artifacts
EXPOSE 5000

# Note: artifacts + backend are configured via docker-compose command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
