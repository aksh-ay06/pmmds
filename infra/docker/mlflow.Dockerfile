# MLflow Tracking Server with PostgreSQL support
# Custom image based on official MLflow with psycopg2 driver

FROM ghcr.io/mlflow/mlflow:v2.10.0

# Install PostgreSQL client library, psycopg2, and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir psycopg2-binary

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
