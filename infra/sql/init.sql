-- PostgreSQL initialization script
-- Creates tables for PMMDS

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Prediction logs table
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    request_id UUID UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    latency_ms FLOAT NOT NULL,
    feature_hash VARCHAR(64) NOT NULL,
    numeric_feature_stats JSONB NOT NULL DEFAULT '{}'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp 
    ON prediction_logs (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_model 
    ON prediction_logs (model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_request_id 
    ON prediction_logs (request_id);

-- Drift detection results table (for future use)
CREATE TABLE IF NOT EXISTS drift_results (
    id SERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    reference_start TIMESTAMPTZ NOT NULL,
    reference_end TIMESTAMPTZ NOT NULL,
    current_start TIMESTAMPTZ NOT NULL,
    current_end TIMESTAMPTZ NOT NULL,
    feature_drift JSONB NOT NULL DEFAULT '{}',
    prediction_drift JSONB NOT NULL DEFAULT '{}',
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_drift_results_timestamp 
    ON drift_results (timestamp DESC);

-- Model registry metadata (supplement to MLflow)
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    mlflow_run_id VARCHAR(100),
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'staging',
    metrics JSONB NOT NULL DEFAULT '{}',
    validation_results JSONB NOT NULL DEFAULT '{}',
    UNIQUE (model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_model_metadata_status 
    ON model_metadata (status);

-- Grant permissions (adjust user as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pmmds;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pmmds;
