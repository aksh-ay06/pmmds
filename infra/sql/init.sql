-- PostgreSQL initialization script
-- Creates tables for PMMDS (NYC Yellow Taxi fare prediction)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Prediction logs table
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    request_id UUID UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Model metadata
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Prediction output
    predicted_fare FLOAT NOT NULL,
    latency_ms FLOAT NOT NULL,

    -- Feature statistics (stored as JSONB for flexibility)
    feature_hash VARCHAR(64) NOT NULL,
    numeric_feature_stats JSONB DEFAULT '{}'::jsonb
);

-- Index for time-based queries (drift monitoring windows)
CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
    ON prediction_logs (timestamp DESC);

-- Index for model-specific queries
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model
    ON prediction_logs (model_name, model_version);

-- Index for feature hash lookups
CREATE INDEX IF NOT EXISTS idx_prediction_logs_feature_hash
    ON prediction_logs (feature_hash);
