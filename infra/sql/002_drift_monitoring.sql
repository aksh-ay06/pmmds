-- Drift monitoring tables migration
-- Adds tables for drift metrics, reference datasets, and alerts

-- Drift metrics table (detailed drift tracking)
CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Model context
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Window information
    reference_window_start TIMESTAMPTZ,
    reference_window_end TIMESTAMPTZ,
    current_window_start TIMESTAMPTZ NOT NULL,
    current_window_end TIMESTAMPTZ NOT NULL,
    reference_sample_count INTEGER NOT NULL,
    current_sample_count INTEGER NOT NULL,
    
    -- Aggregate metrics
    max_psi FLOAT NOT NULL,
    avg_psi FLOAT NOT NULL,
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    features_with_drift JSONB NOT NULL DEFAULT '[]',
    drift_count INTEGER NOT NULL DEFAULT 0,
    
    -- Detailed feature drift (JSON blob)
    feature_drift_details JSONB NOT NULL DEFAULT '{}',
    
    -- Prediction drift
    prediction_psi FLOAT,
    prediction_drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Configuration used
    psi_threshold FLOAT NOT NULL DEFAULT 0.2,
    min_drift_features INTEGER NOT NULL DEFAULT 3
);

-- Indexes for drift_metrics
CREATE INDEX IF NOT EXISTS idx_drift_metrics_timestamp 
    ON drift_metrics (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_model 
    ON drift_metrics (model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_run_id 
    ON drift_metrics (run_id);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_drift_detected 
    ON drift_metrics (drift_detected) WHERE drift_detected = TRUE;


-- Reference datasets table
CREATE TABLE IF NOT EXISTS reference_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    
    -- Dataset info
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    sample_count INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    
    -- Statistics for reference
    dataset_stats JSONB NOT NULL,
    
    -- Path or storage info
    storage_path TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_reference_datasets_name 
    ON reference_datasets (name);

CREATE INDEX IF NOT EXISTS idx_reference_datasets_model 
    ON reference_datasets (model_name);

CREATE INDEX IF NOT EXISTS idx_reference_datasets_active 
    ON reference_datasets (is_active) WHERE is_active = TRUE;


-- Drift alerts table
CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    drift_metric_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    
    -- Alert details
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    features_affected JSONB NOT NULL,
    
    -- Action tracking
    acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    action_taken TEXT,
    retrain_triggered BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_drift_alerts_created 
    ON drift_alerts (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_drift_alerts_severity 
    ON drift_alerts (severity);

CREATE INDEX IF NOT EXISTS idx_drift_alerts_acknowledged 
    ON drift_alerts (acknowledged) WHERE acknowledged = FALSE;

CREATE INDEX IF NOT EXISTS idx_drift_alerts_metric_id 
    ON drift_alerts (drift_metric_id);


-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pmmds;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pmmds;
