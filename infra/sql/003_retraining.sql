-- Automated retraining and promotion tables
-- Tracks retraining runs and promotion decisions

-- Promotion decisions table
CREATE TABLE IF NOT EXISTS promotion_decisions (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Trigger context
    trigger_type VARCHAR(50) NOT NULL,  -- drift, scheduled, manual
    drift_run_id VARCHAR(64),
    
    -- Champion model (current production)
    champion_model_name VARCHAR(100) NOT NULL,
    champion_model_version VARCHAR(50) NOT NULL,
    champion_mlflow_run_id VARCHAR(100),
    champion_metrics JSONB NOT NULL DEFAULT '{}',
    
    -- Challenger model (newly trained)
    challenger_model_name VARCHAR(100) NOT NULL,
    challenger_model_version VARCHAR(50) NOT NULL,
    challenger_mlflow_run_id VARCHAR(100) NOT NULL,
    challenger_metrics JSONB NOT NULL DEFAULT '{}',
    
    -- Comparison results
    validation_passed BOOLEAN NOT NULL,
    metric_improvement BOOLEAN NOT NULL,
    latency_acceptable BOOLEAN NOT NULL,
    primary_metric_name VARCHAR(50) NOT NULL DEFAULT 'roc_auc',
    primary_metric_improvement FLOAT NOT NULL,
    
    -- Decision outcome
    promoted BOOLEAN NOT NULL,
    promotion_reason TEXT NOT NULL,
    rejection_reasons JSONB NOT NULL DEFAULT '[]',
    
    -- Timing
    comparison_duration_seconds FLOAT
);

-- Indexes for promotion_decisions
CREATE INDEX IF NOT EXISTS idx_promotion_decisions_timestamp 
    ON promotion_decisions (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_promotion_decisions_decision_id 
    ON promotion_decisions (decision_id);

CREATE INDEX IF NOT EXISTS idx_promotion_decisions_promoted 
    ON promotion_decisions (promoted);

CREATE INDEX IF NOT EXISTS idx_promotion_decisions_trigger 
    ON promotion_decisions (trigger_type);


-- Retraining runs table
CREATE TABLE IF NOT EXISTS retraining_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Trigger information
    trigger_type VARCHAR(50) NOT NULL,  -- drift, scheduled, manual
    drift_run_id VARCHAR(64),
    drift_features JSONB NOT NULL DEFAULT '[]',
    drift_max_psi FLOAT,
    
    -- Training details
    mlflow_run_id VARCHAR(100),
    model_version VARCHAR(50),
    training_config JSONB NOT NULL DEFAULT '{}',
    
    -- Results
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    metrics JSONB NOT NULL DEFAULT '{}',
    error_message TEXT,
    
    -- Promotion outcome
    promotion_decision_id VARCHAR(64),
    promoted BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timing
    training_duration_seconds FLOAT
);

-- Indexes for retraining_runs
CREATE INDEX IF NOT EXISTS idx_retraining_runs_timestamp 
    ON retraining_runs (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_retraining_runs_run_id 
    ON retraining_runs (run_id);

CREATE INDEX IF NOT EXISTS idx_retraining_runs_status 
    ON retraining_runs (status);

CREATE INDEX IF NOT EXISTS idx_retraining_runs_promoted 
    ON retraining_runs (promoted) WHERE promoted = TRUE;


-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pmmds;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pmmds;
