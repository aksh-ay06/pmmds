-- Migration: Convert prediction_logs from classification (churn) to regression (taxi fare)
-- This migration drops the old prediction/probability columns and adds predicted_fare.

BEGIN;

-- Drop old classification columns
ALTER TABLE prediction_logs DROP COLUMN IF EXISTS prediction;
ALTER TABLE prediction_logs DROP COLUMN IF EXISTS probability;

-- Add regression column
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS predicted_fare FLOAT;

-- Backfill NULLs if any existing rows (unlikely in fresh deploy)
UPDATE prediction_logs SET predicted_fare = 0.0 WHERE predicted_fare IS NULL;

-- Make NOT NULL after backfill
ALTER TABLE prediction_logs ALTER COLUMN predicted_fare SET NOT NULL;

COMMIT;
