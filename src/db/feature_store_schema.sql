-- =============================================================================
-- feature_store_schema.sql
-- PostgreSQL DDL for the churn prediction feature store.
--
-- Tables:
--   user_churn_features   : weekly snapshot of per-user churn features
--   churn_predictions     : XGBoost model output scores per scoring run
--   ab_experiments        : A/B experiment registry
--   ab_assignments        : user-to-variant assignments
--   event_log             : raw interaction event stream
--   model_registry        : artifact versioning table
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS recommender;
SET search_path = recommender;

-- ---------------------------------------------------------------------------
-- 1. User churn feature snapshots
--    Partitioned by snapshot_date for efficient weekly roll-ups.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_churn_features (
    snapshot_date          DATE           NOT NULL,
    user_id                INTEGER        NOT NULL,
    total_ratings          INTEGER        NOT NULL DEFAULT 0,
    avg_rating             NUMERIC(4, 3)  NOT NULL DEFAULT 0,
    std_rating             NUMERIC(4, 3)  NOT NULL DEFAULT 0,
    rating_sessions        INTEGER        NOT NULL DEFAULT 0,
    days_since_first       INTEGER        NOT NULL DEFAULT 0,
    days_since_last        INTEGER        NOT NULL DEFAULT 0,
    avg_session_gap_days   NUMERIC(8, 2)  NOT NULL DEFAULT 0,
    pct_high_rating        NUMERIC(5, 4)  NOT NULL DEFAULT 0,
    genre_diversity        NUMERIC(6, 4)  NOT NULL DEFAULT 0,
    churned                SMALLINT       NOT NULL DEFAULT 0,  -- ground-truth label (1=churned)
    created_at             TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, user_id)
) PARTITION BY RANGE (snapshot_date);

-- Weekly partitions (example for 2024)
CREATE TABLE IF NOT EXISTS user_churn_features_2024
    PARTITION OF user_churn_features
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Index for fast user lookups
CREATE INDEX IF NOT EXISTS idx_ucf_user_date
    ON user_churn_features (user_id, snapshot_date DESC);

-- ---------------------------------------------------------------------------
-- 2. Churn prediction scores
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS churn_predictions (
    id                     BIGSERIAL      PRIMARY KEY,
    scoring_run_id         UUID           NOT NULL,
    snapshot_date          DATE           NOT NULL,
    user_id                INTEGER        NOT NULL,
    churn_probability      NUMERIC(6, 5)  NOT NULL,  -- XGBoost P(churn)
    churn_flag             SMALLINT       NOT NULL,   -- 1 if probability > threshold
    model_version          TEXT           NOT NULL,
    scored_at              TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cp_snapshot
    ON churn_predictions (snapshot_date, user_id);
CREATE INDEX IF NOT EXISTS idx_cp_run
    ON churn_predictions (scoring_run_id);

-- ---------------------------------------------------------------------------
-- 3. A/B experiment registry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ab_experiments (
    experiment_id          TEXT           PRIMARY KEY,
    name                   TEXT           NOT NULL,
    description            TEXT,
    model_control          TEXT           NOT NULL,
    model_treatment        TEXT           NOT NULL,
    traffic_split          NUMERIC(4, 3)  NOT NULL DEFAULT 0.5, -- fraction to treatment
    status                 TEXT           NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','active','paused','complete')),
    started_at             TIMESTAMPTZ,
    ended_at               TIMESTAMPTZ,
    created_at             TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 4. User-to-variant assignments
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ab_assignments (
    experiment_id          TEXT           NOT NULL REFERENCES ab_experiments(experiment_id),
    user_id                INTEGER        NOT NULL,
    variant                TEXT           NOT NULL CHECK (variant IN ('control', 'treatment')),
    assigned_at            TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (experiment_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_ab_assign_exp
    ON ab_assignments (experiment_id, variant);

-- ---------------------------------------------------------------------------
-- 5. Raw interaction event stream
--    Captures clicks, impressions, add-to-cart, purchases for A/B lift
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_log (
    event_id               BIGSERIAL      PRIMARY KEY,
    event_type             TEXT           NOT NULL CHECK (event_type IN (
                                              'impression','click','add_to_cart',
                                              'purchase','rating','recommendation_served')),
    user_id                INTEGER        NOT NULL,
    item_id                INTEGER,
    experiment_id          TEXT,
    variant                TEXT,
    recommendation_source  TEXT,  -- 'personalized','cold_start','cache','fallback'
    session_id             TEXT,
    page_rank              SMALLINT,      -- position of item in recommendation list
    payload                JSONB,         -- extra context (score, A/B flags, etc.)
    occurred_at            TIMESTAMPTZ    NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (occurred_at);

-- Monthly partition template
CREATE TABLE IF NOT EXISTS event_log_2024_01
    PARTITION OF event_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE INDEX IF NOT EXISTS idx_el_user_time
    ON event_log (user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_el_experiment
    ON event_log (experiment_id, variant, occurred_at);
CREATE INDEX IF NOT EXISTS idx_el_type_time
    ON event_log (event_type, occurred_at DESC);

-- ---------------------------------------------------------------------------
-- 6. Model registry (mirrors src/models/registry.py logic in DB)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id                     BIGSERIAL      PRIMARY KEY,
    model_name             TEXT           NOT NULL,
    artifact_path          TEXT           NOT NULL,
    metrics                JSONB,
    is_production          BOOLEAN        NOT NULL DEFAULT FALSE,
    registered_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    promoted_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_mr_name_prod
    ON model_registry (model_name, is_production, registered_at DESC);

-- ---------------------------------------------------------------------------
-- Aggregate view: weekly CTR per experiment variant
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ab_ctr_weekly AS
SELECT
    e.experiment_id,
    e.name                                      AS experiment_name,
    a.variant,
    DATE_TRUNC('week', ev.occurred_at)::DATE    AS week_start,
    COUNT(CASE WHEN ev.event_type = 'impression' THEN 1 END) AS impressions,
    COUNT(CASE WHEN ev.event_type = 'click'      THEN 1 END) AS clicks,
    ROUND(
        COUNT(CASE WHEN ev.event_type = 'click' THEN 1 END)::NUMERIC /
        NULLIF(COUNT(CASE WHEN ev.event_type = 'impression' THEN 1 END), 0),
        4
    )                                           AS ctr
FROM ab_experiments e
JOIN ab_assignments a  USING (experiment_id)
JOIN event_log     ev  ON ev.user_id = a.user_id
                      AND ev.experiment_id = a.experiment_id
GROUP BY 1, 2, 3, 4;

COMMENT ON TABLE user_churn_features  IS 'Weekly per-user feature snapshots for churn model';
COMMENT ON TABLE churn_predictions    IS 'XGBoost churn probability scores per scoring run';
COMMENT ON TABLE ab_experiments       IS 'A/B experiment registry';
COMMENT ON TABLE ab_assignments       IS 'User-to-variant bucket assignments';
COMMENT ON TABLE event_log            IS 'Partitioned interaction event stream';
COMMENT ON TABLE model_registry       IS 'Model artifact versioning';
