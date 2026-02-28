-- ==========================================================================
-- X Virality Optimization Engine — Database Schema
-- PostgreSQL 16 + pgvector
-- ==========================================================================

CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector for embedding storage
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram index for fuzzy text search

-- --------------------------------------------------------------------------
-- Authors
-- --------------------------------------------------------------------------

CREATE TABLE authors (
    id                  BIGSERIAL PRIMARY KEY,
    x_user_id           TEXT UNIQUE NOT NULL,
    username            TEXT NOT NULL,
    display_name        TEXT,
    follower_count      INTEGER NOT NULL DEFAULT 0,
    following_count     INTEGER NOT NULL DEFAULT 0,
    is_verified         BOOLEAN NOT NULL DEFAULT FALSE,
    account_created_at  TIMESTAMPTZ,
    bio                 TEXT,
    location            TEXT,
    avg_engagement_rate DOUBLE PRECISION DEFAULT 0.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_authors_x_user_id ON authors (x_user_id);
CREATE INDEX idx_authors_follower_count ON authors (follower_count);

-- Follower history snapshots for tracking growth
CREATE TABLE author_follower_history (
    id              BIGSERIAL PRIMARY KEY,
    author_id       BIGINT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    follower_count  INTEGER NOT NULL,
    following_count INTEGER NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_author_follower_hist ON author_follower_history (author_id, recorded_at);

-- --------------------------------------------------------------------------
-- Posts
-- --------------------------------------------------------------------------

CREATE TABLE posts (
    id                  BIGSERIAL PRIMARY KEY,
    x_post_id           TEXT UNIQUE NOT NULL,
    author_id           BIGINT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    post_text           TEXT NOT NULL,
    media_type          TEXT NOT NULL DEFAULT 'none'
                        CHECK (media_type IN ('none', 'image', 'video', 'gif', 'poll')),
    media_urls          JSONB DEFAULT '[]'::jsonb,
    language            TEXT DEFAULT 'en',
    posted_at           TIMESTAMPTZ NOT NULL,

    -- Engagement metrics
    impressions         INTEGER NOT NULL DEFAULT 0,
    likes               INTEGER NOT NULL DEFAULT 0,
    reposts             INTEGER NOT NULL DEFAULT 0,
    replies             INTEGER NOT NULL DEFAULT 0,
    quotes              INTEGER NOT NULL DEFAULT 0,
    bookmarks           INTEGER NOT NULL DEFAULT 0,
    url_clicks          INTEGER NOT NULL DEFAULT 0,
    profile_visits      INTEGER NOT NULL DEFAULT 0,

    -- Extracted metadata
    hashtags            TEXT[] DEFAULT '{}',
    mentions            TEXT[] DEFAULT '{}',
    urls                TEXT[] DEFAULT '{}',

    -- Thread / conversation
    conversation_id     TEXT,
    in_reply_to_id      TEXT,

    -- Classification
    is_viral            BOOLEAN GENERATED ALWAYS AS (impressions >= 1000000) STORED,

    -- Timestamps
    collected_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metrics_updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_posts_author ON posts (author_id);
CREATE INDEX idx_posts_posted_at ON posts (posted_at);
CREATE INDEX idx_posts_impressions ON posts (impressions);
CREATE INDEX idx_posts_viral ON posts (posted_at, impressions) WHERE impressions > 100000;
CREATE INDEX idx_posts_hashtags ON posts USING GIN (hashtags);
CREATE INDEX idx_posts_text_search ON posts USING GIN (to_tsvector('english', post_text));
CREATE INDEX idx_posts_composite ON posts (posted_at, impressions);

-- --------------------------------------------------------------------------
-- Post Features (extracted feature vectors)
-- --------------------------------------------------------------------------

CREATE TABLE post_features (
    id                  BIGSERIAL PRIMARY KEY,
    post_id             BIGINT UNIQUE NOT NULL REFERENCES posts(id) ON DELETE CASCADE,

    -- Individual features (all normalized [0, 1])
    hook_strength       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    emotional_intensity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    curiosity_gap       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    controversy_index   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    readability_score   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    structure_score     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    timing_score        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    trend_alignment     DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    -- Detected structure pattern
    structure_pattern   TEXT,
    detected_patterns   JSONB DEFAULT '{}'::jsonb,

    -- NLP outputs
    sentiment_compound  DOUBLE PRECISION,
    emotion_scores      JSONB,                      -- {"joy": 0.8, "anger": 0.1, ...}
    detected_topic      TEXT,
    topic_confidence    DOUBLE PRECISION,

    -- Embedding vector (384-dim for all-MiniLM-L6-v2)
    embedding           vector(384),

    -- Cluster assignment
    cluster_id          INTEGER DEFAULT -1,
    pattern_similarity  DOUBLE PRECISION DEFAULT 0.0,

    -- Composite scores
    virality_score_raw  DOUBLE PRECISION,
    virality_score_final DOUBLE PRECISION,
    predicted_impressions_p50 INTEGER,
    predicted_impressions_p90 INTEGER,

    -- Metadata
    model_version_id    BIGINT,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_post_features_post ON post_features (post_id);
CREATE INDEX idx_post_features_virality ON post_features (virality_score_final);
CREATE INDEX idx_post_features_embedding ON post_features
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX idx_post_features_cluster ON post_features (cluster_id);

-- --------------------------------------------------------------------------
-- Trends
-- --------------------------------------------------------------------------

CREATE TABLE trends (
    id                  BIGSERIAL PRIMARY KEY,
    trend_name          TEXT NOT NULL,
    trend_query         TEXT,
    woeid               INTEGER,                    -- Yahoo WOEID for geo location
    volume              INTEGER DEFAULT 0,
    velocity            DOUBLE PRECISION DEFAULT 0.0, -- rate of volume change
    velocity_normalized DOUBLE PRECISION DEFAULT 0.0,
    peak_time           TIMESTAMPTZ,
    embedding           vector(384),
    category            TEXT,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trends_recorded ON trends (recorded_at);
CREATE INDEX idx_trends_velocity ON trends (velocity_normalized DESC);
CREATE INDEX idx_trends_name ON trends (trend_name, recorded_at);
CREATE INDEX idx_trends_embedding ON trends
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);

-- --------------------------------------------------------------------------
-- Strategies (generated recommendations)
-- --------------------------------------------------------------------------

CREATE TABLE strategies (
    id                  BIGSERIAL PRIMARY KEY,
    post_id             BIGINT REFERENCES posts(id) ON DELETE SET NULL,
    input_text          TEXT NOT NULL,               -- original input post text

    -- Scores
    original_score      DOUBLE PRECISION NOT NULL,
    optimized_score     DOUBLE PRECISION,

    -- Strategy output
    recommendations     JSONB NOT NULL DEFAULT '[]'::jsonb,
    hook_variants       JSONB DEFAULT '[]'::jsonb,
    thread_expansion    JSONB DEFAULT '{}'::jsonb,
    media_recommendation TEXT,
    hashtag_strategy    JSONB DEFAULT '{}'::jsonb,
    engagement_bait_type TEXT,

    -- Outcome tracking (filled after post goes live)
    actual_impressions  INTEGER,
    actual_engagement_rate DOUBLE PRECISION,
    strategy_lift       DOUBLE PRECISION,            -- actual / predicted baseline

    -- Metadata
    model_version_id    BIGINT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_strategies_post ON strategies (post_id);
CREATE INDEX idx_strategies_created ON strategies (created_at);
CREATE INDEX idx_strategies_lift ON strategies (strategy_lift) WHERE strategy_lift IS NOT NULL;

-- --------------------------------------------------------------------------
-- A/B Tests
-- --------------------------------------------------------------------------

CREATE TABLE ab_tests (
    id                  BIGSERIAL PRIMARY KEY,
    name                TEXT,
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'completed', 'cancelled')),
    metric              TEXT NOT NULL DEFAULT 'impressions',
    confidence_level    DOUBLE PRECISION NOT NULL DEFAULT 0.90,
    min_sample_time_h   INTEGER NOT NULL DEFAULT 48,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

CREATE TABLE ab_test_variants (
    id                  BIGSERIAL PRIMARY KEY,
    test_id             BIGINT NOT NULL REFERENCES ab_tests(id) ON DELETE CASCADE,
    variant_type        TEXT NOT NULL,               -- 'control', 'hook_rewrite', 'format_trigger', etc.
    post_text           TEXT NOT NULL,
    x_post_id           TEXT,                        -- filled when actually posted
    posted_at           TIMESTAMPTZ,

    -- Outcome metrics
    impressions         INTEGER,
    likes               INTEGER,
    reposts             INTEGER,
    replies             INTEGER,
    engagement_rate     DOUBLE PRECISION,
    repost_velocity     DOUBLE PRECISION,

    is_winner           BOOLEAN DEFAULT FALSE,
    metrics_collected_at TIMESTAMPTZ
);

CREATE INDEX idx_ab_variants_test ON ab_test_variants (test_id);
CREATE INDEX idx_ab_variants_winner ON ab_test_variants (test_id) WHERE is_winner = TRUE;

-- --------------------------------------------------------------------------
-- Model Versions (ML model registry)
-- --------------------------------------------------------------------------

CREATE TABLE model_versions (
    id                  BIGSERIAL PRIMARY KEY,
    model_name          TEXT NOT NULL,               -- 'xgboost_regressor', 'neural_ranker', etc.
    version             TEXT NOT NULL,
    artifact_path       TEXT NOT NULL,               -- S3 path to model artifact

    -- Training info
    training_data_start TIMESTAMPTZ,
    training_data_end   TIMESTAMPTZ,
    training_samples    INTEGER,
    feature_count       INTEGER,

    -- Performance metrics
    spearman_rho        DOUBLE PRECISION,
    auc_roc             DOUBLE PRECISION,
    mae_log_impressions DOUBLE PRECISION,
    hit_rate_1m         DOUBLE PRECISION,            -- % of 0.85+ scores that hit 1M+

    -- Learned weights (for formula-based scoring)
    learned_weights     JSONB,                       -- {"hook": 0.22, "emotion": 0.14, ...}
    shap_importances    JSONB,                       -- SHAP feature importance values

    -- Status
    is_active           BOOLEAN NOT NULL DEFAULT FALSE,
    deployed_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_model_active ON model_versions (model_name) WHERE is_active = TRUE;
CREATE INDEX idx_model_versions_name ON model_versions (model_name, created_at);

-- --------------------------------------------------------------------------
-- Engagement heatmap (precomputed)
-- --------------------------------------------------------------------------

CREATE TABLE engagement_heatmap (
    id                  BIGSERIAL PRIMARY KEY,
    segment             TEXT NOT NULL DEFAULT 'global', -- 'global', audience_segment_id, author_id
    day_of_week         SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    hour_utc            SMALLINT NOT NULL CHECK (hour_utc BETWEEN 0 AND 23),
    avg_engagement_rate DOUBLE PRECISION NOT NULL,
    sample_count        INTEGER NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (segment, day_of_week, hour_utc)
);

CREATE INDEX idx_heatmap_segment ON engagement_heatmap (segment);

-- --------------------------------------------------------------------------
-- Viral patterns (cluster centroids from historical viral posts)
-- --------------------------------------------------------------------------

CREATE TABLE viral_patterns (
    id                  BIGSERIAL PRIMARY KEY,
    cluster_id          INTEGER UNIQUE NOT NULL,
    centroid            vector(384) NOT NULL,
    post_count          INTEGER NOT NULL DEFAULT 0,
    avg_impressions     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    avg_virality_rate   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    dominant_pattern    TEXT,                         -- structure pattern name
    dominant_emotion    TEXT,
    representative_posts JSONB DEFAULT '[]'::jsonb,  -- top-5 example post IDs
    temporal_distribution JSONB DEFAULT '{}'::jsonb,  -- {day: {hour: count}}
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_viral_patterns_centroid ON viral_patterns
    USING hnsw (centroid vector_cosine_ops) WITH (m = 16, ef_construction = 100);

-- --------------------------------------------------------------------------
-- Outcome feedback (links predictions to actual results)
-- --------------------------------------------------------------------------

CREATE TABLE prediction_outcomes (
    id                  BIGSERIAL PRIMARY KEY,
    post_id             BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    model_version_id    BIGINT NOT NULL REFERENCES model_versions(id),
    predicted_score     DOUBLE PRECISION NOT NULL,
    predicted_impressions_p50 INTEGER,
    actual_impressions  INTEGER,
    absolute_error      DOUBLE PRECISION,            -- |predicted - actual| in log10 space
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_outcomes_model ON prediction_outcomes (model_version_id, recorded_at);
CREATE INDEX idx_outcomes_error ON prediction_outcomes (absolute_error);

-- --------------------------------------------------------------------------
-- Helper functions
-- --------------------------------------------------------------------------

-- Function to compute engagement rate
CREATE OR REPLACE FUNCTION compute_engagement_rate(
    p_likes INTEGER,
    p_reposts INTEGER,
    p_replies INTEGER,
    p_quotes INTEGER,
    p_impressions INTEGER
) RETURNS DOUBLE PRECISION AS $$
BEGIN
    IF p_impressions = 0 THEN
        RETURN 0.0;
    END IF;
    RETURN (p_likes + p_reposts + p_replies + p_quotes)::DOUBLE PRECISION / p_impressions;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to compute repost velocity
CREATE OR REPLACE FUNCTION compute_repost_velocity(
    p_reposts INTEGER,
    p_posted_at TIMESTAMPTZ
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    hours_elapsed DOUBLE PRECISION;
BEGIN
    hours_elapsed := EXTRACT(EPOCH FROM (NOW() - p_posted_at)) / 3600.0;
    IF hours_elapsed <= 0 THEN
        RETURN 0.0;
    END IF;
    RETURN p_reposts::DOUBLE PRECISION / hours_elapsed;
END;
$$ LANGUAGE plpgsql STABLE;
