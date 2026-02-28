# X Virality Optimization Engine — Technical Specification

**Version:** 1.0.0
**Target:** Consistent 1M+ impressions per post
**Architecture:** Event-driven microservices with ML pipeline

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Model & Storage](#2-data-model--storage)
3. [Analytical Engine](#3-analytical-engine)
4. [Virality Scoring Algorithm](#4-virality-scoring-algorithm)
5. [Million-Level Virality Strategy Generator](#5-million-level-virality-strategy-generator)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Development Roadmap](#7-development-roadmap)

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ X API v2 │  │ Trend Tracker│  │ Competitor  │  │ Historical   │ │
│  │ Collector │  │  (Streaming) │  │  Scraper    │  │ Dataset Imp. │ │
│  └────┬─────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘ │
│       └───────────┬────┴────────────────┴─────────────────┘        │
│                   ▼                                                  │
│          [ Message Queue — Kafka / Redis Streams ]                  │
└───────────────────┬─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐    │
│  │  Feature      │  │  NLP Engine  │  │  Pattern Recognition   │    │
│  │  Extraction   │  │  (Embeddings,│  │  (Historical Viral     │    │
│  │  Pipeline     │  │   Sentiment, │  │   Post Matching)       │    │
│  │              │  │   Topics)    │  │                        │    │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘    │
│         └─────────────┬────┴──────────────────────┘                 │
│                       ▼                                              │
│            [ Virality Scoring Engine ]                               │
│                       │                                              │
│                       ▼                                              │
│         [ Strategy Generator Module ]                               │
└───────────────────────┬─────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Rewritten    │  │ Score        │  │ A/B Test Orchestrator   │   │
│  │ Post Drafts  │  │ Predictions  │  │ (Publish + Feedback)    │   │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘   │
│                                                                     │
│              [ Dashboard / API / Webhook Output ]                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Input Data Schema

| Field | Type | Source | Description |
|---|---|---|---|
| `post_text` | string | X API / User | Raw post content (≤280 chars) |
| `media_type` | enum | X API | `none`, `image`, `video`, `gif`, `poll` |
| `media_urls` | string[] | X API | Attached media references |
| `author_id` | string | X API | Unique author identifier |
| `author_follower_count` | int | X API | Followers at time of post |
| `author_following_count` | int | X API | Following at time of post |
| `author_verified` | bool | X API | Verification status |
| `author_account_age_days` | int | X API | Account age in days |
| `posted_at` | datetime | X API | UTC timestamp |
| `language` | string | X API / detected | ISO 639-1 code |
| `impressions` | int | X API | Total impression count |
| `likes` | int | X API | Like count |
| `reposts` | int | X API | Repost count |
| `replies` | int | X API | Reply count |
| `quotes` | int | X API | Quote post count |
| `bookmarks` | int | X API | Bookmark count |
| `url_clicks` | int | X API | Link click count |
| `profile_visits` | int | X API | Profile visit count |
| `hashtags` | string[] | Extracted | Hashtags used |
| `mentions` | string[] | Extracted | User mentions |
| `urls` | string[] | Extracted | Embedded URLs |
| `trend_ids` | string[] | Trend Tracker | Associated trending topics at post time |
| `conversation_id` | string | X API | Thread root ID (null if standalone) |
| `in_reply_to` | string | X API | Parent post ID if reply |

### 1.3 Data Collection Methods

#### X API v2 (Primary)

- **Endpoints used:**
  - `GET /2/tweets/search/recent` — Recent post search with query operators
  - `GET /2/tweets/:id` — Single post lookup with full metrics
  - `GET /2/users/:id/tweets` — User timeline retrieval
  - `GET /2/tweets/counts/recent` — Volume estimation for topics
  - `GET /2/tweets/search/stream` — Filtered real-time stream for trend tracking
- **Rate limit management:** Token bucket with automatic backoff; pool multiple bearer tokens across app registrations
- **Pagination:** Cursor-based with `next_token`; full historical sweep per query

#### Trend Tracking (Streaming)

- Maintain persistent filtered stream connections for top 25 topic categories
- Snapshot `GET /2/trends/by/woeid` every 5 minutes for top 50 geolocations
- Store trend velocity (rate of volume change) not just current volume

#### Historical Dataset Import

- Accept CSV/JSON bulk imports of historical post performance data
- Support Kaggle datasets, exported analytics, third-party data providers
- Normalization pipeline to map external schemas to internal model

### 1.4 Output Formats

```json
{
  "analysis": {
    "virality_score": 0.87,
    "predicted_impressions": { "p50": 450000, "p75": 980000, "p90": 2100000 },
    "feature_breakdown": {
      "hook_strength": 0.92,
      "emotional_intensity": 0.78,
      "curiosity_gap": 0.85,
      "controversy_index": 0.41,
      "readability_score": 0.90,
      "format_score": 0.88,
      "timing_score": 0.73,
      "trend_alignment": 0.65
    },
    "weaknesses": ["timing_score below 0.80 threshold", "trend_alignment low"],
    "category": "high_potential"
  },
  "recommendations": [
    {
      "type": "rewrite",
      "original": "AI will change everything in 2026.",
      "rewritten": "5 jobs that won't exist by 2028 (and what replaces them):",
      "score_delta": +0.21,
      "rationale": "Adds specificity, curiosity gap, listicle format, implied threat"
    },
    {
      "type": "timing",
      "optimal_windows": [
        { "day": "Tuesday", "hour_utc": 14, "score_boost": 0.08 },
        { "day": "Wednesday", "hour_utc": 13, "score_boost": 0.06 }
      ]
    },
    {
      "type": "format",
      "suggestion": "Add line breaks after first sentence. Use → or — for visual scannability.",
      "score_boost": 0.04
    }
  ],
  "strategy": {
    "hook_variants": ["...", "...", "..."],
    "thread_expansion": { "recommended": true, "optimal_length": 5 },
    "media_recommendation": "static_image_with_text_overlay",
    "hashtag_strategy": { "use": 1, "tags": ["#AI"] },
    "engagement_bait_type": "question_close"
  }
}
```

---

## 2. Data Model & Storage

### 2.1 Technology Choice

- **Primary DB:** PostgreSQL 16 (relational data, complex queries, JSONB for flexible fields)
- **Vector Store:** pgvector extension (embedding similarity search in same DB)
- **Cache:** Redis (hot metrics, rate limit counters, session state)
- **Queue:** Redis Streams or Kafka (ingestion pipeline, event-driven processing)
- **Object Store:** S3-compatible (media snapshots, model artifacts)

### 2.2 Core Schema

See `schemas/database.sql` for full DDL. Key tables:

#### `posts`
Primary post storage with all collected metrics.

#### `post_features`
Extracted feature vectors per post (one-to-one with `posts`).

#### `authors`
Author profiles with follower history snapshots.

#### `trends`
Trending topic snapshots with velocity metrics.

#### `strategies`
Generated strategy recommendations linked to input posts.

#### `ab_tests`
A/B test definitions, variants, and outcome tracking.

#### `model_versions`
ML model registry with version tracking and performance metrics.

### 2.3 Indexing Strategy

- B-tree on `posts.posted_at`, `posts.impressions`, `posts.author_id`
- GIN index on `posts.hashtags`, `posts.post_text` (full-text search)
- HNSW index on `post_features.embedding` (pgvector ANN search)
- Composite index on `(posted_at, impressions)` for time-windowed leaderboards
- Partial index on `posts WHERE impressions > 100000` for viral post fast-path queries

---

## 3. Analytical Engine

### 3.1 Feature Extraction Pipeline

All features are extracted per-post and stored in the `post_features` table. Each feature is normalized to [0, 1].

#### 3.1.1 Hook Strength (`hook_strength`)

Measures the stopping power of the first line.

```
hook_strength = w1 * has_number +
                w2 * has_power_word +
                w3 * first_line_brevity +
                w4 * curiosity_gap_signal +
                w5 * pattern_interrupt_score

where:
  has_number         = 1.0 if first line contains a specific number, else 0.0
  has_power_word     = count(power_words ∩ first_line_tokens) / len(first_line_tokens)
  first_line_brevity = 1.0 - (len(first_line) / 100)  # shorter = stronger
  curiosity_gap_signal = classifier_output (see 3.1.3)
  pattern_interrupt_score = 1.0 if first_line breaks expected syntax pattern, else scaled

  power_words = {"secret", "surprising", "shocking", "nobody", "everyone",
                 "actually", "truth", "mistake", "wrong", "free",
                 "instantly", "proven", "warning", "breaking", ...}
                 (full list: 200+ words, empirically validated)

  w1..w5 = learned via logistic regression on viral/non-viral training set
```

#### 3.1.2 Emotional Intensity (`emotional_intensity`)

```
emotional_intensity = α * |sentiment_compound| +
                      β * emotion_max_score +
                      γ * exclamation_density +
                      δ * caps_ratio

where:
  sentiment_compound = VADER or transformer-based sentiment compound score [-1, 1]
  emotion_max_score  = max(joy, anger, fear, surprise, disgust, sadness)
                       from GoEmotions classifier
  exclamation_density = count("!" | "?" | "!!") / word_count
  caps_ratio          = count(ALL_CAPS_WORDS) / word_count, capped at 0.15

  α, β, γ, δ = learned weights (see Section 4 for weight determination)
```

#### 3.1.3 Curiosity Gap (`curiosity_gap`)

Binary and continuous signal that the post withholds information to drive clicks/engagement.

```
Features for curiosity gap classifier:
  - contains_ellipsis (bool)
  - ends_with_colon (bool)
  - has_incomplete_list ("3 things..." without listing all 3)
  - has_cliffhanger_phrase ("you won't believe", "here's what happened", "the result?")
  - information_entropy of post text (high entropy + short length = gap)
  - question_without_answer (bool)

Model: fine-tuned DistilBERT binary classifier
Training data: 50K labeled posts (curiosity_gap = True/False)
Output: probability [0, 1]
```

#### 3.1.4 Controversy Index (`controversy_index`)

```
controversy_index = reply_ratio * polarization_score

where:
  reply_ratio = replies / (likes + reposts + 1)
  # High reply ratio relative to likes indicates divisive content

  polarization_score = sentiment_variance across reply sample
  # Calculated from a sample of up to 100 replies
  # High variance = polarized audience = controversy

  # For new posts without engagement data yet:
  predicted_controversy = topic_controversy_baseline[detected_topic] *
                          stance_strength(post_text)
```

#### 3.1.5 Readability Score (`readability_score`)

```
readability_score = normalize(
  0.4 * (1.0 - flesch_kincaid_grade / 20.0) +   # Target: grade 5-8
  0.3 * short_sentence_ratio +                     # Sentences ≤ 12 words
  0.2 * (1.0 - avg_word_length / 10.0) +          # Shorter words preferred
  0.1 * whitespace_ratio                           # Line breaks / formatting
)

# Optimal readability for X virality: Flesch-Kincaid grade 5-8
# Posts above grade 12 see 60% reduction in virality probability
```

#### 3.1.6 Structure Pattern Score (`structure_score`)

Detects known high-performing structural patterns.

```
Detected patterns (each scored independently):
  - LISTICLE:       "N things/reasons/ways..." → +0.15
  - THREAD_HOOK:    "🧵" or "Thread:" or "(1/N)" → +0.10
  - CONTRARIAN:     "Unpopular opinion:" / "Hot take:" → +0.12
  - STORY_OPEN:     "I just..." / "Last week I..." / "In 2019..." → +0.08
  - ONE_LINER:      Post ≤ 15 words, no links → +0.06
  - QUESTION:       Single question mark at end → +0.07
  - VERSUS:         "X vs Y" / "X or Y?" → +0.09
  - BEFORE_AFTER:   "Before: ... After: ..." or similar → +0.11
  - BOLD_CLAIM:     Declarative + superlative + no hedge words → +0.13

structure_score = max(pattern_scores) + 0.3 * second_max(pattern_scores)
# Primary pattern dominates; secondary pattern provides bonus
```

#### 3.1.7 Timing Score (`timing_score`)

```
timing_score = f(hour_of_day, day_of_week, author_timezone,
                 audience_active_hours, trend_velocity_at_time)

# Precomputed from historical data per audience segment:
engagement_heatmap[day][hour] = avg_engagement_rate for viral posts

timing_score = engagement_heatmap[post_day][post_hour] *
               trend_velocity_multiplier(trend_ids, posted_at)

# trend_velocity_multiplier:
#   1.0 if no trend alignment
#   1.0 + 0.5 * (trend_growth_rate / max_growth_rate) if aligned with rising trend
#   0.8 if aligned with declining trend (late to the trend)
```

#### 3.1.8 Trend Alignment (`trend_alignment`)

```
trend_alignment = max(
  cosine_similarity(post_embedding, trend_embedding_i)
  for trend_i in active_trends
) * trend_i.velocity_normalized

# Uses sentence-transformer embeddings (all-MiniLM-L6-v2 or similar)
# active_trends = trends from last 6 hours with positive velocity
```

### 3.2 NLP Techniques

| Technique | Model / Library | Purpose |
|---|---|---|
| Sentence embeddings | `all-MiniLM-L6-v2` (384-dim) | Semantic similarity, trend alignment, clustering |
| Sentiment analysis | Fine-tuned `cardiffnlp/twitter-roberta-base-sentiment-latest` | Compound sentiment + per-class scores |
| Emotion classification | `SamLowe/roberta-base-go_emotions` | 28-class emotion detection |
| Topic modeling | BERTopic with HDBSCAN | Unsupervised topic clusters over post corpus |
| Named entity recognition | SpaCy `en_core_web_trf` | Entity extraction for trend matching |
| Text generation | Fine-tuned GPT-class model or Claude API | Hook rewriting, post generation |
| Readability | `textstat` library | Flesch-Kincaid, syllable counts |
| Tokenization | `tiktoken` / SpaCy | Token counting, structure detection |

### 3.3 Pattern Recognition from Historical Viral Posts

#### Training Dataset Construction

```
1. Collect N = 500,000+ posts via X API search
2. Label binary:
     viral     = impressions ≥ 1,000,000  (top ~0.1%)
     non_viral = impressions < 100,000
3. Exclude middle band (100K–1M) to sharpen decision boundary
4. Balance dataset: undersample non_viral to 5:1 ratio
5. Split: 70% train, 15% validation, 15% test
6. Stratify by: author_follower_bucket, media_type, language
```

#### Pattern Extraction

```
For each viral post cluster (via BERTopic):
  - Extract centroid embedding
  - Extract top-N representative posts
  - Extract dominant structural pattern (Section 3.1.6)
  - Extract dominant emotional profile
  - Extract temporal distribution (when these go viral)
  - Store as VirPattern(cluster_id, centroid, features, examples)

# VirPattern matching for new posts:
similarity_to_known_patterns = max(
  cosine_sim(new_post_embedding, pattern.centroid) * pattern.avg_virality_rate
  for pattern in vir_patterns
)
```

### 3.4 ML Modeling Approach

#### Primary Model: Gradient Boosted Trees (XGBoost)

**Why:** Handles mixed feature types, interpretable feature importances, fast inference, strong on tabular data.

```
Input features (per post):
  x = [
    hook_strength,          # float [0,1]
    emotional_intensity,    # float [0,1]
    curiosity_gap,          # float [0,1]
    controversy_index,      # float [0,1]
    readability_score,      # float [0,1]
    structure_score,        # float [0,1]
    timing_score,           # float [0,1]
    trend_alignment,        # float [0,1]
    media_type_onehot,      # 5-dim binary vector
    post_length_normalized, # float [0,1]
    hashtag_count,          # int
    mention_count,          # int
    has_url,                # bool
    author_follower_log,    # float (log10 of follower count)
    author_engagement_rate, # float (avg engagement / followers)
    hour_of_day_cyclic_sin, # float
    hour_of_day_cyclic_cos, # float
    day_of_week_cyclic_sin, # float
    day_of_week_cyclic_cos, # float
    pattern_similarity,     # float [0,1]
    embedding_cluster_id,   # int (categorical)
  ]

Target: log10(impressions + 1)  — regression target
Secondary target: binary viral/non_viral — classification

Model configuration:
  xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=50,
  )
```

#### Secondary Model: Neural Ranker

For re-ranking post variants by predicted virality. Uses a pairwise ranking loss on (higher_viral, lower_viral) post pairs.

```
Architecture:
  Input: concatenated feature vector (21-dim) + post embedding (384-dim) = 405-dim
  → Linear(405, 256) → ReLU → Dropout(0.3)
  → Linear(256, 128) → ReLU → Dropout(0.2)
  → Linear(128, 1) → Sigmoid

Loss: MarginRankingLoss on paired samples
Use case: Given 5 candidate rewrites, rank by predicted virality
```

#### Reinforcement Learning Loop (V2+)

```
State:  current post features + audience state + time
Action: select rewrite variant / timing / format change
Reward: actual impressions received (delayed 48h)

Algorithm: Contextual bandit (LinUCB) for action selection
  - Each action arm = a strategy recommendation type
  - Context = post feature vector
  - Reward = normalized impression count

Update policy every 24h with new outcome data
Exploration rate: ε = 0.15, decaying to 0.05 over 90 days
```

---

## 4. Virality Scoring Algorithm

### 4.1 Scoring Formula

```
V(post) = Σ(wi * fi(post)) + interaction_terms + author_baseline

where:
  fi = feature_i extracted value (normalized [0, 1])
  wi = learned weight for feature_i

Primary features and initial weights (refined by training):
  w_hook       = 0.20  # Hook strength
  w_emotion    = 0.15  # Emotional intensity
  w_curiosity  = 0.15  # Curiosity gap
  w_structure  = 0.12  # Structure pattern score
  w_readability = 0.08 # Readability score
  w_timing     = 0.10  # Timing score
  w_trend      = 0.10  # Trend alignment
  w_controversy = 0.05 # Controversy index (diminishing returns above 0.6)
  w_format     = 0.05  # Media/formatting optimization

Interaction terms (captures non-linear effects):
  + 0.08 * hook_strength * curiosity_gap          # Hook + gap compound effect
  + 0.05 * trend_alignment * timing_score          # Right trend + right time
  + 0.04 * emotional_intensity * controversy_index # Emotion + controversy amplifier
  - 0.10 * max(0, controversy_index - 0.7)        # Penalty for extreme controversy

Author baseline:
  author_baseline = log10(follower_count) / 7.0 * 0.3
  # Accounts for organic reach; capped contribution of 0.3
  # A 10M-follower account gets 0.3; a 1K-follower account gets ~0.13

Final score:
  V_final = sigmoid(V_raw) → [0, 1]
  # Mapped to predicted impression buckets:
  #   0.0–0.3  →  < 10K impressions
  #   0.3–0.5  →  10K – 100K
  #   0.5–0.7  →  100K – 500K
  #   0.7–0.85 →  500K – 1M
  #   0.85–1.0 →  1M+ (target zone)
```

### 4.2 Weight Determination

```
Phase 1: Expert initialization
  - Set initial weights based on published engagement research
  - Validate against a 1,000-post hand-labeled calibration set

Phase 2: Data-driven optimization
  - Collect 100K+ posts with known impression outcomes
  - Train XGBoost regressor (Section 3.4)
  - Extract feature importances → normalize to sum = 1.0
  - Use SHAP values for per-prediction weight attribution
  - Update formula weights to match model's learned importances

Phase 3: Continuous refinement
  - Retrain model weekly on latest 90-day window
  - Track weight drift; alert if any weight shifts > 20% in a single week
  - A/B test weight variants on live predictions
  - Bayesian optimization over weight space every 30 days
```

### 4.3 Prediction Validation

```
Offline validation:
  - Metric: Spearman rank correlation between predicted and actual log(impressions)
  - Target: ρ ≥ 0.65 on held-out test set
  - Metric: Binary classification (viral / non-viral) AUC-ROC ≥ 0.82
  - Metric: Mean Absolute Error on log10(impressions) ≤ 0.5

Online validation:
  - Track prediction calibration: predicted percentile vs actual percentile
  - Maintain rolling 7-day calibration curve
  - Alert if Brier score degrades > 10% from baseline

Backtesting:
  - Monthly backtest on most recent 30 days of data (excluded from training)
  - Report: precision@k for top-100 predicted viral posts
  - Report: recall of actual 1M+ posts in top-10% of predictions
```

---

## 5. Million-Level Virality Strategy Generator

### 5.1 Hook Generation Logic

```python
# Hook Generator Pipeline
# Input: topic (string), target_emotion (string), context_keywords (list)
# Output: ranked list of hook candidates with predicted scores

HOOK_TEMPLATES = {
    "numbered_list":     "{N} {adjective} {topic_plural} that {outcome}:",
    "contrarian":        "Stop {common_advice}. Here's what actually works:",
    "story":             "I {past_action} and {unexpected_result}.",
    "question":          "Why do {percentage}% of {group} {fail_verb} at {topic}?",
    "bold_claim":        "{Topic} is {contrarian_adjective}. Here's proof:",
    "before_after":      "{Time_ago}: {before_state}. Today: {after_state}.",
    "secret":            "The {adjective} {topic} trick {authority_group} won't tell you:",
    "challenge":         "You can't {simple_task} without {unexpected_requirement}.",
    "prediction":        "By {year}, {bold_prediction}. Here's why:",
    "mistake":           "The #{N} mistake in {topic} (and {percentage}% don't know it):",
}

def generate_hooks(topic, target_emotion, context, n_candidates=20):
    candidates = []

    # Step 1: Template-based generation
    for template_name, template in HOOK_TEMPLATES.items():
        filled = fill_template(template, topic, context)  # LLM-assisted slot filling
        candidates.append(filled)

    # Step 2: LLM free-generation with few-shot examples
    examples = retrieve_similar_viral_hooks(topic, k=5)  # From viral post DB
    prompt = build_hook_generation_prompt(topic, target_emotion, examples)
    llm_hooks = call_llm(prompt, n=10)
    candidates.extend(llm_hooks)

    # Step 3: Score all candidates
    scored = []
    for hook in candidates:
        features = extract_features(hook)
        score = virality_model.predict(features)
        scored.append((hook, score))

    # Step 4: Diversity filter — remove near-duplicates
    scored = deduplicate_by_embedding(scored, threshold=0.85)

    # Step 5: Return top-N ranked by score
    return sorted(scored, key=lambda x: x[1], reverse=True)[:n_candidates]
```

### 5.2 Psychological Triggers Framework

Each trigger maps to measurable text features and has an associated amplification strategy.

| Trigger | Detection Method | Amplification |
|---|---|---|
| **Loss Aversion** | Presence of loss-framed language ("miss out", "lose", "before it's gone") | Reframe gains as potential losses: "X you're losing" > "X you could gain" |
| **Social Proof** | Numbers + group reference ("10K people", "top 1%") | Inject specific quantities: vague → "87% of founders" |
| **Authority** | Entity recognition of known figures, institutions, credentials | Reference specific credible sources |
| **Curiosity Gap** | Incomplete information + promise of revelation (see 3.1.3) | End first line with colon, ellipsis, or open question |
| **Tribalism** | In-group/out-group language ("real developers", "most people don't") | Strengthen identity signaling for target audience |
| **Novelty** | Temporal markers ("just discovered", "new", "2026") + low corpus frequency | Lead with recency markers; use "first time" framing |
| **Contrarianism** | Negation of common belief + stance classifier | Invert a widely-held assumption; add "actually" |
| **Reciprocity** | Value-first framing ("free", "here's the guide", "sharing my") | Front-load the value; make it feel like a gift |

```python
def apply_psychological_triggers(post_text, target_triggers, max_triggers=3):
    """
    Enhance post text with specified psychological triggers.
    Limit to max_triggers to avoid manipulation signal overload.
    """
    current_triggers = detect_triggers(post_text)
    missing = [t for t in target_triggers if t not in current_triggers]

    for trigger in missing[:max_triggers - len(current_triggers)]:
        post_text = TRIGGER_AMPLIFIERS[trigger](post_text)

    return post_text
```

### 5.3 Formatting Optimization

```python
FORMAT_RULES = {
    "line_break_after_hook": {
        "condition": lambda text: len(text.split('\n')[0]) > 40,
        "action": "Insert line break after first sentence",
        "impact": +0.04
    },
    "visual_separators": {
        "condition": lambda text: text.count('\n') > 2 and '→' not in text,
        "action": "Replace bullet hyphens with → or • for scannability",
        "impact": +0.02
    },
    "whitespace_breathing": {
        "condition": lambda text: max(len(line) for line in text.split('\n')) > 60,
        "action": "Break long lines at natural clause boundaries",
        "impact": +0.03
    },
    "remove_hashtag_spam": {
        "condition": lambda text: text.count('#') > 2,
        "action": "Reduce to 0-1 hashtags; move to end if kept",
        "impact": +0.03
    },
    "emoji_moderation": {
        "condition": lambda text: count_emojis(text) > 3,
        "action": "Reduce to 0-2 emojis; use only as visual anchors",
        "impact": +0.02
    },
    "cta_placement": {
        "condition": lambda text: not has_engagement_cta(text),
        "action": "Add soft CTA: question, 'agree?', 'thoughts?', 'repost if...'",
        "impact": +0.05
    }
}

def optimize_format(post_text):
    for rule_name, rule in FORMAT_RULES.items():
        if rule["condition"](post_text):
            post_text = apply_format_rule(post_text, rule)
    return post_text
```

### 5.4 Timing Optimization

```python
class TimingOptimizer:
    def __init__(self, engagement_history_db):
        # Build engagement heatmap from historical viral posts
        # Shape: (7 days, 24 hours) → avg engagement rate
        self.global_heatmap = self._build_heatmap(engagement_history_db)
        self.audience_heatmaps = {}  # Per-audience-segment heatmaps

    def get_optimal_windows(self, author_id, topic, n_windows=3):
        """
        Returns top-N posting windows ranked by expected engagement.
        Factors: global patterns, audience-specific activity, trend timing.
        """
        # Layer 1: Global engagement patterns
        global_scores = self.global_heatmap.copy()

        # Layer 2: Author's audience activity pattern
        if author_id in self.audience_heatmaps:
            audience_scores = self.audience_heatmaps[author_id]
            combined = 0.4 * global_scores + 0.6 * audience_scores
        else:
            combined = global_scores

        # Layer 3: Trend timing bonus
        active_trends = get_trends_for_topic(topic)
        for trend in active_trends:
            peak_time = trend.predicted_peak_time
            for hour_offset in range(-2, 3):
                t = (peak_time + hour_offset) % 24
                d = peak_time.day_of_week
                combined[d][t] *= (1.0 + 0.3 * trend.velocity * gaussian(hour_offset, 0, 1.5))

        # Layer 4: Competition avoidance
        # Slightly penalize windows where many high-follower accounts typically post
        competition = get_competition_density(topic, combined)
        combined *= (1.0 - 0.1 * competition)

        # Return top windows
        flat = [(d, h, combined[d][h]) for d in range(7) for h in range(24)]
        flat.sort(key=lambda x: x[2], reverse=True)
        return flat[:n_windows]
```

### 5.5 Iterative A/B Testing Loop

```python
class ABTestOrchestrator:
    """
    Manages live A/B testing of post variants to continuously
    improve the virality model and strategy generator.
    """

    def create_test(self, base_post, n_variants=3):
        """
        Generate variants of a post for split testing.
        """
        variants = []

        # Variant A: Original post (control)
        variants.append({"type": "control", "text": base_post})

        # Variant B: Hook rewrite
        new_hook = self.hook_generator.generate_hooks(
            extract_topic(base_post), target_emotion="curiosity", context=[]
        )[0]
        variants.append({
            "type": "hook_rewrite",
            "text": replace_hook(base_post, new_hook)
        })

        # Variant C: Format + trigger optimization
        optimized = optimize_format(base_post)
        optimized = apply_psychological_triggers(optimized, ["curiosity_gap", "social_proof"])
        variants.append({"type": "format_trigger", "text": optimized})

        return ABTest(
            variants=variants,
            metric="impressions",
            min_sample_time_hours=48,
            confidence_level=0.90
        )

    def evaluate_test(self, test_id):
        """
        Evaluate test results after minimum observation window.
        Uses sequential testing (mSPRT) for early stopping.
        """
        test = self.db.get_test(test_id)
        results = []

        for variant in test.variants:
            metrics = self.fetch_metrics(variant.post_id)
            results.append({
                "variant": variant.type,
                "impressions": metrics.impressions,
                "engagement_rate": (metrics.likes + metrics.reposts) / max(metrics.impressions, 1),
                "repost_velocity": metrics.reposts / max(metrics.hours_since_post, 1),
            })

        winner = max(results, key=lambda r: r["impressions"])

        # Feed outcome back into model training
        self.model_trainer.add_outcome(test, results)

        # Update strategy weights based on which variant type won
        self.strategy_weights.update(winner["variant"], reward=1.0)

        return results, winner

    def continuous_loop(self):
        """
        Main loop: generate → test → learn → repeat.
        Runs as a background service.
        """
        while True:
            # 1. Get next post from queue
            post = self.post_queue.dequeue()

            # 2. Generate variants
            test = self.create_test(post.text)

            # 3. Schedule posts at optimal times
            windows = self.timing_optimizer.get_optimal_windows(
                post.author_id, extract_topic(post.text)
            )
            self.scheduler.schedule(test, windows)

            # 4. Wait for minimum observation window
            # (handled by background job, not blocking)
            self.schedule_evaluation(test.id, delay_hours=48)

            # 5. After evaluation, retrain model if enough new data
            if self.outcome_buffer.size() >= RETRAIN_THRESHOLD:
                self.retrain_model()
```

---

## 6. Evaluation Metrics

### 6.1 Post-Level Metrics

| Metric | Formula | Target |
|---|---|---|
| **Impressions** | Raw count from X API | ≥ 1,000,000 |
| **Engagement Rate** | (likes + reposts + replies + quotes) / impressions | ≥ 2.0% |
| **Repost Velocity** | reposts / hours_to_peak | ≥ 500 reposts/hour in first 4h |
| **Virality Coefficient** | reposts / impressions * avg_repost_reach | ≥ 0.05 (each impression generates 0.05 new impressions) |
| **Reply-to-Like Ratio** | replies / likes | 0.05–0.20 (healthy engagement) |
| **Bookmark Rate** | bookmarks / impressions | ≥ 0.5% (indicates save-worthy content) |
| **Profile Click Rate** | profile_visits / impressions | ≥ 0.3% |
| **Follow Conversion** | new_followers_48h / impressions | ≥ 0.01% |

### 6.2 System-Level Metrics

| Metric | Formula | Target |
|---|---|---|
| **Prediction Accuracy** | Spearman ρ (predicted vs actual log impressions) | ≥ 0.65 |
| **Classification AUC** | ROC-AUC for viral/non-viral binary classification | ≥ 0.82 |
| **Hit Rate** | % of posts scoring ≥ 0.85 that achieve 1M+ impressions | ≥ 25% |
| **Strategy Lift** | avg(optimized_impressions) / avg(baseline_impressions) | ≥ 3.0x |
| **Model Freshness** | Time since last retrain | ≤ 7 days |
| **Calibration Error** | Mean absolute deviation of predicted vs actual percentiles | ≤ 5 percentile points |
| **A/B Test Throughput** | Tests completed per week | ≥ 10 |
| **Rewrite Acceptance Rate** | % of suggested rewrites accepted by user | ≥ 60% |

### 6.3 Drift Detection

```
Monitor weekly:
  - Feature distribution shifts (KL divergence per feature)
  - Prediction residual trends (increasing MAE → model stale)
  - Engagement rate baselines (X algorithm changes)
  - API response schema changes (breaking changes)

Alert thresholds:
  - KL divergence > 0.1 on any feature → investigate
  - MAE increase > 15% week-over-week → trigger retrain
  - Engagement rate baseline shift > 20% → recalibrate scoring buckets
```

---

## 7. Development Roadmap

### Phase 1: MVP (Weeks 1–6)

**Goal:** Score existing posts. No generation, no live testing.

| Week | Deliverable |
|---|---|
| 1–2 | X API v2 integration; data ingestion pipeline; PostgreSQL schema; collect 50K posts |
| 3–4 | Feature extraction pipeline (all 8 features); NLP model integration; batch processing |
| 5 | XGBoost model training on initial dataset; scoring API endpoint |
| 6 | Dashboard UI (input post → score + feature breakdown); manual validation on 500 posts |

**Exit criteria:** Spearman ρ ≥ 0.50 on test set; system processes 1,000 posts/minute.

### Phase 2: V2 — Strategy Generation (Weeks 7–14)

**Goal:** Generate actionable rewrite suggestions and timing recommendations.

| Week | Deliverable |
|---|---|
| 7–8 | Hook generator (template + LLM); psychological trigger detection and amplification |
| 9–10 | Formatting optimizer; timing optimizer with engagement heatmaps |
| 11–12 | Full strategy output (JSON format from Section 1.4); variant ranking via neural ranker |
| 13 | Trend tracking streaming pipeline; trend alignment scoring |
| 14 | Integration testing; strategy quality evaluation; user acceptance testing |

**Exit criteria:** Strategy lift ≥ 2.0x on backtested data; rewrite acceptance rate ≥ 50%.

### Phase 3: Scale — Live Testing & Self-Improvement (Weeks 15–24)

**Goal:** Close the feedback loop. Deploy, test, learn, improve automatically.

| Week | Deliverable |
|---|---|
| 15–16 | A/B test orchestrator; post scheduling integration; outcome tracking |
| 17–18 | Reinforcement learning loop (contextual bandit); continuous model retraining pipeline |
| 19–20 | Drift detection system; automated alerting; model versioning and rollback |
| 21–22 | Multi-account support; audience segmentation; per-audience strategy tuning |
| 23–24 | Performance optimization; horizontal scaling; API rate limiting; documentation |

**Exit criteria:** Hit rate ≥ 25% for 1M+ predictions; model retrains automatically weekly; system handles 100 concurrent users.

### Beyond V3: Future Capabilities

- Image/video content analysis for media optimization
- Cross-platform expansion (LinkedIn, Threads, Bluesky)
- Autonomous posting agent (full loop: generate → post → measure → iterate)
- Audience graph analysis (follower overlap, influencer network mapping)
- Real-time algorithm reverse-engineering (detect X feed ranking signal changes)
