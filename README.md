# X Virality Optimization Engine

Data-driven system for analyzing X (Twitter) posts and generating strategies targeting 1M+ impressions.

## Architecture

```
Ingestion → Feature Extraction → Virality Scoring → Strategy Generation → A/B Testing → Feedback Loop
```

## Project Structure

```
├── docs/
│   └── TECHNICAL_SPECIFICATION.md   # Full system design & algorithm specification
├── schemas/
│   └── database.sql                 # PostgreSQL + pgvector schema (12 tables)
├── src/
│   └── core/
│       └── virality_engine.py       # Core algorithm: feature extraction, scoring, strategy generation
└── README.md
```

## Core Components

| Component | Description |
|---|---|
| **Feature Extractor** | 8 normalized features: hook strength, emotional intensity, curiosity gap, controversy index, readability, structure patterns, timing, trend alignment |
| **Virality Scorer** | Weighted formula + XGBoost model blend with interaction terms and author baseline |
| **Strategy Generator** | Hook rewriting, timing optimization, format optimization, psychological trigger framework |
| **A/B Test Orchestrator** | Variant generation, scheduling, outcome tracking, sequential testing |

## Scoring Formula

```
V(post) = Σ(wi * fi) + interaction_terms + author_baseline
→ sigmoid mapping to [0, 1]
→ 0.85+ = predicted 1M+ impressions
```

## Tech Stack

- **Language:** Python 3.11+
- **ML:** XGBoost, sentence-transformers, HuggingFace (RoBERTa, DistilBERT)
- **NLP:** SpaCy, textstat, BERTopic
- **Database:** PostgreSQL 16 + pgvector
- **Queue:** Redis Streams / Kafka
- **API:** X API v2

## Development Roadmap

- **MVP (Weeks 1–6):** Data ingestion, feature extraction, scoring API
- **V2 (Weeks 7–14):** Strategy generation, hook rewriting, timing optimization
- **Scale (Weeks 15–24):** Live A/B testing, reinforcement learning, drift detection

## Evaluation Targets

| Metric | Target |
|---|---|
| Prediction accuracy (Spearman ρ) | ≥ 0.65 |
| Classification AUC-ROC | ≥ 0.82 |
| Hit rate (0.85+ score → 1M+ actual) | ≥ 25% |
| Strategy lift | ≥ 3.0x |

See [`docs/TECHNICAL_SPECIFICATION.md`](docs/TECHNICAL_SPECIFICATION.md) for the full specification.
