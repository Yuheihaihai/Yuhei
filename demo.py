#!/usr/bin/env python3
"""
X Virality Optimization Engine — Runnable Demo
Exercises the full scoring pipeline with mock NLP backends.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field

from src.core.virality_engine import (
    HOOK_TEMPLATES,
    POWER_WORDS,
    STRUCTURE_PATTERNS,
    WEAKNESS_THRESHOLDS,
    WEIGHTS,
    FeatureExtractor,
    FeatureVector,
    MediaType,
    PostInput,
    Recommendation,
    StrategyGenerator,
    StrategyOutput,
    ViralityScore,
    ViralityScorer,
    _clamp,
    _cosine_similarity,
    _get_format_suggestions,
    _get_trigger_suggestions,
    _recommend_engagement_close,
    _recommend_hashtags,
    _recommend_media,
    _recommend_thread,
    _replace_first_line,
    _sigmoid,
)


# ---------------------------------------------------------------------------
# Mock NLP pipeline (replaces real models for demo purposes)
# ---------------------------------------------------------------------------

class MockNLP:
    """Lightweight mock that uses heuristics instead of real ML models."""

    def sentiment_compound(self, text: str) -> float:
        positive = {"great", "amazing", "best", "love", "incredible", "genius", "brilliant"}
        negative = {"worst", "terrible", "hate", "awful", "wrong", "mistake", "fail", "destroyed"}
        words = set(text.lower().split())
        pos = len(positive & words)
        neg = len(negative & words)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def emotion_classify(self, text: str) -> dict[str, float]:
        lower = text.lower()
        scores = {
            "joy": 0.1, "anger": 0.1, "fear": 0.1,
            "surprise": 0.1, "disgust": 0.05, "sadness": 0.05,
        }
        if any(w in lower for w in ["amazing", "love", "great", "genius"]):
            scores["joy"] = 0.75
        if any(w in lower for w in ["shocking", "exposed", "leaked", "breaking"]):
            scores["surprise"] = 0.80
        if any(w in lower for w in ["wrong", "mistake", "warning", "dangerous"]):
            scores["fear"] = 0.65
        if any(w in lower for w in ["hate", "angry", "destroyed", "worst"]):
            scores["anger"] = 0.70
        if "?" in text:
            scores["surprise"] = max(scores["surprise"], 0.40)
        return scores

    def curiosity_gap_predict(self, text: str) -> float:
        score = 0.1
        if text.strip().endswith(":"):
            score += 0.35
        if "..." in text:
            score += 0.20
        if any(p in text.lower() for p in ["you won't believe", "here's what", "the result", "here's why"]):
            score += 0.30
        if text.strip().endswith("?"):
            score += 0.15
        if re.search(r'\d+\s+(things|reasons|ways|tips|secrets|mistakes)', text.lower()):
            score += 0.25
        return min(score, 1.0)

    def detect_topic(self, text: str) -> str:
        topics = {
            "tech": ["ai", "code", "software", "tech", "startup", "saas", "app", "developer"],
            "finance": ["money", "invest", "stock", "crypto", "wealth", "income", "revenue"],
            "career": ["job", "career", "hire", "interview", "resume", "work", "salary"],
            "health": ["health", "fitness", "diet", "sleep", "mental", "exercise"],
            "marketing": ["brand", "marketing", "growth", "viral", "audience", "content"],
        }
        lower = text.lower()
        best_topic, best_count = "general", 0
        for topic, keywords in topics.items():
            count = sum(1 for k in keywords if k in lower)
            if count > best_count:
                best_topic, best_count = topic, count
        return best_topic

    def stance_strength(self, text: str) -> float:
        strong = {"always", "never", "every", "nobody", "everyone", "only", "must", "guaranteed"}
        words = set(text.lower().split())
        return min(len(strong & words) * 0.25, 1.0)

    def flesch_kincaid_grade(self, text: str) -> float:
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        words = text.split()
        word_count = max(len(words), 1)
        syllables = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in words)
        return 0.39 * (word_count / sentences) + 11.8 * (syllables / word_count) - 15.59

    def sentence_split(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]

    def embed(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding based on character hashing."""
        random.seed(hash(text) % (2**31))
        return [random.gauss(0, 1) for _ in range(384)]

    def pattern_interrupt_score(self, text: str) -> float:
        if text.startswith(("What if", "Imagine", "Picture this", "Nobody")):
            return 0.7
        if any(text.startswith(str(n)) for n in range(10)):
            return 0.5
        return 0.2


class MockTrendStore:
    """Mock trend store with simulated active trends."""

    @dataclass
    class Trend:
        name: str
        embedding: list[float]
        velocity_normalized: float

    def __init__(self):
        random.seed(42)
        self.trends = [
            self.Trend("AI agents", self._rand_emb("AI agents"), 0.85),
            self.Trend("startup layoffs", self._rand_emb("startup layoffs"), 0.60),
            self.Trend("remote work debate", self._rand_emb("remote work debate"), 0.45),
        ]

    def _rand_emb(self, seed: str) -> list[float]:
        random.seed(hash(seed) % (2**31))
        return [random.gauss(0, 1) for _ in range(384)]

    def get_active(self, hours: int = 6) -> list:
        return self.trends

    def engagement_heatmap_lookup(self, dow: int, hour: int) -> float:
        # Peak: Tue-Thu 13-16 UTC
        day_factor = {0: 0.6, 1: 0.85, 2: 0.90, 3: 0.85, 4: 0.70, 5: 0.40, 6: 0.35}
        hour_peak = math.exp(-0.5 * ((hour - 14.5) / 3.0) ** 2)
        return _clamp(day_factor.get(dow, 0.5) * hour_peak)


class MockPatternDB:
    """Mock viral pattern database."""

    @dataclass
    class VirPattern:
        cluster_id: int
        centroid: list[float]
        avg_virality_rate: float

    def __init__(self):
        random.seed(99)
        self.patterns = [
            self.VirPattern(0, self._rand_emb(0), 0.75),
            self.VirPattern(1, self._rand_emb(1), 0.60),
            self.VirPattern(2, self._rand_emb(2), 0.50),
        ]

    def _rand_emb(self, seed: int) -> list[float]:
        random.seed(seed + 1000)
        return [random.gauss(0, 1) for _ in range(384)]

    def get_viral_patterns(self) -> list:
        return self.patterns

    def nearest_cluster(self, emb: list[float]) -> int:
        best_id, best_sim = -1, -1.0
        for p in self.patterns:
            sim = _cosine_similarity(emb, p.centroid)
            if sim > best_sim:
                best_id, best_sim = p.cluster_id, sim
        return best_id

    def topic_controversy_baseline(self, topic: str) -> float:
        baselines = {"tech": 0.3, "finance": 0.4, "career": 0.35, "health": 0.25, "marketing": 0.2, "general": 0.15}
        return baselines.get(topic, 0.15)


class MockHookGenerator:
    """Generate hook variants using templates (no LLM needed)."""

    def generate(self, topic: str, target_emotion: str, context: list[str]) -> list[tuple[str, float]]:
        hooks = [
            (f"5 {topic} mistakes that cost you everything:", 0.88),
            (f"I spent 10,000 hours studying {topic}. Here's what nobody tells you:", 0.85),
            (f"The {topic} trick that 99% of people don't know:", 0.82),
            (f"Stop doing {topic} wrong. Here's the truth:", 0.79),
            (f"Why {topic} will look completely different by 2028:", 0.76),
        ]
        return hooks


class MockTimingOptimizer:
    """Mock timing optimizer."""

    def get_optimal_windows(self, author_id, topic: str, n_windows: int = 3) -> list[tuple]:
        return [
            ("Tuesday", 14, 0.92),
            ("Wednesday", 13, 0.88),
            ("Thursday", 15, 0.85),
        ]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

SAMPLE_POSTS = [
    {
        "text": "AI will change everything in 2026.",
        "label": "Weak: vague claim, no hook, no curiosity gap",
    },
    {
        "text": "5 AI tools that replaced my entire marketing team:\n\n1. Copy → ChatGPT\n2. Design → Midjourney\n3. Analytics → Obviously AI\n4. Scheduling → Typefully\n5. Research → Perplexity\n\nTotal cost: $200/month vs $35K/month.\n\nThe future is already here.",
        "label": "Strong: listicle, specific numbers, curiosity hook, contrast",
    },
    {
        "text": "Unpopular opinion: Most startup advice is wrong.\n\nVCs tell you to \"move fast and break things.\"\n\nBut the founders who win?\n\nThey move slow and build things that don't break.\n\nSpeed is overrated. Durability is underrated.",
        "label": "Strong: contrarian frame, pattern interrupt, rhythm, bold claim",
    },
    {
        "text": "Just had a meeting that could have been an email",
        "label": "Moderate: relatable, but no hook or curiosity gap",
    },
    {
        "text": "I analyzed 10,000 viral posts on X.\n\n87% shared one thing in common:\n\nThe first line made you stop scrolling.\n\nHere's the exact formula (thread):",
        "label": "Strong: data-backed, curiosity gap, thread hook, specific number",
    },
]


def format_score_bar(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:.2f}"


def print_separator(char: str = "═", width: int = 72):
    print(char * width)


def run_demo():
    print()
    print_separator()
    print("  X VIRALITY OPTIMIZATION ENGINE — LIVE DEMO")
    print_separator()
    print()

    # Initialize with mocks
    nlp = MockNLP()
    trends = MockTrendStore()
    patterns = MockPatternDB()

    extractor = FeatureExtractor(nlp, trends, patterns)
    scorer = ViralityScorer(model=None, weights=WEIGHTS)
    hook_gen = MockHookGenerator()
    timing_opt = MockTimingOptimizer()
    strategy_gen = StrategyGenerator(extractor, scorer, hook_gen, timing_opt)

    for i, sample in enumerate(SAMPLE_POSTS, 1):
        text = sample["text"]
        label = sample["label"]

        print(f"  POST #{i}")
        print(f"  Label: {label}")
        print(f"  ┌{'─' * 68}┐")
        for line in text.split("\n"):
            print(f"  │ {line:<66} │")
        print(f"  └{'─' * 68}┘")
        print()

        # --- Feature extraction ---
        post = PostInput(
            text=text,
            author_follower_count=50_000,
            author_engagement_rate=0.025,
            posted_at="2026-02-25T14:00:00Z",
        )
        fv = extractor.extract(post)
        score = scorer.score(fv)

        # --- Feature breakdown ---
        print("  Feature Breakdown:")
        for feat_name, feat_val in score.feature_breakdown.items():
            print(f"    {feat_name:<22} {format_score_bar(feat_val)}")
        print()

        # --- Virality score ---
        category_icons = {
            "high_potential": "🟢",
            "medium_potential": "🟡",
            "moderate": "🟠",
            "low_potential": "🔴",
        }
        icon = category_icons.get(score.category, "⚪")
        print(f"  Virality Score:       {format_score_bar(score.final_score)}")
        print(f"  Category:             {icon} {score.category}")
        print(f"  Predicted Impressions: p50={score.predicted_impressions_p50:>10,}  "
              f"p75={score.predicted_impressions_p75:>10,}  "
              f"p90={score.predicted_impressions_p90:>10,}")
        print()

        # --- Weaknesses ---
        if score.weaknesses:
            print("  Weaknesses Detected:")
            for w in score.weaknesses:
                print(f"    ⚠  {w}")
            print()

        # --- Strategy recommendations ---
        strategy = strategy_gen.generate(post)

        if strategy.recommendations:
            print("  Recommendations:")
            for rec in strategy.recommendations:
                print(f"    [{rec.rec_type.upper()}] score_delta: {rec.score_delta:+.3f}")
                if rec.rec_type == "rewrite" and "rewritten" in rec.detail:
                    print(f"      Original:  \"{rec.detail['original']}\"")
                    print(f"      Rewritten: \"{rec.detail['rewritten']}\"")
                    print(f"      Rationale: {rec.detail.get('rationale', '')}")
                elif rec.rec_type == "timing":
                    windows = rec.detail.get("optimal_windows", [])
                    for w in windows[:3]:
                        print(f"      {w[0]} at {w[1]}:00 UTC (score: {w[2]:.2f})")
                elif rec.rec_type == "format":
                    for s in rec.detail.get("suggestions", []):
                        print(f"      → {s['action']} (impact: +{s['impact']:.2f})")
                elif rec.rec_type == "trigger":
                    for t in rec.detail.get("triggers", []):
                        print(f"      → {t}")
            print()

        if strategy.hook_variants:
            print("  Hook Variants (ranked by predicted score):")
            for j, hook in enumerate(strategy.hook_variants[:3], 1):
                print(f"    {j}. \"{hook}\"")
            print()

        print(f"  Media:    {strategy.media_recommendation}")
        print(f"  Hashtags: {json.dumps(strategy.hashtag_strategy)}")
        print(f"  CTA type: {strategy.engagement_bait_type}")
        print(f"  Thread:   {'Recommended (' + str(strategy.thread_expansion.get('optimal_length', 0)) + ' posts)' if strategy.thread_expansion.get('recommended') else 'Not recommended'}")

        print()
        print_separator("─")
        print()

    # --- Summary table ---
    print("  COMPARISON SUMMARY")
    print_separator("─")
    print(f"  {'#':<4} {'Score':<8} {'Category':<18} {'p50 Impressions':<18} {'First 40 chars...'}")
    print_separator("─")

    for i, sample in enumerate(SAMPLE_POSTS, 1):
        post = PostInput(
            text=sample["text"],
            author_follower_count=50_000,
            author_engagement_rate=0.025,
            posted_at="2026-02-25T14:00:00Z",
        )
        fv = extractor.extract(post)
        score = scorer.score(fv)
        first_40 = sample["text"][:40].replace("\n", " ")
        print(f"  {i:<4} {score.final_score:<8.3f} {score.category:<18} {score.predicted_impressions_p50:<18,} {first_40}...")

    print()
    print_separator()
    print("  Engine demo complete. All systems operational.")
    print_separator()
    print()


if __name__ == "__main__":
    run_demo()
