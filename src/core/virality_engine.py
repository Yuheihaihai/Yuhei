"""
X Virality Optimization Engine — Core Algorithm
Production-ready pseudocode with complete pipeline logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class MediaType(Enum):
    NONE = "none"
    IMAGE = "image"
    VIDEO = "video"
    GIF = "gif"
    POLL = "poll"


@dataclass
class PostInput:
    text: str
    media_type: MediaType = MediaType.NONE
    media_urls: list[str] = field(default_factory=list)
    author_follower_count: int = 0
    author_engagement_rate: float = 0.0
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    posted_at: Optional[str] = None  # ISO-8601 UTC
    language: str = "en"


@dataclass
class FeatureVector:
    hook_strength: float = 0.0
    emotional_intensity: float = 0.0
    curiosity_gap: float = 0.0
    controversy_index: float = 0.0
    readability_score: float = 0.0
    structure_score: float = 0.0
    timing_score: float = 0.0
    trend_alignment: float = 0.0
    media_type_vec: list[float] = field(default_factory=lambda: [0.0] * 5)
    post_length_norm: float = 0.0
    hashtag_count: int = 0
    mention_count: int = 0
    has_url: bool = False
    author_follower_log: float = 0.0
    author_engagement_rate: float = 0.0
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0
    pattern_similarity: float = 0.0
    cluster_id: int = -1

    def to_list(self) -> list[float]:
        return [
            self.hook_strength,
            self.emotional_intensity,
            self.curiosity_gap,
            self.controversy_index,
            self.readability_score,
            self.structure_score,
            self.timing_score,
            self.trend_alignment,
            *self.media_type_vec,
            self.post_length_norm,
            float(self.hashtag_count),
            float(self.mention_count),
            float(self.has_url),
            self.author_follower_log,
            self.author_engagement_rate,
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
            self.pattern_similarity,
            float(self.cluster_id),
        ]


@dataclass
class ViralityScore:
    raw_score: float
    final_score: float  # sigmoid-mapped [0, 1]
    predicted_impressions_p50: int
    predicted_impressions_p75: int
    predicted_impressions_p90: int
    feature_breakdown: dict[str, float] = field(default_factory=dict)
    weaknesses: list[str] = field(default_factory=list)
    category: str = "unknown"


@dataclass
class Recommendation:
    rec_type: str  # "rewrite" | "timing" | "format" | "trigger"
    detail: dict = field(default_factory=dict)
    score_delta: float = 0.0


@dataclass
class StrategyOutput:
    score: ViralityScore
    recommendations: list[Recommendation]
    hook_variants: list[str]
    thread_expansion: dict = field(default_factory=dict)
    media_recommendation: str = ""
    hashtag_strategy: dict = field(default_factory=dict)
    engagement_bait_type: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POWER_WORDS = frozenset({
    "secret", "surprising", "shocking", "nobody", "everyone",
    "actually", "truth", "mistake", "wrong", "free", "instantly",
    "proven", "warning", "breaking", "urgent", "exclusive", "hidden",
    "banned", "illegal", "controversial", "insane", "genius",
    "underrated", "overrated", "deadly", "critical", "massive",
    "ultimate", "revolutionary", "destroyed", "exposed", "leaked",
})

HOOK_TEMPLATES = {
    "numbered_list":  "{n} {adjective} {topic_plural} that {outcome}:",
    "contrarian":     "Stop {common_advice}. Here's what actually works:",
    "story":          "I {past_action} and {unexpected_result}.",
    "question":       "Why do {percentage}% of {group} {fail_verb} at {topic}?",
    "bold_claim":     "{topic} is {contrarian_adjective}. Here's proof:",
    "before_after":   "{time_ago}: {before_state}. Today: {after_state}.",
    "secret":         "The {adjective} {topic} trick {authority_group} won't tell you:",
    "challenge":      "You can't {simple_task} without {unexpected_requirement}.",
    "prediction":     "By {year}, {bold_prediction}. Here's why:",
    "mistake":        "The #{n} mistake in {topic} (and {percentage}% don't know it):",
}

STRUCTURE_PATTERNS = {
    "LISTICLE":    0.15,
    "THREAD_HOOK": 0.10,
    "CONTRARIAN":  0.12,
    "STORY_OPEN":  0.08,
    "ONE_LINER":   0.06,
    "QUESTION":    0.07,
    "VERSUS":      0.09,
    "BEFORE_AFTER": 0.11,
    "BOLD_CLAIM":  0.13,
}

# Scoring formula weights (initial; refined by training — see spec §4.2)
WEIGHTS = {
    "hook":         0.20,
    "emotion":      0.15,
    "curiosity":    0.15,
    "structure":    0.12,
    "readability":  0.08,
    "timing":       0.10,
    "trend":        0.10,
    "controversy":  0.05,
    "format":       0.05,
}

INTERACTION_COEFFICIENTS = {
    "hook_x_curiosity":     0.08,
    "trend_x_timing":       0.05,
    "emotion_x_controversy": 0.04,
    "controversy_penalty":  -0.10,
}

IMPRESSION_BUCKETS = [
    (0.00, 0.30, 10_000),
    (0.30, 0.50, 100_000),
    (0.50, 0.70, 500_000),
    (0.70, 0.85, 1_000_000),
    (0.85, 1.00, 5_000_000),
]

WEAKNESS_THRESHOLDS = {
    "hook_strength":       0.70,
    "emotional_intensity": 0.50,
    "curiosity_gap":       0.50,
    "readability_score":   0.70,
    "timing_score":        0.80,
    "trend_alignment":     0.40,
    "structure_score":     0.50,
}


# ---------------------------------------------------------------------------
# Feature extraction (stubs — each calls the real NLP / stats libraries)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts all features from a PostInput.
    In production, each method delegates to the appropriate NLP model.
    """

    def __init__(self, nlp_pipeline, trend_store, viral_pattern_db):
        self.nlp = nlp_pipeline
        self.trends = trend_store
        self.patterns = viral_pattern_db

    def extract(self, post: PostInput) -> FeatureVector:
        fv = FeatureVector()
        fv.hook_strength = self._hook_strength(post.text)
        fv.emotional_intensity = self._emotional_intensity(post.text)
        fv.curiosity_gap = self._curiosity_gap(post.text)
        fv.controversy_index = self._controversy_index(post.text)
        fv.readability_score = self._readability(post.text)
        fv.structure_score = self._structure_score(post.text)
        fv.timing_score = self._timing_score(post.posted_at)
        fv.trend_alignment = self._trend_alignment(post.text)
        fv.media_type_vec = self._media_onehot(post.media_type)
        fv.post_length_norm = min(len(post.text) / 280.0, 1.0)
        fv.hashtag_count = len(post.hashtags)
        fv.mention_count = len(post.mentions)
        fv.has_url = len(post.urls) > 0
        fv.author_follower_log = (
            math.log10(max(post.author_follower_count, 1))
        )
        fv.author_engagement_rate = post.author_engagement_rate
        fv.pattern_similarity = self._pattern_similarity(post.text)
        fv.cluster_id = self._cluster_id(post.text)
        if post.posted_at:
            fv.hour_sin, fv.hour_cos = self._cyclic_hour(post.posted_at)
            fv.dow_sin, fv.dow_cos = self._cyclic_dow(post.posted_at)
        return fv

    # -- Hook strength -------------------------------------------------------

    def _hook_strength(self, text: str) -> float:
        first_line = text.split("\n")[0].strip()
        tokens = first_line.lower().split()

        has_number = float(any(tok.isdigit() for tok in tokens))
        power_word_ratio = (
            len(POWER_WORDS.intersection(tokens)) / max(len(tokens), 1)
        )
        brevity = max(0.0, 1.0 - len(first_line) / 100.0)
        curiosity_signal = self._curiosity_gap_classifier(first_line)
        pattern_interrupt = self._pattern_interrupt(first_line)

        # Weights from logistic regression on viral/non-viral first lines
        w = [0.20, 0.25, 0.20, 0.25, 0.10]
        raw = (
            w[0] * has_number
            + w[1] * power_word_ratio
            + w[2] * brevity
            + w[3] * curiosity_signal
            + w[4] * pattern_interrupt
        )
        return _clamp(raw)

    # -- Emotional intensity -------------------------------------------------

    def _emotional_intensity(self, text: str) -> float:
        sentiment = self.nlp.sentiment_compound(text)  # [-1, 1]
        emotion_scores = self.nlp.emotion_classify(text)  # dict[str, float]
        emotion_max = max(emotion_scores.values()) if emotion_scores else 0.0

        words = text.split()
        exclamation_density = (
            sum(1 for w in words if w.endswith(("!", "?", "!!"))) / max(len(words), 1)
        )
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        caps_ratio = min(caps_words / max(len(words), 1), 0.15)

        a, b, g, d = 0.30, 0.35, 0.20, 0.15
        raw = (
            a * abs(sentiment)
            + b * emotion_max
            + g * exclamation_density
            + d * caps_ratio
        )
        return _clamp(raw)

    # -- Curiosity gap -------------------------------------------------------

    def _curiosity_gap(self, text: str) -> float:
        return self._curiosity_gap_classifier(text)

    def _curiosity_gap_classifier(self, text: str) -> float:
        """Fine-tuned DistilBERT binary classifier → probability [0, 1]."""
        return self.nlp.curiosity_gap_predict(text)

    # -- Controversy index ---------------------------------------------------

    def _controversy_index(self, text: str) -> float:
        topic = self.nlp.detect_topic(text)
        topic_baseline = self.patterns.topic_controversy_baseline(topic)
        stance_strength = self.nlp.stance_strength(text)
        return _clamp(topic_baseline * stance_strength)

    # -- Readability ---------------------------------------------------------

    def _readability(self, text: str) -> float:
        fk_grade = self.nlp.flesch_kincaid_grade(text)
        sentences = self.nlp.sentence_split(text)
        short_ratio = (
            sum(1 for s in sentences if len(s.split()) <= 12) / max(len(sentences), 1)
        )
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        whitespace_ratio = text.count("\n") / max(len(text) / 50, 1)

        raw = (
            0.4 * max(0, 1.0 - fk_grade / 20.0)
            + 0.3 * short_ratio
            + 0.2 * max(0, 1.0 - avg_word_len / 10.0)
            + 0.1 * min(whitespace_ratio, 1.0)
        )
        return _clamp(raw)

    # -- Structure score -----------------------------------------------------

    def _structure_score(self, text: str) -> float:
        detected = self._detect_patterns(text)
        if not detected:
            return 0.0
        scores = sorted(detected.values(), reverse=True)
        primary = scores[0]
        secondary = scores[1] if len(scores) > 1 else 0.0
        return _clamp(primary + 0.3 * secondary)

    def _detect_patterns(self, text: str) -> dict[str, float]:
        """Return dict of pattern_name → score for patterns detected in text."""
        results = {}
        lower = text.lower().strip()
        words = lower.split()
        lines = text.strip().split("\n")

        # LISTICLE: starts with a number + "things/reasons/ways/..."
        if words and words[0].isdigit():
            results["LISTICLE"] = STRUCTURE_PATTERNS["LISTICLE"]

        # THREAD_HOOK
        if any(marker in lower for marker in ["🧵", "thread:", "(1/"]):
            results["THREAD_HOOK"] = STRUCTURE_PATTERNS["THREAD_HOOK"]

        # CONTRARIAN
        if any(lower.startswith(p) for p in ["unpopular opinion", "hot take", "controversial take"]):
            results["CONTRARIAN"] = STRUCTURE_PATTERNS["CONTRARIAN"]

        # STORY_OPEN
        if any(lower.startswith(p) for p in ["i just", "last week", "last month", "in 20"]):
            results["STORY_OPEN"] = STRUCTURE_PATTERNS["STORY_OPEN"]

        # ONE_LINER
        if len(words) <= 15 and len(lines) == 1 and "http" not in lower:
            results["ONE_LINER"] = STRUCTURE_PATTERNS["ONE_LINER"]

        # QUESTION
        if text.strip().endswith("?"):
            results["QUESTION"] = STRUCTURE_PATTERNS["QUESTION"]

        # VERSUS
        if " vs " in lower or " or " in lower and lower.endswith("?"):
            results["VERSUS"] = STRUCTURE_PATTERNS["VERSUS"]

        # BEFORE_AFTER
        if "before:" in lower and "after:" in lower:
            results["BEFORE_AFTER"] = STRUCTURE_PATTERNS["BEFORE_AFTER"]

        # BOLD_CLAIM: declarative, contains superlative, no hedging
        hedge_words = {"maybe", "perhaps", "might", "could", "possibly", "i think"}
        superlatives = {"best", "worst", "most", "biggest", "fastest", "only", "never", "always", "every"}
        if (
            not text.strip().endswith("?")
            and not hedge_words.intersection(set(words))
            and superlatives.intersection(set(words))
        ):
            results["BOLD_CLAIM"] = STRUCTURE_PATTERNS["BOLD_CLAIM"]

        return results

    # -- Timing --------------------------------------------------------------

    def _timing_score(self, posted_at: Optional[str]) -> float:
        if not posted_at:
            return 0.5  # neutral if unknown
        hour, dow = _parse_time_components(posted_at)
        base = self.trends.engagement_heatmap_lookup(dow, hour)
        return _clamp(base)

    # -- Trend alignment -----------------------------------------------------

    def _trend_alignment(self, text: str) -> float:
        post_emb = self.nlp.embed(text)
        active_trends = self.trends.get_active(hours=6)
        if not active_trends:
            return 0.0
        best = 0.0
        for trend in active_trends:
            sim = _cosine_similarity(post_emb, trend.embedding)
            score = sim * trend.velocity_normalized
            best = max(best, score)
        return _clamp(best)

    # -- Pattern similarity --------------------------------------------------

    def _pattern_similarity(self, text: str) -> float:
        post_emb = self.nlp.embed(text)
        patterns = self.patterns.get_viral_patterns()
        if not patterns:
            return 0.0
        best = 0.0
        for pat in patterns:
            sim = _cosine_similarity(post_emb, pat.centroid)
            score = sim * pat.avg_virality_rate
            best = max(best, score)
        return _clamp(best)

    # -- Helpers -------------------------------------------------------------

    def _cluster_id(self, text: str) -> int:
        emb = self.nlp.embed(text)
        return self.patterns.nearest_cluster(emb)

    def _pattern_interrupt(self, text: str) -> float:
        return self.nlp.pattern_interrupt_score(text)

    @staticmethod
    def _media_onehot(media_type: MediaType) -> list[float]:
        mapping = {
            MediaType.NONE:  [1, 0, 0, 0, 0],
            MediaType.IMAGE: [0, 1, 0, 0, 0],
            MediaType.VIDEO: [0, 0, 1, 0, 0],
            MediaType.GIF:   [0, 0, 0, 1, 0],
            MediaType.POLL:  [0, 0, 0, 0, 1],
        }
        return [float(v) for v in mapping.get(media_type, [1, 0, 0, 0, 0])]

    @staticmethod
    def _cyclic_hour(posted_at: str) -> tuple[float, float]:
        hour = _parse_time_components(posted_at)[0]
        return (math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24))

    @staticmethod
    def _cyclic_dow(posted_at: str) -> tuple[float, float]:
        dow = _parse_time_components(posted_at)[1]
        return (math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7))


# ---------------------------------------------------------------------------
# Virality scoring
# ---------------------------------------------------------------------------

class ViralityScorer:
    """
    Computes the composite virality score from a FeatureVector.
    Uses the weighted formula defined in spec §4.1.
    """

    def __init__(self, model=None, weights: dict | None = None):
        self.weights = weights or WEIGHTS
        self.model = model  # Optional XGBoost model for ML-based scoring

    def score(self, fv: FeatureVector) -> ViralityScore:
        # --- Weighted linear combination ---
        raw = (
            self.weights["hook"]        * fv.hook_strength
            + self.weights["emotion"]   * fv.emotional_intensity
            + self.weights["curiosity"] * fv.curiosity_gap
            + self.weights["structure"] * fv.structure_score
            + self.weights["readability"] * fv.readability_score
            + self.weights["timing"]    * fv.timing_score
            + self.weights["trend"]     * fv.trend_alignment
            + self.weights["controversy"] * fv.controversy_index
            + self.weights["format"]    * _format_score(fv)
        )

        # --- Interaction terms ---
        raw += (
            INTERACTION_COEFFICIENTS["hook_x_curiosity"]
            * fv.hook_strength * fv.curiosity_gap
        )
        raw += (
            INTERACTION_COEFFICIENTS["trend_x_timing"]
            * fv.trend_alignment * fv.timing_score
        )
        raw += (
            INTERACTION_COEFFICIENTS["emotion_x_controversy"]
            * fv.emotional_intensity * fv.controversy_index
        )
        raw += (
            INTERACTION_COEFFICIENTS["controversy_penalty"]
            * max(0.0, fv.controversy_index - 0.7)
        )

        # --- Author baseline ---
        author_baseline = min(fv.author_follower_log / 7.0 * 0.3, 0.3)
        raw += author_baseline

        # --- ML model override (if available) ---
        if self.model is not None:
            ml_pred = self.model.predict([fv.to_list()])[0]
            # Blend: 40% formula, 60% ML model
            raw = 0.4 * raw + 0.6 * ml_pred

        # --- Sigmoid mapping ---
        final = _sigmoid(raw * 5 - 2.5)  # Scale & shift for useful range

        # --- Impression prediction ---
        p50, p75, p90 = self._predict_impressions(final)

        # --- Feature breakdown ---
        breakdown = {
            "hook_strength":       fv.hook_strength,
            "emotional_intensity": fv.emotional_intensity,
            "curiosity_gap":       fv.curiosity_gap,
            "controversy_index":   fv.controversy_index,
            "readability_score":   fv.readability_score,
            "structure_score":     fv.structure_score,
            "timing_score":        fv.timing_score,
            "trend_alignment":     fv.trend_alignment,
        }

        # --- Weakness detection ---
        weaknesses = []
        for feat, threshold in WEAKNESS_THRESHOLDS.items():
            value = breakdown.get(feat, 0)
            if value < threshold:
                weaknesses.append(f"{feat} = {value:.2f} (below {threshold} threshold)")

        # --- Category ---
        if final >= 0.85:
            category = "high_potential"
        elif final >= 0.70:
            category = "medium_potential"
        elif final >= 0.50:
            category = "moderate"
        else:
            category = "low_potential"

        return ViralityScore(
            raw_score=raw,
            final_score=final,
            predicted_impressions_p50=p50,
            predicted_impressions_p75=p75,
            predicted_impressions_p90=p90,
            feature_breakdown=breakdown,
            weaknesses=weaknesses,
            category=category,
        )

    @staticmethod
    def _predict_impressions(final_score: float) -> tuple[int, int, int]:
        for low, high, ceiling in IMPRESSION_BUCKETS:
            if low <= final_score < high:
                frac = (final_score - low) / (high - low)
                p50 = int(ceiling * frac * 0.5)
                p75 = int(ceiling * frac * 0.8)
                p90 = int(ceiling * frac * 1.2)
                return p50, p75, p90
        return 5_000_000, 8_000_000, 15_000_000


# ---------------------------------------------------------------------------
# Strategy generator
# ---------------------------------------------------------------------------

class StrategyGenerator:
    """
    Given a post and its scored features, produce actionable recommendations
    to push it toward 1M+ impressions.
    """

    def __init__(self, feature_extractor, scorer, hook_generator, timing_optimizer):
        self.extractor = feature_extractor
        self.scorer = scorer
        self.hooks = hook_generator
        self.timing = timing_optimizer

    def generate(self, post: PostInput) -> StrategyOutput:
        fv = self.extractor.extract(post)
        score = self.scorer.score(fv)

        recommendations = []

        # 1. Hook rewrite if hook is weak
        if fv.hook_strength < WEAKNESS_THRESHOLDS["hook_strength"]:
            hook_variants = self.hooks.generate(
                topic=self.extractor.nlp.detect_topic(post.text),
                target_emotion="curiosity",
                context=post.hashtags,
            )
            if hook_variants:
                best_hook = hook_variants[0]
                rewritten = _replace_first_line(post.text, best_hook[0])
                new_fv = self.extractor.extract(
                    PostInput(text=rewritten, **_copy_metadata(post))
                )
                new_score = self.scorer.score(new_fv)
                recommendations.append(Recommendation(
                    rec_type="rewrite",
                    detail={
                        "original": post.text.split("\n")[0],
                        "rewritten": best_hook[0],
                        "rationale": "Stronger hook with higher curiosity gap and power words",
                    },
                    score_delta=new_score.final_score - score.final_score,
                ))
        else:
            hook_variants = []

        # 2. Timing recommendation
        windows = self.timing.get_optimal_windows(
            author_id=None,
            topic=self.extractor.nlp.detect_topic(post.text),
            n_windows=3,
        )
        recommendations.append(Recommendation(
            rec_type="timing",
            detail={"optimal_windows": windows},
            score_delta=max(0, windows[0][2] - fv.timing_score) * WEIGHTS["timing"]
            if windows else 0.0,
        ))

        # 3. Format optimization
        format_suggestions = _get_format_suggestions(post.text)
        if format_suggestions:
            total_boost = sum(s["impact"] for s in format_suggestions)
            recommendations.append(Recommendation(
                rec_type="format",
                detail={"suggestions": format_suggestions},
                score_delta=total_boost,
            ))

        # 4. Psychological trigger enhancement
        trigger_suggestions = _get_trigger_suggestions(post.text, fv)
        if trigger_suggestions:
            recommendations.append(Recommendation(
                rec_type="trigger",
                detail={"triggers": trigger_suggestions},
                score_delta=0.05 * len(trigger_suggestions),
            ))

        # 5. Media recommendation
        media_rec = _recommend_media(post, fv)

        # 6. Thread expansion recommendation
        thread_rec = _recommend_thread(post.text, score)

        # 7. Hashtag strategy
        hashtag_strategy = _recommend_hashtags(post.hashtags, fv)

        # 8. Engagement bait
        bait_type = _recommend_engagement_close(post.text, fv)

        return StrategyOutput(
            score=score,
            recommendations=recommendations,
            hook_variants=[h[0] for h in hook_variants[:5]] if hook_variants else [],
            thread_expansion=thread_rec,
            media_recommendation=media_rec,
            hashtag_strategy=hashtag_strategy,
            engagement_bait_type=bait_type,
        )


# ---------------------------------------------------------------------------
# Core pipeline — main entry point
# ---------------------------------------------------------------------------

class ViralityEngine:
    """
    Top-level orchestrator. Call `analyze()` for single-post analysis
    or `batch_analyze()` for bulk processing.
    """

    def __init__(self, config: dict):
        self.nlp = _init_nlp_pipeline(config)
        self.trend_store = _init_trend_store(config)
        self.pattern_db = _init_pattern_db(config)
        self.extractor = FeatureExtractor(self.nlp, self.trend_store, self.pattern_db)
        self.scorer = ViralityScorer(
            model=_load_ml_model(config.get("model_path")),
            weights=config.get("weights", WEIGHTS),
        )
        self.hook_gen = _init_hook_generator(config)
        self.timing_opt = _init_timing_optimizer(config)
        self.strategy_gen = StrategyGenerator(
            self.extractor, self.scorer, self.hook_gen, self.timing_opt
        )

    def analyze(self, post: PostInput) -> StrategyOutput:
        """Full analysis: extract features → score → generate strategy."""
        return self.strategy_gen.generate(post)

    def score_only(self, post: PostInput) -> ViralityScore:
        """Lightweight: extract features → score. No strategy generation."""
        fv = self.extractor.extract(post)
        return self.scorer.score(fv)

    def batch_analyze(self, posts: list[PostInput]) -> list[StrategyOutput]:
        """Batch analysis for historical data or comparison."""
        return [self.analyze(p) for p in posts]

    def compare_variants(self, variants: list[str], base_post: PostInput) -> list[ViralityScore]:
        """Score multiple text variants of the same post."""
        results = []
        for variant_text in variants:
            variant_post = PostInput(
                text=variant_text,
                media_type=base_post.media_type,
                author_follower_count=base_post.author_follower_count,
                author_engagement_rate=base_post.author_engagement_rate,
                posted_at=base_post.posted_at,
                language=base_post.language,
            )
            results.append(self.score_only(variant_post))
        return sorted(results, key=lambda s: s.final_score, reverse=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_time_components(iso_str: str) -> tuple[int, int]:
    """Extract (hour, day_of_week) from ISO-8601 string."""
    from datetime import datetime
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.hour, dt.weekday()


def _format_score(fv: FeatureVector) -> float:
    """Compute format score from media type and post structure."""
    media_bonus = {0: 0.0, 1: 0.15, 2: 0.20, 3: 0.08, 4: 0.10}
    active_idx = fv.media_type_vec.index(1.0) if 1.0 in fv.media_type_vec else 0
    return _clamp(media_bonus.get(active_idx, 0.0) + 0.3 * (1.0 - fv.post_length_norm))


def _replace_first_line(text: str, new_first_line: str) -> str:
    lines = text.split("\n")
    lines[0] = new_first_line
    return "\n".join(lines)


def _copy_metadata(post: PostInput) -> dict:
    return {
        "media_type": post.media_type,
        "media_urls": post.media_urls,
        "author_follower_count": post.author_follower_count,
        "author_engagement_rate": post.author_engagement_rate,
        "hashtags": post.hashtags,
        "mentions": post.mentions,
        "urls": post.urls,
        "posted_at": post.posted_at,
        "language": post.language,
    }


def _get_format_suggestions(text: str) -> list[dict]:
    suggestions = []
    lines = text.split("\n")

    if len(lines[0]) > 40:
        suggestions.append({
            "rule": "line_break_after_hook",
            "action": "Insert line break after first sentence",
            "impact": 0.04,
        })

    if text.count("\n") > 2 and "→" not in text:
        suggestions.append({
            "rule": "visual_separators",
            "action": "Replace bullet hyphens with → or • for scannability",
            "impact": 0.02,
        })

    if text.count("#") > 2:
        suggestions.append({
            "rule": "remove_hashtag_spam",
            "action": "Reduce to 0-1 hashtags; move to end if kept",
            "impact": 0.03,
        })

    return suggestions


def _get_trigger_suggestions(text: str, fv: FeatureVector) -> list[str]:
    suggestions = []
    lower = text.lower()

    if fv.curiosity_gap < 0.5 and not lower.endswith(":"):
        suggestions.append("curiosity_gap: End with colon or open question")

    if "%" not in text and not any(c.isdigit() for c in text):
        suggestions.append("social_proof: Add specific numbers or percentages")

    if fv.emotional_intensity < 0.5:
        suggestions.append("loss_aversion: Reframe as what reader stands to lose")

    return suggestions


def _recommend_media(post: PostInput, fv: FeatureVector) -> str:
    if post.media_type != MediaType.NONE:
        return post.media_type.value
    if fv.structure_score > 0.10:
        return "static_image_with_text_overlay"
    return "none_or_screenshot"


def _recommend_thread(text: str, score: ViralityScore) -> dict:
    word_count = len(text.split())
    if score.final_score >= 0.60 and word_count > 30:
        return {"recommended": True, "optimal_length": min(max(3, word_count // 40), 10)}
    return {"recommended": False, "optimal_length": 0}


def _recommend_hashtags(current: list[str], fv: FeatureVector) -> dict:
    if len(current) > 2:
        return {"use": 1, "tags": current[:1], "note": "Reduce to max 1 hashtag"}
    if len(current) == 0 and fv.trend_alignment > 0.5:
        return {"use": 1, "tags": ["[top_aligned_trend_tag]"]}
    return {"use": len(current), "tags": current}


def _recommend_engagement_close(text: str, fv: FeatureVector) -> str:
    if text.strip().endswith("?"):
        return "question_already_present"
    if fv.controversy_index > 0.5:
        return "agree_disagree"
    return "question_close"


# ---------------------------------------------------------------------------
# Initialization — real implementations backed by lightweight NLP + SQLite
# ---------------------------------------------------------------------------

_shared_store = None


def _get_or_create_store(config: dict):
    global _shared_store
    if _shared_store is None:
        from src.storage.database import Database
        db_path = config.get("db_path",
                             config.get("storage", {}).get("db_path", "data/virality.db"))
        _shared_store = Database(db_path)
    return _shared_store


def _init_nlp_pipeline(config: dict):
    """Initialize NLP models: VADER sentiment, textstat readability, hash embeddings."""
    from src.nlp.pipeline import NLPPipeline
    dim = config.get("nlp", {}).get("embedding_dim", 384) if isinstance(config, dict) else 384
    return NLPPipeline(embedding_dim=dim)


def _init_trend_store(config: dict):
    """Initialize trend store backed by SQLite."""
    return _get_or_create_store(config)


def _init_pattern_db(config: dict):
    """Initialize viral pattern database backed by SQLite."""
    return _get_or_create_store(config)


def _load_ml_model(model_path: str | None):
    """Load trained ML model. Returns None for MVP (formula-only scoring)."""
    if model_path is None:
        return None
    import logging
    logging.getLogger(__name__).warning(
        "ML model path specified (%s) but model loading not yet implemented. "
        "Using formula-only scoring.", model_path
    )
    return None


def _init_hook_generator(config: dict):
    """Initialize template-based hook generator."""
    from src.strategy.hook_generator import HookGenerator
    nlp = _init_nlp_pipeline(config)
    return HookGenerator(nlp)


def _init_timing_optimizer(config: dict):
    """Initialize timing optimizer backed by engagement heatmap data."""
    from src.strategy.timing import TimingOptimizer
    store = _get_or_create_store(config)
    return TimingOptimizer(store)
