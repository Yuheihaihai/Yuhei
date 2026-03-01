"""
LLM-powered viral post analyzer and rewriter.

Uses Claude to deeply analyze posts and rewrite them for
maximum virality targeting 1M+ impressions / likes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the world's #1 X (Twitter) virality engineer.
You have reverse-engineered the X algorithm and studied 500,000+ posts that achieved 1M+ impressions.

Your ONLY goal: rewrite any post so it GUARANTEES maximum virality (1M+ impressions, 100K+ likes).

Core principles you follow:
1. HOOK — The first 5-8 words decide everything. Must trigger pattern interrupt + curiosity gap.
2. EMOTION — Posts that trigger strong emotions (awe, anger, fear, surprise) get 5-10x more shares.
3. SPECIFICITY — "87% of founders" > "most founders". Numbers create credibility.
4. STRUCTURE — Short lines. Visual breathing room. One idea per line. Never a wall of text.
5. CONTROVERSY — Slightly polarizing > neutral. But never offensive. Aim for "I disagree but I'll share this."
6. SOCIAL PROOF — Imply authority, data, or experience. "After analyzing 10,000 posts..." beats "I think..."
7. CURIOSITY GAP — Promise a revelation. Withhold just enough to force engagement.
8. CALL TO ACTION — End with engagement bait: question, challenge, or "repost if you agree."
9. TIMING — Reference current trends, zeitgeist, or pain points.
10. FORMAT — Use line breaks aggressively. No hashtags (or max 1). No links in the main body.

The X algorithm rewards:
- High reply-to-impression ratio (controversial/debatable content)
- Early engagement velocity (first 30 minutes)
- Bookmarks (signal high-value content)
- Quote reposts (signal debate-worthy content)
- Dwell time (longer posts people actually read)

You respond ONLY in valid JSON. No markdown fences. No explanation outside JSON."""

ANALYZE_PROMPT = """\
Analyze this X post and provide a complete virality optimization.

POST:
\"\"\"
{post_text}
\"\"\"

CURRENT METRICS:
- Author followers: {followers:,}
- Feature scores: {features}

Respond in this exact JSON format:
{{
  "diagnosis": {{
    "current_viral_probability": "<0-100% estimate>",
    "fatal_flaws": ["<most critical issues preventing virality>"],
    "strengths": ["<what's already working>"],
    "algorithm_signals": "<how X algorithm will treat this post>"
  }},
  "rewrites": [
    {{
      "version": "maximum_viral",
      "text": "<completely rewritten post optimized for 1M+ impressions>",
      "strategy": "<1-sentence explanation of why this version will go viral>",
      "predicted_engagement": "<estimated impressions>"
    }},
    {{
      "version": "hook_optimized",
      "text": "<same core message but with a killer hook>",
      "strategy": "<1-sentence explanation>",
      "predicted_engagement": "<estimated impressions>"
    }},
    {{
      "version": "controversy_play",
      "text": "<slightly polarizing angle that drives debate>",
      "strategy": "<1-sentence explanation>",
      "predicted_engagement": "<estimated impressions>"
    }}
  ],
  "hooks": [
    "<hook variant 1 — curiosity gap>",
    "<hook variant 2 — bold claim>",
    "<hook variant 3 — data-backed>",
    "<hook variant 4 — story opener>",
    "<hook variant 5 — contrarian>"
  ],
  "thread_strategy": {{
    "should_thread": true/false,
    "thread_outline": ["<tweet 1>", "<tweet 2>", "..."],
    "why": "<reason>"
  }},
  "posting_advice": {{
    "best_time": "<specific recommendation>",
    "media": "<what image/video to attach and why>",
    "engagement_strategy": "<what to do in first 30 min after posting>",
    "follow_up": "<what to post 2-4 hours later to boost>"
  }},
  "viral_score": <0-100 integer — your honest estimate of 1M+ potential after optimization>
}}"""


@dataclass
class LLMAnalysis:
    """Result from LLM analysis."""
    diagnosis: dict = field(default_factory=dict)
    rewrites: list[dict] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    thread_strategy: dict = field(default_factory=dict)
    posting_advice: dict = field(default_factory=dict)
    viral_score: int = 0
    raw_response: str = ""
    error: str | None = None


class LLMAnalyzer:
    """
    LLM-powered deep analysis using Claude API.
    Set ANTHROPIC_API_KEY environment variable to enable.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Sanitize user input to mitigate prompt injection."""
        # Truncate to reasonable length
        text = text[:10000]
        # Escape triple-quote delimiters that could break out of the prompt
        text = text.replace('"""', '‟‟‟')
        text = text.replace("'''", "‛‛‛")
        return text

    def analyze(self, post_text: str, followers: int = 10000,
                feature_scores: dict | None = None) -> LLMAnalysis:
        """Run deep LLM analysis on a post."""
        if not self.is_available:
            return LLMAnalysis(error="ANTHROPIC_API_KEY not set")

        post_text = self._sanitize_input(post_text)
        followers = max(0, min(followers, 500_000_000))

        features_str = ""
        if feature_scores:
            features_str = ", ".join(
                f"{k}: {v:.0%}" for k, v in feature_scores.items()
            )

        user_prompt = ANALYZE_PROMPT.format(
            post_text=post_text,
            followers=followers,
            features=features_str or "not yet scored",
        )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = response.content[0].text.strip()

            # Parse JSON from response
            data = self._parse_json(raw)

            return LLMAnalysis(
                diagnosis=data.get("diagnosis", {}),
                rewrites=data.get("rewrites", []),
                hooks=data.get("hooks", []),
                thread_strategy=data.get("thread_strategy", {}),
                posting_advice=data.get("posting_advice", {}),
                viral_score=data.get("viral_score", 0),
                raw_response=raw,
            )

        except Exception as e:
            logger.error("LLM analysis failed: %s", e)
            return LLMAnalysis(error=str(e))

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from LLM response, handling common formatting issues."""
        # Strip markdown fences if present
        if "```" in text:
            lines = text.split("\n")
            json_lines = []
            inside = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside or not any(line.strip().startswith("```") for _ in [0]):
                    json_lines.append(line)
            text = "\n".join(json_lines)

        # Try direct parse
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Find first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {}
