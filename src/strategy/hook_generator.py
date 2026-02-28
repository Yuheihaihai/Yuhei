"""Template-based hook generation — no LLM dependency."""

from __future__ import annotations

import random

from src.core.virality_engine import HOOK_TEMPLATES, POWER_WORDS


# Slot value dictionaries per topic
_SLOT_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "tech": {
        "adjective": ["surprising", "hidden", "critical", "underrated", "counterintuitive"],
        "outcome": ["will change everything", "most devs ignore", "no one teaches you"],
        "common_advice": "following generic tech advice",
        "authority_group": ["senior engineers", "FAANG devs", "YC founders"],
        "fail_verb": ["fail", "struggle", "burn out"],
    },
    "finance": {
        "adjective": ["costly", "hidden", "dangerous", "overlooked"],
        "outcome": ["destroy your portfolio", "most investors miss", "the wealthy exploit"],
        "common_advice": "saving 10% of your income",
        "authority_group": ["Wall Street", "hedge funds", "financial advisors"],
        "fail_verb": ["lose money", "go broke", "stay poor"],
    },
    "career": {
        "adjective": ["silent", "career-ending", "overlooked", "critical"],
        "outcome": ["kill your career", "top performers know", "get you promoted"],
        "common_advice": "updating your resume",
        "authority_group": ["recruiters", "hiring managers", "CEOs"],
        "fail_verb": ["get rejected", "stall out", "stay stuck"],
    },
    "marketing": {
        "adjective": ["surprising", "untapped", "proven", "counterintuitive"],
        "outcome": ["10x your growth", "most marketers ignore", "actually convert"],
        "common_advice": "posting every day",
        "authority_group": ["growth hackers", "top creators", "agencies"],
        "fail_verb": ["plateau", "waste money", "lose followers"],
    },
}


class HookGenerator:
    """Generate and score hook variants using templates + heuristics."""

    def __init__(self, nlp_pipeline):
        self.nlp = nlp_pipeline

    def generate(
        self, topic: str, target_emotion: str, context: list[str], n_candidates: int = 10
    ) -> list[tuple[str, float]]:
        candidates = []

        # 1. Fill each template
        slots = self._build_slots(topic)
        for _name, template in HOOK_TEMPLATES.items():
            try:
                filled = template.format(**slots)
                candidates.append(filled)
            except KeyError:
                continue

        # 2. Emotion-targeted variations
        emotion_hooks = {
            "curiosity": [
                f"Here's what nobody tells you about {topic}:",
                f"The {topic} truth that changes everything:",
                f"I was wrong about {topic}. Here's what I found:",
            ],
            "fear": [
                f"Warning: {topic} is about to change forever.",
                f"Why {topic} might be your biggest mistake:",
                f"The {topic} crisis no one is talking about:",
            ],
            "surprise": [
                f"The {topic} data that shocked me:",
                f"I spent 10,000 hours studying {topic}. One thing stands out:",
                f"Everything you know about {topic} is wrong:",
            ],
        }
        for hook in emotion_hooks.get(target_emotion, emotion_hooks["curiosity"]):
            candidates.append(hook)

        # 3. Score all candidates
        scored = []
        for hook in candidates:
            score = self._score_hook(hook)
            scored.append((hook, score))

        # 4. Sort descending by score, return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n_candidates]

    def _build_slots(self, topic: str) -> dict[str, str]:
        overrides = _SLOT_OVERRIDES.get(topic, _SLOT_OVERRIDES.get("tech", {}))

        def pick(key: str, default: str) -> str:
            vals = overrides.get(key, [default]) if isinstance(overrides.get(key), list) else [overrides.get(key, default)]
            return random.choice(vals)

        return {
            "n": str(random.choice([3, 5, 7, 10])),
            "adjective": pick("adjective", "surprising"),
            "topic_plural": f"{topic} strategies",
            "outcome": pick("outcome", "will change everything"),
            "common_advice": overrides.get("common_advice", f"following generic {topic} advice")
            if isinstance(overrides.get("common_advice"), str) else f"following generic {topic} advice",
            "past_action": f"spent 1,000 hours studying {topic}",
            "unexpected_result": "this is what I found",
            "percentage": str(random.choice([87, 92, 95, 99])),
            "group": "people",
            "fail_verb": pick("fail_verb", "fail"),
            "topic": topic,
            "contrarian_adjective": random.choice(["overrated", "broken", "backwards"]),
            "time_ago": random.choice(["6 months ago", "1 year ago", "In 2023"]),
            "before_state": "struggling",
            "after_state": "everything changed",
            "authority_group": pick("authority_group", "experts"),
            "simple_task": f"master {topic}",
            "unexpected_requirement": "this one mindset shift",
            "year": str(random.choice([2027, 2028, 2030])),
            "bold_prediction": f"{topic} will look nothing like today",
        }

    def _score_hook(self, hook: str) -> float:
        words = hook.lower().split()
        word_set = set(words)

        score = 0.0
        # Power word density
        score += 0.25 * len(POWER_WORDS & word_set) / max(len(words), 1)
        # Brevity bonus
        score += 0.20 * max(0.0, 1.0 - len(hook) / 100.0)
        # Curiosity gap
        score += 0.25 * self.nlp.curiosity_gap_predict(hook)
        # Numeric specificity
        score += 0.15 * float(any(w.isdigit() for w in words))
        # Pattern interrupt
        score += 0.15 * self.nlp.pattern_interrupt_score(hook)

        return min(score, 1.0)
