"""Built-in cognitive personas and a starter-library builder.

Personas are the cheapest source of diversity (design doc §3 ①, "almost free").
Each persona is a thinking *style* — the doc's "developmental-trait-like" roles:
detail-hyperfocus, lateral leaps, literal reading, top-down structure, and an
always-on disconfirmer. Crossing a handful of personas with a few base models
yields the 10-30 hand-built agents of Phase 1.
"""
from __future__ import annotations

from typing import Dict, List

from .base import AgentSpec
from .library import AgentLibrary

# name -> (system prompt, behavior descriptor on cognitive axes in [0,1])
# axes: abstract_concrete, convergent_divergent, cautious_bold, detail_global, literal_figurative
BUILTIN_PERSONAS: Dict[str, Dict] = {
    "detail_hyperfocus": {
        "prompt": (
            "You fixate on fine detail. Check every specific, edge case, and exact "
            "wording before committing. Distrust sweeping generalizations."
        ),
        "descriptor": {
            "abstract_concrete": 0.85, "convergent_divergent": 0.3,
            "cautious_bold": 0.25, "detail_global": 0.05, "literal_figurative": 0.3,
        },
    },
    "lateral_leaper": {
        "prompt": (
            "You think by surprising sideways association. Connect the problem to "
            "distant domains and propose non-obvious framings others would miss."
        ),
        "descriptor": {
            "abstract_concrete": 0.4, "convergent_divergent": 0.95,
            "cautious_bold": 0.85, "detail_global": 0.6, "literal_figurative": 0.9,
        },
    },
    "literalist": {
        "prompt": (
            "You take statements at exact face value. Answer only what is literally "
            "asked; refuse to infer unstated intent. Flag ambiguity instead of guessing."
        ),
        "descriptor": {
            "abstract_concrete": 0.7, "convergent_divergent": 0.2,
            "cautious_bold": 0.2, "detail_global": 0.3, "literal_figurative": 0.02,
        },
    },
    "systems_holist": {
        "prompt": (
            "You start from the whole structure. Map the system, its parts and their "
            "relations first; only then descend to specifics."
        ),
        "descriptor": {
            "abstract_concrete": 0.15, "convergent_divergent": 0.5,
            "cautious_bold": 0.5, "detail_global": 0.95, "literal_figurative": 0.55,
        },
    },
    "disconfirmer": {
        "prompt": (
            "You hunt for the reason the obvious answer is wrong. Actively seek "
            "counter-evidence and the strongest case against the leading option."
        ),
        "descriptor": {
            "abstract_concrete": 0.5, "convergent_divergent": 0.7,
            "cautious_bold": 0.9, "detail_global": 0.5, "literal_figurative": 0.5,
        },
    },
    "pragmatist": {
        "prompt": (
            "You weigh trade-offs and pick what works under real constraints. Prefer "
            "the robust, good-enough answer over the elegant or extreme one."
        ),
        "descriptor": {
            "abstract_concrete": 0.55, "convergent_divergent": 0.35,
            "cautious_bold": 0.45, "detail_global": 0.6, "literal_figurative": 0.45,
        },
    },
}

# Distinct open-weights *families* — different lineages have lower error
# correlation than two frontier models (design doc §1.1 note).
DEFAULT_BASE_MODELS: List[str] = [
    "llama-3.1-8b",
    "qwen2.5-7b",
    "mistral-7b",
    "gemma2-9b",
    "phi-3.5-mini",
]


def build_starter_library(
    base_models: List[str] | None = None,
    personas: List[str] | None = None,
    temperatures: List[float] | None = None,
) -> AgentLibrary:
    """Hand-build the Phase 1 library: base models x personas x sampling.

    Caps at a sensible size; the Diversity Engine grows it further later.
    """
    base_models = base_models or DEFAULT_BASE_MODELS
    personas = personas or list(BUILTIN_PERSONAS.keys())
    temperatures = temperatures or [0.4, 0.9]

    specs: List[AgentSpec] = []
    for bm in base_models:
        for pname in personas:
            p = BUILTIN_PERSONAS[pname]
            for ti, temp in enumerate(temperatures):
                agent_id = f"{bm}__{pname}__t{ti}"
                specs.append(
                    AgentSpec(
                        id=agent_id,
                        base_model=bm,
                        persona=pname,
                        system_prompt=p["prompt"],
                        temperature=temp,
                        top_p=0.95,
                        behavior_descriptor=dict(p["descriptor"]),
                    )
                )
    return AgentLibrary(specs)
