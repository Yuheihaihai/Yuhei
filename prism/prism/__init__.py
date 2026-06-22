"""Prism — Cognitive Diversity Orchestration.

Intentionally uncorrelated, 'quirky' AI agents, bundled without collapsing
their diversity, to reduce a group's blind spots. The opposite reward of a
convergence-optimising ensemble: Prism optimises for the *preservation of
diversity*, measured statistically (ensemble error decomposition), not for
converging on one answer.

Five components:
  ① Agent Library          prism.agents
  ② Cognitive Fingerprint  prism.fingerprint
  ③ Diversity Engine       prism.diversity
  ④ Panel Selector         prism.orchestrator
  ⑤ Deliberation+Aggregate prism.deliberation
  + Evaluation loop        prism.eval
"""
from .agents import AgentLibrary, AgentSpec, MockBackend, build_starter_library
from .agents.mock import SimWorld
from .pipeline import Prism, PrismAnswer

__version__ = "0.1.0"

__all__ = [
    "Prism",
    "PrismAnswer",
    "AgentLibrary",
    "AgentSpec",
    "MockBackend",
    "SimWorld",
    "build_starter_library",
]
