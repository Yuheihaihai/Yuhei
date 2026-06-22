"""Run probes across a library and turn the results into cognitive prints.

Output of this module feeds both the visualisation (are agents spread out, or
clumped = redundant?) and the Diversity Engine (error-correlation matrix).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..agents.base import AgentBackend, AgentSpec
from ..agents.library import AgentLibrary
from ..agents.mock import Task
from ..utils.linalg import pca, safe_corrcoef


@dataclass
class ProbeResults:
    """Raw agent-by-probe matrices — the substrate for every diversity metric."""

    agent_ids: List[str]
    tasks: List[Task]
    choices: np.ndarray      # int   [n_agents, n_probes] chosen option index
    correct: np.ndarray      # bool  [n_agents, n_probes] choice == ground truth
    confidence: np.ndarray   # float [n_agents, n_probes]

    @property
    def n_agents(self) -> int:
        return len(self.agent_ids)

    @property
    def n_probes(self) -> int:
        return len(self.tasks)

    def accuracy(self) -> np.ndarray:
        """Per-agent accuracy over the probe battery."""
        return self.correct.mean(axis=1)


def run_probes(
    library: AgentLibrary,
    backend: AgentBackend,
    tasks: List[Task],
) -> ProbeResults:
    specs: List[AgentSpec] = library.specs()
    n_a, n_p = len(specs), len(tasks)
    choices = np.zeros((n_a, n_p), dtype=int)
    correct = np.zeros((n_a, n_p), dtype=bool)
    confidence = np.zeros((n_a, n_p), dtype=float)

    for ai, spec in enumerate(specs):
        for ti, task in enumerate(tasks):
            r = backend.answer(spec, task.question, task.options)
            choices[ai, ti] = r.choice
            correct[ai, ti] = r.choice == task.answer
            confidence[ai, ti] = r.confidence

    return ProbeResults(
        agent_ids=[s.id for s in specs],
        tasks=tasks,
        choices=choices,
        correct=correct,
        confidence=confidence,
    )


def cognitive_fingerprints(results: ProbeResults) -> np.ndarray:
    """Concatenate the error vector and a behaviour vector into one print.

    * error vector  : 1 where the agent was wrong (which problems, how)
    * behaviour vec : mean-centred per-probe choice + confidence signature
      (what it attended to / how sure it was), a cheap stand-in for the output
      embedding in the doc. Together these separate agents that are *right or
      wrong on the same things* from those that are not.
    """
    err = (~results.correct).astype(float)
    # behaviour: choice pattern (as fraction of options) + confidence, centred
    n_opts = results.choices.max(initial=1) + 1
    choice_sig = results.choices.astype(float) / max(1, n_opts - 1)
    conf = results.confidence
    behaviour = np.hstack([choice_sig, conf])
    behaviour = behaviour - behaviour.mean(axis=0, keepdims=True)
    fp = np.hstack([err, behaviour])
    return fp


def error_correlation(results: ProbeResults) -> np.ndarray:
    """Pairwise correlation of agents' *error* signals — the quantity that

    determines effective sample size. High correlation = same blind spots.
    """
    err = (~results.correct).astype(float)
    return safe_corrcoef(err)


def reduce_2d(fingerprints: np.ndarray, method: str = "auto") -> np.ndarray:
    """2D embedding of fingerprints for the "are we spread out?" plot.

    Tries UMAP if requested/available; always falls back to pure-numpy PCA so
    the pipeline never hard-depends on optional libraries.
    """
    if method in ("auto", "umap"):
        try:  # pragma: no cover - optional dep
            import umap  # type: ignore

            n = fingerprints.shape[0]
            reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(2, n - 1)))
            return reducer.fit_transform(fingerprints)
        except Exception:
            if method == "umap":
                raise
    return pca(fingerprints, 2)
