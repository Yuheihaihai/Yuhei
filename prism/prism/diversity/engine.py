"""Quality-Diversity library growth with a saturation stopping rule.

MAP-Elites over *cognitive axes* (behaviour descriptors), but the fitness is
diversity, not accuracy (design doc §3 ③): a mutated agent is kept only if it
fills an empty cell of the archive or lowers the panel's error correlation. The
StoppingRule is the cost lever — once new agents stop adding diversity, growth
halts even though no "final answer" has been produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..agents.base import AgentBackend, AgentSpec
from ..agents.library import AgentLibrary
from ..agents.mock import Task
from .metrics import effective_sample_size


def _evaluate(spec: AgentSpec, backend: AgentBackend, tasks: List[Task]) -> Tuple[np.ndarray, np.ndarray]:
    choices = np.array([backend.answer(spec, t.question, t.options).choice for t in tasks], dtype=int)
    answers = np.array([t.answer for t in tasks], dtype=int)
    return choices, (choices == answers)


class StoppingRule:
    """Stop when the collective diversity metric stalls for `patience` steps."""

    def __init__(self, patience: int = 6, min_improve: float = 1e-2):
        self.patience = patience
        self.min_improve = min_improve
        self.best = -np.inf
        self.stale = 0

    def update(self, value: float) -> bool:
        if value > self.best + self.min_improve:
            self.best = value
            self.stale = 0
        else:
            self.stale += 1
        return self.stale >= self.patience


@dataclass
class GrowthResult:
    library: AgentLibrary
    history: List[float] = field(default_factory=list)
    accepted: int = 0
    proposed: int = 0
    stopped_early: bool = False
    reason: str = ""


class DiversityEngine:
    def __init__(
        self,
        backend: AgentBackend,
        probe_tasks: List[Task],
        axes: Tuple[str, str] = ("convergent_divergent", "detail_global"),
        bins: int = 5,
        base_models: Optional[List[str]] = None,
        metric_fn: Callable[[np.ndarray], float] = effective_sample_size,
        seed: int = 0,
    ):
        self.backend = backend
        self.probe_tasks = probe_tasks
        self.axes = axes
        self.bins = bins
        self.metric_fn = metric_fn
        self.rng = np.random.default_rng(seed)
        from ..agents.personas import DEFAULT_BASE_MODELS

        self.base_models = base_models or DEFAULT_BASE_MODELS
        self._answers = np.array([t.answer for t in probe_tasks], dtype=int)

    def _cell(self, spec: AgentSpec) -> Tuple[int, int]:
        d = spec.behavior_descriptor
        x = int(np.clip(d.get(self.axes[0], 0.5) * self.bins, 0, self.bins - 1))
        y = int(np.clip(d.get(self.axes[1], 0.5) * self.bins, 0, self.bins - 1))
        return x, y

    def _mutate(self, parent: AgentSpec, n: int) -> AgentSpec:
        desc = dict(parent.behavior_descriptor)
        for k in desc:
            desc[k] = float(np.clip(desc[k] + self.rng.normal(0, 0.18), 0.0, 1.0))
        base = parent.base_model
        # switch lineage often: a different base model is the cheapest route to
        # genuinely uncorrelated errors (same-family agents fail alike).
        if self.rng.random() < 0.75:
            others = [b for b in self.base_models if b != parent.base_model] or self.base_models
            base = str(self.rng.choice(others))
        temp = float(np.clip(parent.temperature + self.rng.normal(0, 0.2), 0.1, 1.3))
        # a fresh persona name gives the agent a genuinely new bias structure
        persona = f"{parent.persona}_m{n}"
        return AgentSpec(
            id=f"qd_{n}_{base}_{persona}",
            base_model=base,
            persona=persona,
            system_prompt=parent.system_prompt + f" (variant {n})",
            temperature=temp,
            top_p=parent.top_p,
            lora_adapter=parent.lora_adapter,
            behavior_descriptor=desc,
        )

    def grow(
        self,
        library: AgentLibrary,
        max_new: int = 60,
        stopping: Optional[StoppingRule] = None,
    ) -> GrowthResult:
        stopping = stopping or StoppingRule()
        specs: List[AgentSpec] = library.specs()
        correct_rows: List[np.ndarray] = []
        archive: Dict[Tuple[int, int], str] = {}
        for s in specs:
            _, corr = _evaluate(s, self.backend, self.probe_tasks)
            correct_rows.append(corr)
            archive.setdefault(self._cell(s), s.id)

        correct = np.array(correct_rows)
        result = GrowthResult(library=AgentLibrary(specs), history=[self.metric_fn(correct)])

        for n in range(max_new):
            result.proposed += 1
            parent = specs[int(self.rng.integers(0, len(specs)))]
            cand = self._mutate(parent, n)
            _, corr = _evaluate(cand, self.backend, self.probe_tasks)

            cell = self._cell(cand)
            trial = np.vstack([correct, corr])
            gain = self.metric_fn(trial)
            fills_empty = cell not in archive
            improves = gain > result.history[-1] + 1e-9
            # keep the trajectory monotone: accept improvements, and accept an
            # empty-cell explorer only if it does not erode the collective metric
            accept = improves or (fills_empty and gain >= result.history[-1] - 1e-9)

            if accept:
                specs.append(cand)
                correct = trial
                archive.setdefault(cell, cand.id)
                result.library.add(cand)
                result.accepted += 1
                result.history.append(gain)
                if stopping.update(gain):
                    result.stopped_early = True
                    result.reason = (
                        f"diversity saturated: no >{stopping.min_improve} gain "
                        f"in {stopping.patience} accepted agents"
                    )
                    break
            else:
                # proposal rejected; still counts toward saturation patience
                if stopping.update(result.history[-1]):
                    result.stopped_early = True
                    result.reason = "diversity saturated: proposals stopped improving the panel"
                    break

        return result
