"""End-to-end Prism pipeline tying the five components together.

    library + fingerprint -> diversity-aware panel selection
        -> collapse-prevented deliberation -> correlation-corrected aggregation
        -> answer + minority report + coverage

This is the shape of the Phase-5 product API (§4): a question goes in; an
answer, the principled dissent, and a coverage indicator come out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .agents.base import AgentBackend
from .agents.library import AgentLibrary
from .agents.mock import Task
from .deliberation.aggregate import aggregate
from .deliberation.protocol import deliberate
from .fingerprint.fingerprint import ProbeResults, run_probes
from .orchestrator.selector import select_panel
from .utils.linalg import cosine_similarity_matrix


@dataclass
class PrismAnswer:
    labels: np.ndarray
    coverage: np.ndarray
    minority_reports: list
    panel_ids: List[str]
    deliberation_collapse: float


class Prism:
    def __init__(
        self,
        library: AgentLibrary,
        backend: AgentBackend,
        n_classes: int,
        panel_size: int = 30,
        selection_method: str = "dpp",
    ):
        self.library = library
        self.backend = backend
        self.n_classes = n_classes
        self.panel_size = panel_size
        self.selection_method = selection_method
        self._fp: Optional[ProbeResults] = None
        self._similarity: Optional[np.ndarray] = None
        self._quality: Optional[np.ndarray] = None

    def fit(self, probe_tasks: List[Task]) -> "Prism":
        """Fingerprint the whole library: build similarity + quality for selection."""
        self._fp = run_probes(self.library, self.backend, probe_tasks)
        self._similarity = cosine_similarity_matrix(self._fp.correct.astype(float))
        self._quality = self._fp.accuracy()
        return self

    def answer(self, tasks: List[Task]) -> PrismAnswer:
        if self._fp is None:
            raise RuntimeError("call fit() with a probe battery before answer()")

        sel = select_panel(
            self.library.ids, self._similarity, self.panel_size,
            quality=self._quality, method=self.selection_method,
        )
        panel = self.library.subset(sel.agent_ids)
        res = run_probes(panel, self.backend, tasks)

        delib = deliberate(res.choices, res.confidence, self.n_classes, prevent_collapse=True)
        agg = aggregate(delib.final_choices, self.n_classes, agent_ids=res.agent_ids,
                        correlation_correction=True)
        return PrismAnswer(
            labels=agg.labels,
            coverage=agg.coverage(),
            minority_reports=agg.minority_reports,
            panel_ids=sel.agent_ids,
            deliberation_collapse=delib.collapse,
        )
