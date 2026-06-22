"""Per-question panel selection.

The design principle (§3 ④): don't run ten thousand agents — pick the few dozen
whose combined coverage is greatest for *this* question. We maximise a
determinantal (volume) objective, which rewards picking agents that are both
capable (quality) and mutually complementary (low similarity). Cost is capped
by the panel size, not the library size.

Two selectors are provided:
* ``greedy_dpp``  — fast greedy MAP inference for a Determinantal Point Process
  (Chen et al. 2018), O(N * k^2), the recommended diversity-aware choice.
* ``facility_location`` — a plain submodular coverage greedy, as a baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def _build_kernel(similarity: np.ndarray, quality: Optional[np.ndarray]) -> np.ndarray:
    """L = diag(q) . S . diag(q): a quality-scaled similarity kernel for the DPP."""
    S = np.array(similarity, dtype=float)
    n = S.shape[0]
    q = np.ones(n) if quality is None else np.clip(np.asarray(quality, float), 1e-3, None)
    return (q[:, None] * S) * q[None, :]


def greedy_dpp(
    similarity: np.ndarray,
    k: int,
    quality: Optional[np.ndarray] = None,
) -> List[int]:
    """Greedy MAP inference for a DPP — returns up to ``k`` complementary indices."""
    L = _build_kernel(similarity, quality)
    n = L.shape[0]
    k = min(k, n)
    d2 = np.diag(L).astype(float).copy()
    c = np.zeros((n, k))
    selected: List[int] = []
    remaining = set(range(n))

    j = int(np.argmax(d2))
    selected.append(j)
    remaining.discard(j)

    for it in range(1, k):
        if not remaining:
            break
        rem = np.fromiter(remaining, dtype=int)
        # e_i = (L[j,i] - c[j,:].c[i,:]) / sqrt(d2[j])
        prev = selected[-1]
        denom = np.sqrt(max(d2[prev], 1e-12))
        e = (L[prev, rem] - c[rem, :it] @ c[prev, :it]) / denom
        c[rem, it - 1] = e
        d2[rem] -= e ** 2
        d2[rem] = np.clip(d2[rem], 0.0, None)

        nxt = rem[int(np.argmax(d2[rem]))]
        if d2[nxt] <= 1e-10:
            # diversity capacity exhausted: top up the fixed panel budget with the
            # remaining agent least similar to those already chosen.
            sub = L[np.ix_(rem, selected)]
            nxt = rem[int(np.argmin(sub.max(axis=1)))]
        selected.append(int(nxt))
        remaining.discard(int(nxt))

    return selected


def facility_location(similarity: np.ndarray, k: int, quality: Optional[np.ndarray] = None) -> List[int]:
    """Submodular coverage greedy: each agent should be 'represented' by the panel."""
    S = np.array(similarity, dtype=float)
    n = S.shape[0]
    k = min(k, n)
    q = np.ones(n) if quality is None else np.asarray(quality, float)
    covered = np.zeros(n)
    selected: List[int] = []
    for _ in range(k):
        best, best_gain = -1, -np.inf
        for i in range(n):
            if i in selected:
                continue
            gain = np.maximum(covered, S[i]).sum() * (0.5 + 0.5 * q[i]) - covered.sum()
            if gain > best_gain:
                best_gain, best = gain, i
        if best < 0:
            break
        selected.append(best)
        covered = np.maximum(covered, S[best])
    return selected


@dataclass
class PanelSelection:
    indices: List[int]
    agent_ids: List[str]
    method: str


def select_panel(
    agent_ids: Sequence[str],
    similarity: np.ndarray,
    k: int,
    quality: Optional[np.ndarray] = None,
    method: str = "dpp",
) -> PanelSelection:
    """Pick a panel of size ``k`` from the library by complementarity.

    ``similarity`` is an agent-by-agent similarity matrix (e.g. cosine of
    correctness signatures or fingerprints); ``quality`` optionally folds in
    per-agent capability or this question's relevance.
    """
    if method == "dpp":
        idx = greedy_dpp(similarity, k, quality)
    elif method == "facility":
        idx = facility_location(similarity, k, quality)
    else:
        raise ValueError(f"unknown method: {method}")
    return PanelSelection(indices=idx, agent_ids=[agent_ids[i] for i in idx], method=method)
