"""Collapse-prevention deliberation protocol.

Left alone, deliberating agents converge and diversity evaporates (norm
collapse / tyranny of the majority, design doc §1.3 B). This module models the
exchange dynamics so the prevention machinery can be tested and tuned offline:

1. independent initial opinions (already collected before any exchange);
2. structured exchange where agents may conform toward the weighted majority,
   the more so the less confident they are;
3. a permanent devil's advocate that never conforms and shifts to the strongest
   minority view;
4. minority preservation — every distinct opinion is recorded before it can be
   absorbed;
5. premature-consensus penalty — items that snap to near-unanimity early are
   flagged and frozen instead of collapsing further.

In a live system step 2 is real LLM turns; here it is a conformity model whose
*effect* on measured diversity is what we validate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _opinion_diversity(choices_col: np.ndarray, n_classes: int) -> float:
    """Gini diversity of one item's opinions: 1 - sum_c f_c^2 (0=unanimous)."""
    counts = np.bincount(choices_col, minlength=n_classes).astype(float)
    f = counts / counts.sum()
    return float(1.0 - np.sum(f ** 2))


def _mean_diversity(choices: np.ndarray, n_classes: int) -> float:
    return float(np.mean([_opinion_diversity(choices[:, j], n_classes) for j in range(choices.shape[1])]))


@dataclass
class DeliberationOutcome:
    initial_choices: np.ndarray
    final_choices: np.ndarray
    diversity_before: float
    diversity_after: float
    collapse: float                                   # before - after (>=0 is loss)
    premature_consensus_items: List[int] = field(default_factory=list)
    minority_opinions: Dict[int, List[Dict]] = field(default_factory=dict)


def deliberate(
    choices: np.ndarray,
    confidence: np.ndarray,
    n_classes: int,
    rounds: int = 2,
    conformity: float = 0.5,
    prevent_collapse: bool = True,
    devils_advocate: Optional[int] = 0,
    early_consensus_threshold: float = 0.85,
    seed: int = 0,
) -> DeliberationOutcome:
    """Run the exchange and report how much diversity was lost.

    With ``prevent_collapse=True`` the devil's advocate, minority records and
    early-consensus freeze should yield a markedly smaller collapse than the
    naive conformity dynamics with it switched off.
    """
    rng = np.random.default_rng(seed)
    n_agents, n_items = choices.shape
    cur = choices.copy()
    before = _mean_diversity(cur, n_classes)

    minority: Dict[int, List[Dict]] = {}
    flagged: List[int] = []
    frozen = np.zeros(n_items, dtype=bool)

    for r in range(rounds):
        for item in range(n_items):
            col = cur[:, item]
            counts = np.bincount(col, minlength=n_classes).astype(float)
            # weighted majority (by confidence) for this item
            wcounts = np.zeros(n_classes)
            for a in range(n_agents):
                wcounts[col[a]] += confidence[a, item]
            majority = int(wcounts.argmax())
            agree_frac = counts.max() / counts.sum()

            # minority preservation: record distinct non-majority opinions
            for label in np.nonzero(counts)[0]:
                if label != majority:
                    holders = [a for a in range(n_agents) if col[a] == label]
                    minority.setdefault(item, [])
                    if not any(m["label"] == int(label) for m in minority[item]):
                        minority[item].append({"label": int(label), "agents": holders})

            # premature-consensus penalty
            if prevent_collapse and r == 0 and agree_frac >= early_consensus_threshold:
                flagged.append(item)
                frozen[item] = True
                continue
            if prevent_collapse and frozen[item]:
                continue

            for a in range(n_agents):
                if prevent_collapse and a == devils_advocate:
                    # never conform; shift to the strongest minority view
                    minlabels = [c for c in range(n_classes) if c != majority and counts[c] > 0]
                    if minlabels:
                        cur[a, item] = int(max(minlabels, key=lambda c: counts[c]))
                    continue
                # conform with prob rising as own confidence falls
                p = conformity * (1.0 - confidence[a, item])
                if rng.random() < p:
                    cur[a, item] = majority

    after = _mean_diversity(cur, n_classes)
    return DeliberationOutcome(
        initial_choices=choices,
        final_choices=cur,
        diversity_before=before,
        diversity_after=after,
        collapse=before - after,
        premature_consensus_items=sorted(set(flagged)),
        minority_opinions=minority,
    )
