"""Compare single-best / naive-majority / Prism on a ground-truth benchmark.

The Phase-2 gate (the project's go/no-go): does Prism reduce blind spots — the
correct answers a single strong model misses — more than a naive majority vote?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..deliberation.aggregate import aggregate
from ..deliberation.protocol import deliberate


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between stated confidence and outcome (lower=better)."""
    return float(np.mean((confidence - correct.astype(float)) ** 2))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy across confidence bins."""
    conf = np.asarray(confidence, float)
    corr = correct.astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(conf)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(conf[mask].mean() - corr[mask].mean())
    return float(ece)


@dataclass
class MethodScore:
    name: str
    accuracy: float
    blind_spot_coverage: float   # of items single-best missed, fraction recovered
    brier: float
    ece: float
    extra: Dict = None

    def row(self) -> str:
        return (f"{self.name:<16} acc={self.accuracy:.3f}  "
                f"coverage={self.blind_spot_coverage:.3f}  "
                f"brier={self.brier:.3f}  ece={self.ece:.3f}")


def _single_best(choices: np.ndarray, correct: np.ndarray) -> int:
    """Index of the agent with the highest accuracy (the 'single strongest model')."""
    return int(correct.mean(axis=1).argmax())


def _coverage(method_labels: np.ndarray, single_labels: np.ndarray, answers: np.ndarray) -> float:
    missed = single_labels != answers
    if missed.sum() == 0:
        return 1.0
    recovered = (method_labels[missed] == answers[missed])
    return float(recovered.mean())


def compare_methods(
    choices: np.ndarray,
    confidence: np.ndarray,
    answers: np.ndarray,
    n_classes: int,
    agent_ids: Optional[List[str]] = None,
    deliberation_rounds: int = 2,
) -> Dict[str, MethodScore]:
    """Score the three methods on the same panel responses."""
    correct = choices == answers[None, :]
    best = _single_best(choices, correct)
    single_labels = choices[best]
    single_conf = confidence[best]

    # naive majority vote + vote-share confidence
    maj_labels = np.array([np.bincount(choices[:, j], minlength=n_classes).argmax()
                           for j in range(choices.shape[1])])
    maj_conf = np.array([np.bincount(choices[:, j], minlength=n_classes).max() / choices.shape[0]
                         for j in range(choices.shape[1])])

    # Prism: collapse-prevented deliberation, then correlation-corrected Dawid-Skene
    delib = deliberate(choices, confidence, n_classes, rounds=deliberation_rounds,
                       prevent_collapse=True)
    agg = aggregate(delib.final_choices, n_classes, agent_ids=agent_ids,
                    correlation_correction=True)
    prism_labels = agg.labels
    # posterior margin (top1 - top2) is a less saturated, better-calibrated
    # confidence than the raw top posterior, which Dawid-Skene drives toward 1.
    prism_conf = agg.coverage()

    def score(name, labels, conf) -> MethodScore:
        corr = (labels == answers)
        return MethodScore(
            name=name,
            accuracy=float(corr.mean()),
            blind_spot_coverage=_coverage(labels, single_labels, answers),
            brier=brier_score(conf, corr),
            ece=expected_calibration_error(conf, corr),
        )

    s_single = score("single_best", single_labels, single_conf)
    s_single.blind_spot_coverage = float("nan")  # coverage is defined relative to it
    results = {
        "single_best": s_single,
        "naive_majority": score("naive_majority", maj_labels, maj_conf),
        "prism": score("prism", prism_labels, prism_conf),
    }
    results["prism"].extra = {
        "n_minority_reports": len(agg.minority_reports),
        "deliberation_collapse": delib.collapse,
        "n_premature_consensus": len(delib.premature_consensus_items),
    }
    return results


def gate2_decision(results: Dict[str, MethodScore], margin: float = 0.0) -> Dict:
    """Go/no-go: Prism must beat both baselines on blind-spot coverage & accuracy."""
    prism = results["prism"]
    naive = results["naive_majority"]
    single = results["single_best"]
    beats_single_acc = prism.accuracy >= single.accuracy + margin
    beats_naive_cov = prism.blind_spot_coverage >= naive.blind_spot_coverage + margin
    recovers_blindspots = prism.blind_spot_coverage > 0.0
    go = bool(beats_single_acc and beats_naive_cov and recovers_blindspots)
    return {
        "go": go,
        "decision": "GO -> Phase 3" if go else "NO-GO -> revisit diversity/collapse/aggregation",
        "prism_accuracy": prism.accuracy,
        "single_best_accuracy": single.accuracy,
        "prism_blind_spot_coverage": prism.blind_spot_coverage,
        "naive_blind_spot_coverage": naive.blind_spot_coverage,
    }
