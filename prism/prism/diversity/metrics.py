"""Diversity metrics.

These quantify the design doc's central claim: ensemble error decomposes into
average individual error *minus* diversity, so diversity is a subtractable
quantity you can measure and optimise. All inputs are the per-(agent, item)
oracle outputs (correct/incorrect) and chosen options from a ProbeResults-like
matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..utils.linalg import cosine_similarity_matrix, log_volume, safe_corrcoef


def _pair_counts(correct: np.ndarray):
    """For every agent pair, count (both right, one right, both wrong) cells.

    Returns a, b_plus_c, d matrices where a=both correct, d=both wrong,
    b=#(i right, j wrong), c=#(i wrong, j right).
    """
    c = correct.astype(float)
    w = 1.0 - c
    a = c @ c.T            # both correct
    d = w @ w.T            # both wrong
    b = c @ w.T            # i correct, j wrong
    cc = w @ c.T           # i wrong, j correct
    return a, b, cc, d


def q_statistic_matrix(correct: np.ndarray) -> np.ndarray:
    """Yule's Q for every agent pair. Q -> 0 means independent errors (good)."""
    a, b, c, d = _pair_counts(correct)
    num = a * d - b * c
    den = a * d + b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(den > 0, num / den, 0.0)
    np.fill_diagonal(Q, 1.0)
    return Q


def disagreement_matrix(choices: np.ndarray) -> np.ndarray:
    """Fraction of items on which two agents pick *different* options."""
    n = choices.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = (choices[i][None, :] != choices).mean(axis=1)
    return D


def double_fault_matrix(correct: np.ndarray) -> np.ndarray:
    """Fraction of items where *both* agents are wrong — the costly coincidence."""
    a, b, c, d = _pair_counts(correct)
    n_items = correct.shape[1]
    return d / n_items


def mean_pairwise_error_correlation(correct: np.ndarray) -> float:
    """Average off-diagonal correlation of error signals across the panel.

    This is the number that throttles the effective sample size in the
    correlation-corrected Condorcet bound.
    """
    err = (~correct.astype(bool)).astype(float)
    C = safe_corrcoef(err)
    n = C.shape[0]
    if n < 2:
        return 0.0
    off = C[~np.eye(n, dtype=bool)]
    return float(off.mean())


def effective_sample_size(correct: np.ndarray) -> float:
    """Correlation-corrected effective number of independent voters.

    N_eff = n / (1 + (n-1) * rho_bar). With rho_bar ~ 0 this is ~n; with high
    correlation it collapses toward 1 — "ten thousand that fail alike == a few
    dozen independent ones" (design doc §1.4).
    """
    n = correct.shape[0]
    if n < 2:
        return float(n)
    rho = max(0.0, mean_pairwise_error_correlation(correct))
    return float(n / (1.0 + (n - 1) * rho))


def ambiguity_decomposition(
    choices: np.ndarray, answers: np.ndarray, n_options: int
) -> Dict[str, float]:
    """Exact ambiguity decomposition for squared loss over one-hot predictions.

    Identity (holds exactly for an averaging ensemble):
        ensemble_error == avg_individual_error - diversity
    where diversity is the mean squared spread of members around the ensemble
    mean. This is the doc's "diversity is a subtractable quantity" made literal.
    """
    n_agents, n_items = choices.shape
    # one-hot predictions: [agents, items, options]
    P = np.zeros((n_agents, n_items, n_options))
    ai, ii = np.indices((n_agents, n_items))
    P[ai, ii, choices] = 1.0
    T = np.zeros((n_items, n_options))
    T[np.arange(n_items), answers] = 1.0

    pbar = P.mean(axis=0)                                  # [items, options]
    indiv_err = ((P - T[None]) ** 2).sum(axis=2).mean()    # avg over agents+items
    ens_err = ((pbar - T) ** 2).sum(axis=1).mean()         # avg over items
    diversity = ((P - pbar[None]) ** 2).sum(axis=2).mean()
    return {
        "avg_individual_error": float(indiv_err),
        "ensemble_error": float(ens_err),
        "diversity": float(diversity),
        # residual should be ~0; surfaced so tests can assert the identity
        "identity_residual": float(indiv_err - diversity - ens_err),
    }


@dataclass
class DiversitySummary:
    n_agents: int
    mean_q: float
    mean_disagreement: float
    mean_double_fault: float
    mean_error_correlation: float
    effective_sample_size: float
    log_volume: float
    diversity_term: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "n_agents": self.n_agents,
            "mean_q": self.mean_q,
            "mean_disagreement": self.mean_disagreement,
            "mean_double_fault": self.mean_double_fault,
            "mean_error_correlation": self.mean_error_correlation,
            "effective_sample_size": self.effective_sample_size,
            "log_volume": self.log_volume,
            "diversity_term": self.diversity_term,
        }


def _offdiag_mean(M: np.ndarray) -> float:
    n = M.shape[0]
    if n < 2:
        return 0.0
    return float(M[~np.eye(n, dtype=bool)].mean())


def summarize_diversity(
    choices: np.ndarray, correct: np.ndarray, answers: np.ndarray, n_options: int
) -> DiversitySummary:
    Q = q_statistic_matrix(correct)
    D = disagreement_matrix(choices)
    DF = double_fault_matrix(correct)
    # similarity for volume: agents represented by their correctness signature
    sim = cosine_similarity_matrix(correct.astype(float)) if correct.shape[1] else np.eye(len(correct))
    decomp = ambiguity_decomposition(choices, answers, n_options)
    return DiversitySummary(
        n_agents=choices.shape[0],
        mean_q=_offdiag_mean(Q),
        mean_disagreement=_offdiag_mean(D),
        mean_double_fault=_offdiag_mean(DF),
        mean_error_correlation=mean_pairwise_error_correlation(correct),
        effective_sample_size=effective_sample_size(correct),
        log_volume=log_volume(sim + 1.0),  # shift to keep similarity PSD-ish
        diversity_term=decomp["diversity"],
    )
