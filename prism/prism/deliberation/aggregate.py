"""Confidence-weighted aggregation — not majority vote (design doc §3 ⑤).

Dawid-Skene jointly estimates each agent's confusion matrix (its reliability)
and the latent true label, so a well-calibrated minority can outvote a
confident-but-wrong majority. We add a correlation correction: agents whose raw
responses are redundant with a cluster are down-weighted, blunting the
"frontier models fail alike" attack on naive voting. The output carries a
minority report — principled dissent that the consensus overrode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def response_redundancy_weights(responses: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Down-weight agents whose answers are redundant with the rest of the panel.

    Without ground truth we cannot measure error correlation directly, so we use
    response agreement as its observable proxy: an agent that agrees with almost
    everyone adds little independent signal and, if part of a correlated cluster,
    would otherwise dominate the vote. Weight = (1 - excess_agreement) ** strength.
    """
    n_agents, n_items = responses.shape
    if n_agents < 2:
        return np.ones(n_agents)
    agree = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        agree[i] = (responses[i][None, :] == responses).mean(axis=1)
    np.fill_diagonal(agree, 0.0)
    mean_agree = agree.sum(axis=1) / (n_agents - 1)
    # excess over the panel-average agreement; clip to [0,1]
    excess = np.clip(mean_agree - mean_agree.mean(), 0.0, 1.0)
    return np.clip((1.0 - excess) ** strength, 0.05, 1.0)


def dawid_skene(
    responses: np.ndarray,
    n_classes: int,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """EM for the Dawid-Skene model.

    responses: int matrix [n_agents, n_items] of chosen class indices.
    weights:   optional per-agent tempering in [0,1] (correlation correction).
    Returns (T, confusion, priors): posterior over true labels [items, classes],
    per-agent confusion matrices [agents, classes, classes], class priors.
    """
    n_agents, n_items = responses.shape
    w = np.ones(n_agents) if weights is None else np.asarray(weights, float)

    # init posterior T from (weighted) vote
    T = np.zeros((n_items, n_classes))
    for a in range(n_agents):
        T[np.arange(n_items), responses[a]] += w[a]
    T /= T.sum(axis=1, keepdims=True)

    prev_ll = -np.inf
    confusion = np.zeros((n_agents, n_classes, n_classes))
    priors = np.full(n_classes, 1.0 / n_classes)

    for _ in range(max_iter):
        # M-step: priors and confusion matrices
        priors = T.mean(axis=0)
        priors = np.clip(priors, 1e-9, None)
        priors /= priors.sum()
        for a in range(n_agents):
            obs = np.zeros((n_classes, n_classes))
            for l in range(n_classes):
                mask = responses[a] == l
                obs[:, l] = T[mask].sum(axis=0)
            obs += 1e-3  # Laplace smoothing
            confusion[a] = obs / obs.sum(axis=1, keepdims=True)

        # E-step: log-posterior with per-agent tempering by weight
        log_T = np.log(priors)[None, :].repeat(n_items, axis=0)
        for a in range(n_agents):
            ll_a = np.log(confusion[a][:, responses[a]].T)  # [items, classes]
            log_T += w[a] * ll_a
        log_T -= log_T.max(axis=1, keepdims=True)
        T = np.exp(log_T)
        T /= T.sum(axis=1, keepdims=True)

        ll = float(np.log(np.exp(log_T).sum(axis=1)).sum())
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return T, confusion, priors


@dataclass
class AggregationResult:
    labels: np.ndarray                 # MAP estimate per item
    posteriors: np.ndarray             # [items, classes]
    agent_reliability: np.ndarray      # mean diagonal of each confusion matrix
    agent_weights: np.ndarray          # correlation-correction weights used
    minority_reports: List[Dict] = field(default_factory=list)

    def coverage(self) -> np.ndarray:
        """Per-item posterior margin (top1 - top2) as a confidence/coverage signal."""
        srt = np.sort(self.posteriors, axis=1)
        return srt[:, -1] - srt[:, -2]


def aggregate(
    responses: np.ndarray,
    n_classes: int,
    agent_ids: Optional[List[str]] = None,
    correlation_correction: bool = True,
    minority_reliability_floor: float = 0.5,
) -> AggregationResult:
    """Full aggregation: correlation-corrected Dawid-Skene + minority report."""
    n_agents = responses.shape[0]
    agent_ids = agent_ids or [f"a{i}" for i in range(n_agents)]
    weights = response_redundancy_weights(responses) if correlation_correction else np.ones(n_agents)

    T, confusion, _ = dawid_skene(responses, n_classes, weights=weights)
    labels = T.argmax(axis=1)
    reliability = np.array([np.mean(np.diag(confusion[a])) for a in range(n_agents)])

    # minority report: surface only *principled* dissent — a reliable cluster
    # (>= min_cluster agents) that agrees on one alternative to the consensus.
    # A lone reliable voice is noise; a concentrated reliable minority is a
    # signal that the consensus may itself be a blind spot.
    min_cluster = 2
    min_support_ratio = 0.34   # alt must carry >= this share of consensus support
    minority: List[Dict] = []
    for item in range(responses.shape[1]):
        consensus = int(labels[item])
        # reliability-weighted support for each option among confident agents
        support = np.zeros(n_classes)
        for a in range(n_agents):
            if reliability[a] >= minority_reliability_floor:
                support[responses[a, item]] += reliability[a]
        cons_support = support[consensus]
        if cons_support <= 0:
            continue
        alt = int(np.argmax([s if c != consensus else -1 for c, s in enumerate(support)]))
        cluster = [
            agent_ids[a] for a in range(n_agents)
            if responses[a, item] == alt and reliability[a] >= minority_reliability_floor
        ]
        # surface only a *credible* minority: a reliable cluster carrying real weight
        if len(cluster) >= min_cluster and support[alt] >= min_support_ratio * cons_support:
            minority.append({
                "item": item,
                "consensus": consensus,
                "consensus_posterior": float(T[item, consensus]),
                "dissent_label": alt,
                "dissenters": cluster,
                "n_dissent": len(cluster),
            })

    return AggregationResult(
        labels=labels,
        posteriors=T,
        agent_reliability=reliability,
        agent_weights=weights,
        minority_reports=minority,
    )
