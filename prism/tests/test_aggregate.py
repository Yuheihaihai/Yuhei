import numpy as np

from prism.deliberation.aggregate import (
    aggregate,
    dawid_skene,
    response_redundancy_weights,
)


def test_dawid_skene_recovers_truth():
    # 7 agents, mostly-correct, should recover the ground truth labels
    rng = np.random.default_rng(0)
    n_items, n_classes = 80, 4
    truth = rng.integers(0, n_classes, size=n_items)
    responses = []
    for _ in range(7):
        r = truth.copy()
        flip = rng.random(n_items) < 0.25
        r[flip] = rng.integers(0, n_classes, size=flip.sum())
        responses.append(r)
    responses = np.array(responses)
    T, conf, priors = dawid_skene(responses, n_classes)
    acc = (T.argmax(axis=1) == truth).mean()
    assert acc > 0.9


def test_reliable_minority_beats_confident_majority():
    # 4 correlated agents always say class 0 (wrong); 3 reliable agents say truth.
    n_items, n_classes = 60, 3
    rng = np.random.default_rng(1)
    truth = rng.integers(1, n_classes, size=n_items)  # truth never 0
    block = np.zeros((4, n_items), dtype=int)          # the correlated wrong cluster
    good = np.tile(truth, (3, 1))                       # reliable minority
    # give the good agents some noise so DS can estimate their reliability
    for a in range(3):
        flip = rng.random(n_items) < 0.1
        good[a, flip] = 0
    responses = np.vstack([block, good])
    res = aggregate(responses, n_classes, correlation_correction=True)
    acc = (res.labels == truth).mean()
    # correlation correction should down-weight the 4-agent bloc enough to win
    assert acc > 0.7
    assert res.agent_weights[:4].mean() < res.agent_weights[4:].mean()


def test_redundancy_weights_downweight_duplicates():
    rng = np.random.default_rng(2)
    unique = rng.integers(0, 4, size=(1, 100))
    dup = np.repeat(unique, 6, axis=0)            # 6 identical agents
    diverse = rng.integers(0, 4, size=(3, 100))   # 3 independent agents
    responses = np.vstack([dup, diverse])
    w = response_redundancy_weights(responses)
    assert w[:6].mean() < w[6:].mean()
