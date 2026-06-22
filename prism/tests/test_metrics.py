import numpy as np

from prism.diversity.metrics import (
    ambiguity_decomposition,
    effective_sample_size,
    mean_pairwise_error_correlation,
    q_statistic_matrix,
)


def test_ambiguity_decomposition_identity():
    # ensemble_error == avg_individual_error - diversity must hold exactly
    rng = np.random.default_rng(0)
    choices = rng.integers(0, 4, size=(8, 50))
    answers = rng.integers(0, 4, size=50)
    d = ambiguity_decomposition(choices, answers, 4)
    assert abs(d["identity_residual"]) < 1e-9
    assert d["diversity"] >= 0.0


def test_q_statistic_identical_vs_independent():
    # identical agents -> Q == 1 (max correlation)
    correct = np.tile(np.array([[1, 0, 1, 1, 0, 1]]), (3, 1)).astype(bool)
    Q = q_statistic_matrix(correct)
    assert np.allclose(Q[0, 1], 1.0)

    # independent random agents -> mean off-diagonal Q near 0
    rng = np.random.default_rng(1)
    correct = rng.random((20, 400)) < 0.6
    Q = q_statistic_matrix(correct)
    off = Q[~np.eye(20, dtype=bool)]
    assert abs(off.mean()) < 0.2


def test_effective_sample_size_collapses_with_correlation():
    # 10 identical agents -> effective size ~ 1
    correct = np.tile((np.arange(100) % 3 == 0), (10, 1)).astype(bool)
    assert effective_sample_size(correct) < 2.0

    # 10 independent agents -> effective size close to 10
    rng = np.random.default_rng(2)
    correct = rng.random((10, 2000)) < 0.6
    assert effective_sample_size(correct) > 7.0


def test_error_correlation_sign():
    rng = np.random.default_rng(3)
    base = rng.random((1, 500)) < 0.5
    correlated = np.repeat(base, 5, axis=0)
    assert mean_pairwise_error_correlation(correlated) > 0.9
