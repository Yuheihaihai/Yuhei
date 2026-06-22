import numpy as np

from prism.orchestrator.selector import greedy_dpp, select_panel
from prism.utils.linalg import cosine_similarity_matrix


def test_greedy_dpp_returns_distinct_indices():
    rng = np.random.default_rng(0)
    X = rng.random((30, 12))
    S = cosine_similarity_matrix(X)
    idx = greedy_dpp(S, k=8)
    assert len(idx) == 8
    assert len(set(idx)) == 8


def test_dpp_prefers_diverse_over_duplicates():
    # three distinct prototypes, each duplicated 4x -> DPP should spread picks
    protos = np.eye(3)
    X = np.repeat(protos, 4, axis=0) + 1e-3
    S = cosine_similarity_matrix(X)
    idx = greedy_dpp(S, k=3)
    groups = {i // 4 for i in idx}
    assert len(groups) == 3  # one from each distinct cluster


def test_select_panel_caps_size():
    rng = np.random.default_rng(1)
    ids = [f"a{i}" for i in range(50)]
    S = cosine_similarity_matrix(rng.random((50, 8)))
    sel = select_panel(ids, S, k=20, method="dpp")
    assert len(sel.agent_ids) == 20
    assert set(sel.agent_ids).issubset(set(ids))
