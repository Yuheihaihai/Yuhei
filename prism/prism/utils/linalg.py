"""Lightweight linear-algebra helpers used across Prism.

Everything here is numpy-only so the statistical backbone runs without
scipy/sklearn. Heavier optional dependencies (UMAP) are looked up lazily by
callers; these functions are the always-available fallback.
"""
from __future__ import annotations

import numpy as np


def pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project rows of ``X`` onto their top principal components.

    Pure-numpy PCA via SVD. Used as the always-available fallback for the
    fingerprint visualisation when UMAP is not installed.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("pca expects a 2D array")
    n_components = min(n_components, *X.shape)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # economy SVD; columns of Vt are principal directions
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


def log_volume(similarity: np.ndarray, eps: float = 1e-9) -> float:
    """Log-determinant of a similarity/kernel matrix = log of spanned volume.

    The determinant of a similarity matrix is the squared volume of the
    parallelepiped spanned by the members; a larger volume means the members
    cover more of the space (more diverse). We return the log-determinant for
    numerical stability and add ``eps`` to the diagonal so a rank-deficient
    (perfectly redundant) panel yields a very negative — not undefined — value.
    """
    S = np.asarray(similarity, dtype=float)
    n = S.shape[0]
    S = S + eps * np.eye(n)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``X`` (range roughly [-1, 1])."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def safe_corrcoef(X: np.ndarray) -> np.ndarray:
    """Correlation matrix that tolerates zero-variance rows (returns 0 there)."""
    X = np.asarray(X, dtype=float)
    std = X.std(axis=1)
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    # rows/cols with no variance: force correlation to 0 except self
    zero = std == 0
    C[zero, :] = 0.0
    C[:, zero] = 0.0
    np.fill_diagonal(C, 1.0)
    return C
