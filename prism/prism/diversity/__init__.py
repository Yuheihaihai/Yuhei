"""Diversity Engine (component ③): measure diversity and grow the library."""
from .metrics import (
    DiversitySummary,
    ambiguity_decomposition,
    double_fault_matrix,
    effective_sample_size,
    mean_pairwise_error_correlation,
    q_statistic_matrix,
    summarize_diversity,
)
from .engine import DiversityEngine, StoppingRule

__all__ = [
    "q_statistic_matrix",
    "double_fault_matrix",
    "ambiguity_decomposition",
    "mean_pairwise_error_correlation",
    "effective_sample_size",
    "DiversitySummary",
    "summarize_diversity",
    "DiversityEngine",
    "StoppingRule",
]
