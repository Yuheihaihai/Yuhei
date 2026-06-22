"""Evaluation loop (§7): measure blind-spot reduction vs baselines."""
from .benchmark import make_benchmark
from .compare import (
    MethodScore,
    brier_score,
    compare_methods,
    expected_calibration_error,
    gate2_decision,
)

__all__ = [
    "make_benchmark",
    "MethodScore",
    "compare_methods",
    "gate2_decision",
    "brier_score",
    "expected_calibration_error",
]
