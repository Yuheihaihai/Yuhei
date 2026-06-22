"""Ground-truth benchmark generation.

Blind-spot reduction is only measurable where the answer is decidable, so the
benchmark is a battery of decidable multiple-choice items drawn from the same
SimWorld the agents live in (live: bring your own ground-truth QA / reasoning /
error-detection sets).
"""
from __future__ import annotations

from typing import List

from ..agents.mock import SimWorld, Task


def make_benchmark(world: SimWorld, n: int = 200) -> List[Task]:
    return world.make_tasks(n, kind="bench")
