"""Probe batteries.

A probe set is a battery of small, decidable tasks (logic, common sense,
ambiguity, trade-off judgement, error detection ...). Offline we draw them from
a SimWorld; live, supply your own ground-truth Tasks built the same way.
"""
from __future__ import annotations

from typing import List

from ..agents.mock import SimWorld, Task


def default_probe_battery(world: SimWorld, n: int = 60) -> List[Task]:
    """A reproducible probe battery covering the world's topics."""
    return world.make_tasks(n, kind="probe")
