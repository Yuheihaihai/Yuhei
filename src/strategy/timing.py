"""Timing optimizer — determines optimal posting windows."""

from __future__ import annotations

from src.storage.database import Database

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class TimingOptimizer:
    """Determines optimal posting windows from engagement heatmap data."""

    def __init__(self, store: Database):
        self.store = store

    def get_optimal_windows(
        self, author_id, topic: str, n_windows: int = 3
    ) -> list[tuple[str, int, float]]:
        """
        Return top-N posting windows as (day_name, hour_utc, score).
        Uses stored heatmap data; falls back to global heuristic.
        """
        heatmap: dict[tuple[int, int], float] = {}
        for dow in range(7):
            for hour in range(24):
                heatmap[(dow, hour)] = self.store.engagement_heatmap_lookup(dow, hour)

        ranked = sorted(heatmap.items(), key=lambda x: x[1], reverse=True)

        results = []
        for (dow, hour), score in ranked[:n_windows]:
            results.append((_DAY_NAMES[dow], hour, round(score, 2)))

        return results
