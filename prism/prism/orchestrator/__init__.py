"""Panel Selector / Orchestrator (component ④)."""
from .selector import PanelSelection, greedy_dpp, select_panel

__all__ = ["greedy_dpp", "select_panel", "PanelSelection"]
