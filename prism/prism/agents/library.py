"""Agent Library: an in-memory + on-disk collection of AgentSpecs.

Storing an agent is almost free (it is config + an optional adapter path), so
the library can hold thousands of variants while the Orchestrator only ever
instantiates a few dozen per question.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from .base import AgentSpec


class AgentLibrary:
    def __init__(self, specs: Iterable[AgentSpec] = ()):
        self._specs: Dict[str, AgentSpec] = {}
        for s in specs:
            self.add(s)

    def add(self, spec: AgentSpec) -> None:
        if spec.id in self._specs:
            raise ValueError(f"duplicate agent id: {spec.id}")
        self._specs[spec.id] = spec

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[AgentSpec]:
        return iter(self._specs.values())

    def __getitem__(self, agent_id: str) -> AgentSpec:
        return self._specs[agent_id]

    @property
    def ids(self) -> List[str]:
        return list(self._specs.keys())

    def specs(self) -> List[AgentSpec]:
        return list(self._specs.values())

    def subset(self, ids: Iterable[str]) -> "AgentLibrary":
        return AgentLibrary(self._specs[i] for i in ids)

    # ---- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = [asdict(s) for s in self._specs.values()]
        # tuples (tools) are not round-tripped by json as tuples; store as lists
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "AgentLibrary":
        data = json.loads(Path(path).read_text())
        specs = []
        for d in data:
            d = dict(d)
            d["tools"] = tuple(d.get("tools", ()) or ())
            specs.append(AgentSpec(**d))
        return cls(specs)
