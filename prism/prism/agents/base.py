"""Core agent data model.

Per the design doc (§3 ①), one agent = (base model id + persona/values prompt
+ sampling settings + tool permissions + optional LoRA adapter). It is almost
entirely a config file plus an adapter, so storage cost is ~zero. The
``behavior_descriptor`` holds the cognitive axes (abstract<->concrete,
convergent<->divergent, ...) used by the Diversity Engine's MAP-Elites archive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class AgentSpec:
    """A stored agent definition — cheap to keep, instantiated only when called."""

    id: str
    base_model: str
    persona: str
    system_prompt: str
    temperature: float = 0.7
    top_p: float = 1.0
    tools: tuple = ()
    lora_adapter: Optional[str] = None
    # cognitive axes in [0, 1], e.g. {"abstract_concrete": 0.2, "convergent_divergent": 0.9}
    behavior_descriptor: Dict[str, float] = field(default_factory=dict)

    def short(self) -> str:
        return f"{self.id}[{self.base_model}/{self.persona}]"


@dataclass
class AgentResponse:
    """A single agent's answer to a multiple-choice probe/question."""

    agent_id: str
    choice: int  # index into the answer options
    confidence: float  # self-reported, in [0, 1]
    rationale: str = ""


class AgentBackend(Protocol):
    """Anything that can turn an AgentSpec + question into an AgentResponse.

    Implementations: MockBackend (offline, deterministic biased agents) and
    AnthropicBackend (live persona agents over the API).
    """

    def answer(
        self,
        spec: AgentSpec,
        question: str,
        options: List[str],
    ) -> AgentResponse:  # pragma: no cover - protocol
        ...
