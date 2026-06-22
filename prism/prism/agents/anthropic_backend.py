"""Live agent backend over the Anthropic API.

This realises the cheapest real-world diversity source from the design doc:
persona + sampling settings on top of a shared base model. It is optional —
``anthropic`` only needs to be installed when you actually run live panels; the
offline MockBackend covers tests and the demo. For genuinely uncorrelated
errors the doc recommends mixing open-weights *families*; wire those in behind
this same AgentBackend interface (e.g. a vLLM backend) as a drop-in.
"""
from __future__ import annotations

import json
import os
import re
from typing import List

from .base import AgentResponse, AgentSpec


class AnthropicBackend:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "AnthropicBackend requires the 'anthropic' package: pip install anthropic"
            ) from e
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def answer(self, spec: AgentSpec, question: str, options: List[str]) -> AgentResponse:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        user = (
            f"{question}\n\nOptions:\n{opts}\n\n"
            "Respond ONLY with JSON: "
            '{"choice": <int index>, "confidence": <0..1>, "rationale": "<one line>"}'
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=spec.temperature,
            top_p=spec.top_p,
            system=spec.system_prompt,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return self._parse(spec.id, text, len(options))

    @staticmethod
    def _parse(agent_id: str, text: str, n_options: int) -> AgentResponse:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                choice = int(data.get("choice", 0))
                conf = float(data.get("confidence", 0.5))
                return AgentResponse(
                    agent_id=agent_id,
                    choice=max(0, min(n_options - 1, choice)),
                    confidence=max(0.0, min(1.0, conf)),
                    rationale=str(data.get("rationale", ""))[:200],
                )
            except (ValueError, json.JSONDecodeError):
                pass
        return AgentResponse(agent_id=agent_id, choice=0, confidence=0.0, rationale="parse-failed")
