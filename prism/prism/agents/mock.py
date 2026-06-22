"""Offline, deterministic agent simulator.

The design doc's value claim is statistical: a panel of agents whose *errors*
are uncorrelated covers blind spots a single strong model misses. To validate
that machinery without GPUs or model downloads, this module simulates a world
of multiple-choice tasks and agents with structured, correlated blind spots:

* Each task belongs to a *topic* with a latent skill demand and one seductive
  *trap* option.
* Each agent has a competence vector over skills plus a set of *blind topics*.
  Blind topics have a component shared by all agents on the same base model
  (-> correlated errors) and an idiosyncratic per-persona component
  (-> diversity). On a blind topic an agent tends to fall for the trap *and*
  stays over-confident — exactly the "smart models make the same mistake"
  failure mode the doc warns about.

The result is a faithful testbed: a single best model is accurate overall yet
systematically wrong on its blind topics; an uncorrelated panel, aggregated
with Dawid-Skene, recovers those cases. Nothing here is needed in production —
it is the simulation harness for the statistical backbone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .base import AgentBackend, AgentResponse, AgentSpec


def _seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:16], 16)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Task:
    """A decidable multiple-choice item (probe or benchmark question)."""

    id: int
    topic: int
    question: str
    options: List[str]
    answer: int            # ground-truth option index
    trap: int              # seductive wrong option for this topic
    skill_demand: np.ndarray


@dataclass
class _Profile:
    competence: np.ndarray
    blind_topics: frozenset
    noise: float


@dataclass
class SimWorld:
    """Generates tasks and resolves agent responses deterministically."""

    n_topics: int = 12
    n_skills: int = 6
    n_options: int = 4
    seed: int = 7
    # difficulty knobs (tuned so overall accuracy is realistic, blind topics fail)
    alpha: float = 6.0     # weight on skill match
    beta: float = 3.2      # penalty on a blind topic
    gamma: float = 1.0     # global difficulty offset
    blind_frac: float = 0.18
    _topic_skill: np.ndarray = field(init=False)
    _topic_trap: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._topic_skill = rng.dirichlet(np.ones(self.n_skills), size=self.n_topics)
        # trap = a fixed wrong option index per topic (answer is rotated in make_tasks)
        self._topic_trap = rng.integers(1, self.n_options, size=self.n_topics)

    # ---- task generation -------------------------------------------------
    def make_tasks(self, n: int, *, kind: str = "task", offset: int = 0) -> List[Task]:
        rng = np.random.default_rng(_seed(str(self.seed), kind, str(offset)))
        tasks: List[Task] = []
        for i in range(n):
            topic = int(rng.integers(0, self.n_topics))
            answer = int(rng.integers(0, self.n_options))
            trap = (answer + int(self._topic_trap[topic])) % self.n_options
            options = [f"option {chr(65 + k)}" for k in range(self.n_options)]
            tasks.append(
                Task(
                    id=offset + i,
                    topic=topic,
                    question=f"[{kind}#{offset + i}] topic={topic} :: choose the correct option",
                    options=options,
                    answer=answer,
                    trap=trap,
                    skill_demand=self._topic_skill[topic],
                )
            )
        return tasks

    # ---- agent profile ---------------------------------------------------
    def profile(self, spec: AgentSpec) -> _Profile:
        model_rng = np.random.default_rng(_seed("model", spec.base_model))
        pers_rng = np.random.default_rng(_seed("persona", spec.base_model, spec.persona))

        base_comp = 0.35 + 0.6 * model_rng.beta(2.5, 2.0, size=self.n_skills)
        pers_shift = 0.25 * (pers_rng.random(self.n_skills) - 0.5)
        competence = np.clip(base_comp + pers_shift, 0.05, 0.99)

        k = max(1, int(round(self.blind_frac * self.n_topics)))
        model_blind = set(model_rng.choice(self.n_topics, size=k, replace=False).tolist())
        persona_blind = set(pers_rng.choice(self.n_topics, size=k, replace=False).tolist())
        # a LoRA adapter deepens *structural* bias: keep persona blind spots sharp
        if spec.lora_adapter:
            extra = pers_rng.choice(self.n_topics, size=1).tolist()
            persona_blind |= set(extra)
        blind = frozenset(model_blind | persona_blind)

        noise = float(np.clip(spec.temperature, 0.0, 1.5)) * 0.35
        return _Profile(competence=competence, blind_topics=blind, noise=noise)

    def respond(self, spec: AgentSpec, task: Task) -> AgentResponse:
        prof = self.profile(spec)
        rng = np.random.default_rng(_seed("resp", spec.id, str(task.id)))
        skill = float(np.dot(prof.competence, task.skill_demand))
        blind = task.topic in prof.blind_topics
        logit = self.alpha * (skill - 0.5) - (self.beta if blind else 0.0) + self.gamma
        logit -= prof.noise  # sampling temperature erodes reliability
        p_correct = _sigmoid(logit)

        if rng.random() < p_correct:
            choice = task.answer
            confidence = float(np.clip(0.6 + 0.35 * p_correct + 0.1 * rng.random(), 0, 1))
        elif blind:
            # systematic failure: fall for the shared trap, stay over-confident
            choice = task.trap
            confidence = float(np.clip(0.55 + 0.3 * rng.random(), 0, 1))
        else:
            wrong = [o for o in range(len(task.options)) if o != task.answer]
            choice = int(rng.choice(wrong))
            confidence = float(np.clip(0.35 + 0.25 * rng.random(), 0, 1))

        return AgentResponse(
            agent_id=spec.id,
            choice=choice,
            confidence=confidence,
            rationale=f"(sim) skill={skill:.2f} blind={blind}",
        )


class MockBackend(AgentBackend):
    """AgentBackend over a SimWorld; resolves questions by exact text match."""

    def __init__(self, world: SimWorld, tasks: Optional[List[Task]] = None):
        self.world = world
        self._index: Dict[str, Task] = {}
        if tasks:
            self.register(tasks)

    def register(self, tasks: List[Task]) -> None:
        for t in tasks:
            self._index[t.question] = t

    def answer(self, spec: AgentSpec, question: str, options: List[str]) -> AgentResponse:
        task = self._index.get(question)
        if task is None:
            raise KeyError(
                "MockBackend received an unregistered question; register its Task first."
            )
        return self.world.respond(spec, task)
