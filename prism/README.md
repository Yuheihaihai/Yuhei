# Prism — Cognitive Diversity Orchestration

> 「正解に収束する」型の逆を行く。意図的に相関していない “クセのある” AIエージェント群を作り、崩さず束ねて、集団の盲点を減らすオーケストレーション・システム。

Prism builds intentionally **uncorrelated**, quirky AI agents and bundles them
*without collapsing their diversity*, to reduce a group's **blind spots**. The
reward is inverted from a convergence-optimising ensemble: Prism optimises for
the **preservation of diversity**, and treats diversity as a measurable,
subtractable quantity (ensemble error decomposition) rather than a vibe.

This repository is the **Phase 0–2 implementation** of the design doc, with
working previews of Phases 3–5. The statistical backbone runs on **numpy alone**
— no GPU, no model downloads — using an offline agent simulator so the central
claim ("a diverse panel reduces the blind spots a single strong model misses")
can actually be *measured*. A live `AnthropicBackend` is included for real panels.

## The two hard problems (and how the code solves them)

| Design-doc problem | Where it's solved in code |
|---|---|
| **(A) Measurement** — "did the group avoid a blind spot?" is hard to score | `prism/diversity/metrics.py` — ensemble error decomposition (diversity is `avg_individual_error − ensemble_error`), Q-statistic, double-fault, error correlation, correlation-corrected effective sample size |
| **(B) Collapse** — deliberation converges and diversity vanishes | `prism/deliberation/protocol.py` — independent initial opinions, permanent devil's advocate, minority preservation, premature-consensus penalty; measures the diversity drop |

The backbone is statistics: **ensemble error decomposition**, **Condorcet +
correlation correction**, and **Dawid–Skene** crowd aggregation.

## The five components

| # | Component | Module | What it does |
|---|---|---|---|
| ① | Agent Library | `prism/agents/` | Cheap storage of many agents = (base model + persona prompt + sampling + tools + optional LoRA). Built-in cognitive personas: detail-hyperfocus, lateral-leaper, literalist, systems-holist, disconfirmer, pragmatist. |
| ② | Cognitive Fingerprinting | `prism/fingerprint/` | Run a probe battery; record each agent's error vector + behaviour vector → a `cognitive_fingerprint`; reduce to 2D (UMAP if available, else PCA). |
| ③ | Diversity Engine | `prism/diversity/` | Measure diversity; grow the library with Quality-Diversity / MAP-Elites **selecting for low correlation, not accuracy**; **saturation stopping rule** caps cost. |
| ④ | Panel Selector / Orchestrator | `prism/orchestrator/` | Per question, pick the few-dozen most complementary agents via greedy **DPP** (or submodular facility-location). Cost is capped by panel size, not library size. |
| ⑤ | Deliberation + Aggregation | `prism/deliberation/` | Collapse-prevented deliberation, then **Dawid–Skene with correlation correction** (not majority vote). Output = answer **+ minority report + coverage**. |
| + | Evaluation loop | `prism/eval/` | Single-best vs naive-majority vs Prism on a ground-truth benchmark: accuracy, **blind-spot coverage**, calibration (Brier/ECE), and the **Gate-2 go/no-go**. |

## Quickstart

```bash
pip install numpy            # the only hard dependency
PYTHONPATH=. python examples/run_phase1.py    # fingerprints + diversity (Gate 1)
PYTHONPATH=. python examples/run_phase2.py    # blind-spot reduction (Gate 2 ★)
python -m pytest -q                            # 13 tests
```

### Library API (the Phase-5 product shape: question → answer + dissent + coverage)

```python
from prism import SimWorld, MockBackend, build_starter_library, Prism
from prism.fingerprint import default_probe_battery
from prism.eval import make_benchmark

world   = SimWorld(seed=7)
backend = MockBackend(world)                  # swap for AnthropicBackend() to go live
library = build_starter_library()             # base models × personas × sampling
probes  = default_probe_battery(world, 80); backend.register(probes)
tasks   = make_benchmark(world, 200);        backend.register(tasks)

prism = Prism(library, backend, n_classes=world.n_options, panel_size=30).fit(probes)
answer = prism.answer(tasks)
print(answer.labels, answer.minority_reports, answer.coverage.mean())
```

## What the demo proves (Gate 2, the project's go/no-go)

On a seeded ground-truth benchmark (a strong single model with real blind spots):

| Method | Accuracy | Blind-spot coverage | Brier | ECE |
|---|---|---|---|---|
| single best model | 0.745 | — | 0.109 | 0.177 |
| naive majority | 0.900 | 0.667 | 0.134 | 0.233 |
| **Prism** | **0.955** | **0.824** | **0.045** | **0.045** |

Prism recovers **82%** of the correct answers the single strongest model missed
(vs 67% for naive majority) **and** is the best-calibrated — the design doc's
core thesis, made measurable. *(Numbers are from the deterministic simulator;
they validate the machinery, not a real-world benchmark.)*

The Diversity Engine then grows a 6-agent seed library while effective sample
size climbs (~4.6 → ~60) and the **saturation stopping rule** halts growth once
new agents stop adding diversity beyond threshold — the cost lever.

## Going live

`prism/agents/anthropic_backend.py` runs persona agents over the Anthropic API
(`pip install anthropic`, set `ANTHROPIC_API_KEY`). For genuinely uncorrelated
errors the design doc recommends mixing **open-weights families** (Llama / Qwen /
Mistral / Gemma / Phi / DeepSeek) behind the same `AgentBackend` interface — e.g.
a vLLM backend — plus LoRA/QLoRA bias injection in Phase 4.

## Ethics (design doc §9 — the backbone of the project)

Not built to replace people. The two sanctioned uses are **monoculture
auditing** (show, in numbers, how much a single AI / single viewpoint misses)
and **strengths matching**. The goal is to move the justification for neurodiversity
from "useful" to "a right."

## Status

Design-confirmed; this is the implementation through Gate 2. See the full design
doc for Phases 3–5 and the production stack (vLLM, PEFT, pyribs, crowd-kit, dppy).
Per the doc, update at each gate as results come in.
