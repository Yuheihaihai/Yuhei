"""Phase 2 (★ most important) — prove blind-spot reduction, the project go/no-go.

Compares single best model / naive majority / Prism on a ground-truth benchmark,
runs the Diversity Engine (QD growth + saturation stop) and per-question panel
selection, and prints the Gate-2 decision.

Run:  python examples/run_phase2.py
"""
from __future__ import annotations

import numpy as np

from prism import MockBackend, Prism, SimWorld, build_starter_library
from prism.diversity import DiversityEngine, StoppingRule
from prism.eval import compare_methods, gate2_decision, make_benchmark
from prism.fingerprint import default_probe_battery, run_probes


def main() -> None:
    world = SimWorld(seed=7)
    library = build_starter_library()
    backend = MockBackend(world)

    probes = default_probe_battery(world, n=80)
    bench = make_benchmark(world, n=200)
    backend.register(probes)
    backend.register(bench)

    answers = np.array([t.answer for t in bench])

    # --- Phase 2 core: single best vs naive majority vs Prism -------------
    full = run_probes(library, backend, bench)
    results = compare_methods(full.choices, full.confidence, answers,
                              world.n_options, agent_ids=full.agent_ids)
    print("=== Phase 2: blind-spot reduction (whole library) ===")
    for s in results.values():
        print("  " + s.row())
    decision = gate2_decision(results)
    print(f"\n  minority reports: {results['prism'].extra['n_minority_reports']}  "
          f"| deliberation collapse: {results['prism'].extra['deliberation_collapse']:.3f}  "
          f"| premature-consensus items frozen: {results['prism'].extra['n_premature_consensus']}")
    print(f"\n  GATE 2: {decision['decision']}")
    print(f"    Prism acc {decision['prism_accuracy']:.3f} vs single-best "
          f"{decision['single_best_accuracy']:.3f}")
    print(f"    Prism blind-spot coverage {decision['prism_blind_spot_coverage']:.3f} "
          f"vs naive {decision['naive_blind_spot_coverage']:.3f}")

    # --- Phase 3 preview: QD growth + saturation stopping rule ------------
    # Grow from a small hand-built seed so the effect is visible: diversity
    # rises as uncorrelated agents are bred, then saturates and auto-stops.
    print("\n=== Diversity Engine: QD growth with saturation stop ===")
    seed_lib = build_starter_library(
        base_models=["llama-3.1-8b", "qwen2.5-7b"],
        personas=["detail_hyperfocus", "lateral_leaper", "disconfirmer"],
        temperatures=[0.7],
    )
    engine = DiversityEngine(backend, probes, seed=1)
    grown = engine.grow(seed_lib, max_new=120, stopping=StoppingRule(patience=8, min_improve=1.3))
    print(f"  proposed={grown.proposed} accepted={grown.accepted} "
          f"library {len(seed_lib)} -> {len(grown.library)}")
    print(f"  effective-sample-size trajectory: "
          f"{grown.history[0]:.2f} -> {max(grown.history):.2f} "
          f"(final {grown.history[-1]:.2f})")
    print(f"  stopped_early={grown.stopped_early} ({grown.reason or 'reached max_new'})")

    # --- Phase 4/5 preview: per-question panel selection (capped cost) ----
    print("\n=== Orchestrated panel (cost capped at panel_size) ===")
    prism = Prism(library, backend, world.n_options, panel_size=30).fit(probes)
    ans = prism.answer(bench)
    acc = float((ans.labels == answers).mean())
    print(f"  selected panel: {len(ans.panel_ids)} agents from {len(grown.library)} in library")
    print(f"  Prism accuracy: {acc:.3f} | minority reports: {len(ans.minority_reports)} "
          f"| mean coverage: {ans.coverage.mean():.3f}")


if __name__ == "__main__":
    main()
