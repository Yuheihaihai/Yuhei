"""Phase 1 — hand-built diverse agents + fingerprinting + diversity visualisation.

Gate 1: do the fingerprints spread out (and where do they clump = redundancy)?
Run:  python examples/run_phase1.py
"""
from __future__ import annotations

import numpy as np

from prism import MockBackend, SimWorld, build_starter_library
from prism.diversity import summarize_diversity
from prism.fingerprint import (
    cognitive_fingerprints,
    default_probe_battery,
    reduce_2d,
    run_probes,
)
from prism.fingerprint.fingerprint import error_correlation


def main() -> None:
    world = SimWorld(seed=7)
    library = build_starter_library()
    backend = MockBackend(world)
    probes = default_probe_battery(world, n=80)
    backend.register(probes)

    print(f"Library: {len(library)} agents (base models x personas x sampling)")
    results = run_probes(library, backend, probes)
    answers = np.array([t.answer for t in probes])

    summ = summarize_diversity(results.choices, results.correct, answers, world.n_options)
    print("\nDiversity summary:")
    for k, v in summ.as_dict().items():
        print(f"  {k:<26} {v:.3f}" if isinstance(v, float) else f"  {k:<26} {v}")

    ec = error_correlation(results)
    off = ec[~np.eye(len(ec), dtype=bool)]
    print(f"\nError correlation: mean={off.mean():.3f}  max={off.max():.3f}")
    print("  (low mean = independent errors = real diversity, not just different personas)")

    fp = cognitive_fingerprints(results)
    coords = reduce_2d(fp)  # UMAP if available, else PCA
    spread = float(np.linalg.norm(coords - coords.mean(0), axis=1).mean())
    print(f"\n2D fingerprint spread (mean radius): {spread:.3f}")

    gate1 = off.mean() < 0.25 and spread > 0
    print(f"\nGATE 1: {'PASS' if gate1 else 'FAIL'} — "
          f"fingerprints vary and error correlation is low.")


if __name__ == "__main__":
    main()
