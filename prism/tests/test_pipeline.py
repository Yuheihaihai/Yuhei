import numpy as np

from prism import MockBackend, Prism, SimWorld, build_starter_library
from prism.deliberation.protocol import deliberate
from prism.eval import compare_methods, gate2_decision, make_benchmark
from prism.fingerprint import default_probe_battery, run_probes


def _setup():
    world = SimWorld(seed=7)
    lib = build_starter_library()
    backend = MockBackend(world)
    probes = default_probe_battery(world, 60)
    bench = make_benchmark(world, 150)
    backend.register(probes)
    backend.register(bench)
    return world, lib, backend, probes, bench


def test_collapse_prevention_reduces_collapse():
    _, lib, backend, _, bench = _setup()
    res = run_probes(lib, backend, bench)
    off = deliberate(res.choices, res.confidence, 4, prevent_collapse=False, seed=0)
    on = deliberate(res.choices, res.confidence, 4, prevent_collapse=True, seed=0)
    assert on.collapse <= off.collapse
    assert on.diversity_after >= off.diversity_after


def test_gate2_prism_beats_baselines():
    world, lib, backend, _, bench = _setup()
    res = run_probes(lib, backend, bench)
    answers = np.array([t.answer for t in bench])
    results = compare_methods(res.choices, res.confidence, answers, world.n_options,
                              agent_ids=res.agent_ids)
    decision = gate2_decision(results)
    assert decision["go"] is True
    assert results["prism"].accuracy >= results["single_best"].accuracy
    assert results["prism"].blind_spot_coverage >= results["naive_majority"].blind_spot_coverage


def test_prism_pipeline_runs():
    world, lib, backend, probes, bench = _setup()
    prism = Prism(lib, backend, world.n_options, panel_size=25).fit(probes)
    ans = prism.answer(bench)
    answers = np.array([t.answer for t in bench])
    assert len(ans.panel_ids) == 25
    assert (ans.labels == answers).mean() > 0.8
