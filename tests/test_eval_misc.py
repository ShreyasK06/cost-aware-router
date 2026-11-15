import numpy as np
import pandas as pd

from router.eval.bootstrap import bootstrap_ci, bootstrap_many
from router.eval.calibration import expected_calibration_error, reliability_curve
from router.eval.latency import measure_overhead
from router.eval.metrics import EvalData, frontier
from router.eval.baselines import oracle_scores, random_scores
from router.eval.plots import plot_frontier, plot_per_task, plot_reliability
from router.eval.robustness import perturb_case, perturb_filler, perturb_typos

PROMPTS = ["What is the capital of France?", "Solve 2x + 3 = 11.", "Write a function to sort a list."]


def test_bootstrap_contains_point_and_deterministic():
    x = np.random.default_rng(0).random(300)
    f = lambda idx: x[idx].mean()
    a = bootstrap_ci(f, 300, n_boot=200, seed=4)
    b = bootstrap_ci(f, 300, n_boot=200, seed=4)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["n_boot"] == 200
    c = bootstrap_ci(f, 300, n_boot=200, seed=5)
    assert c["lo"] != a["lo"]


def test_bootstrap_many_and_nan_ignored():
    x = np.random.default_rng(1).random(200)
    out = bootstrap_many(lambda idx: {"m": x[idx].mean(), "s": x[idx].std()}, 200, n_boot=100)
    assert set(out) == {"m", "s"}
    for v in out.values():
        assert v["lo"] <= v["point"] <= v["hi"]
    r = bootstrap_ci(lambda idx: np.nan if idx[0] % 7 == 0 else 1.0, 50, n_boot=100)
    assert r["lo"] == 1.0


def test_ece_calibrated_vs_miscalibrated():
    probs = np.repeat([0.1, 0.5, 0.9], 1000)
    labels = np.concatenate([
        np.r_[np.ones(100), np.zeros(900)],
        np.r_[np.ones(500), np.zeros(500)],
        np.r_[np.ones(900), np.zeros(100)],
    ])
    assert expected_calibration_error(probs, labels) < 1e-9
    assert expected_calibration_error(probs, 1 - labels) > 0.3
    c = reliability_curve(probs, labels)
    assert list(c.columns) == ["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"]
    assert len(c) == 3 and c["count"].sum() == 3000


def test_perturbations():
    for fn in (perturb_typos, perturb_case, perturb_filler):
        a, b = fn(PROMPTS, seed=3), fn(PROMPTS, seed=3)
        assert a == b and len(a) == len(PROMPTS)
    assert perturb_typos(PROMPTS, rate=0.3, seed=1) != PROMPTS
    assert perturb_typos(PROMPTS, rate=0.0, seed=1) == PROMPTS
    assert all(p == p.lower() for p in perturb_case(PROMPTS))
    f = perturb_filler(PROMPTS)
    assert all(p in o and o.endswith("Thanks in advance!") for p, o in zip(PROMPTS, f))


def test_measure_overhead():
    calls = []
    r = measure_overhead(lambda p: calls.append(p), PROMPTS, n=30, warmup=5)
    assert set(r) == {"p50_ms", "p95_ms", "mean_ms", "n"}
    assert r["n"] == 30 and len(calls) == 35
    assert r["p95_ms"] >= r["p50_ms"] >= 0


def test_plots_write_pngs(tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    cheap = (rng.random(n) < 0.5).astype(float)
    strong = np.maximum(cheap, (rng.random(n) < 0.8).astype(float))
    data = EvalData(cheap, strong, np.full(n, 1e-4), np.full(n, 1e-3))
    curves = {"Oracle": frontier(oracle_scores(data), data),
              "Router": frontier(random_scores(n, 0), data)}
    q = curves["Router"]["quality"].to_numpy()
    for x in ("cost", "frac"):
        out = tmp_path / "sub" / f"frontier_{x}.png"
        plot_frontier(curves, data, out, "t", x=x, bands={"Router": (q - 0.02, q + 0.02)})
        assert out.stat().st_size > 1000
    probs = rng.random(500)
    labels = (rng.random(500) < probs).astype(int)
    cur = reliability_curve(probs, labels)
    out = tmp_path / "rel.png"
    plot_reliability(cur, expected_calibration_error(probs, labels), out, "rel")
    assert out.stat().st_size > 1000
    table = pd.DataFrame({"task_family": ["mmlu", "gsm8k"], "cheap": [0.5, 0.3],
                          "strong": [0.8, 0.9], "router": [0.75, 0.8], "frac_strong": [0.4, 0.6]})
    out = tmp_path / "task.png"
    plot_per_task(table, out)
    assert out.stat().st_size > 1000
