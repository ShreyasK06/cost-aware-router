import numpy as np
import pytest

from router.data.synthetic import make_synthetic_pair_table
from router.eval.baselines import (
    always_cheap, always_strong, oracle_scores, random_quality_at_cost, random_scores,
    heuristic_scores,
)
from router.eval.metrics import (
    EvalData, apgr, auc_cost_quality, calibrate_threshold, cost_at_quality, cpt, frontier,
    operating_point, quality_at_budget, route_at_threshold,
)


def make_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    cheap = (rng.random(n) < 0.55).astype(float)
    strong = np.where(rng.random(n) < 0.5, 1.0, (rng.random(n) < 0.8).astype(float))
    strong = np.maximum(strong, (rng.random(n) < 0.9).astype(float) * cheap)
    return EvalData(cheap, strong, np.full(n, 1e-4), np.full(n, 1e-3))


@pytest.fixture(scope="module")
def data():
    return make_data()


def test_from_frame_subset_len():
    df = make_synthetic_pair_table(n=200, seed=1)
    d = EvalData.from_frame(df)
    assert len(d) == 200
    s = d.subset(np.arange(50))
    assert len(s) == 50
    assert np.allclose(s.cheap_score, df["cheap_score"].to_numpy()[:50])


def test_frontier_endpoints(data):
    f = frontier(random_scores(len(data), 1), data)
    assert len(f) == 101
    assert f["frac_strong"].iloc[0] == 0 and f["frac_strong"].iloc[-1] == 1
    assert f["quality"].iloc[0] == pytest.approx(always_cheap(data)["quality"])
    assert f["quality"].iloc[-1] == pytest.approx(always_strong(data)["quality"])
    assert f["cost"].iloc[0] == pytest.approx(always_cheap(data)["cost"])
    assert f["cost"].iloc[-1] == pytest.approx(always_strong(data)["cost"])
    assert f["rel_quality"].iloc[-1] == pytest.approx(1.0)
    assert f["rel_cost"].iloc[-1] == pytest.approx(1.0)
    assert f["pgr"].iloc[0] == pytest.approx(0.0) and f["pgr"].iloc[-1] == pytest.approx(1.0)


def test_random_apgr_near_half(data):
    a = apgr(random_scores(len(data), 3), data)
    assert abs(a - 0.5) < 0.05


def test_oracle_apgr_dominates(data):
    o = apgr(oracle_scores(data), data)
    assert o >= 0.5
    for seed in range(3):
        assert o >= apgr(random_scores(len(data), seed), data)
    assert o >= apgr(data.strong_score - data.cheap_score + 0.3 * random_scores(len(data), 9), data)


def test_perfect_router_beats_random_at_matched_cost(data):
    sc = oracle_scores(data)
    f = frontier(sc, data, n_points=21)
    mid = f.iloc[10]
    assert mid["quality"] > random_quality_at_cost(mid["cost"], data) + 0.01


def test_auc_oracle_beats_random(data):
    assert auc_cost_quality(oracle_scores(data), data) > auc_cost_quality(random_scores(len(data), 0), data)


def test_cpt_edges(data):
    assert cpt(oracle_scores(data), data, 0.0) == 0.0
    assert cpt(oracle_scores(data), data, 0.5) < 0.5
    assert cpt(oracle_scores(data), data, 1.0) <= 1.0
    # unreachable target
    assert cpt(random_scores(len(data), 0), data, 5.0) == 1.0


def test_cost_at_quality_edges(data):
    r = cost_at_quality(oracle_scores(data), data, 0.95)
    assert r["rel_quality"] >= 0.95 - 1e-9
    assert 0 < r["cost_saving"] < 1
    # unreachable -> always-strong point
    r2 = cost_at_quality(oracle_scores(data), data, 10.0)
    assert r2["frac_strong"] == 1.0 and r2["cost_saving"] == pytest.approx(0.0)
    r3 = cost_at_quality(random_scores(len(data), 0), data, 0.0)
    assert r3["frac_strong"] == 0.0


def test_quality_at_budget_edges(data):
    full = quality_at_budget(oracle_scores(data), data, 1.0)
    assert full["quality"] >= always_strong(data)["quality"] - 1e-12
    none = quality_at_budget(oracle_scores(data), data, 0.0)
    assert none["frac_strong"] == 0.0
    assert none["quality"] == pytest.approx(always_cheap(data)["quality"])
    half = quality_at_budget(oracle_scores(data), data, 0.5)
    assert half["cost"] <= 0.5 * always_strong(data)["cost"] + 1e-12


def test_calibrate_threshold_roundtrip(data):
    sc = oracle_scores(data) + 1e-3 * random_scores(len(data), 5)
    t = calibrate_threshold(sc, data, 0.95)
    op = operating_point(route_at_threshold(sc, t), data)
    assert op["rel_quality"] >= 0.95 - 1e-9
    assert op["frac_strong"] < 1.0


def test_calibrate_threshold_edges(data):
    sc = random_scores(len(data), 0)
    assert calibrate_threshold(sc, data, 0.0) == float("inf")
    assert calibrate_threshold(sc, data, 5.0) == float("-inf")
    assert not route_at_threshold(sc, float("inf")).any()
    assert route_at_threshold(sc, float("-inf")).all()


def test_operating_point_keys_and_lift(data):
    sc = oracle_scores(data) + 1e-3 * random_scores(len(data), 5)
    t = calibrate_threshold(sc, data, 0.9)
    op = operating_point(route_at_threshold(sc, t), data)
    for k in ("quality", "cost", "frac_strong", "rel_quality", "rel_cost", "cost_saving",
              "pgr", "random_quality", "lift_vs_random"):
        assert k in op
    assert op["lift_vs_random"] > 0
    assert op["lift_vs_random"] == pytest.approx(op["quality"] - op["random_quality"])


def test_random_quality_at_cost_endpoints(data):
    c, s = always_cheap(data), always_strong(data)
    assert random_quality_at_cost(c["cost"], data) == pytest.approx(c["quality"])
    assert random_quality_at_cost(s["cost"], data) == pytest.approx(s["quality"])
    assert random_quality_at_cost(10.0, data) == pytest.approx(s["quality"])
    assert random_quality_at_cost(-1.0, data) == pytest.approx(c["quality"])
    mid = 0.5 * (c["cost"] + s["cost"])
    assert random_quality_at_cost(mid, data) == pytest.approx(0.5 * (c["quality"] + s["quality"]))


def test_heuristic_scores_deterministic_and_ordered():
    ps = ["hi", "Solve 3x + 4 = 10 step by step and explain why each step holds.",
          "def f(x):\n    return x"]
    a, b = heuristic_scores(ps), heuristic_scores(ps)
    assert a.shape == (3,) and np.array_equal(a, b)
    assert a[1] > a[0]


def test_frontier_fast_on_large_n():
    d = make_data(n=7000, seed=2)
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        apgr(random_scores(7000, 1), d)
    assert time.perf_counter() - t0 < 5.0
