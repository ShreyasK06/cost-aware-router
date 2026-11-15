"""Unit tests for routing policies (pure numpy, no I/O)."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from router.serve.policy import MIN_WARM, BudgetPolicy, ThresholdPolicy


def test_threshold_policy_boundary():
    pol = ThresholdPolicy(0.5)
    assert pol.decide(0.5) is True
    assert pol.decide(0.4999) is False
    assert pol.decide(0.1, threshold=0.05) is True  # per-request override


@pytest.mark.parametrize("target", [0.1, 0.3, 0.5, 0.8])
def test_budget_policy_uniform_scores_hit_target(target):
    rng = np.random.default_rng(0)
    pol = BudgetPolicy(target, window=500)
    decisions = [pol.decide(float(s)) for s in rng.uniform(size=5000)]
    assert abs(np.mean(decisions) - target) < 0.03


@pytest.mark.parametrize("target", [0.2, 0.4])
def test_budget_policy_skewed_scores_hit_target(target):
    rng = np.random.default_rng(1)
    pol = BudgetPolicy(target, window=500)
    decisions = [pol.decide(float(s)) for s in rng.beta(0.5, 4.0, size=5000)]
    assert abs(np.mean(decisions) - target) < 0.03


def test_budget_policy_uses_init_threshold_until_warm():
    pol = BudgetPolicy(0.5, init_threshold=0.9)
    for _ in range(MIN_WARM - 1):
        assert pol.decide(0.8) is False  # below init threshold, window not warm yet
    assert pol.threshold == pytest.approx(0.9)


def test_budget_policy_tracks_distribution_shift():
    pol = BudgetPolicy(0.3, window=200)
    rng = np.random.default_rng(2)
    for s in rng.uniform(0.0, 0.3, size=400):
        pol.decide(float(s))
    low_thr = pol.threshold
    for s in rng.uniform(0.7, 1.0, size=400):
        pol.decide(float(s))
    assert pol.threshold > low_thr + 0.3


def test_budget_policy_extreme_targets():
    rng = np.random.default_rng(3)
    never, always = BudgetPolicy(0.0), BudgetPolicy(1.0)
    scores = rng.uniform(size=200)
    n = [(never.decide(float(s)), always.decide(float(s))) for s in scores]
    assert not any(a for a, _ in n[MIN_WARM:])  # first MIN_WARM use init_threshold
    assert all(b for _, b in n[MIN_WARM:])


def test_budget_policy_thread_safe():
    pol = BudgetPolicy(0.5, window=100)
    out: list[bool] = []

    def worker(seed: int):
        rng = np.random.default_rng(seed)
        res = [pol.decide(float(s)) for s in rng.uniform(size=500)]
        out.extend(res)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 4000
    assert 0.35 < np.mean(out) < 0.65
