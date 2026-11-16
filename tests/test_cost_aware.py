import numpy as np
import pytest

from router.eval import baselines as B
from router.eval import metrics as M
from router.models.artifact import LinearRouterArtifact
from router.models.cost_aware import CostAwareLogisticRouter
from router.models.routers import LogisticRouter


def _toy(n=4000, seed=0):
    """Feature 0 drives the quality gap, feature 1 drives the strong model's extra cost."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    p_gain = 1 / (1 + np.exp(-1.5 * X[:, 0]))
    cheap = (rng.random(n) > p_gain).astype(float)
    strong = np.ones(n)
    extra = np.exp(-7 + 1.5 * X[:, 1] + 0.1 * rng.normal(size=n))
    data = M.EvalData(cheap, strong, np.full(n, 1e-4), 1e-4 + extra)
    return X, strong - cheap, extra, data


def test_alpha_zero_matches_plain_logistic():
    X, gap, extra, _ = _toy()
    plain = LogisticRouter(C=0.1).fit(X, gap).score(X)
    ca = CostAwareLogisticRouter(alpha=0.0, C=0.1).fit(X, gap, extra)
    np.testing.assert_allclose(ca.score(X), plain, atol=1e-6)
    np.testing.assert_allclose(ca.proba(X), plain, atol=1e-6)


def test_cost_awareness_improves_cost_quality_auc_when_costs_vary():
    X, gap, extra, data = _toy()
    tr, te = np.arange(0, 3000), np.arange(3000, 4000)
    r = CostAwareLogisticRouter(alpha=0.0).fit(X[tr], gap[tr], extra[tr])
    d = data.subset(te)
    auc0 = M.auc_cost_quality(r.score(X[te]), d)
    auc1 = M.auc_cost_quality(r.with_alpha(1.0).score(X[te]), d)
    assert auc1 > auc0 + 0.02
    assert auc1 > 0.5


def test_predicted_extra_cost_tracks_truth():
    X, gap, extra, _ = _toy()
    r = CostAwareLogisticRouter().fit(X, gap, extra)
    assert np.corrcoef(np.log(r.predicted_extra_cost(X)), np.log(extra))[0, 1] > 0.95


def test_export_round_trip_matches_score(tmp_path):
    X, gap, extra, _ = _toy(800)
    r = CostAwareLogisticRouter(alpha=0.75).fit(X, gap, extra)
    art = LinearRouterArtifact.from_router(r, {"kind": "linear", "embedder": "hashing"})
    art.save(tmp_path / "ca")
    loaded = LinearRouterArtifact.load(tmp_path / "ca")
    np.testing.assert_allclose(loaded.predict_proba(X), r.score(X), atol=1e-6)


def test_fit_requires_cost_and_score_requires_fit():
    X, gap, extra, _ = _toy(200)
    with pytest.raises(ValueError):
        CostAwareLogisticRouter().fit(X, gap)
    with pytest.raises(RuntimeError):
        CostAwareLogisticRouter().score(X)


def test_cost_aware_oracle_dominates_gap_oracle_in_cost_space():
    _, _, _, data = _toy(2000, seed=3)
    a = M.auc_cost_quality(B.oracle_cost_aware_scores(data), data)
    b = M.auc_cost_quality(B.oracle_scores(data), data)
    assert a >= b - 1e-12
    # never routes a prompt where the strong model does not help before one where it does
    s = B.oracle_cost_aware_scores(data)
    gap = data.strong_score - data.cheap_score
    assert s[gap > 0].min() > s[gap <= 0].max()
