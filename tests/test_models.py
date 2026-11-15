import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from router.data.synthetic import make_synthetic_pair_table
from router.features import Featurizer, HashingEmbedder
from router.models import (GapRegressionRouter, KNNRouter, LinearRouterArtifact, LogisticRouter,
                           MLPRouter)


@pytest.fixture(scope="module")
def data():
    df = make_synthetic_pair_table(600)
    X = Featurizer(HashingEmbedder(), True).transform(df["prompt"].tolist())
    tr, te = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    return X, df, tr, te


@pytest.mark.parametrize("make,bar", [
    (lambda: LogisticRouter(), 0.7),
    (lambda: MLPRouter(max_iter=30), 0.8),
    (lambda: GapRegressionRouter(), 0.7),
    (lambda: KNNRouter(k=20), 0.7),
])
def test_router_auc(data, make, bar):
    X, df, tr, te = data
    r = make().fit(X[tr], df["gap"].to_numpy()[tr])
    s = r.score(X[te])
    assert s.shape == (te.sum(),) and np.isfinite(s).all()
    assert roc_auc_score(df["needs_strong"].to_numpy()[te], s) > bar
    assert isinstance(r.name, str) and r.name


def test_probabilistic_flags_and_range(data):
    X, df, tr, te = data
    p = LogisticRouter().fit(X[tr], df["gap"].to_numpy()[tr])
    s = p.score(X[te])
    assert p.is_probabilistic and s.min() >= 0 and s.max() <= 1
    assert not GapRegressionRouter().is_probabilistic and not KNNRouter().is_probabilistic


def test_custom_name():
    assert LogisticRouter(name="foo").name == "foo"
    assert KNNRouter(name="bar").name == "bar"


def test_knn_k_larger_than_train():
    X = np.random.default_rng(0).normal(size=(5, 4))
    r = KNNRouter(k=40).fit(X, np.array([0, 1, 0, 1, 1.0]))
    assert np.allclose(r.score(X), 0.6)


def test_export_linear_matches_score(data):
    X, df, tr, te = data
    r = LogisticRouter(C=0.5).fit(X[tr], df["gap"].to_numpy()[tr])
    lin = r.export_linear()
    assert lin["coef"].ndim == 1 and lin["intercept"].shape == (1,)
    assert all(v.dtype == np.float64 for v in lin.values())
    z = ((X[te].astype(np.float64) - lin["mean"]) / lin["scale"]) @ lin["coef"] + lin["intercept"]
    assert np.allclose(1 / (1 + np.exp(-z)), r.score(X[te]), atol=1e-6)


def test_artifact_roundtrip(data, tmp_path):
    X, df, tr, te = data
    r = LogisticRouter().fit(X[tr], df["gap"].to_numpy()[tr])
    meta = {"kind": "linear", "embedder": "hashing", "max_seq_length": 0, "use_handcrafted": True,
            "thresholds": {"rel95": 0.4, "rel90": 0.6}, "created": "2026-01-01"}
    art = LinearRouterArtifact.from_router(r, meta)
    art.save(tmp_path / "sub" / "router")
    assert (tmp_path / "sub" / "router.npz").exists() and (tmp_path / "sub" / "router.json").exists()
    for stem in (tmp_path / "sub" / "router", tmp_path / "sub" / "router.npz"):
        loaded = LinearRouterArtifact.load(stem)
        assert loaded.meta["n_features"] == X.shape[1] and loaded.meta["embedder"] == "hashing"
        assert np.allclose(loaded.predict_proba(X[te]), r.score(X[te]), atol=1e-6)


def test_artifact_feature_mismatch_and_stability():
    art = LinearRouterArtifact(np.zeros(3), np.ones(3), np.array([1.0, 1.0, 1.0]), np.zeros(1), {})
    with pytest.raises(ValueError):
        art.predict_proba(np.zeros((2, 4)))
    p = art.predict_proba(np.array([[1000.0] * 3, [-1000.0] * 3]))
    assert np.isfinite(p).all() and p[0] == 1.0 and p[1] == 0.0


@pytest.mark.parametrize("make", [LogisticRouter, MLPRouter, GapRegressionRouter, KNNRouter])
def test_score_before_fit_raises(make):
    with pytest.raises(RuntimeError, match="not fitted"):
        make().score(np.zeros((2, 3)))


@pytest.mark.parametrize("make", [LogisticRouter, lambda: MLPRouter(max_iter=5)])
def test_single_class_fit_constant(make):
    X = np.random.default_rng(0).normal(size=(20, 4))
    r = make().fit(X, np.zeros(20))
    assert np.allclose(r.score(X), 0.0)
    r = make().fit(X, np.ones(20))
    assert np.allclose(r.score(X), 1.0)


def test_finetune_module_imports():
    # torch/transformers are imported lazily inside functions, so this stays light
    import router.models.finetune as ft
    assert callable(ft.finetune_encoder) and ft.FineTunedRouter.name == "Fine-tuned MiniLM"
