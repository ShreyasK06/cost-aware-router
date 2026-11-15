"""Pure-numpy serving artifact (numpy + json + pathlib only; no pickles)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _stem(path_stem) -> Path:
    p = Path(path_stem)
    if p.suffix in (".npz", ".json"):
        p = p.with_suffix("")
    return p


class LinearRouterArtifact:
    def __init__(self, mean, scale, coef, intercept, meta: dict):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
        self.meta = dict(meta)
        self.meta["n_features"] = int(self.coef.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.coef.shape[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept[0]
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def save(self, path_stem) -> None:
        stem = _stem(path_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stem.with_name(stem.name + ".npz"), mean=self.mean, scale=self.scale,
                 coef=self.coef, intercept=self.intercept)
        stem.with_name(stem.name + ".json").write_text(
            json.dumps(self.meta, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path_stem) -> "LinearRouterArtifact":
        stem = _stem(path_stem)
        with np.load(stem.with_name(stem.name + ".npz"), allow_pickle=False) as z:
            arrays = {k: z[k] for k in ("mean", "scale", "coef", "intercept")}
        meta = json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))
        return cls(arrays["mean"], arrays["scale"], arrays["coef"], arrays["intercept"], meta)

    @classmethod
    def from_router(cls, router, meta: dict) -> "LinearRouterArtifact":
        lin = router.export_linear()
        return cls(lin["mean"], lin["scale"], lin["coef"], lin["intercept"], meta)
