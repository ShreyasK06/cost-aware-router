"""Routers operating on precomputed feature matrices."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseRouter(ABC):
    name: str = "router"
    is_probabilistic: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, gap: np.ndarray) -> "BaseRouter": ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def _check_fitted(self, attr: str = "_fitted") -> None:
        if not getattr(self, attr, False):
            raise RuntimeError(f"{type(self).__name__} '{self.name}' is not fitted; call fit() before score()")


class _ClassifierRouter(BaseRouter):
    """Shared logic for gap > epsilon classifiers with a single-class fallback."""
    is_probabilistic = True

    def __init__(self, epsilon: float = 0.0, name: str | None = None):
        self.epsilon = epsilon
        self.name = name or "router"
        self._fitted = False
        self._const: float | None = None
        self.model_ = None

    def _make_model(self):  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, gap):
        X = np.asarray(X, dtype=np.float64)
        y = (np.asarray(gap, dtype=np.float64) > self.epsilon).astype(int)
        if len(np.unique(y)) < 2:
            self._const = float(y.mean()) if len(y) else 0.0
            self.model_ = None
        else:
            self._const = None
            self.model_ = self._make_model().fit(X, y)
        self._fitted = True
        return self

    def score(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if self._const is not None:
            return np.full(len(X), self._const, dtype=np.float64)
        return self.model_.predict_proba(X)[:, 1]


class LogisticRouter(_ClassifierRouter):
    def __init__(self, epsilon: float = 0.0, C: float = 1.0, class_weight=None, name: str | None = None):
        super().__init__(epsilon, name or "Logistic")
        self.C = C
        self.class_weight = class_weight

    def _make_model(self):
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=self.C, max_iter=2000,
                                                    class_weight=self.class_weight))])

    def export_linear(self) -> dict:
        self._check_fitted()
        if self.model_ is None:
            raise RuntimeError("single-class fit: no linear model to export")
        sc, clf = self.model_.named_steps["scaler"], self.model_.named_steps["clf"]
        return {"mean": np.asarray(sc.mean_, dtype=np.float64),
                "scale": np.asarray(sc.scale_, dtype=np.float64),
                "coef": np.asarray(clf.coef_, dtype=np.float64).reshape(-1),
                "intercept": np.asarray(clf.intercept_, dtype=np.float64).reshape(1)}


class MLPRouter(_ClassifierRouter):
    def __init__(self, epsilon: float = 0.0, hidden=(256, 64), max_iter: int = 60, seed: int = 0,
                 name: str | None = None):
        super().__init__(epsilon, name or "MLP")
        self.hidden = tuple(hidden)
        self.max_iter = max_iter
        self.seed = seed

    def _make_model(self):
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=self.hidden, early_stopping=True,
                                               max_iter=self.max_iter, random_state=self.seed))])


class GapRegressionRouter(BaseRouter):
    is_probabilistic = False

    def __init__(self, alpha: float = 10.0, name: str | None = None):
        self.alpha = alpha
        self.name = name or "Gap regression"
        self._fitted = False

    def fit(self, X, gap):
        self.model_ = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=self.alpha))])
        self.model_.fit(np.asarray(X, dtype=np.float64), np.asarray(gap, dtype=np.float64))
        self._fitted = True
        return self

    def score(self, X):
        self._check_fitted()
        return self.model_.predict(np.asarray(X, dtype=np.float64))


class KNNRouter(BaseRouter):
    is_probabilistic = False

    def __init__(self, k: int = 40, name: str | None = None):
        self.k = k
        self.name = name or f"kNN (k={k})"
        self._fitted = False

    def fit(self, X, gap):
        X = np.asarray(X, dtype=np.float32)
        self._gap = np.asarray(gap, dtype=np.float64)
        self._k_eff = max(1, min(self.k, len(X)))
        self.nn_ = NearestNeighbors(n_neighbors=self._k_eff, metric="cosine", algorithm="brute").fit(X)
        self._fitted = True
        return self

    def score(self, X):
        self._check_fitted()
        _, idx = self.nn_.kneighbors(np.asarray(X, dtype=np.float32), n_neighbors=self._k_eff)
        return self._gap[idx].mean(axis=1)
