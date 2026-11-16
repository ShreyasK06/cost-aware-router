"""Cost-aware logistic router.

A plain router ranks prompts by P(needs strong) and ignores that prompts differ in how much the
strong model costs. When per-prompt cost varies a lot (in RouterBench one task family carries about
half of the strong-model bill), the better ranking is "expected benefit per extra dollar".

Two linear heads share one StandardScaler:
  * logistic regression:  z(x) = logit P(strong beats cheap by more than epsilon)
  * ridge regression:     c(x) = predicted log(extra cost of the strong model in USD)
Routing score = sigmoid(z(x) - alpha * (c(x) - mean log extra cost)).
alpha = 0 recovers the plain logistic router; alpha is chosen on validation data.
Because both heads are linear in the same standardised features, the combination is again a single
linear model and exports to the same pure-numpy serving artifact.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from router.models.routers import BaseRouter

MIN_COST = 1e-9


class CostAwareLogisticRouter(BaseRouter):
    is_probabilistic = False  # the score is a cost-adjusted ranking score in (0, 1), not a probability

    def __init__(self, alpha: float = 1.0, epsilon: float = 0.0, C: float = 0.1,
                 ridge_alpha: float = 10.0, name: str | None = None):
        self.alpha = float(alpha)
        self.epsilon = epsilon
        self.C = C
        self.ridge_alpha = ridge_alpha
        self.name = name or "Cost-aware logistic"
        self._fitted = False

    def fit(self, X: np.ndarray, gap: np.ndarray, extra_cost: np.ndarray | None = None
            ) -> "CostAwareLogisticRouter":
        if extra_cost is None:
            raise ValueError("CostAwareLogisticRouter.fit needs extra_cost = strong_cost - cheap_cost")
        X = np.asarray(X, dtype=np.float64)
        y = (np.asarray(gap, dtype=np.float64) > self.epsilon).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("training labels contain a single class")
        log_cost = np.log(np.maximum(np.asarray(extra_cost, dtype=np.float64), MIN_COST))
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000).fit(Xs, y)
        self.cost_ = Ridge(alpha=self.ridge_alpha).fit(Xs, log_cost)
        self.mean_log_cost_ = float(log_cost.mean())
        self._fitted = True
        return self

    def with_alpha(self, alpha: float) -> "CostAwareLogisticRouter":
        """Same fitted heads, different cost exponent (no refit)."""
        self._check_fitted()
        other = CostAwareLogisticRouter(alpha, self.epsilon, self.C, self.ridge_alpha, self.name)
        other.scaler_, other.clf_, other.cost_ = self.scaler_, self.clf_, self.cost_
        other.mean_log_cost_ = self.mean_log_cost_
        other._fitted = True
        return other

    def _heads(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self.clf_.decision_function(Xs), self.cost_.predict(Xs)

    def proba(self, X: np.ndarray) -> np.ndarray:
        """Plain P(needs strong), ignoring cost."""
        self._check_fitted()
        z, _ = self._heads(X)
        return 1.0 / (1.0 + np.exp(-z))

    def predicted_extra_cost(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.exp(self._heads(X)[1])

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        z, c = self._heads(X)
        s = z - self.alpha * (c - self.mean_log_cost_)
        return 1.0 / (1.0 + np.exp(-np.clip(s, -700, 700)))

    def export_linear(self) -> dict:
        """Single linear head equivalent to score(): sigmoid(((X - mean) / scale) @ coef + intercept)."""
        self._check_fitted()
        coef = self.clf_.coef_.reshape(-1) - self.alpha * self.cost_.coef_.reshape(-1)
        intercept = (float(self.clf_.intercept_[0])
                     - self.alpha * (float(self.cost_.intercept_) - self.mean_log_cost_))
        return {"mean": np.asarray(self.scaler_.mean_, dtype=np.float64),
                "scale": np.asarray(self.scaler_.scale_, dtype=np.float64),
                "coef": np.asarray(coef, dtype=np.float64),
                "intercept": np.asarray([intercept], dtype=np.float64)}
