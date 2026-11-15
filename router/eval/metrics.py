"""Frontier metrics for a cheap-vs-strong router (all numpy, vectorised)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalData:
    cheap_score: np.ndarray
    strong_score: np.ndarray
    cheap_cost: np.ndarray
    strong_cost: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "EvalData":
        return cls(
            cheap_score=df["cheap_score"].to_numpy(dtype=float),
            strong_score=df["strong_score"].to_numpy(dtype=float),
            cheap_cost=df["cheap_cost"].to_numpy(dtype=float),
            strong_cost=df["strong_cost"].to_numpy(dtype=float),
        )

    def subset(self, idx) -> "EvalData":
        idx = np.asarray(idx)
        return EvalData(
            self.cheap_score[idx],
            self.strong_score[idx],
            self.cheap_cost[idx],
            self.strong_cost[idx],
        )

    def __len__(self) -> int:
        return int(len(self.cheap_score))


def _trapz(y, x) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _order(scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices sorted by descending score; ties broken by a seeded permutation."""
    scores = np.asarray(scores, dtype=float)
    perm = np.random.default_rng(seed).permutation(len(scores))
    return perm[np.argsort(-scores[perm], kind="stable")]


def _full_curve(scores, data: EvalData, seed: int = 0):
    """Return (frac, quality, cost) arrays for k = 0..n strong prompts."""
    n = len(data)
    order = _order(scores, seed)
    dq = (data.strong_score - data.cheap_score)[order]
    dc = (data.strong_cost - data.cheap_cost)[order]
    q = (data.cheap_score.sum() + np.concatenate([[0.0], np.cumsum(dq)])) / n
    c = (data.cheap_cost.sum() + np.concatenate([[0.0], np.cumsum(dc)])) / n
    frac = np.arange(n + 1) / n
    return frac, q, c


def _pgr(q, q_cheap, q_strong):
    denom = q_strong - q_cheap
    if denom == 0:
        return np.zeros_like(q)
    return (q - q_cheap) / denom


def frontier(scores, data: EvalData, n_points: int = 101, seed: int = 0) -> pd.DataFrame:
    n = len(data)
    frac, q, c = _full_curve(scores, data, seed)
    ks = np.round(np.linspace(0, 1, n_points) * n).astype(int)
    q_cheap, q_strong, c_strong = q[0], q[-1], c[-1]
    qk, ck = q[ks], c[ks]
    return pd.DataFrame(
        {
            "frac_strong": frac[ks],
            "cost": ck,
            "quality": qk,
            "rel_quality": qk / q_strong if q_strong != 0 else np.nan,
            "rel_cost": ck / c_strong if c_strong != 0 else np.nan,
            "pgr": _pgr(qk, q_cheap, q_strong),
        }
    )


def apgr(scores, data: EvalData, seed: int = 0) -> float:
    frac, q, _ = _full_curve(scores, data, seed)
    return _trapz(_pgr(q, q[0], q[-1]), frac)


def auc_cost_quality(scores, data: EvalData, seed: int = 0) -> float:
    _, q, c = _full_curve(scores, data, seed)
    span = c[-1] - c[0]
    if span == 0:
        return float(q.mean())
    x = (c - c[0]) / span
    o = np.argsort(x, kind="stable")
    return _trapz(q[o], x[o])


def cpt(scores, data: EvalData, target_pgr: float = 0.5, seed: int = 0) -> float:
    frac, q, _ = _full_curve(scores, data, seed)
    pgr = _pgr(q, q[0], q[-1])
    hit = np.nonzero(pgr >= target_pgr - 1e-12)[0]
    return float(frac[hit[0]]) if len(hit) else 1.0


def cost_at_quality(scores, data: EvalData, rel_quality: float = 0.95, seed: int = 0) -> dict:
    frac, q, c = _full_curve(scores, data, seed)
    q_strong, c_strong = q[-1], c[-1]
    ok = np.nonzero(q >= rel_quality * q_strong - 1e-12)[0]
    i = len(q) - 1 if len(ok) == 0 else ok[np.argmin(c[ok])]
    return {
        "cost": float(c[i]),
        "cost_saving": float(1 - c[i] / c_strong) if c_strong else 0.0,
        "frac_strong": float(frac[i]),
        "quality": float(q[i]),
        "rel_quality": float(q[i] / q_strong) if q_strong else float("nan"),
    }


def quality_at_budget(scores, data: EvalData, budget_frac: float = 0.5, seed: int = 0) -> dict:
    frac, q, c = _full_curve(scores, data, seed)
    q_strong, c_strong = q[-1], c[-1]
    ok = np.nonzero(c <= budget_frac * c_strong + 1e-15)[0]
    i = 0 if len(ok) == 0 else ok[np.argmax(q[ok])]
    return {
        "quality": float(q[i]),
        "rel_quality": float(q[i] / q_strong) if q_strong else float("nan"),
        "cost": float(c[i]),
        "frac_strong": float(frac[i]),
    }


def calibrate_threshold(val_scores, val_data: EvalData, rel_quality: float, seed: int = 0) -> float:
    val_scores = np.asarray(val_scores, dtype=float)
    _, q, _ = _full_curve(val_scores, val_data, seed)
    ok = np.nonzero(q >= rel_quality * q[-1] - 1e-12)[0]
    if len(ok) == 0:
        return float("-inf")
    k = int(ok[0])
    if k == 0:
        return float("inf")
    return float(np.sort(val_scores)[::-1][k - 1])


def route_at_threshold(scores, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) >= threshold


def operating_point(route_strong, data: EvalData) -> dict:
    from router.eval.baselines import random_quality_at_cost

    r = np.asarray(route_strong, dtype=bool)
    quality = float(np.where(r, data.strong_score, data.cheap_score).mean())
    cost = float(np.where(r, data.strong_cost, data.cheap_cost).mean())
    q_cheap = float(data.cheap_score.mean())
    q_strong = float(data.strong_score.mean())
    c_strong = float(data.strong_cost.mean())
    rq = float(random_quality_at_cost(cost, data))
    denom = q_strong - q_cheap
    return {
        "quality": quality,
        "cost": cost,
        "frac_strong": float(r.mean()),
        "rel_quality": quality / q_strong if q_strong else float("nan"),
        "rel_cost": cost / c_strong if c_strong else float("nan"),
        "cost_saving": 1 - cost / c_strong if c_strong else 0.0,
        "pgr": (quality - q_cheap) / denom if denom != 0 else 0.0,
        "random_quality": rq,
        "lift_vs_random": quality - rq,
    }
