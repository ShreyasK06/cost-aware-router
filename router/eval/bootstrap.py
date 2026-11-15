"""Percentile bootstrap confidence intervals."""
from __future__ import annotations

from typing import Callable

import numpy as np


def _ci(samples: np.ndarray, alpha: float):
    lo = float(np.nanpercentile(samples, 100 * alpha / 2))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float], n: int, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(n)))
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        vals[b] = stat_fn(rng.integers(0, n, n))
    lo, hi = _ci(vals, alpha)
    return {"point": point, "lo": lo, "hi": hi, "n_boot": int(n_boot)}


def bootstrap_many(stat_fn: Callable[[np.ndarray], dict], n: int, n_boot: int = 1000,
                   seed: int = 0, alpha: float = 0.05) -> dict:
    """Several statistics share the same resamples. stat_fn(idx) -> dict[str, float]."""
    rng = np.random.default_rng(seed)
    point = {k: float(v) for k, v in stat_fn(np.arange(n)).items()}
    vals = {k: np.empty(n_boot, dtype=float) for k in point}
    for b in range(n_boot):
        res = stat_fn(rng.integers(0, n, n))
        for k in point:
            vals[k][b] = res[k]
    out = {}
    for k in point:
        lo, hi = _ci(vals[k], alpha)
        out[k] = {"point": point[k], "lo": lo, "hi": hi, "n_boot": int(n_boot)}
    return out
