"""Probability calibration diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(probs, labels, n_bins: int = 10) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        cnt = int(m.sum())
        if cnt == 0:
            continue
        rows.append({
            "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
            "mean_pred": float(probs[m].mean()), "frac_pos": float(labels[m].mean()),
            "count": cnt,
        })
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"])


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    curve = reliability_curve(probs, labels, n_bins)
    if curve.empty:
        return 0.0
    w = curve["count"] / curve["count"].sum()
    return float((w * (curve["mean_pred"] - curve["frac_pos"]).abs()).sum())
