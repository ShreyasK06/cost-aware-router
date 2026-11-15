"""Routing policies: map a P(needs strong) score to a strong/cheap decision. Thread-safe."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

MIN_WARM = 20  # scores needed before BudgetPolicy trusts its own quantile


class ThresholdPolicy:
    """Fixed cut-off: route to the strong model iff p_strong >= threshold."""

    name = "threshold"

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def decide(self, p_strong: float, threshold: float | None = None) -> bool:
        thr = self.threshold if threshold is None else float(threshold)
        return bool(p_strong >= thr)


class BudgetPolicy:
    """Online policy that keeps the long-run strong share near ``target_frac``.

    Every score is pushed into a sliding window; the threshold is the (1 - target_frac)
    quantile of that window, so about ``target_frac`` of recent traffic scores at or above it.
    Until ``MIN_WARM`` scores have been seen, ``init_threshold`` is used.
    """

    name = "budget"

    def __init__(self, target_frac: float, window: int = 500, init_threshold: float = 0.5):
        self.target_frac = float(min(max(target_frac, 0.0), 1.0))
        self.window = int(window)
        self.init_threshold = float(init_threshold)
        self._scores: deque[float] = deque(maxlen=self.window)
        self._threshold = self.init_threshold
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float:
        with self._lock:
            return self._threshold

    def decide(self, p_strong: float, threshold: float | None = None) -> bool:
        """``threshold`` (per-request override) bypasses the quantile but still feeds the window."""
        p = float(p_strong)
        with self._lock:
            self._scores.append(p)
            n = len(self._scores)
            if n >= MIN_WARM:
                self._threshold = float(
                    np.quantile(np.fromiter(self._scores, dtype=float, count=n), 1.0 - self.target_frac)
                )
            else:
                self._threshold = self.init_threshold
            thr = self._threshold if threshold is None else float(threshold)
            if threshold is None and n >= MIN_WARM:
                if self.target_frac <= 0.0:
                    return False
                if self.target_frac >= 1.0:
                    return True
            return bool(p >= thr)
