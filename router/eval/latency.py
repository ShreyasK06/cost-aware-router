"""Router overhead measurement."""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np


def measure_overhead(fn: Callable[[str], Any], prompts: Sequence[str], n: int = 200,
                     warmup: int = 10) -> dict:
    prompts = list(prompts)
    m = len(prompts)
    for i in range(warmup):
        fn(prompts[i % m])
    times = np.empty(n, dtype=float)
    for i in range(n):
        p = prompts[i % m]
        t0 = time.perf_counter()
        fn(p)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "n": int(n),
    }
