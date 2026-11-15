"""Baseline routers and reference curves."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from router.eval.metrics import EvalData


def always_cheap(data: EvalData) -> dict:
    return {"quality": float(data.cheap_score.mean()), "cost": float(data.cheap_cost.mean())}


def always_strong(data: EvalData) -> dict:
    return {"quality": float(data.strong_score.mean()), "cost": float(data.strong_cost.mean())}


def oracle_scores(data: EvalData) -> np.ndarray:
    """gap = strong - cheap; ties (gap 0) go to strong after all positive gaps."""
    return data.strong_score - data.cheap_score


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def random_quality_at_cost(cost: float, data: EvalData) -> float:
    """Expected quality of random mixing that matches `cost`."""
    c0, c1 = float(data.cheap_cost.mean()), float(data.strong_cost.mean())
    q0, q1 = float(data.cheap_score.mean()), float(data.strong_score.mean())
    if c1 == c0:
        return q0
    p = float(np.clip((cost - c0) / (c1 - c0), 0.0, 1.0))
    return q0 + p * (q1 - q0)


_MATH = re.compile(r"\d|[+\-*/=^<>%]|\b(prove|derive|calculate|compute|solve|equation|integral|sum|probability)\b", re.I)
_CODE = re.compile(r"```|\bdef\b|\bfunction\b|\bclass\b|\breturn\b|\bimport\b|\bpython\b", re.I)
_MULTI = re.compile(
    r"\b(must|at least|at most|exactly|step by step|explain why|constraint|each|all of|compare|analy[sz]e)\b", re.I
)
_REASON = re.compile(r"\b(why|how)\b", re.I)


def heuristic_scores(prompts: Sequence[str]) -> np.ndarray:
    """Training-free difficulty score (higher = harder). Deterministic."""
    out = np.zeros(len(prompts), dtype=float)
    for i, p in enumerate(prompts):
        p = str(p)
        words = len(p.split())
        s = np.log1p(words) / np.log(1000.0)
        s += 0.15 * min(len(_MATH.findall(p)), 5) / 5
        s += 0.3 * (1.0 if _CODE.search(p) else 0.0)
        s += 0.1 * min(len(_MULTI.findall(p)), 3)
        s += 0.1 * (1.0 if _REASON.search(p) else 0.0)
        out[i] = s
    return out
