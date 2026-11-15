"""Small synthetic pair table for tests and offline smoke runs.

NOT a benchmark: scores are simulated so that prompt text carries a learnable difficulty signal.
Never report numbers computed on this data as results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from router.data.dataset import add_labels, make_id_split

_EASY = ["What is the capital of {x}?", "Spell the word {x}.", "Say hello to {x}.",
         "Name one colour of {x}.", "Is {x} a noun?"]
_HARD = ["Prove that the sequence defined by {x} converges and derive its limit step by step.",
         "Write a function to compute the longest common subsequence of {x} with unit tests.",
         "Solve the integral of {x} and explain every step of the derivation carefully.",
         "Given the constraints {x}, design an algorithm and analyse its complexity.",
         "Derive the equation relating {x} and justify each algebraic transformation."]
_FAMILIES = ["mmlu", "gsm8k", "hellaswag", "mbpp"]
_WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa", "lambda", "zeta"]


def make_synthetic_pair_table(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        hard = rng.random() < 0.45
        tpl = (_HARD if hard else _EASY)[rng.integers(5)]
        x = " ".join(rng.choice(_WORDS, size=rng.integers(1, 4))) + f" {i}"
        fam = _FAMILIES[rng.integers(len(_FAMILIES))]
        p_cheap = 0.25 if hard else 0.9
        p_strong = 0.85 if hard else 0.95
        rows.append({
            "sample_id": f"syn.{i}",
            "prompt": tpl.format(x=x),
            "eval_name": fam,
            "task_family": fam,
            "cheap_score": float(rng.random() < p_cheap),
            "strong_score": float(rng.random() < p_strong),
            "cheap_cost": 1e-4 * (1 + rng.random()),
            "strong_cost": 1.5e-3 * (1 + rng.random()),
        })
    df = add_labels(pd.DataFrame(rows), 0.0)
    df["split"] = make_id_split(df, seed)
    return df
