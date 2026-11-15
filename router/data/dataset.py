"""Pair table, labels and splits for a (cheap, strong) model pair on RouterBench."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

PAIR_COLUMNS = [
    "sample_id", "prompt", "eval_name", "task_family",
    "cheap_score", "strong_score", "cheap_cost", "strong_cost",
]

_FAMILY_EXACT = {
    "hellaswag": "hellaswag",
    "grade-school-math": "gsm8k",
    "arc-challenge": "arc",
    "winogrande": "winogrande",
    "mbpp": "mbpp",
}


def task_family(eval_name: str) -> str:
    """Map RouterBench's 86 eval names onto 9 task families used for OOD splits."""
    if eval_name in _FAMILY_EXACT:
        return _FAMILY_EXACT[eval_name]
    low = eval_name.lower()
    if low.startswith("mmlu"):
        return "mmlu"
    if low.startswith("chinese"):
        return "chinese"
    if low.startswith("mtbench"):
        return "mtbench"
    return "misc"


def _prompt_text(raw: str) -> str:
    """RouterBench stores the prompt as a stringified list of message strings."""
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parts = ast.literal_eval(s)
            if isinstance(parts, (list, tuple)):
                return "\n".join(str(p) for p in parts)
        except (ValueError, SyntaxError):
            pass
    return raw


def build_pair_table(raw_parquet: str | Path, cheap_model: str, strong_model: str) -> pd.DataFrame:
    raw = pd.read_parquet(raw_parquet)
    for col in (cheap_model, strong_model, f"{cheap_model}|total_cost", f"{strong_model}|total_cost"):
        if col not in raw.columns:
            raise KeyError(f"column {col!r} not in RouterBench table")
    df = pd.DataFrame({
        "sample_id": raw["sample_id"].astype(str),
        "prompt": raw["prompt"].astype(str).map(_prompt_text),
        "eval_name": raw["eval_name"].astype(str),
        "cheap_score": raw[cheap_model].astype(float),
        "strong_score": raw[strong_model].astype(float),
        "cheap_cost": raw[f"{cheap_model}|total_cost"].astype(float),
        "strong_cost": raw[f"{strong_model}|total_cost"].astype(float),
    })
    df["task_family"] = df["eval_name"].map(task_family)
    df = df.dropna(subset=["cheap_score", "strong_score", "cheap_cost", "strong_cost"])
    return df[PAIR_COLUMNS].reset_index(drop=True)


def add_labels(df: pd.DataFrame, epsilon: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out["gap"] = out["strong_score"] - out["cheap_score"]
    out["needs_strong"] = (out["gap"] > epsilon).astype(int)
    out["both_fail"] = (out["cheap_score"] == 0) & (out["strong_score"] == 0)
    return out


def make_id_split(df: pd.DataFrame, seed: int = 0,
                  fracs: Sequence[float] = (0.6, 0.2, 0.2)) -> pd.Series:
    """Random train/val/test split stratified by task_family; identical prompts stay together."""
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must sum to 1")
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=df.index, dtype=object)
    for _, fam in df.groupby("task_family", sort=True):
        # assign per unique prompt so the same prompt text never straddles splits
        keys = sorted(fam["prompt"].unique())
        n = len(keys)
        order = rng.permutation(n)
        n_train = int(round(fracs[0] * n))
        n_val = int(round(fracs[1] * n))
        assign = np.empty(n, dtype=object)
        assign[order[:n_train]] = "train"
        assign[order[n_train:n_train + n_val]] = "val"
        assign[order[n_train + n_val:]] = "test"
        split.loc[fam.index] = fam["prompt"].map(dict(zip(keys, assign))).to_numpy()
    return split


def lofo_folds(df: pd.DataFrame, families: Sequence[str] | None = None
               ) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Leave-one-family-out folds built on top of the ID split.

    Yields (family, train_idx, val_idx, test_idx) as positional indices:
    train/val come from the other families' train/val rows, test = held-out family's test rows.
    """
    fam = df["task_family"].to_numpy()
    split = df["split"].to_numpy()
    for f in (families or sorted(set(fam))):
        tr = np.flatnonzero((fam != f) & (split == "train"))
        va = np.flatnonzero((fam != f) & (split == "val"))
        te = np.flatnonzero((fam == f) & (split == "test"))
        if len(te):
            yield f, tr, va, te


def load_dataset(cfg: dict, root: str | Path = ".") -> pd.DataFrame:
    """Download (if needed) and return the labelled, split pair table for the configured pair."""
    from router.data.download import download_routerbench

    root = Path(root)
    raw = download_routerbench(root / cfg["data"]["raw_dir"])
    df = build_pair_table(raw, cfg["pair"]["cheap"], cfg["pair"]["strong"])
    df = add_labels(df, cfg["labels"]["epsilon"])
    df["split"] = make_id_split(df, cfg["split"]["seed"], cfg["split"]["fracs"])
    return df
