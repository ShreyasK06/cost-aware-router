"""Prompt perturbations for robustness checks (deterministic given a seed)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

_LETTERS = "abcdefghijklmnopqrstuvwxyz"

_PREFIXES = [
    "Hi there, could you please help me with the following? ",
    "Hello! I hope you are doing well. Quick question for you: ",
    "Sorry to bother you, but would you mind helping with this? ",
]


def _typo(text: str, rate: float, rng: np.random.Generator) -> str:
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and rng.random() < rate:
            op = int(rng.integers(0, 3))
            if op == 0 and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.extend([chars[i + 1], ch])
                i += 2
                continue
            if op == 1:
                i += 1
                continue
            sub = _LETTERS[int(rng.integers(0, 26))]
            out.append(sub.upper() if ch.isupper() else sub)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def perturb_typos(prompts: Sequence[str], rate: float = 0.05, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [_typo(str(p), rate, rng) for p in prompts]


def perturb_case(prompts: Sequence[str], seed: int = 0) -> list[str]:
    return [str(p).lower() for p in prompts]


def perturb_filler(prompts: Sequence[str], seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [_PREFIXES[int(rng.integers(0, len(_PREFIXES)))] + str(p) + " Thanks in advance!"
            for p in prompts]
