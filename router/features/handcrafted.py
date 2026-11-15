"""Cheap, deterministic handcrafted prompt features (pure python / re / numpy)."""
from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

FEATURE_NAMES: list[str] = [
    "log_chars", "log_words", "avg_word_len", "log_lines", "digit_ratio", "log_numbers",
    "math_ops", "math_keyword", "code_markers", "has_code_fence", "multiple_choice",
    "question_marks", "q_why", "q_how", "q_wh", "constraint_words", "step_by_step",
    "non_ascii_ratio", "upper_ratio", "punct_ratio", "unique_word_ratio",
]

_WORD = re.compile(r"\w+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_MATH_OPS = re.compile(r"[+\-*/^=<>%×÷√∑∫]|\\frac|\\sqrt")
_MATH_KW = re.compile(
    r"\b(prove|proof|derive|calculate|compute|solve|integral|derivative|equation|probability|"
    r"theorem|simplify|evaluate|matrix|polynomial|sum of|how many|remainder|factor)\b")
_CODE = re.compile(r"(def |function|class |return\b|import |#include|=>|\bnull\b|\bvoid\b|;\s*$|[{}])",
                   re.M)
_MC = re.compile(r"(^\s*\(?[A-D][\.\)]\s)|\boptions?\b|\bchoices?\b|\banswer:", re.M | re.I)
_WHY = re.compile(r"\bwhy\b")
_HOW = re.compile(r"\bhow\b")
_WH = re.compile(r"\b(what|which|who|when|where)\b")
_CONSTRAINT = re.compile(r"\b(must|at least|at most|exactly|only|without|each|every|ensure|constraint)\b")
_STEP = re.compile(r"step[- ]by[- ]step|\bexplain\b|\bjustify\b|\breasoning\b")
_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _one(text: str) -> list[float]:
    n = len(text)
    low = text.lower()
    words = _WORD.findall(low)
    nw = len(words)
    nl = text.count("\n") + 1
    nums = len(_NUM.findall(text))
    d = max(n, 1)
    return [
        math.log1p(n),
        math.log1p(nw),
        (sum(map(len, words)) / nw) if nw else 0.0,
        math.log1p(nl),
        sum(c.isdigit() for c in text) / d,
        math.log1p(nums),
        float(len(_MATH_OPS.findall(text))),
        float(bool(_MATH_KW.search(low))),
        float(len(_CODE.findall(text))),
        float("```" in text),
        float(bool(_MC.search(text))),
        float(text.count("?")),
        float(bool(_WHY.search(low))),
        float(bool(_HOW.search(low))),
        float(bool(_WH.search(low))),
        float(len(_CONSTRAINT.findall(low))),
        float(bool(_STEP.search(low))),
        sum(ord(c) > 127 for c in text) / d,
        sum(c.isupper() for c in text) / d,
        sum(c in _PUNCT for c in text) / d,
        (len(set(words)) / nw) if nw else 0.0,
    ]


def handcrafted_features(prompts: Sequence[str]) -> np.ndarray:
    """Return (n, len(FEATURE_NAMES)) float32 matrix."""
    out = np.zeros((len(prompts), len(FEATURE_NAMES)), dtype=np.float64)
    for i, p in enumerate(prompts):
        out[i] = _one(p if isinstance(p, str) else str(p))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
