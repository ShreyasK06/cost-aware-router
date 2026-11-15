"""Combine embeddings and handcrafted features into one matrix."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from router.features.handcrafted import handcrafted_features


def combine_features(emb: np.ndarray | None, hand: np.ndarray | None) -> np.ndarray:
    parts = [a for a in (emb, hand) if a is not None]
    if not parts:
        raise ValueError("combine_features needs at least one of emb / hand")
    return np.hstack(parts).astype(np.float32, copy=False)


class Featurizer:
    def __init__(self, embedder, use_handcrafted: bool):
        self.embedder = embedder
        self.use_handcrafted = use_handcrafted

    def transform(self, prompts: Sequence[str]) -> np.ndarray:
        prompts = list(prompts)
        emb = self.embedder.encode(prompts) if self.embedder is not None else None
        hand = handcrafted_features(prompts) if self.use_handcrafted else None
        return combine_features(emb, hand)
