"""Prompt embedders (sentence-transformers wrapper + offline hashing) and an on-disk cache."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np


class Embedder:
    """Lazy sentence-transformers wrapper; L2-normalised float32 output."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_seq_length: int = 256, device: str = "cpu"):
        self.name = model_name
        self.max_seq_length = max_seq_length
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.name, device=self.device)
            model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    @property
    def dim(self) -> int:
        m = self._load()
        d = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return int(d())

    def encode(self, prompts: Sequence[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        model = self._load()
        emb = model.encode(list(prompts), batch_size=batch_size, show_progress_bar=show_progress,
                           convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class HashingEmbedder:
    """Char n-gram hashing embedder; no downloads."""
    name = "hashing"
    max_seq_length = 0

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.dim = dim
        self._vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=dim,
                                      alternate_sign=True, norm="l2")

    def encode(self, prompts: Sequence[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        return self._vec.transform(list(prompts)).toarray().astype(np.float32)


def get_embedder(name: str, max_seq_length: int = 256):
    if name == "hashing":
        return HashingEmbedder()
    return Embedder(name, max_seq_length)


def cached_embeddings(prompts: Sequence[str], embedder, cache_dir) -> np.ndarray:
    prompts = list(prompts)
    h = hashlib.sha1()
    h.update(str(embedder.name).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(getattr(embedder, "max_seq_length", 0)).encode("utf-8"))
    for p in prompts:
        h.update(b"\x00")
        h.update(p.encode("utf-8", errors="replace"))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(embedder.name)).strip("-")[-40:] or "emb"
    cache_dir = Path(cache_dir)
    path = cache_dir / f"emb_{slug}_{h.hexdigest()[:16]}.npy"
    if path.exists():
        return np.load(path, allow_pickle=False)
    emb = embedder.encode(prompts, batch_size=64, show_progress=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
