"""Router engine: prompt text -> features -> P(needs strong), with overhead timing."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from router.features.embeddings import get_embedder
from router.features.featurizer import combine_features
from router.features.handcrafted import handcrafted_features
from router.features.text import routing_view
from router.models.artifact import LinearRouterArtifact

MAX_CONTEXT_MESSAGES = 4


def _content_text(content: Any) -> str:
    """Extract text from an OpenAI message content (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def routing_text(messages: list[dict]) -> str:
    """Text the router scores: system/user message contents joined with newlines.

    Only the last ``MAX_CONTEXT_MESSAGES`` messages are considered when the conversation is longer.
    """
    recent = list(messages)[-MAX_CONTEXT_MESSAGES:]
    texts = [
        _content_text(m.get("content"))
        for m in recent
        if isinstance(m, dict) and m.get("role") in ("system", "user")
    ]
    return "\n".join(t for t in texts if t)


class RouterEngine:
    def __init__(self, artifact_stem: str, embedder_name: str | None = None):
        self.artifact = LinearRouterArtifact.load(artifact_stem)
        self.meta: dict = self.artifact.meta
        self.use_handcrafted = bool(self.meta.get("use_handcrafted", True))
        name = embedder_name or self.meta["embedder"]
        self.embedder = get_embedder(name, max_seq_length=int(self.meta.get("max_seq_length", 256)))
        thresholds = self.meta.get("thresholds") or {}
        self.default_threshold: float = float(thresholds.get("rel95", 0.5))

    def predict(self, prompt: str) -> tuple[float, float]:
        """Return (p_strong, overhead_ms). Featurisation mirrors training."""
        t0 = time.perf_counter()
        emb = np.asarray(self.embedder.encode([routing_view(prompt)]), dtype=np.float32)
        hand = handcrafted_features([prompt]) if self.use_handcrafted else None
        x = combine_features(emb, hand)
        p = float(np.asarray(self.artifact.predict_proba(x)).reshape(-1)[0])
        return p, (time.perf_counter() - t0) * 1000.0

    def warmup(self) -> None:
        self.predict("warm-up prompt: what is 2 + 2?")
