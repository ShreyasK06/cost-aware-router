"""Text view used for embedding.

Benchmark prompts often start with a long shared instruction / few-shot prefix and put the actual
question at the end (e.g. GSM8K in RouterBench has an ~890-character shared prefix). A head-truncated
encoder would then see almost only the prefix. We therefore embed a short head plus a long tail.
The same view is applied at training time and in the serving engine.
"""
from __future__ import annotations

from typing import Sequence

HEAD_CHARS = 160
TAIL_CHARS = 840


def routing_view(prompt: str, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> str:
    if len(prompt) <= head + tail:
        return prompt
    return prompt[:head] + "\n...\n" + prompt[-tail:]


def routing_views(prompts: Sequence[str], head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> list[str]:
    return [routing_view(p, head, tail) for p in prompts]
