"""Serving settings, resolved from configs/default.yaml and environment variables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from router.config import REPO_ROOT, load_config

log = logging.getLogger(__name__)

OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
BACKENDS = ("mock", "openai", "anthropic")
POLICIES = ("threshold", "budget")


class SettingsError(ValueError):
    """Raised for invalid or incomplete serving configuration."""


@dataclass
class Settings:
    backend: str = "mock"
    cheap_model: str = "gpt-3.5-turbo-1106"
    strong_model: str = "gpt-4-1106-preview"
    cheap_base_url: str = OPENAI_DEFAULT_URL
    strong_base_url: str = OPENAI_DEFAULT_URL
    cheap_api_key: str | None = None
    strong_api_key: str | None = None
    artifact_path: str = "artifacts/router_lr"
    threshold: float | None = None  # None -> artifact thresholds["rel95"]
    policy: str = "threshold"
    budget_frac: float = 0.3
    db_path: str = "data/requests.sqlite"
    embedder: str | None = None  # optional override of the artifact's embedder name
    prices: dict[str, dict[str, float]] = field(default_factory=dict)

    def price(self, model: str) -> tuple[float, float]:
        """(input, output) USD per 1M tokens; unknown models cost 0."""
        p = self.prices.get(model)
        if p is None:
            log.warning("No price configured for model %r; assuming 0 USD", model)
            return 0.0, 0.0
        return float(p.get("input_per_mtok", 0.0)), float(p.get("output_per_mtok", 0.0))

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = self.price(model)
        return (prompt_tokens * pin + completion_tokens * pout) / 1e6

    @classmethod
    def from_env(cls, cfg: dict[str, Any] | None = None) -> "Settings":
        cfg = cfg if cfg is not None else load_config()
        serve = cfg.get("serve", {})
        # Empty values (e.g. docker-compose pass-through of unset vars) count as unset.
        env = {k: v for k, v in os.environ.items() if v.strip() != ""}

        backend = env.get("ROUTER_BACKEND", serve.get("backend", "mock")).strip().lower()
        if backend not in BACKENDS:
            raise SettingsError(f"ROUTER_BACKEND must be one of {BACKENDS}, got {backend!r}")
        policy = env.get("ROUTER_POLICY", serve.get("policy", "threshold")).strip().lower()
        if policy not in POLICIES:
            raise SettingsError(f"ROUTER_POLICY must be one of {POLICIES}, got {policy!r}")

        default_url = ANTHROPIC_DEFAULT_URL if backend == "anthropic" else OPENAI_DEFAULT_URL
        fallback_key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(backend)
        fallback_key = env.get(fallback_key_var) if fallback_key_var else None
        cheap_key = env.get("CHEAP_API_KEY") or fallback_key
        strong_key = env.get("STRONG_API_KEY") or fallback_key
        if backend != "mock":
            for tier, key in (("CHEAP", cheap_key), ("STRONG", strong_key)):
                if not key:
                    raise SettingsError(
                        f"ROUTER_BACKEND={backend} requires an API key: set {tier}_API_KEY "
                        f"(or {fallback_key_var}) for the {tier.lower()} model"
                    )

        cheap_model = env.get("CHEAP_MODEL", serve.get("cheap_model", "gpt-3.5-turbo-1106"))
        strong_model = env.get("STRONG_MODEL", serve.get("strong_model", "gpt-4-1106-preview"))

        prices = {
            m: dict(v) for m, v in (cfg.get("prices", {}).get("models", {}) or {}).items()
        }
        for tier, model in (("CHEAP", cheap_model), ("STRONG", strong_model)):
            pin, pout = env.get(f"{tier}_PRICE_IN"), env.get(f"{tier}_PRICE_OUT")
            if pin is not None or pout is not None:
                cur = prices.setdefault(model, {"input_per_mtok": 0.0, "output_per_mtok": 0.0})
                if pin is not None:
                    cur["input_per_mtok"] = _to_float(pin, f"{tier}_PRICE_IN")
                if pout is not None:
                    cur["output_per_mtok"] = _to_float(pout, f"{tier}_PRICE_OUT")
        for model in (cheap_model, strong_model):
            if model not in prices:
                log.warning("No price for model %r (set *_PRICE_IN/*_PRICE_OUT); using 0", model)
                prices[model] = {"input_per_mtok": 0.0, "output_per_mtok": 0.0}

        thr = env.get("ROUTER_THRESHOLD")
        threshold = _to_float(thr, "ROUTER_THRESHOLD") if thr not in (None, "") else None
        budget = env.get("ROUTER_BUDGET_FRAC")
        budget_frac = (
            _to_float(budget, "ROUTER_BUDGET_FRAC") if budget not in (None, "")
            else float(serve.get("budget_frac", 0.3))
        )

        artifact = env.get("ROUTER_ARTIFACT", serve.get("artifact", "artifacts/router_lr"))
        db_path = env.get("ROUTER_DB", serve.get("db_path", "data/requests.sqlite"))
        return cls(
            backend=backend,
            cheap_model=cheap_model,
            strong_model=strong_model,
            cheap_base_url=env.get("CHEAP_BASE_URL", default_url),
            strong_base_url=env.get("STRONG_BASE_URL", default_url),
            cheap_api_key=cheap_key,
            strong_api_key=strong_key,
            artifact_path=_resolve(artifact),
            threshold=threshold,
            policy=policy,
            budget_frac=budget_frac,
            db_path=_resolve(db_path),
            embedder=env.get("ROUTER_EMBEDDER") or None,
            prices=prices,
        )


def _to_float(value: str, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{name} must be a number, got {value!r}") from None


def _resolve(path: str) -> str:
    """Relative paths are anchored at the repo root (cwd-independent)."""
    return path if os.path.isabs(path) else str(REPO_ROOT / path)
