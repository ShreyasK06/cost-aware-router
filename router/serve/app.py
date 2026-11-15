"""OpenAI-compatible routing proxy (FastAPI).

Run with::

    uvicorn router.serve.app:get_app --factory --host 0.0.0.0 --port 8000

Configuration comes from configs/default.yaml and environment variables (see .env.example).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from router.serve.backends import BackendError, build_backends
from router.serve.engine import RouterEngine, routing_text
from router.serve.logging_db import RequestLog, hash_prompt
from router.serve.policy import BudgetPolicy, ThresholdPolicy
from router.serve.settings import Settings

_PASSTHROUGH_SKIP = {"model", "messages", "stream"}


def _error(status: int, message: str, etype: str = "invalid_request_error", code: str | None = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "code": code}},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = RouterEngine(settings.artifact_path, settings.embedder)
    engine.warmup()
    default_threshold = settings.threshold if settings.threshold is not None else engine.default_threshold
    if settings.policy == "budget":
        policy: Any = BudgetPolicy(settings.budget_frac, init_threshold=default_threshold)
    else:
        policy = ThresholdPolicy(default_threshold)
    cheap_backend, strong_backend = build_backends(settings)
    req_log = RequestLog(settings.db_path)

    app = FastAPI(title="cost-aware-router", version="0.1.0")
    app.state.settings = settings
    app.state.engine = engine
    app.state.policy = policy
    app.state.log = req_log
    app.state.backends = (cheap_backend, strong_backend)

    def forced_tier(model: Any) -> str | None:
        if model in ("cheap", settings.cheap_model):
            return "cheap"
        if model in ("strong", settings.strong_model):
            return "strong"
        return None  # "auto", omitted, or unknown -> route

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error(400, "Request body must be valid JSON")
        if not isinstance(body, dict):
            return _error(400, "Request body must be a JSON object")
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(isinstance(m, dict) and "role" in m for m in messages)
        ):
            return _error(400, "'messages' must be a non-empty list of {role, content} objects")

        override: float | None = None
        hdr = request.headers.get("x-router-threshold")
        if hdr is not None:
            try:
                override = float(hdr)
            except ValueError:
                return _error(400, "x-router-threshold must be a number")

        text = routing_text(messages)
        p_strong, overhead_ms = await run_in_threadpool(engine.predict, text)

        forced = forced_tier(body.get("model"))
        if forced is not None:
            strong = forced == "strong"
            policy_name = "forced"
            threshold = default_threshold
        else:
            strong = policy.decide(p_strong, override)
            policy_name = settings.policy
            threshold = override if override is not None else (
                policy.threshold if hasattr(policy, "threshold") else default_threshold
            )
        route = "strong" if strong else "cheap"
        model = settings.strong_model if strong else settings.cheap_model
        backend = strong_backend if strong else cheap_backend
        params = {k: v for k, v in body.items() if k not in _PASSTHROUGH_SKIP}

        row = dict(
            query_hash=hash_prompt(text), route=route, model=model, p_strong=p_strong,
            threshold=threshold, policy=policy_name, overhead_ms=overhead_ms,
        )
        t0 = time.perf_counter()
        try:
            result = await backend.complete(model, messages, **params)
        except BackendError as exc:
            req_log.log(
                **row, latency_ms=(time.perf_counter() - t0) * 1000.0, prompt_tokens=0,
                completion_tokens=0, cost_usd=0.0, strong_cost_usd=0.0, status="error",
            )
            return _error(502, f"Upstream error ({exc.status_code}): {exc.message}",
                          "upstream_error", str(exc.status_code))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        cost = settings.cost(model, result.prompt_tokens, result.completion_tokens)
        strong_cost = settings.cost(settings.strong_model, result.prompt_tokens, result.completion_tokens)
        req_log.log(
            **row, latency_ms=latency_ms, prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens, cost_usd=cost, strong_cost_usd=strong_cost,
            status="ok",
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        router_info = {
            "route": route, "p_strong": p_strong, "threshold": threshold,
            "policy": policy_name, "overhead_ms": overhead_ms,
        }
        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
        headers = {"x-router-route": route, "x-router-p-strong": f"{p_strong:.6f}"}

        if body.get("stream"):
            # Note: the upstream call above is NOT streamed token by token. The full completion
            # is obtained first and then emitted as a single content chunk (plus a stop chunk),
            # which is enough for OpenAI-SDK clients that set stream=True.
            def sse():
                def chunk(delta: dict, finish: str | None, extra: dict | None = None) -> str:
                    payload = {
                        "id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": result.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    if extra:
                        payload.update(extra)
                    return f"data: {json.dumps(payload)}\n\n"

                yield chunk({"role": "assistant", "content": result.text}, None)
                yield chunk({}, "stop", {"router": router_info})
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

        return JSONResponse(
            content={
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": result.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }],
                "usage": usage,
                "router": router_info,
            },
            headers=headers,
        )

    @app.get("/v1/models")
    async def list_models():
        now = int(time.time())
        ids = ["auto", settings.cheap_model, settings.strong_model]
        return {
            "object": "list",
            "data": [{"id": i, "object": "model", "created": now, "owned_by": "cost-aware-router"} for i in ids],
        }

    @app.get("/health")
    async def health():
        thr = policy.threshold if settings.policy == "budget" else default_threshold
        return {
            "status": "ok", "backend": settings.backend, "policy": settings.policy,
            "threshold": thr, "artifact": settings.artifact_path,
        }

    @app.get("/stats")
    async def stats():
        return await run_in_threadpool(req_log.summary)

    return app


def get_app() -> FastAPI:
    """Factory for uvicorn: ``uvicorn router.serve.app:get_app --factory``."""
    return create_app()
