"""Serving-layer tests: offline, hashing embedder + mock backend + httpx.MockTransport."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import router.serve.app as app_mod
from router.features.embeddings import get_embedder
from router.features.featurizer import combine_features
from router.features.handcrafted import handcrafted_features
from router.models.artifact import LinearRouterArtifact
from router.serve.app import create_app
from router.serve.backends import (
    AnthropicBackend,
    BackendError,
    MockBackend,
    OpenAICompatBackend,
    build_backends,
)
from router.serve.engine import RouterEngine, routing_text
from router.serve.logging_db import RequestLog
from router.serve.settings import Settings, SettingsError

CHEAP, STRONG = "cheap-m", "strong-m"
PRICES = {
    CHEAP: {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
    STRONG: {"input_per_mtok": 10.0, "output_per_mtok": 30.0},
}
SECRET_PROMPT = "zebra-quokka-unique-prompt-9137 what is the capital of France?"
SAMPLE = [
    "What is 2+2?", "Hi there", "Write a python function that sorts a list and explain complexity.",
    "A train leaves at 3pm going 60 mph; another leaves at 4pm going 80 mph. When do they meet?",
    "Which of the following is a noble gas? A) Neon B) Iron C) Salt D) Water",
    "Prove that the square root of 2 is irrational, step by step.",
]


@pytest.fixture(scope="module")
def artifact_stem(tmp_path_factory) -> str:
    emb = get_embedder("hashing").encode(SAMPLE)
    x = combine_features(emb, handcrafted_features(SAMPLE))
    rng = np.random.default_rng(0)
    art = LinearRouterArtifact(
        mean=x.mean(0), scale=x.std(0) + 1e-3, coef=rng.normal(0, 0.05, x.shape[1]),
        intercept=np.array([0.0]),
        meta={
            "kind": "linear", "embedder": "hashing", "max_seq_length": 256,
            "use_handcrafted": True, "thresholds": {"rel95": 0.5, "rel90": 0.6},
            "frontier": [
                {"frac_strong": 0.0, "rel_quality": 0.7, "rel_cost": 0.1},
                {"frac_strong": 1.0, "rel_quality": 1.0, "rel_cost": 1.0},
            ],
            "cheap_model": CHEAP, "strong_model": STRONG,
        },
    )
    stem = tmp_path_factory.mktemp("artifact") / "router_lr"
    art.save(stem)
    return str(stem)


def make_settings(stem: str, tmp_path: Path, **kw) -> Settings:
    base = dict(
        backend="mock", cheap_model=CHEAP, strong_model=STRONG, artifact_path=stem,
        db_path=str(tmp_path / "req.sqlite"), embedder="hashing", prices=PRICES,
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def client(artifact_stem, tmp_path):
    return TestClient(create_app(make_settings(artifact_stem, tmp_path)))


def chat(client, content=SECRET_PROMPT, headers=None, **body):
    body["messages"] = [{"role": "user", "content": content}]
    return client.post("/v1/chat/completions", json=body, headers=headers)


# ---------------------------------------------------------------- engine / helpers
def test_routing_text_string_and_parts():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}},
                                     {"type": "text", "text": "b"}]},
        {"role": "user", "content": "last"},
    ]
    assert routing_text(msgs) == "sys\na\nb\nlast"


def test_routing_text_keeps_last_four_messages():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    assert routing_text(msgs) == "m2\nm3\nm4\nm5"


def test_engine_predict_and_meta(artifact_stem):
    eng = RouterEngine(artifact_stem, "hashing")
    eng.warmup()
    p, ms = eng.predict("What is 2+2?")
    assert 0.0 <= p <= 1.0 and ms >= 0.0
    assert eng.default_threshold == 0.5
    assert eng.predict("What is 2+2?")[0] == p  # deterministic


# ---------------------------------------------------------------- HTTP surface
def test_completion_schema(client):
    r = chat(client)
    assert r.status_code == 200
    d = r.json()
    assert d["id"].startswith("chatcmpl-") and d["object"] == "chat.completion"
    assert isinstance(d["created"], int)
    ch = d["choices"][0]
    assert ch["message"]["role"] == "assistant" and ch["finish_reason"] == "stop"
    assert d["usage"]["total_tokens"] == d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"]
    info = d["router"]
    assert set(info) == {"route", "p_strong", "threshold", "policy", "overhead_ms"}
    assert info["route"] in ("cheap", "strong")
    assert d["model"] == (STRONG if info["route"] == "strong" else CHEAP)
    assert r.headers["x-router-route"] == info["route"]
    assert float(r.headers["x-router-p-strong"]) == pytest.approx(info["p_strong"], abs=1e-5)
    assert ch["message"]["content"].startswith(f"[mock:{d['model']}]")


@pytest.mark.parametrize("model,route,actual", [
    ("cheap", "cheap", CHEAP), ("strong", "strong", STRONG),
    (CHEAP, "cheap", CHEAP), (STRONG, "strong", STRONG),
])
def test_forced_tier(client, model, route, actual):
    d = chat(client, model=model).json()
    assert d["router"]["route"] == route and d["router"]["policy"] == "forced"
    assert d["model"] == actual
    assert 0.0 <= d["router"]["p_strong"] <= 1.0


def test_auto_and_unknown_model_route(client):
    for m in ("auto", "gpt-unknown-9"):
        assert chat(client, model=m).json()["router"]["policy"] == "threshold"


def test_threshold_header_overrides(client):
    hi = chat(client, headers={"x-router-threshold": "0.0"})
    lo = chat(client, headers={"x-router-threshold": "1.01"})
    assert hi.json()["router"]["route"] == "strong"
    assert lo.json()["router"]["route"] == "cheap"
    assert hi.json()["router"]["threshold"] == 0.0
    assert chat(client, headers={"x-router-threshold": "abc"}).status_code == 400


def test_streaming_sse(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [e for e in r.text.split("\n\n") if e.strip()]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[len("data: "):]) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["content"].startswith("[mock:")
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("body", [{"messages": []}, {}, {"messages": "hi"}, {"messages": [1]}])
def test_bad_request_400(client, body):
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    err = r.json()["error"]
    assert {"message", "type", "code"} <= set(err)


def test_models_health_stats(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["auto", CHEAP, STRONG]
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["backend"] == "mock" and h["policy"] == "threshold"
    assert h["threshold"] == 0.5 and "artifact" in h
    assert client.get("/stats").json()["n_requests"] == 0
    chat(client)
    chat(client, model="strong")
    s = client.get("/stats").json()
    assert s["n_requests"] == 2 and s["n_strong"] >= 1
    assert s["total_tokens"] > 0 and s["overhead_ms_p50"] is not None


def test_log_row_has_hash_and_no_raw_prompt(artifact_stem, tmp_path):
    c = TestClient(create_app(make_settings(artifact_stem, tmp_path)))
    chat(c)
    rows = c.app.state.log.rows()
    assert len(rows) == 1 and len(rows[0]["query_hash"]) == 64 and rows[0]["status"] == "ok"
    c.app.state.log.close()
    files = list(tmp_path.glob("req.sqlite*"))
    assert files
    for f in files:
        assert b"zebra-quokka" not in f.read_bytes()


def test_cost_math(artifact_stem, tmp_path):
    c = TestClient(create_app(make_settings(artifact_stem, tmp_path)))
    d = chat(c, model="cheap").json()
    row = c.app.state.log.rows()[0]
    u = d["usage"]
    assert row["cost_usd"] == pytest.approx((u["prompt_tokens"] * 1.0 + u["completion_tokens"] * 2.0) / 1e6)
    assert row["strong_cost_usd"] == pytest.approx((u["prompt_tokens"] * 10.0 + u["completion_tokens"] * 30.0) / 1e6)
    assert row["strong_cost_usd"] >= row["cost_usd"]
    s = c.get("/stats").json()
    assert 0.0 < s["cost_saving_frac"] < 1.0


def test_budget_policy_end_to_end(artifact_stem, tmp_path):
    c = TestClient(create_app(make_settings(
        artifact_stem, tmp_path, policy="budget", budget_frac=0.3)))
    rng = np.random.default_rng(5)
    words = ["alpha", "beta", "sum", "python", "why", "cat", "integral", "list", "quantum", "def", "7", "tax"]
    n_strong = 0
    for i in range(300):
        text = " ".join(rng.choice(words, size=int(rng.integers(3, 30)))) + f" #{i}"
        n_strong += chat(c, content=text).json()["router"]["route"] == "strong"
    assert abs(n_strong / 300 - 0.3) < 0.1


# ---------------------------------------------------------------- backends
def run(coro):
    return asyncio.run(coro)


def test_mock_backend_deterministic():
    msgs = [{"role": "user", "content": "x" * 200}]
    a = run(MockBackend().complete("m1", msgs))
    b = run(MockBackend().complete("m1", msgs))
    assert a == b and a.text == "[mock:m1] " + "x" * 80
    assert a.prompt_tokens == 50 and a.completion_tokens == -(-len(a.text) // 4)


def test_openai_backend_request_and_usage():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hey"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    be = OpenAICompatBackend("https://api.example.com/v1/", "sk-test", transport=httpx.MockTransport(handler))
    msgs = [{"role": "user", "content": "hi"}]
    res = run(be.complete("m", msgs, temperature=0.2, max_tokens=5, bogus=1, top_p=None))
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {"model": "m", "messages": msgs, "temperature": 0.2, "max_tokens": 5}
    assert (res.text, res.prompt_tokens, res.completion_tokens) == ("hey", 7, 3)


def test_anthropic_backend_translation():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"] = str(req.url), req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "foo"}, {"type": "tool_use"}, {"type": "text", "text": "bar"}],
            "usage": {"input_tokens": 11, "output_tokens": 4}})

    be = AnthropicBackend("ak-test", "https://api.anthropic.com", transport=httpx.MockTransport(handler))
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "no markdown"},
    ]
    res = run(be.complete("claude-x", msgs))
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "ak-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    body = seen["body"]
    assert body["system"] == "be brief\n\nno markdown"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 1024 and body["model"] == "claude-x"
    assert (res.text, res.prompt_tokens, res.completion_tokens) == ("foobar", 11, 4)


def test_backend_error_on_500_and_app_maps_to_502(artifact_stem, tmp_path, monkeypatch):
    transport = httpx.MockTransport(lambda r: httpx.Response(500, json={"error": {"message": "boom"}}))
    for be in (OpenAICompatBackend("http://x/v1", "k", transport), AnthropicBackend("k", "http://x", transport)):
        with pytest.raises(BackendError) as ei:
            run(be.complete("m", [{"role": "user", "content": "hi"}]))
        assert ei.value.status_code == 500 and "boom" in ei.value.message

    bad = OpenAICompatBackend("http://x/v1", "k", transport)
    monkeypatch.setattr(app_mod, "build_backends", lambda s: (bad, bad))
    c = TestClient(create_app(make_settings(artifact_stem, tmp_path)))
    r = chat(c)
    assert r.status_code == 502
    assert r.json()["error"]["message"] and r.json()["error"]["type"]
    assert c.app.state.log.rows()[0]["status"] == "error"


def test_build_backends_by_provider():
    s = Settings(backend="anthropic", cheap_api_key="a", strong_api_key="b",
                 cheap_base_url="https://api.anthropic.com", strong_base_url="https://api.anthropic.com")
    assert all(isinstance(b, AnthropicBackend) for b in build_backends(s))
    s2 = Settings(backend="openai", cheap_api_key="a", strong_api_key="b")
    assert all(isinstance(b, OpenAICompatBackend) for b in build_backends(s2))
    assert all(isinstance(b, MockBackend) for b in build_backends(Settings()))


# ---------------------------------------------------------------- settings
_ENV = ["ROUTER_BACKEND", "CHEAP_MODEL", "STRONG_MODEL", "CHEAP_BASE_URL", "STRONG_BASE_URL",
        "CHEAP_API_KEY", "STRONG_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ROUTER_ARTIFACT",
        "ROUTER_THRESHOLD", "ROUTER_POLICY", "ROUTER_BUDGET_FRAC", "ROUTER_DB", "ROUTER_EMBEDDER",
        "CHEAP_PRICE_IN", "CHEAP_PRICE_OUT", "STRONG_PRICE_IN", "STRONG_PRICE_OUT"]


@pytest.fixture
def clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_settings_defaults_from_config(clean_env):
    s = Settings.from_env()
    assert s.backend == "mock" and s.policy == "threshold" and s.threshold is None
    assert s.cheap_model in s.prices and s.strong_model in s.prices
    assert s.cheap_base_url == "https://api.openai.com/v1"


def test_settings_missing_key_raises_clear_error(clean_env):
    clean_env.setenv("ROUTER_BACKEND", "openai")
    with pytest.raises(SettingsError, match="CHEAP_API_KEY"):
        Settings.from_env()
    clean_env.setenv("ROUTER_BACKEND", "anthropic")
    with pytest.raises(SettingsError, match="ANTHROPIC_API_KEY"):
        Settings.from_env()


def test_settings_env_overrides(clean_env):
    clean_env.setenv("ROUTER_BACKEND", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "k")
    clean_env.setenv("STRONG_API_KEY", "own")
    clean_env.setenv("ROUTER_THRESHOLD", "0.7")
    clean_env.setenv("ROUTER_POLICY", "budget")
    clean_env.setenv("CHEAP_MODEL", "tiny")
    clean_env.setenv("CHEAP_PRICE_IN", "0.5")
    s = Settings.from_env()
    assert s.cheap_api_key == "k" and s.strong_api_key == "own"
    assert s.strong_base_url == "https://api.anthropic.com"
    assert s.threshold == 0.7 and s.policy == "budget"
    assert s.prices["tiny"]["input_per_mtok"] == 0.5 and s.prices["tiny"]["output_per_mtok"] == 0.0


def test_request_log_summary_empty(tmp_path):
    log = RequestLog(tmp_path / "sub" / "x.sqlite")
    s = log.summary()
    assert s["n_requests"] == 0 and s["frac_strong"] == 0.0 and s["overhead_ms_p50"] is None


def test_dashboard_compiles():
    import py_compile
    py_compile.compile(str(Path(__file__).resolve().parents[1] / "router" / "dashboards" / "app.py"), doraise=True)
