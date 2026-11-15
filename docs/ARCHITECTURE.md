# Architecture and interface contracts

Scope: a binary router (cheap vs strong LLM), trained and evaluated offline on RouterBench,
served as an OpenAI-compatible proxy. Python 3.11, CPU only. Everything must run with no API keys.

## 1. Module boundaries

```
configs/default.yaml          model pair, prices (dated), epsilon, seeds, embedder, eval settings
router/config.py              load_config(path) -> dict
router/data/                  download (HF -> parquet), pair table, labels, splits, synthetic data
router/features/              handcrafted features, embedders (MiniLM + offline hashing), caching
router/models/                routers on feature matrices, pure-numpy serving artifact, fine-tune
router/eval/                  frontier metrics, baselines, bootstrap CIs, calibration, plots,
                              latency, robustness perturbations
router/serve/                 FastAPI app, settings, policies, backends (mock/openai/anthropic),
                              SQLite request log, router engine
router/dashboards/app.py      Streamlit dashboard over the SQLite log
scripts/reproduce.py          one command: data -> features -> routers -> eval -> figure + results.json
scripts/demo.py               < 2 min mock-backend demo of the proxy
tests/                        pytest; synthetic data + hashing embedder only (no network, no keys)
artifacts/                    small committed router artifact (npz + json) used by serve/demo
assets/                       committed figures; results/ committed metrics JSON + markdown tables
```

Dependency direction: `data`, `features` -> `models` -> `eval` -> `scripts`; `serve` depends on
`features` + `models.artifact` only (never on pandas-heavy training code or `eval`).

## 2. Data schemas

### Pair table (`pd.DataFrame`, one row per prompt)
| column | type | meaning |
|---|---|---|
| sample_id | str | RouterBench id |
| prompt | str | plain prompt text (RouterBench message list joined with `\n`) |
| eval_name | str | source eval (86 values) |
| task_family | str | one of `mmlu, hellaswag, gsm8k, arc, winogrande, mbpp, chinese, mtbench, misc` |
| cheap_score, strong_score | float in [0,1] | per-prompt quality of each model |
| cheap_cost, strong_cost | float USD | per-prompt cost recorded by RouterBench |
| gap | float | `strong_score - cheap_score` |
| needs_strong | int {0,1} | `gap > epsilon` |
| both_fail | bool | `cheap_score == 0 and strong_score == 0` |
| split | str | `train` / `val` / `test` (in-distribution split, stratified by task_family) |

### Splits
* **In-distribution (ID):** 60/20/20 random split stratified by `task_family`, identical prompts kept
  in the same split. Seeded.
* **OOD (leave-one-family-out, LOFO):** for each family `f`: train on `train` rows with family != f,
  calibrate thresholds on `val` rows with family != f, test on `test` rows with family == f.
  OOD predictions for all test rows are pooled, so ID and OOD numbers are on the *same test rows*.

### Router artifact (serving)
`<name>.npz` with arrays `mean, scale, coef, intercept` and `<name>.json` with
`{"kind": "linear", "embedder": str, "max_seq_length": int, "use_handcrafted": bool,
"n_features": int, "thresholds": {"rel95": float, "rel90": float},
"frontier": [{"frac_strong": float, "rel_quality": float, "rel_cost": float}, ...],
"trained_on": str, "cheap_model": str, "strong_model": str, "created": "YYYY-MM-DD"}`.
No pickles.

### Request log (SQLite table `requests`)
`id INTEGER PK, ts REAL (unix), query_hash TEXT (sha256 of prompt, no raw text stored), route TEXT
('cheap'|'strong'), model TEXT, p_strong REAL, threshold REAL, policy TEXT, overhead_ms REAL,
latency_ms REAL, prompt_tokens INT, completion_tokens INT, cost_usd REAL, strong_cost_usd REAL
(counterfactual always-strong cost), status TEXT`.

## 3. Function signatures

### router/data  (owner: lead)
```python
build_pair_table(raw_parquet, cheap_model, strong_model) -> DataFrame   # columns up to strong_cost
add_labels(df, epsilon=0.0) -> DataFrame                                # + gap, needs_strong, both_fail
make_id_split(df, seed=0, fracs=(0.6, 0.2, 0.2)) -> pd.Series           # values train/val/test
lofo_folds(df, families=None) -> Iterator[tuple[str, ndarray, ndarray, ndarray]]  # (family, train_idx, val_idx, test_idx), positional
load_dataset(cfg) -> DataFrame                                          # all of the above
make_synthetic_pair_table(n=600, seed=0) -> DataFrame                   # same schema, for tests
```

### router/features  (owner: features+models agent)
```python
# handcrafted.py
FEATURE_NAMES: list[str]
handcrafted_features(prompts: Sequence[str]) -> np.ndarray   # (n, len(FEATURE_NAMES)) float32; pure python/numpy/re
# embeddings.py
class Embedder:            # sentence-transformers wrapper
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256, device="cpu")
    name: str; dim: int
    def encode(self, prompts: Sequence[str], batch_size=64, show_progress=False) -> np.ndarray  # (n, dim) float32, L2-normalised
class HashingEmbedder:     # same interface; char n-gram hashing -> 256 dims; no downloads; name == "hashing"
get_embedder(name: str, max_seq_length=256) -> Embedder | HashingEmbedder    # "hashing" -> HashingEmbedder
cached_embeddings(prompts, embedder, cache_dir) -> np.ndarray   # npy cache keyed by sha1(model name, max_seq_length, prompts)
# featurizer.py
combine_features(emb: np.ndarray | None, hand: np.ndarray | None) -> np.ndarray   # hstack, emb first
class Featurizer:
    def __init__(self, embedder, use_handcrafted: bool)
    def transform(self, prompts: Sequence[str]) -> np.ndarray
```

### router/models  (owner: features+models agent)
All routers work on precomputed feature matrices and expose the same interface:
```python
class BaseRouter(ABC):
    name: str
    is_probabilistic: bool                 # True if score() is P(needs_strong) in [0,1]
    def fit(self, X: np.ndarray, gap: np.ndarray) -> "BaseRouter"   # gap = strong_score - cheap_score
    def score(self, X: np.ndarray) -> np.ndarray                    # (n,), higher = send to strong
class LogisticRouter(BaseRouter)        # __init__(epsilon=0.0, C=1.0, class_weight=None); StandardScaler + LogisticRegression
    def export_linear(self) -> dict     # {"mean","scale","coef","intercept"} numpy arrays
class MLPRouter(BaseRouter)             # __init__(epsilon=0.0, hidden=(256, 64), max_iter=60, seed=0); sklearn MLPClassifier, early stopping
class GapRegressionRouter(BaseRouter)   # __init__(alpha=10.0); StandardScaler + Ridge on gap; score = predicted gap
class KNNRouter(BaseRouter)             # __init__(k=40); mean gap of k cosine nearest neighbours (RouterBench-style)
# artifact.py  (numpy + json only; imported by serve)
class LinearRouterArtifact:
    def __init__(self, mean, scale, coef, intercept, meta: dict)
    def predict_proba(self, X: np.ndarray) -> np.ndarray   # sigmoid(((X-mean)/scale) @ coef + intercept)
    def save(self, path_stem) -> None                      # writes <stem>.npz and <stem>.json
    @classmethod
    def load(cls, path_stem) -> "LinearRouterArtifact"
    @classmethod
    def from_router(cls, router: LogisticRouter, meta: dict) -> "LinearRouterArtifact"
# finetune.py (torch + transformers; optional, CPU-sized)
finetune_encoder(train_prompts, train_labels, val_prompts, val_labels, model_name=..., epochs=1,
                 max_len=128, batch_size=32, lr=5e-5, seed=0) -> FineTunedRouter
class FineTunedRouter:  def score_prompts(self, prompts, batch_size=64) -> np.ndarray   # P(needs_strong)
```

### router/eval  (owner: eval agent)
Routing convention: for a strong-fraction `f`, the `round(f*n)` prompts with the highest score go to
the strong model; ties are broken by a fixed seeded permutation. Quality = mean per-prompt score of the
chosen model; cost = mean per-prompt cost of the chosen model.
```python
# metrics.py
@dataclass(frozen=True)
class EvalData: cheap_score, strong_score, cheap_cost, strong_cost: np.ndarray
    @classmethod from_frame(df) ; def subset(idx) ; def __len__
frontier(scores, data, n_points=101, seed=0) -> DataFrame[frac_strong, cost, quality, rel_quality, rel_cost, pgr]
apgr(scores, data) -> float                  # trapezoid of PGR over frac_strong in [0,1]; random = 0.5
auc_cost_quality(scores, data) -> float      # trapezoid of quality over cost normalised to [0,1] between cheap and strong
cpt(scores, data, target_pgr=0.5) -> float   # min frac_strong with PGR >= target (RouteLLM call-performance threshold)
cost_at_quality(scores, data, rel_quality=0.95) -> dict   # test-sweep (optimistic): cost, cost_saving, frac_strong, quality
quality_at_budget(scores, data, budget_frac=0.5) -> dict  # best quality with cost <= budget_frac * strong-only cost
calibrate_threshold(val_scores, val_data, rel_quality) -> float   # score threshold; route strong iff score >= threshold
operating_point(route_strong: np.ndarray[bool], data) -> dict
    # quality, cost, frac_strong, rel_quality (vs always-strong), cost_saving (1 - cost/strong_cost),
    # random_quality (random routing at the SAME cost), lift_vs_random
# baselines.py
always_cheap(data) / always_strong(data) -> dict(quality, cost)
oracle_scores(data) -> np.ndarray            # gap, ties favour cheap; upper bound
random_scores(n, seed) -> np.ndarray
random_quality_at_cost(cost, data) -> float  # analytic expectation of random mixing at matched cost
heuristic_scores(prompts) -> np.ndarray      # length / keyword router, no training
# bootstrap.py
bootstrap_ci(stat_fn: Callable[[np.ndarray], float], n: int, n_boot=1000, seed=0, alpha=0.05) -> dict(point, lo, hi)
    # stat_fn receives an index array (resampled with replacement); point uses arange(n)
# calibration.py
expected_calibration_error(probs, labels, n_bins=10) -> float
reliability_curve(probs, labels, n_bins=10) -> DataFrame[bin_lo, bin_hi, mean_pred, frac_pos, count]
# robustness.py
perturb_typos(prompts, rate=0.05, seed=0) / perturb_case(prompts, seed=0) / perturb_filler(prompts, seed=0) -> list[str]
# latency.py
measure_overhead(fn: Callable[[str], Any], prompts, n=200, warmup=10) -> dict(p50_ms, p95_ms, mean_ms, n)
# plots.py (matplotlib Agg)
plot_frontier(curves: dict[str, DataFrame], data: EvalData, out_path, title, x="cost", bands=None) -> None
plot_reliability(curve_df, ece, out_path, title) -> None
plot_per_task(table: DataFrame, out_path) -> None
```

### router/serve  (owner: serving agent)
```python
# settings.py
@dataclass
class Settings: backend ("mock"|"openai"|"anthropic"), cheap_model, strong_model, cheap_base_url, strong_base_url,
    cheap_api_key, strong_api_key, artifact_path, threshold (float|None -> artifact rel95), policy ("threshold"|"budget"),
    budget_frac, db_path, prices (dict model -> {input_per_mtok, output_per_mtok})
    @classmethod from_env(cfg: dict | None = None) -> Settings
# policy.py
class ThresholdPolicy(threshold);            decide(p_strong) -> bool
class BudgetPolicy(target_frac, window=500, init_threshold=0.5);  decide(p_strong) -> bool; .threshold
    # online: threshold = (1 - target_frac) quantile of the last `window` scores -> strong share tracks target
# engine.py
class RouterEngine:  __init__(artifact_stem);  predict(prompt: str) -> tuple[float, float]   # (p_strong, overhead_ms)
# backends.py
@dataclass class ChatResult: text, prompt_tokens, completion_tokens, model
class MockBackend / OpenAICompatBackend(base_url, api_key) / AnthropicBackend(api_key, base_url=...):
    async def complete(self, model: str, messages: list[dict], **params) -> ChatResult
# logging_db.py
class RequestLog(db_path):  log(**row) -> None;  summary() -> dict;  rows(limit=None) -> list[dict]
# app.py
create_app(settings: Settings | None = None) -> FastAPI
```
HTTP surface: `POST /v1/chat/completions` (OpenAI schema; `model` = `auto` routes, `cheap`/`strong`
force a tier; optional `stream`), `GET /v1/models`, `GET /health`, `GET /stats`. Responses carry a
`router` object (`route, p_strong, threshold, policy, overhead_ms`) and `x-router-route` header.

## 4. Config format (`configs/default.yaml`)
See the file; keys: `data`, `pair`, `labels`, `split`, `features`, `eval`, `serve`, `prices`.
Environment variables override `serve.*` at runtime (`ROUTER_BACKEND`, `CHEAP_MODEL`, ...).

## 5. Evaluation protocol (what the headline number means)
1. Fit routers on ID `train`.
2. **Frontier metrics** (APGR, AUC, CPT, cost@quality) are computed by sweeping the strong fraction on
   ID `test`; they describe the whole curve and involve no threshold choice.
3. **Deployed operating point (headline):** the threshold is calibrated on `val` to hit a target
   relative quality (95% / 90% of always-strong), then frozen and applied to `test`. Reported:
   realised quality, cost saving, share of traffic to cheap, and the lift over random routing at
   the same realised cost, each with a 95% bootstrap CI over test prompts.
4. The same is repeated OOD (LOFO): model and threshold never see the held-out family.
