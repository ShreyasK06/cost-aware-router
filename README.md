# Cost-aware LLM router

![Cost vs quality frontier, in-distribution and out-of-distribution](assets/frontier.png)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 102 passed](https://img.shields.io/badge/tests-102%20passed-brightgreen.svg)](tests)

A small classifier decides, per query and before any generation, whether to call a cheap LLM or a
strong one. This repo measures the cost vs quality frontier of several such routers against
baselines on [RouterBench](https://huggingface.co/datasets/withmartian/routerbench) (GPT-3.5-turbo
vs GPT-4-turbo, 36,497 prompts), with bootstrap confidence intervals and an out-of-distribution
test, and ships the best one as an OpenAI-compatible proxy with request logging and a dashboard.
The main finding is a negative one as much as a positive one: the in-distribution saving is large,
but most of it comes from predicting which prompts are *expensive*, and it does not survive unseen
task types.

The test badge is the result of running `python -m pytest -q` on the committed code (102 passed;
synthetic data and a hashing embedder, no network). The figure shows ID and OOD frontiers; both are
on the same 7,298 test prompts.

## Headline results

Pair: `gpt-3.5-turbo-1106` (cheap) vs `gpt-4-1106-preview` (strong). Thresholds are set on
validation data to hit a quality target, frozen, then applied to the test prompts. Brackets are 95%
bootstrap CIs over test prompts (1000 resamples). "Lift" is quality points above random routing at
the same realised cost. Numbers are copied from [results/tables.md](results/tables.md).

**Cost-aware logistic router (the headline router), deployed operating points**

| Target | Split | Quality retained vs always-strong | Cost saving vs always-strong | Traffic to cheap | Lift vs random (points) |
|---|---|---|---|---|---|
| 95% of strong quality | ID test | 95.6% [95.0, 96.3] | 67.8% [66.7, 68.8] | 37.6% [36.5, 38.7] | +8.23 [+7.58, +8.97] |
| 95% of strong quality | OOD (leave-one-family-out) | 92.0% [91.1, 92.8] | 21.6% [20.4, 22.9] | 25.7% [24.8, 26.7] | -2.59 [-3.13, -2.06] |
| 90% of strong quality | ID test | 90.9% [90.0, 91.8] | 77.1% [76.3, 77.9] | 56.5% [55.3, 57.6] | +6.13 [+5.47, +6.86] |
| 90% of strong quality | OOD (leave-one-family-out) | 87.7% [86.6, 88.7] | 42.1% [40.4, 43.6] | 47.8% [46.6, 49.0] | -2.40 [-2.98, -1.85] |

**What each part contributes (95% target; ID test vs OOD on the same test prompts)**

| Router | ID: quality | ID: cost saving | ID: lift (points) | OOD: quality | OOD: cost saving | OOD: lift (points) |
|---|---|---|---|---|---|---|
| Plain classifier: LR (embedding + handcrafted) | 95.1% [94.3, 95.8] | 22.4% [21.1, 23.7] | -0.01 [-0.56, +0.48] | 87.5% [86.4, 88.6] | 24.3% [23.1, 25.6] | -5.58 [-6.25, -4.94] |
| Cost only: predicted cost, no quality model | 95.9% [95.2, 96.5] | 66.5% [65.4, 67.5] | +8.19 [+7.52, +8.95] | 91.2% [90.3, 92.1] | 40.1% [38.7, 41.7] | +0.00 [-0.57, +0.60] |
| Cost-aware LR (headline) | 95.6% [95.0, 96.3] | 67.8% [66.7, 68.8] | +8.23 [+7.58, +8.97] | 92.0% [91.1, 92.8] | 21.6% [20.4, 22.9] | -2.59 [-3.13, -2.06] |

On the test split always-cheap scores 0.624 and always-strong 0.783; the strong model costs 13.6x
more per query. Every other router, the oracles, per-family breakdowns, calibration, robustness and
the other model pairs are in the [full generated results](#full-generated-results) and in
[results/tables.md](results/tables.md); [docs/RESULTS.md](docs/RESULTS.md) explains how to read them.

## What this does and does not show

**It shows**

- On RouterBench, in distribution, a router can keep about 95% of GPT-4 quality at about two thirds
  lower cost (ID test, threshold frozen on validation, CIs above).
- A baseline with no quality model at all, which only predicts the strong model's extra cost per
  prompt, gets a 66.5% saving at the same target; the headline router gets 67.8%.
  **The in-distribution saving is cost prediction, not difficulty prediction.** One task family
  (GSM8K) carries roughly half of the GPT-4 bill while GPT-4 is only about 5 points better on it,
  so everything from that family goes to the cheap model.
- **A plain "does this need the strong model?" classifier is no better than random at matched
  cost** (lift -0.01 points, CI [-0.56, +0.48], ID test). It looks good on call-count metrics (APGR
  0.637 vs 0.5 for random) because the prompts it sends to the strong model are the expensive ones.
- **Out of distribution, the headline router loses to random routing at the same cost** (-2.59
  points, CI [-3.13, -2.06]) and the plain classifier loses by 5.6 points. The cost-only baseline is
  exactly at random (+0.00). Calibration degrades too (ECE 0.015 ID vs 0.110 OOD).
- Within a single task family the plain classifier's ranking of prompts is modest in distribution
  (within-family APGR 0.609, random is 0.5) and gone when the family is held out (0.486). The router
  is mostly identifying the dataset, not the difficulty.

**It does not show**

- That the saving transfers to your traffic. There are no live LLM calls, no own holdout set and no
  LLM judge (see [Not validated](#not-validated)).
- That routing on difficulty works for these models. On held-out families the data say it does not.
- That the numbers hold for current models or prices. The pair and prices are from 2024.

## Architecture

```mermaid
flowchart LR
    D["RouterBench 0-shot<br/>per-prompt score and cost<br/>for every model"] --> F["Features<br/>MiniLM embedding<br/>+ 21 handcrafted"]
    F --> R["Router<br/>one linear head:<br/>logit P(needs strong)<br/>minus alpha x predicted log extra cost"]
    R --> P["Policy<br/>fixed threshold<br/>or online budget"]
    P --> X["Dispatcher"]
    X --> C["Cheap LLM"]
    X --> S["Strong LLM"]
    X --> L[("Logger<br/>SQLite")]
    L --> B["Streamlit dashboard"]
```

Training and evaluation use the data and features offline (label: `needs_strong = strong_score -
cheap_score > 0`). Serving runs features, router, policy and dispatcher per request through
`POST /v1/chat/completions`. Module contracts and schemas: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate                 # Windows; Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m pytest -q                    # 102 tests, synthetic data, no network
python scripts/demo.py                 # routing demo, mock LLM backend, no keys
python scripts/reproduce.py            # regenerates every number, table and figure
```

`scripts/reproduce.py` downloads RouterBench (about 100 MB, no token), embeds 36,497 prompts on CPU
once (cached under `data/cache/`, roughly 10 to 40 minutes on a laptop), then takes about 10
minutes. `--quick` uses fewer resamples and skips the MLP and robustness steps. It rewrites
`results/`, `assets/` and the generated block at the bottom of this README.

`scripts/demo.py` uses the committed router artifact. The first run downloads the MiniLM model
(about 90 MB). Real output from a run on this repo, abridged (mock backend). The router was trained
on benchmark-templated prompts, so its choices on these hand-written prompts are not sensible: it
sends "What is 17 + 25?" to the strong model and a proof request to the cheap one.

```text
policy=threshold  threshold=0.223  backend=mock

prompt                                                   p_strong  route  overhead ms
--------------------------------------------------------------------------------------
What is the capital of France?                              0.144  cheap        17.12
What is 17 + 25?                                            0.564  strong       18.17
Write a Python function that returns the n-th Fibon...      0.028  cheap        17.60
Which of the following is a noble gas? A) Neon B) I...      0.350  strong       29.87
Prove that the sum of two even integers is even, th...      0.054  cheap        16.79
... (24 prompts in the full run)

requests:                24
routed to cheap:         66.7%  (16 of 24)
mock cost:               $0.006481
always-strong cost:      $0.020330
cost saving:             68.1%
routing overhead p50/p95: 17.65 / 28.94 ms
```

The 68.1% is on 24 hand-written prompts, priced from the config table with estimated token counts.
It is a plumbing check, not a result.

## Serving

```bash
# mock backend (default): no keys, canned answers, real routing decisions
uvicorn router.serve.app:get_app --factory --port 8000
curl -s localhost:8000/v1/chat/completions -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 17 + 25?"}]}'
streamlit run router/dashboards/app.py          # reads the SQLite log (ROUTER_DB)

docker compose up --build   # written, NOT built or run; see Not validated
```

`model: "auto"` routes; `"cheap"` / `"strong"` force a tier. Every response carries a `router`
object (`route`, `p_strong`, threshold, policy, overhead) and an `x-router-route` header. The
`x-router-threshold` header overrides the threshold per request, and `ROUTER_POLICY=budget` tracks a
target share of strong calls online. `p_strong` is the cost-adjusted routing score in (0, 1), not a
calibrated probability. Only a SHA-256 hash of the prompt is logged, never its text.

![Streamlit dashboard over a 24-request mock-backend demo log](docs/screenshots/dashboard.png)

_The dashboard over the 24-request demo log above (cropped to the main panel). "Estimated quality"
is an offline estimate from the benchmark frontier, not measured on live traffic._

### Pointing the proxy at real providers

Copy `.env.example` to `.env`, or export the variables. None of this has been run against a live
API (see [Not validated](#not-validated)).

```bash
# OpenAI-compatible endpoints (OpenAI, Groq, Together, vLLM, Ollama, Gemini's compat endpoint)
export ROUTER_BACKEND=openai
export CHEAP_MODEL=<cheap model id>  STRONG_MODEL=<strong model id>
export CHEAP_BASE_URL=<url>  STRONG_BASE_URL=<url>   # optional; default https://api.openai.com/v1
export OPENAI_API_KEY=...                            # or CHEAP_API_KEY / STRONG_API_KEY per tier
export CHEAP_PRICE_IN=... CHEAP_PRICE_OUT=... STRONG_PRICE_IN=... STRONG_PRICE_OUT=...  # USD per 1M tokens
uvicorn router.serve.app:get_app --factory --port 8000

# Anthropic
export ROUTER_BACKEND=anthropic ANTHROPIC_API_KEY=... CHEAP_MODEL=... STRONG_MODEL=...
```

Other settings: `ROUTER_POLICY` (`threshold` or `budget`), `ROUTER_THRESHOLD` (default: the
artifact's 95% threshold), `ROUTER_BUDGET_FRAC`, `ROUTER_ARTIFACT`, `ROUTER_DB`. A non-mock backend
without a key fails at startup with a message naming the missing variable. Models without a
configured price are costed at 0, so set the prices. The shipped artifact was trained on the
GPT-3.5/GPT-4 pair; pointing it at a different pair keeps the routing scores but not their meaning.

## Repo layout

```
router/data         download, pair table, labels, splits, synthetic data
router/features     handcrafted features, MiniLM / hashing embedders, cache
router/models       routers (LR, ridge, kNN, MLP, cost-aware), numpy serving artifact, fine-tune
router/eval         frontier metrics, baselines, bootstrap, calibration, robustness, latency, plots
router/serve        FastAPI app, policies, backends (mock/openai/anthropic), SQLite log, engine
router/dashboards   Streamlit dashboard
configs/            default.yaml: model pair, dated prices, seeds, eval settings
scripts/            reproduce.py (data -> results), demo.py
tests/              pytest, synthetic data only
artifacts/          small committed router artifact (npz + json)
assets/  results/   figures; results.json and tables.md
docs/               ARCHITECTURE.md, PLAN.md, RESULTS.md, screenshots/
```

## Not validated

- **No live LLM calls were made.** The OpenAI-compatible and Anthropic backends are tested only
  against mocked HTTP transports; the demo and end-to-end tests use the mock backend.
- **No own holdout set.** The planned self-collected prompts answered by a current model pair were
  not generated, so there is no evidence about current models.
- **No LLM judge.** All quality numbers are RouterBench's own automatic scores.
- **No RouteLLM or LMSYS preference data**, and RouteLLM's published routers were not reproduced.
  The kNN router is a reimplementation in the style of the RouterBench paper.
- **The fine-tuned encoder** (`router/models/finetune.py`, `--finetune`) is implemented but was not
  run (CPU-only machine), so no number is reported for it.
- **`docker compose up` was written but not built or run.**
- **RouterBench licensing.** Its dataset card states no licence. It is downloaded at run time and
  not redistributed here; check its terms before reusing the data.
- Serving-layer dollar figures use the dated price table in `configs/default.yaml`; benchmark
  numbers use the per-prompt costs recorded in RouterBench in early 2024.

## Limitations

- **The headline saving is in-distribution and benchmark-specific.** It depends on one task family
  being expensive for the strong model and gaining little from it. The OOD rows are the better guide
  for new task types.
- **Free-form prompts are out of distribution too.** The router saw benchmark-templated prompts;
  retrain on your own traffic before using it for anything real.
- **Old model pair** (GPT-3.5-turbo-1106 vs GPT-4-1106-preview, 2024 prices).
- **Benchmark, not user traffic.** Mostly multiple-choice and short-answer tasks; little open-ended
  chat or code. Contamination and saturation of these evals are likely.
- **Pooled OOD scores come from nine fold models**, so the pooled OOD frontier curve is an
  approximation; the deployed operating points use per-fold thresholds and are the cleaner number.
- **Robustness tests are surface-level** (typos, lowercasing, polite filler). There is no paraphrase
  or adversarial set.
- **Routing overhead** (p95 108.4 ms in the results; about 29 ms in the demo above) was measured
  in-process on one laptop CPU under different conditions each time. It is not a server benchmark.

## Roadmap

- Collect an own holdout: current model pair, real prompts, judged outputs (including a
  position-swapped LLM judge for open-ended chat).
- Find a difficulty signal that survives held-out families, and make within-family evaluation a
  first-class gate.
- Live-validate the OpenAI-compatible and Anthropic backends; build and run the compose stack.
- Run the encoder fine-tune on a GPU; compare against RouteLLM routers.
- Three-tier routing, cascading, drift monitoring.

## License

Code: MIT (see [LICENSE](LICENSE)). RouterBench is not redistributed here (see
[Not validated](#not-validated)). Reference: Hu et al., 2024, "RouterBench: A Benchmark for
Multi-LLM Routing System".

## Full generated results

<details>
<summary>All tables written by <code>scripts/reproduce.py</code> (do not edit by hand)</summary>

<!-- RESULTS:START -->
_Generated by `python scripts/reproduce.py` on 2026-11-15; do not edit by hand._

Benchmark: RouterBench 0-shot (withmartian/routerbench), pair `gpt-3.5-turbo-1106` (cheap) vs `gpt-4-1106-preview` (strong); 36,497 prompts, split 21,900/7,299/7,298 train/val/test. On the test split always-cheap scores 0.624 and always-strong 0.783 (cheap = 79.7% of strong quality) and the strong model costs 13.6x more per query. Brackets are 95% bootstrap CIs over test prompts (1000 resamples). Router: Cost-aware LR (embedding + handcrafted). ID and OOD rows use the same 7,298 test prompts.

### Headline: deployed operating points

| Target (set on validation) | Split | Quality retained vs always-strong | Cost saving vs always-strong | Traffic to cheap model | Quality lift vs random routing at the same cost (points) |
|---|---|---|---|---|---|
| 95% of strong quality | ID test | 95.6% [95.0, 96.3] | 67.8% [66.7, 68.8] | 37.6% [36.5, 38.7] | +8.23 [+7.58, +8.97] |
| 95% of strong quality | OOD (leave-one-family-out) | 92.0% [91.1, 92.8] | 21.6% [20.4, 22.9] | 25.7% [24.8, 26.7] | -2.59 [-3.13, -2.06] |
| 90% of strong quality | ID test | 90.9% [90.0, 91.8] | 77.1% [76.3, 77.9] | 56.5% [55.3, 57.6] | +6.13 [+5.47, +6.86] |
| 90% of strong quality | OOD (leave-one-family-out) | 87.7% [86.6, 88.7] | 42.1% [40.4, 43.6] | 47.8% [46.6, 49.0] | -2.40 [-2.98, -1.85] |

### Routers and baselines: frontier metrics, in-distribution test split

| Router | Split | APGR (random = 0.5) | AUC cost-quality | CPT(50%) | CPT(80%) | Cost saving @95% quality (test sweep) | Cost saving @90% quality (test sweep) | Quality retained at 50% of strong-only cost |
|---|---|---|---|---|---|---|---|---|
| Random | ID test | 0.498 [0.479, 0.518] | 0.703 [0.695, 0.711] | 49.5% [46.0, 52.9] | 79.6% [76.7, 83.0] | 23.1% [20.1, 25.7] | 46.3% [41.6, 50.9] | 89.2% [88.2, 90.2] |
| Length/keyword heuristic | ID test | 0.520 [0.503, 0.537] | 0.669 [0.661, 0.678] | 46.8% [44.7, 49.2] | 66.2% [63.8, 71.4] | 13.3% [11.4, 15.1] | 21.1% [18.8, 23.3] | 82.3% [81.0, 83.4] |
| LR (handcrafted only) | ID test | 0.624 [0.605, 0.643] | 0.712 [0.704, 0.721] | 34.4% [29.2, 38.9] | 65.4% [60.9, 70.1] | 18.9% [16.9, 20.4] | 48.8% [37.9, 64.5] | 90.0% [89.0, 90.9] |
| LR (embedding) | ID test | 0.634 [0.614, 0.653] | 0.719 [0.711, 0.727] | 28.1% [25.5, 31.9] | 67.3% [64.0, 70.2] | 21.6% [17.7, 30.4] | 64.1% [57.4, 70.5] | 92.0% [90.9, 92.9] |
| LR (embedding + handcrafted) | ID test | 0.637 [0.618, 0.656] | 0.720 [0.712, 0.728] | 28.3% [25.5, 31.3] | 66.9% [62.7, 70.4] | 23.2% [18.5, 29.7] | 65.9% [59.5, 71.2] | 92.2% [91.3, 93.1] |
| LR (embedding + handcrafted), both-fail rows dropped from training | ID test | 0.645 [0.627, 0.665] | 0.726 [0.718, 0.734] | 27.6% [25.4, 30.0] | 63.4% [60.2, 68.5] | 32.2% [26.3, 41.0] | 71.6% [66.9, 75.2] | 93.5% [92.6, 94.3] |
| Gap regression, ridge (embedding + handcrafted) | ID test | 0.673 [0.655, 0.691] | 0.746 [0.738, 0.753] | 25.3% [23.5, 27.7] | 57.2% [52.8, 60.2] | 66.0% [62.3, 69.1] | 77.6% [75.9, 79.3] | 97.1% [96.6, 97.7] |
| kNN, k=40 (embedding) | ID test | 0.667 [0.649, 0.686] | 0.745 [0.737, 0.752] | 26.7% [24.3, 29.5] | 58.5% [53.7, 62.1] | 64.7% [59.1, 66.5] | 76.9% [74.9, 78.8] | 96.8% [96.2, 97.4] |
| MLP (embedding + handcrafted) | ID test | 0.628 [0.608, 0.647] | 0.717 [0.709, 0.725] | 30.0% [26.8, 34.2] | 67.1% [63.4, 71.4] | 20.9% [18.3, 26.3] | 61.4% [51.8, 68.8] | 91.2% [90.2, 92.2] |
| Cost only: predicted cost, no quality model (embedding + handcrafted) | ID test | 0.516 [0.499, 0.533] | 0.750 [0.742, 0.758] | 51.5% [49.2, 53.1] | 67.0% [65.2, 69.0] | 68.6% [66.9, 70.0] | 77.1% [75.6, 78.9] | 98.6% [98.2, 98.9] |
| Cost-aware LR (embedding + handcrafted) | ID test | 0.603 [0.586, 0.620] | 0.753 [0.745, 0.761] | 39.6% [36.6, 42.4] | 63.8% [61.1, 65.8] | 69.5% [67.6, 71.2] | 78.5% [77.0, 80.5] | 98.6% [98.3, 98.9] |
| Oracle (quality gap only) | ID test | 1.184 [1.144, 1.228] | 0.821 [0.815, 0.828] | 8.0% [7.5, 8.5] | 12.7% [11.9, 13.6] | 86.3% [85.7, 87.0] | 88.3% [87.8, 89.0] | 106.9% [106.2, 107.5] |
| Oracle (gain per dollar, upper bound) | ID test | 1.184 [1.144, 1.228] | 0.825 [0.818, 0.831] | 8.0% [7.5, 8.5] | 12.7% [11.9, 13.6] | 88.6% [88.0, 89.1] | 90.5% [90.1, 90.8] | 106.9% [106.2, 107.5] |

### Same metrics out of distribution (leave-one-task-family-out, pooled over the same test prompts)

| Router | Split | APGR (random = 0.5) | AUC cost-quality | CPT(50%) | CPT(80%) | Cost saving @95% quality (test sweep) | Cost saving @90% quality (test sweep) | Quality retained at 50% of strong-only cost |
|---|---|---|---|---|---|---|---|---|
| Length/keyword heuristic | OOD (LOFO) | 0.520 [0.503, 0.537] | 0.669 [0.661, 0.678] | 46.8% [44.7, 49.2] | 66.2% [63.8, 71.4] | 13.3% [11.4, 15.1] | 21.1% [18.8, 23.3] | 82.3% [81.0, 83.4] |
| LR (handcrafted only) | OOD (LOFO) | 0.544 [0.526, 0.564] | 0.690 [0.682, 0.699] | 49.6% [46.1, 52.5] | 72.6% [69.6, 75.9] | 13.7% [12.1, 15.5] | 22.2% [19.8, 24.8] | 86.6% [85.4, 87.6] |
| LR (embedding) | OOD (LOFO) | 0.429 [0.411, 0.445] | 0.662 [0.653, 0.671] | 58.5% [55.8, 61.2] | 82.4% [80.3, 83.9] | 8.8% [7.6, 10.6] | 18.8% [17.0, 20.8] | 80.9% [79.7, 82.1] |
| LR (embedding + handcrafted) | OOD (LOFO) | 0.432 [0.414, 0.449] | 0.665 [0.656, 0.673] | 58.8% [55.9, 62.0] | 82.0% [80.4, 84.1] | 9.4% [8.3, 10.8] | 19.3% [17.2, 21.2] | 81.5% [80.3, 82.7] |
| Gap regression, ridge (embedding + handcrafted) | OOD (LOFO) | 0.462 [0.446, 0.478] | 0.673 [0.664, 0.681] | 55.4% [53.0, 57.9] | 80.4% [77.8, 82.5] | 13.3% [11.4, 15.1] | 23.5% [21.6, 25.6] | 82.4% [81.1, 83.5] |
| kNN, k=40 (embedding) | OOD (LOFO) | 0.429 [0.413, 0.447] | 0.665 [0.656, 0.674] | 59.2% [55.8, 61.9] | 81.9% [80.0, 84.1] | 10.2% [8.9, 11.6] | 19.5% [17.7, 21.7] | 81.5% [80.3, 82.7] |
| Cost only: predicted cost, no quality model (embedding + handcrafted) | OOD (LOFO) | 0.473 [0.454, 0.491] | 0.728 [0.720, 0.736] | 52.6% [50.8, 55.0] | 82.6% [79.3, 84.8] | 36.9% [28.5, 47.7] | 72.4% [70.5, 74.8] | 94.1% [93.3, 94.8] |
| Cost-aware LR (embedding + handcrafted) | OOD (LOFO) | 0.456 [0.436, 0.476] | 0.712 [0.704, 0.721] | 55.2% [51.3, 60.8] | 86.9% [85.3, 88.6] | 15.7% [13.0, 18.3] | 60.3% [50.6, 68.7] | 91.0% [90.1, 91.8] |

### Deployed 95% operating point for every router, ID vs OOD

| Router | Split | Quality retained | Cost saving | Lift vs random at same cost (points) |
|---|---|---|---|---|
| Length/keyword heuristic | ID test | 96.5% [95.8, 97.2] | 8.8% [8.1, 9.7] | -1.18 [-1.70, -0.71] |
| Length/keyword heuristic | OOD (LOFO) | 94.8% [93.9, 95.5] | 10.6% [9.9, 11.5] | -2.28 [-2.90, -1.73] |
| LR (handcrafted only) | ID test | 96.1% [95.4, 96.8] | 16.1% [14.8, 17.3] | -0.28 [-0.81, +0.23] |
| LR (handcrafted only) | OOD (LOFO) | 90.9% [90.0, 91.8] | 19.8% [18.6, 21.0] | -3.71 [-4.32, -3.10] |
| LR (embedding) | ID test | 95.0% [94.2, 95.8] | 21.6% [20.3, 22.9] | -0.20 [-0.74, +0.29] |
| LR (embedding) | OOD (LOFO) | 87.9% [86.8, 89.0] | 22.7% [21.7, 23.9] | -5.54 [-6.26, -4.87] |
| LR (embedding + handcrafted) | ID test | 95.1% [94.3, 95.8] | 22.4% [21.1, 23.7] | -0.01 [-0.56, +0.48] |
| LR (embedding + handcrafted) | OOD (LOFO) | 87.5% [86.4, 88.6] | 24.3% [23.1, 25.6] | -5.58 [-6.25, -4.94] |
| LR (embedding + handcrafted), both-fail rows dropped from training | ID test | 95.4% [94.7, 96.1] | 27.8% [26.4, 29.2] | +1.21 [+0.70, +1.70] |
| Gap regression, ridge (embedding + handcrafted) | ID test | 95.2% [94.5, 95.9] | 65.2% [64.0, 66.2] | +7.44 [+6.80, +8.10] |
| Gap regression, ridge (embedding + handcrafted) | OOD (LOFO) | 91.9% [91.1, 92.8] | 20.0% [18.7, 21.2] | -2.88 [-3.47, -2.30] |
| kNN, k=40 (embedding) | ID test | 95.4% [94.6, 96.1] | 61.7% [60.4, 63.0] | +6.96 [+6.30, +7.67] |
| kNN, k=40 (embedding) | OOD (LOFO) | 94.5% [93.7, 95.3] | 11.2% [10.4, 12.1] | -2.39 [-2.92, -1.84] |
| MLP (embedding + handcrafted) | ID test | 95.9% [95.2, 96.5] | 18.2% [17.1, 19.5] | -0.11 [-0.63, +0.35] |
| Cost only: predicted cost, no quality model (embedding + handcrafted) | ID test | 95.9% [95.2, 96.5] | 66.5% [65.4, 67.5] | +8.19 [+7.52, +8.95] |
| Cost only: predicted cost, no quality model (embedding + handcrafted) | OOD (LOFO) | 91.2% [90.3, 92.1] | 40.1% [38.7, 41.7] | +0.00 [-0.57, +0.60] |
| Cost-aware LR (embedding + handcrafted) | ID test | 95.6% [95.0, 96.3] | 67.8% [66.7, 68.8] | +8.23 [+7.58, +8.97] |
| Cost-aware LR (embedding + handcrafted) | OOD (LOFO) | 92.0% [91.1, 92.8] | 21.6% [20.4, 22.9] | -2.59 [-3.13, -2.06] |

### Per task family (headline router, 95% target)

| Task family | n test | Cheap | Strong | Within-family APGR, plain LR, ID | Within-family APGR, plain LR, OOD | Within-family APGR, heuristic | Router quality ID | To strong ID | Router quality OOD | To strong OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| arc | 294 | 0.847 | 0.969 | 0.592 | 0.582 | 0.580 | 0.959 | 90% | 0.963 | 91% |
| chinese | 157 | 0.287 | 0.503 | 0.561 | 0.441 | 0.484 | 0.465 | 73% | 0.363 | 43% |
| gsm8k | 1490 | 0.604 | 0.658 | 0.573 | 0.425 | 0.536 | 0.604 | 0% | 0.653 | 96% |
| hellaswag | 2009 | 0.598 | 0.837 | 0.612 | 0.443 | 0.640 | 0.794 | 86% | 0.653 | 43% |
| mbpp | 85 | 0.635 | 0.635 | 0.000 | 0.000 | 0.000 | 0.635 | 1% | 0.624 | 95% |
| misc | 185 | 0.580 | 0.599 | 0.697 | 0.720 | 1.372 | 0.585 | 2% | 0.585 | 2% |
| mmlu | 2808 | 0.652 | 0.818 | 0.638 | 0.537 | 0.598 | 0.794 | 78% | 0.800 | 88% |
| mtbench | 16 | 0.744 | 0.906 | 0.663 | 0.620 | 0.483 | 0.744 | 0% | 0.750 | 6% |
| winogrande | 254 | 0.594 | 0.823 | 0.524 | 0.524 | 0.479 | 0.823 | 100% | 0.819 | 95% |

Within-family APGR ranks prompts inside one task family only (0.5 = random), so it measures difficulty prediction rather than dataset identification. Test-size-weighted mean over arc, chinese, gsm8k, hellaswag, mmlu, winogrande (families with n >= 150 and a real quality gap), plain LR: 0.609 in distribution, 0.486 when the family was held out of training; length/keyword heuristic: 0.589. The 'Router quality' and 'To strong' columns are the headline cost-aware router at its 95% operating point.

### Calibration and overhead

- Calibration of P(needs strong), LR (embedding + handcrafted): ECE 0.015 on ID test, 0.110 OOD (LOFO); base rate 0.251.
- Routing overhead (embed + features + classify, single query, in-process, CPU `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`): p50 52.3 ms, p95 108.4 ms over 300 test prompts.

### Robustness to surface perturbations (ID test subset)

Clean APGR on the 1,500-prompt subset: 0.593.

| Perturbation | APGR | Routes flipped at the 95% threshold | Mean abs change in P | Quality retained (clean -> perturbed) | Cost saving (clean -> perturbed) |
|---|---|---|---|---|---|
| typos_5pct | 0.593 | 13.3% | 0.067 | 95.5% -> 93.8% | 67.4% -> 72.9% |
| lowercase | 0.595 | 1.2% | 0.009 | 95.5% -> 95.4% | 67.4% -> 67.9% |
| polite_filler | 0.589 | 6.5% | 0.034 | 95.5% -> 94.3% | 67.4% -> 70.1% |

### Stretch: budget-constrained online policy (simulated on the ID test stream)

| Target share to strong | Realised share | Quality retained | Cost saving | Lift vs random (points) | Quality retained, hindsight threshold at same share |
|---|---|---|---|---|---|
| 20% | 20.4% | 85.5% | 86.5% | +3.52 | 85.6% |
| 40% | 40.2% | 89.8% | 78.5% | +5.49 | 90.0% |
| 60% | 59.8% | 95.1% | 69.2% | +8.02 | 95.1% |

### Other model pairs (same splits and features, ID test)

| Pair (cheap -> strong) | Cheap / strong quality (ID test) | APGR (ID test) | Quality retained @95% target | Cost saving @95% target | Lift vs random (points) |
|---|---|---|---|---|---|
| mistralai/mixtral-8x7b-chat -> gpt-4-1106-preview | 0.549 / 0.783 | 0.596 [0.584, 0.609] | 95.2% [94.6, 95.7] | 65.3% [64.1, 66.4] | +12.16 [+11.47, +12.91] |
| claude-instant-v1 -> claude-v2 | 0.599 / 0.632 | 0.563 [0.477, 0.640] | 95.6% [94.3, 97.1] | 87.8% [87.5, 88.0] | +0.45 [+0.13, +0.77] |

### Resume bullets (filled from the numbers above)

- Built a cost-aware LLM router (MiniLM sentence embedding + handcrafted features + logistic regression with a predicted-cost term) retaining 95.6% of strong-model accuracy while routing 38% of traffic to a cheaper model, cutting cost 68% on RouterBench (gpt-3.5-turbo-1106 vs gpt-4-1106-preview, in-distribution test split, n=7,298; +8.2 quality points vs random routing at equal cost). Leave-one-task-family-out evaluation showed the gain does not transfer to unseen task types (92.0% quality at a 22% cost cut, -2.6 points vs random), reported with bootstrap CIs.
- Deployed as an OpenAI-compatible FastAPI proxy with 108 ms p95 routing overhead (CPU, in-process), request logging, and a cost/quality dashboard.
<!-- RESULTS:END -->

</details>
