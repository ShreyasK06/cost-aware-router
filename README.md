# Cost-aware LLM router

![Cost vs quality frontier, in-distribution and out-of-distribution](assets/frontier.png)

A small classifier decides, per query and before any generation, whether to call a cheap LLM or a
strong one. This repo measures the cost vs quality frontier of several such routers against
baselines on [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), and ships the
best one as an OpenAI-compatible proxy with request logging and a dashboard.

**Result in one paragraph.** On RouterBench (GPT-3.5-turbo vs GPT-4-turbo, 36,497 prompts), a
logistic-regression router over a MiniLM embedding, 21 handcrafted features and a predicted-cost
term kept **95.6% of GPT-4 quality at 67.8% lower cost on the in-distribution test split**
(threshold frozen on validation; +8.2 quality points over random routing at the same cost).
**It did not generalize to unseen task types:** when each task family was held out of training, the
same recipe kept 92.0% of quality at a 21.6% cost cut, which is 2.6 points *worse* than random
routing at the same cost. Both numbers are on the same 7,298 test prompts and have bootstrap CIs
in the tables below.

## What the experiments say

1. **Most of the in-distribution saving comes from knowing which prompts are expensive, not which
   are hard.** A baseline with no quality model at all, which only predicts the strong model's
   extra cost per prompt, gets a 66.5% cost cut at the 95% target (ID test), almost the same as the
   full router's 67.8%. One task family (GSM8K) carries roughly half of the GPT-4 bill while GPT-4
   is only about 5 points better on it, so the router sends all of it to the cheap model.
2. **A plain "does this need the strong model?" classifier is no better than random at matched
   cost.** Embedding + logistic regression gets APGR 0.637 (random is 0.5) when you count calls,
   but at its 95% operating point its quality lift over random routing at the same dollar cost is
   -0.01 points [-0.56, +0.48] (ID test), because the prompts it sends to the strong model are the
   expensive ones.
3. **Routers learn the dataset more than the difficulty.** Held-out task families (OOD) push every
   learned router to or below random routing at matched cost. The cost-only baseline is exactly at
   random (+0.00 points), and the plain classifier is 5.6 points below. Calibration degrades too:
   ECE 0.015 in distribution vs 0.110 OOD.
4. The headline router was fixed before the final run; the cost exponent and all thresholds are
   chosen on validation data only.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

More figures: [all routers](assets/frontier_all_routers.png),
[reliability ID](assets/reliability_id.png), [reliability OOD](assets/reliability_ood.png),
[per task family](assets/per_task.png). Raw numbers: [results/results.json](results/results.json).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate                 # Windows; on Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/demo.py                 # routing proxy demo, mock LLM backend, no keys needed
python -m pytest -q                    # tests: synthetic data + hashing embedder, no network
python scripts/reproduce.py            # regenerates every number, table and figure above
```

`scripts/demo.py` uses the committed router artifact and finishes in well under two minutes. Its
first run downloads the MiniLM embedding model (about 90 MB) from Hugging Face; pass
`--embedder hashing` only to smoke-test without any download (routing decisions are then
meaningless, because the artifact was trained on MiniLM features).

`scripts/reproduce.py` downloads RouterBench (about 100 MB, no token needed), embeds 36,497 prompts
on CPU once (cached under `data/cache/`, roughly 10 to 40 minutes on a laptop) and then takes about
10 minutes. `--quick` uses fewer bootstrap resamples and skips the MLP and robustness steps.

## How it works

```
client -> router service (FastAPI, OpenAI-compatible /v1/chat/completions)
            |-- featurizer: MiniLM sentence embedding (head + tail of the prompt) + 21 handcrafted features
            |-- router: one linear head = logit P(needs strong) - alpha * predicted log extra cost
            |-- policy: fixed threshold (calibrated offline) or online budget policy
            |-- dispatcher -> cheap LLM / strong LLM (mock, OpenAI-compatible, or Anthropic backend)
            `-- logger: SQLite (prompt hash, route, score, latency, tokens, cost, counterfactual cost)
dashboard (Streamlit): cost vs always-strong, route mix, latency, offline quality estimate
```

* **Data and labels.** RouterBench 0-shot gives every model's score and dollar cost on every prompt,
  so routers are trained and evaluated without paying for calls. `needs_strong = (strong_score -
  cheap_score) > 0`. Prompts where both models fail carry no routing signal; a variant that drops
  them from training is reported.
* **Splits.** In-distribution: 60/20/20 random split, stratified by task family, duplicate prompts
  kept together. OOD: leave-one-task-family-out, where the model and its threshold never see the
  held-out family; predictions are pooled so ID and OOD are compared on the same test prompts.
* **Evaluation harness** (`router/eval/`, built before any router): always-cheap, always-strong,
  random routing at matched cost, a training-free length/keyword heuristic, two oracles, frontier
  metrics (APGR, cost-quality AUC, CPT, cost at 95%/90% of strong quality, quality at half budget),
  deployed operating points with thresholds frozen on validation, bootstrap CIs, calibration (ECE),
  per-family breakdown, surface-perturbation robustness and routing overhead.
* **Routers** (`router/models/`): logistic regression on embeddings, on handcrafted features, on
  both; ridge regression of the quality gap; kNN; MLP; and the cost-aware logistic router.
* Interfaces and schemas: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Original plan:
  [docs/PLAN.md](docs/PLAN.md).

## Serving

```bash
# mock backend (default): no keys, canned answers, real routing decisions
uvicorn router.serve.app:get_app --factory --port 8000
curl -s localhost:8000/v1/chat/completions -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 17 + 25?"}]}'
streamlit run router/dashboards/app.py

# real providers: copy .env.example to .env and set the backend, models, keys and prices
ROUTER_BACKEND=openai CHEAP_MODEL=... STRONG_MODEL=... OPENAI_API_KEY=... uvicorn router.serve.app:get_app --factory

docker compose up --build              # router on :8000, dashboard on :8501
```

`model: "auto"` routes; `"cheap"` / `"strong"` force a tier. Every response carries a `router`
object (`route`, score, threshold, policy, overhead) and an `x-router-route` header. The header
`x-router-threshold` overrides the threshold per request, and `ROUTER_POLICY=budget` switches to an
online policy that tracks a target share of strong calls. The value the service calls `p_strong` is
the cost-adjusted routing score in (0, 1), not a calibrated probability. Only a SHA-256 hash of the
prompt is logged, never its text.

## What was not validated against live APIs

No API keys were available while building this, so:

* **Live LLM calls were never made.** The OpenAI-compatible and Anthropic backends are tested only
  against mocked HTTP transports. The demo and the end-to-end tests use the mock backend.
* **No own holdout set.** The plan's self-collected prompts answered by a current model pair were
  not generated, so there is no evidence about today's models.
* **No LLM judge.** Open-ended chat quality with a position-swapped judge was not run; all quality
  numbers are RouterBench's own automatic scores.
* **RouteLLM / LMSYS preference data was not used**, and RouteLLM's published routers (MF, BERT)
  were not reproduced; the kNN router is a reimplementation in the style of the RouterBench paper.
* **`docker compose up` was written but not built or run** on the development machine.
* **The fine-tuned encoder is implemented (`router/models/finetune.py`, `--finetune`) but was not
  run** (CPU-only machine), so no number is reported for it.
* Dollar figures in the serving layer use the price table in `configs/default.yaml` (dated); the
  benchmark numbers use the per-prompt costs recorded in RouterBench in early 2024.

## Limitations

* **The headline saving is in-distribution and benchmark-specific.** It depends on a cost structure
  where one task family is expensive for the strong model and gains little from it. Traffic without
  that structure should expect much less, and the OOD rows are the better guide for new task types.
* **OOD routing is at or below random at matched cost.** Within a task family the classifier's
  ranking of prompts is only slightly better than chance (see the per-family table), so the router
  is mostly identifying the dataset.
* **Old model pair.** GPT-3.5-turbo-1106 vs GPT-4-1106-preview with 2024 prices. Quality gaps and
  price ratios of current models differ.
* **Benchmark, not user traffic.** Mostly multiple-choice and short-answer tasks with templated
  prompts; little open-ended chat or code. Contamination and saturation of these evals are likely.
* **Pooled OOD scores come from nine different fold models**, so the pooled OOD frontier curve is
  an approximation; the deployed operating points use per-fold thresholds and are the cleaner number.
* **Robustness tests are surface-level** (typos, lowercasing, polite filler). No paraphrase or
  adversarial "easy-looking hard prompt" set was built.
* **Routing overhead was measured in-process on one laptop CPU** while other work was running; it is
  dominated by the MiniLM forward pass and is not a server benchmark.
* Stretch items not done: three-tier routing, cascading, drift monitoring. The budget policy is
  evaluated only in simulation on the test stream.

## Repo layout

```
router/data  router/features  router/models  router/eval  router/serve  router/dashboards
configs/  scripts/reproduce.py  scripts/demo.py  tests/  artifacts/  assets/  results/  docs/
```

## License and data

Code: MIT (see [LICENSE](LICENSE)). RouterBench is downloaded at run time from
`withmartian/routerbench` and is not redistributed here; its dataset card states no license, so
check its terms before reusing the data. RouterBench: Hu et al., 2024, "RouterBench: A Benchmark for
Multi-LLM Routing System".
