# Cost-Aware Model Router: Project Plan

**One-liner:** A small classifier predicts whether a query needs a strong (expensive) LLM or a cheap one. Measure the cost vs. quality frontier against baselines and ship it as an OpenAI-compatible routing service.

**Resume headline target:** "Cut LLM spend X% at Y% of strong-model quality on [benchmark], with a routing overhead of under N ms."

## 1. Core question and hypothesis
Many queries are easy enough for a cheap model. If a fast classifier can tell which ones, you pay the strong-model price only when it matters.
**Hypothesis:** a lightweight router (embedding + logistic regression or small MLP, or a fine-tuned small encoder) can recover most of the strong model's quality while sending a large share of traffic to the cheap model.

## 2. Scope decisions
| Decision | Recommendation |
|---|---|
| Model pair | One cheap + one strong from the same provider family (e.g. Haiku-class vs Opus/Sonnet-class, or open 8B vs 70B via API). Third tier only in stretch. |
| Task mix | Mixed: general QA/reasoning (MMLU-style), math (GSM8K), code (HumanEval/MBPP-style), open-ended chat. |
| Quality definition | Per-task automatic metric where available (exact match, pass@1). Open-ended chat: LLM judge with position-swap. |
| Routing granularity | Per-query, before generation (no extra model call). Cascading is a stretch comparison. |

## 3. Data
1. **Primary: RouterBench** (precomputed responses/scores of many LLMs across tasks). Train/evaluate routers without paying per call. Check license and which models it covers.
2. **Secondary: RouteLLM / LMSYS preference data** for the chat portion.
3. **Own holdout:** responses from the chosen pair on a few hundred self-collected prompts, to verify generalization (requires API keys).

**Labels:** `needs_strong = (strong_score - cheap_score) > epsilon`. Soft alternative: regress the quality gap, then threshold (tunable knob, smooth frontier).
Split by **query cluster or source dataset** (not just random) to report in-distribution vs OOD.

## 4. Architecture
```
client -> router service (FastAPI, OpenAI-compatible /v1/chat/completions)
            |-- featurizer (embedding model, optional handcrafted features)
            |-- router model (predicts P(needs_strong))
            |-- policy (threshold tau, budget-aware variant)
            |-- dispatcher -> cheap LLM / strong LLM
            `-- logger (SQLite/Postgres: query hash, route, latency, tokens, cost)
dashboard (Streamlit): cost, route mix, latency, quality estimates
```
Features in order: (1) frozen sentence embedding + logistic regression (baseline router); (2) add handcrafted features: length, code/math markers, constraints count, question type; (3) fine-tune small encoder (MiniLM/DistilBERT class); (4) stretch: cheap model's own signal (logprobs, self-consistency).

## 5. Baselines (non-negotiable)
Always-cheap, always-strong; random routing at matched cost; length/keyword heuristic router; oracle router (upper bound); published routers (RouteLLM MF/BERT) if reproducible.

## 6. Evaluation
**Primary plot:** x = avg cost per query (or % to strong), y = avg quality; curve per router plus baselines. Report: area under cost-quality curve; cost at 95%/90% of strong-model quality (headline); quality at fixed budget (e.g. 50% of strong-only cost); APGR/CPT-style metrics from RouteLLM.
**Secondary:** per-task breakdown; calibration of P(needs_strong) (reliability diagram, ECE); OOD test (train on some task families, test on held-out); router overhead p50/p95 latency; robustness (paraphrases, typos, adversarial easy-looking-hard prompts); bootstrap CIs on every headline number.

## 7. Milestones
0 Setup (loaders, config for pair + pricing; always-cheap/strong/oracle numbers) -> 1 Baseline router (embedding+LR, frontier plot, beats random-at-matched-cost with CI) -> 2 Better routers (handcrafted, fine-tuned encoder, regression variant, ablation table) -> 3 Rigorous eval (OOD, calibration, overhead, per-task, failure cases, CIs) -> 4 Serving (FastAPI OpenAI-compatible proxy, logging, dashboard, `docker compose up`) -> 5 Stretch (pick one: 3-tier, cascade comparison, budget-constrained online policy, drift monitoring) -> 6 Write-up (README with plots, reproducible scripts).

## 8. Risks
Label noise (both-fail queries: treat separately, report both versions). Contamination/saturation (mixed set + own holdout). Router learns dataset not difficulty (OOD splits; say so honestly if only in-distribution). LLM-judge bias (swap positions, different judge family, hand spot-check). Prices change (store prices in config with a date; report tokens plus dollars).

## 9. Repo layout
```
router/
  data/ features/ models/ eval/ serve/ dashboards/ configs/ notebooks/ tests/
  README.md   # headline plot first
```

## 10. Resume bullets (fill in ONLY with real measured results)
- Built a cost-aware LLM router (embedding + [model]) retaining [Y]% of strong-model accuracy while routing [Z]% of traffic to a cheaper model, cutting cost [X]% on [benchmark], with bootstrap CIs and OOD splits.
- Deployed as an OpenAI-compatible FastAPI proxy with [N] ms p95 routing overhead, request logging, and a cost/quality dashboard.

## 11. Definition of done
One headline figure and number, reproducible from a single script. Honest limitations section. Working demo runnable in under 2 minutes.

**Shared tip:** build the evaluation harness before fancy modeling.
