"""Reproduce every number and figure in the README from scratch.

    python scripts/reproduce.py            # full run (downloads RouterBench + MiniLM on first use)
    python scripts/reproduce.py --quick    # fewer bootstrap resamples, skips MLP / robustness
    python scripts/reproduce.py --finetune # additionally fine-tune a small encoder on CPU (slow)

Outputs: assets/*.png, results/results.json, results/tables.md, artifacts/router_lr.{npz,json},
and the auto-generated results block in README.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from router.config import load_config  # noqa: E402
from router.data.dataset import add_labels, build_pair_table, load_dataset, lofo_folds  # noqa: E402
from router.eval import baselines as B  # noqa: E402
from router.eval import metrics as M  # noqa: E402
from router.eval.bootstrap import bootstrap_many  # noqa: E402
from router.eval.calibration import expected_calibration_error, reliability_curve  # noqa: E402
from router.eval.latency import measure_overhead  # noqa: E402
from router.eval.robustness import perturb_case, perturb_filler, perturb_typos  # noqa: E402
from router.features.embeddings import cached_embeddings, get_embedder  # noqa: E402
from router.features.featurizer import combine_features  # noqa: E402
from router.features.handcrafted import handcrafted_features  # noqa: E402
from router.features.text import routing_views  # noqa: E402
from router.models.artifact import LinearRouterArtifact  # noqa: E402
from router.models.cost_aware import CostAwareLogisticRouter  # noqa: E402
from router.models.routers import (GapRegressionRouter, KNNRouter, LogisticRouter,  # noqa: E402
                                   MLPRouter)

HEADLINE = "Cost-aware LR (embedding + handcrafted)"
PLAIN = "LR (embedding + handcrafted)"
HEUR = "Length/keyword heuristic"
ORACLE = "Oracle (gain per dollar, upper bound)"
ORACLE_GAP = "Oracle (quality gap only)"
COST_ONLY = "Cost only: predicted cost, no quality model (embedding + handcrafted)"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
NO_BOTH_FAIL = "LR (embedding + handcrafted), both-fail rows dropped from training"
FINETUNED = "Fine-tuned MiniLM (1 epoch, CPU-sized)"
TARGETS = {"rel95": 0.95, "rel90": 0.90}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- evaluation helpers
def frontier_stats(scores: np.ndarray, data: M.EvalData, n_boot: int, seed: int, budget: float) -> dict:
    """Curve-level metrics on a test set (no threshold choice), with bootstrap CIs."""
    scores = np.asarray(scores, dtype=float)

    def stat(idx):
        s, d = scores[idx], data.subset(idx)
        a = M.apgr(s, d)
        return {
            "apgr": a,
            "apgr_minus_random": a - 0.5,
            "auc_cost_quality": M.auc_cost_quality(s, d),
            "cpt50": M.cpt(s, d, 0.5),
            "cpt80": M.cpt(s, d, 0.8),
            "sweep_cost_saving_at_rel95": M.cost_at_quality(s, d, 0.95)["cost_saving"],
            "sweep_cost_saving_at_rel90": M.cost_at_quality(s, d, 0.90)["cost_saving"],
            "rel_quality_at_half_budget": M.quality_at_budget(s, d, budget)["rel_quality"],
        }

    return bootstrap_many(stat, len(data), n_boot=n_boot, seed=seed)


def op_stats(route: np.ndarray, data: M.EvalData, n_boot: int, seed: int) -> dict:
    """Realised operating point for frozen routing decisions, with bootstrap CIs."""
    route = np.asarray(route, dtype=bool)

    def stat(idx):
        op = M.operating_point(route[idx], data.subset(idx))
        return {
            "rel_quality": op["rel_quality"],
            "quality": op["quality"],
            "cost_saving": op["cost_saving"],
            "frac_cheap": 1.0 - op["frac_strong"],
            "random_quality": op["random_quality"],
            "lift_vs_random": op["lift_vs_random"],
        }

    return bootstrap_many(stat, len(data), n_boot=n_boot, seed=seed)


def frontier_band(scores, data, n_boot=200, seed=0, n_points=101):
    rng = np.random.default_rng(seed)
    n = len(data)
    qs = np.empty((n_boot, n_points))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        qs[b] = M.frontier(scores[idx], data.subset(idx), n_points)["quality"].to_numpy()
    return np.percentile(qs, 2.5, axis=0), np.percentile(qs, 97.5, axis=0)


def fmt(ci: dict, pct: bool = False, digits: int = 3) -> str:
    k = 100.0 if pct else 1.0
    d = 1 if pct else digits
    u = "%" if pct else ""
    return f"{ci['point'] * k:.{d}f}{u} [{ci['lo'] * k:.{d}f}, {ci['hi'] * k:.{d}f}]"


def fmt_pts(ci: dict) -> str:
    return f"{ci['point'] * 100:+.2f} [{ci['lo'] * 100:+.2f}, {ci['hi'] * 100:+.2f}]"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- router zoo
def fit_cost_aware(x_tr, gap_tr, extra_tr, x_va, d_va):
    """Fit both heads once, then pick the cost exponent alpha on validation (never on test)."""
    base = CostAwareLogisticRouter(alpha=0.0, C=0.1).fit(x_tr, gap_tr, extra_tr)
    aucs = {a: M.auc_cost_quality(base.with_alpha(a).score(x_va), d_va) for a in ALPHAS}
    return base.with_alpha(max(aucs, key=aucs.get))


class _CostOnly:
    """Baseline with no quality model: send the prompts that are cheapest for the strong model first."""

    def __init__(self, base: CostAwareLogisticRouter):
        self.base = base

    def score(self, x):
        return -np.log(self.base.predicted_extra_cost(x))


def fit_cost_only(x_tr, gap_tr, extra_tr, x_va, d_va):
    return _CostOnly(CostAwareLogisticRouter(alpha=0.0, C=0.1).fit(x_tr, gap_tr, extra_tr))


def _plain(make):
    return lambda x_tr, gap_tr, extra_tr, x_va, d_va: make().fit(x_tr, gap_tr)


def make_routers(quick: bool) -> list[tuple[str, str, object]]:
    """(display name, feature set, fit function). Feature sets: E=embedding, H=handcrafted, EH=both.

    fit function signature: (X_train, gap_train, extra_cost_train, X_val, EvalData_val) -> fitted router.
    """
    zoo = [
        ("LR (embedding)", "E", _plain(lambda: LogisticRouter(C=0.1))),
        ("LR (handcrafted only)", "H", _plain(lambda: LogisticRouter(C=1.0))),
        (PLAIN, "EH", _plain(lambda: LogisticRouter(C=0.1))),
        (HEADLINE, "EH", fit_cost_aware),
        (COST_ONLY, "EH", fit_cost_only),
        ("Gap regression, ridge (embedding + handcrafted)", "EH",
         _plain(lambda: GapRegressionRouter(alpha=100.0))),
        ("kNN, k=40 (embedding)", "E", _plain(lambda: KNNRouter(k=40))),
    ]
    if not quick:
        zoo.append(("MLP (embedding + handcrafted)", "EH",
                    _plain(lambda: MLPRouter(hidden=(256, 64), seed=0))))
    return zoo


OOD_ROUTERS = ["LR (embedding)", "LR (handcrafted only)", PLAIN, HEADLINE, COST_ONLY,
               "Gap regression, ridge (embedding + handcrafted)", "kNN, k=40 (embedding)"]


def evaluate_scores(val_scores, test_scores, data_val, data_test, n_boot, seed, budget) -> dict:
    out = {"val_apgr": M.apgr(val_scores, data_val),
           "frontier": frontier_stats(test_scores, data_test, n_boot, seed, budget),
           "thresholds": {}, "deployed": {}}
    for key, target in TARGETS.items():
        t = M.calibrate_threshold(val_scores, data_val, target)
        out["thresholds"][key] = t
        out["deployed"][key] = op_stats(M.route_at_threshold(test_scores, t), data_test, n_boot, seed)
    return out


# ----------------------------------------------------------------------------- figures
def headline_figure(panels: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, grey = "#0072B2", "#D55E00", "#7f7f7f"
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    k = 1000.0
    for ax, p in zip(axes, panels):
        d = p["data"]
        ch, st = B.always_cheap(d), B.always_strong(d)
        ax.plot(p["oracle"]["cost"] * k, p["oracle"]["quality"], color="black", ls="--", lw=1.2,
                label="Oracle (upper bound)")
        ax.plot([ch["cost"] * k, st["cost"] * k], [ch["quality"], st["quality"]], color=grey, ls="--",
                lw=1.4, label="Random routing")
        ax.plot(p["heuristic"]["cost"] * k, p["heuristic"]["quality"], color=orange, lw=1.6,
                label="Length/keyword heuristic")
        fr = p["router"]
        if p.get("band") is not None:
            ax.fill_between(fr["cost"] * k, p["band"][0], p["band"][1], color=blue, alpha=0.18, lw=0)
        ax.plot(fr["cost"] * k, fr["quality"], color=blue, lw=2.4, label=f"Router: {p['router_name']}")
        ax.scatter([ch["cost"] * k], [ch["quality"]], marker="s", s=70, color="#009E73", zorder=5,
                   label=f"Always cheap ({p['cheap']})")
        ax.scatter([st["cost"] * k], [st["quality"]], marker="D", s=70, color="#CC79A7", zorder=5,
                   label=f"Always strong ({p['strong']})")
        ax.axhline(0.95 * st["quality"], color=grey, ls=":", lw=1)
        ax.text(ch["cost"] * k, 0.95 * st["quality"], "95% of strong quality", va="bottom", ha="left",
                fontsize=8.5, color=grey)
        op = p.get("op")
        if op is not None:
            ax.scatter([op["cost"] * k], [op["quality"]], marker="*", s=260, color=blue, edgecolor="white",
                       zorder=6)
            ax.annotate(p["op_label"], xy=(op["cost"] * k, op["quality"]),
                        xytext=(0.40, 0.08), textcoords="axes fraction", fontsize=9.3,
                        arrowprops=dict(arrowstyle="->", color=blue, lw=1.1),
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=blue, lw=0.9))
        ax.set_title(p["title"], fontsize=11.5)
        ax.set_xlabel("Average cost per 1,000 queries (USD, costs recorded in RouterBench)")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Average quality (benchmark score)")
    axes[0].legend(loc="upper left", fontsize=8.3, frameon=False)
    fig.suptitle("Cost vs quality frontier on RouterBench (0-shot); shaded: 95% bootstrap band", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap resamples; skip MLP and robustness")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--finetune", action="store_true", help="also fine-tune a small encoder (CPU, slow)")
    ap.add_argument("--no-readme", action="store_true", help="do not rewrite the README results block")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render results/tables.md and the README block from results/results.json")
    args = ap.parse_args()

    if args.render_only:
        res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
        tables = render_tables(res)
        (ROOT / "results" / "tables.md").write_text(tables["all"], encoding="utf-8")
        inject_readme(tables["readme"])
        print(tables["headline"])
        return

    cfg = load_config(args.config)
    seed = cfg["eval"]["seed"]
    n_boot = args.n_boot or (200 if args.quick else cfg["eval"]["n_boot"])
    budget = cfg["eval"]["budget_frac"]
    cheap_name, strong_name = cfg["pair"]["cheap"], cfg["pair"]["strong"]
    assets, results_dir, art_dir = ROOT / "assets", ROOT / "results", ROOT / "artifacts"
    for d in (assets, results_dir, art_dir):
        d.mkdir(exist_ok=True)

    # ---- 1. data
    log("loading RouterBench (downloads on first run)")
    df = load_dataset(cfg, ROOT)
    split = df["split"].to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    gap = df["gap"].to_numpy()
    extra = (df["strong_cost"] - df["cheap_cost"]).to_numpy()
    prompts = df["prompt"].tolist()
    data_all = M.EvalData.from_frame(df)
    data_val, data_test = data_all.subset(va), data_all.subset(te)
    fam_test = df["task_family"].to_numpy()[te]
    log(f"n={len(df)} train/val/test={len(tr)}/{len(va)}/{len(te)}; pair {cheap_name} -> {strong_name}")

    # ---- 2. features
    cache_dir = ROOT / cfg["data"]["cache_dir"]
    embedder = get_embedder(cfg["features"]["embedder"], cfg["features"]["max_seq_length"])
    log("embedding prompts (cached after the first run)")
    emb = cached_embeddings(routing_views(prompts), embedder, cache_dir)
    hand = handcrafted_features(prompts)
    feats = {"E": emb, "H": hand, "EH": combine_features(emb, hand)}

    res: dict = {"meta": {
        "generated": dt.date.today().isoformat(),
        "benchmark": "RouterBench 0-shot (withmartian/routerbench)",
        "cheap_model": cheap_name, "strong_model": strong_name,
        "n_total": int(len(df)), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "n_boot": int(n_boot), "epsilon": cfg["labels"]["epsilon"],
        "embedder": cfg["features"]["embedder"], "split_seed": cfg["split"]["seed"],
        "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        "quick": bool(args.quick),
    }}

    # ---- 3. baselines first (evaluation harness before routers)
    ch, st = B.always_cheap(data_test), B.always_strong(data_test)
    res["dataset"] = {
        "test": {"always_cheap": ch, "always_strong": st,
                 "cheap_rel_quality": ch["quality"] / st["quality"],
                 "cost_ratio_strong_over_cheap": st["cost"] / ch["cost"]},
        "label_rates_all": {
            "needs_strong": float(df["needs_strong"].mean()),
            "cheap_better": float((df["gap"] < 0).mean()),
            "both_fail": float(df["both_fail"].mean()),
            "tie": float((df["gap"] == 0).mean()),
        },
        "families": {f: int(n) for f, n in df["task_family"].value_counts().items()},
    }
    log(f"test: always-cheap q={ch['quality']:.4f} cost={ch['cost']:.6f} | always-strong q={st['quality']:.4f} "
        f"cost={st['cost']:.6f}")

    id_res: dict = {}
    test_scores: dict[str, np.ndarray] = {}
    val_scores: dict[str, np.ndarray] = {}
    heur_all = B.heuristic_scores(prompts)
    base_scores = {
        ORACLE: (B.oracle_cost_aware_scores(data_val), B.oracle_cost_aware_scores(data_test)),
        ORACLE_GAP: (B.oracle_scores(data_val), B.oracle_scores(data_test)),
        "Random": (B.random_scores(len(va), seed), B.random_scores(len(te), seed + 1)),
        HEUR: (heur_all[va], heur_all[te]),
    }
    for name, (sv, s_te) in base_scores.items():
        val_scores[name], test_scores[name] = sv, s_te
        id_res[name] = evaluate_scores(sv, s_te, data_val, data_test, n_boot, seed, budget)
        log(f"baseline: {name}: test APGR {fmt(id_res[name]['frontier']['apgr'])}")

    # ---- 4. routers, in-distribution
    fitted: dict[str, object] = {}
    for name, fs, fit_fn in make_routers(args.quick):
        t0 = time.time()
        r = fit_fn(feats[fs][tr], gap[tr], extra[tr], feats[fs][va], data_val)
        fitted[name] = r
        val_scores[name], test_scores[name] = r.score(feats[fs][va]), r.score(feats[fs][te])
        id_res[name] = evaluate_scores(val_scores[name], test_scores[name], data_val, data_test, n_boot, seed,
                                       budget)
        id_res[name]["features"] = fs
        if hasattr(r, "alpha") and isinstance(r, CostAwareLogisticRouter):
            id_res[name]["cost_alpha"] = r.alpha
        log(f"router: {name}: test AUC {fmt(id_res[name]['frontier']['auc_cost_quality'])} "
            f"APGR {fmt(id_res[name]['frontier']['apgr'])} ({time.time() - t0:.0f}s)")

    # label-noise variant: drop both-fail rows from training (they carry no routing signal)
    keep = tr[~df["both_fail"].to_numpy()[tr]]
    r_nb = LogisticRouter(C=0.1).fit(feats["EH"][keep], gap[keep])
    val_scores[NO_BOTH_FAIL], test_scores[NO_BOTH_FAIL] = r_nb.score(feats["EH"][va]), r_nb.score(feats["EH"][te])
    id_res[NO_BOTH_FAIL] = evaluate_scores(val_scores[NO_BOTH_FAIL], test_scores[NO_BOTH_FAIL], data_val,
                                           data_test, n_boot, seed, budget)
    id_res[NO_BOTH_FAIL]["features"] = "EH"
    res["dataset"]["both_fail_train_rows_dropped"] = int(len(tr) - len(keep))

    # optional fine-tuned encoder (scores are cached so later default runs can include the row)
    ft_cache = cache_dir / "finetune_scores.npz"
    if args.finetune:
        from router.models.finetune import finetune_encoder
        views = routing_views(prompts, head=100, tail=400)
        y = df["needs_strong"].to_numpy()
        rng = np.random.default_rng(seed)
        sub = rng.choice(tr, size=min(8000, len(tr)), replace=False)
        vsub = rng.choice(va, size=min(1000, len(va)), replace=False)
        log(f"fine-tuning encoder on {len(sub)} prompts (CPU)")
        t0 = time.time()
        ft = finetune_encoder([views[i] for i in sub], y[sub], [views[i] for i in vsub], y[vsub],
                              model_name=cfg["features"]["embedder"], epochs=1, max_len=128, batch_size=32,
                              lr=5e-5, seed=seed)
        sv = ft.score_prompts([views[i] for i in va])
        s_te = ft.score_prompts([views[i] for i in te])
        np.savez(ft_cache, val=sv, test=s_te, n_train=len(sub), minutes=(time.time() - t0) / 60.0,
                 n_test=len(te))
    if ft_cache.exists():
        z = np.load(ft_cache, allow_pickle=False)
        if int(z["n_test"]) == len(te):
            val_scores[FINETUNED], test_scores[FINETUNED] = z["val"], z["test"]
            id_res[FINETUNED] = evaluate_scores(z["val"], z["test"], data_val, data_test, n_boot, seed, budget)
            id_res[FINETUNED].update(features="raw text", train_rows=int(z["n_train"]),
                                     train_minutes=float(z["minutes"]))
            log(f"router: {FINETUNED}: test APGR {fmt(id_res[FINETUNED]['frontier']['apgr'])}")
    res["id"] = id_res

    # ---- 5. OOD: leave-one-family-out on the same test rows
    log("OOD: leave-one-task-family-out")
    pos_in_test = {int(i): j for j, i in enumerate(te)}
    zoo = {n: (fs, f) for n, fs, f in make_routers(True)}
    ood_alphas: dict[str, float] = {}
    ood_scores = {n: np.full(len(te), np.nan) for n in OOD_ROUTERS}
    ood_val_apgr = {n: [] for n in OOD_ROUTERS}
    ood_route = {n: {k: np.zeros(len(te), dtype=bool) for k in TARGETS} for n in OOD_ROUTERS}
    heur_route = {k: np.zeros(len(te), dtype=bool) for k in TARGETS}
    for fam, f_tr, f_va, f_te in lofo_folds(df):
        j = np.array([pos_in_test[int(i)] for i in f_te])
        d_va = data_all.subset(f_va)
        for name in OOD_ROUTERS:
            fs, fit_fn = zoo[name]
            r = fit_fn(feats[fs][f_tr], gap[f_tr], extra[f_tr], feats[fs][f_va], d_va)
            if name == HEADLINE:
                ood_alphas[fam] = r.alpha
            sv, s_te = r.score(feats[fs][f_va]), r.score(feats[fs][f_te])
            ood_scores[name][j] = s_te
            ood_val_apgr[name].append(M.apgr(sv, d_va))
            for k, target in TARGETS.items():
                ood_route[name][k][j] = M.route_at_threshold(s_te, M.calibrate_threshold(sv, d_va, target))
        for k, target in TARGETS.items():  # the heuristic needs no training, but its threshold does
            heur_route[k][j] = M.route_at_threshold(heur_all[f_te],
                                                    M.calibrate_threshold(heur_all[f_va], d_va, target))
        log(f"  held out {fam}: n_test={len(f_te)}")
    ood_res: dict = {}
    for name in OOD_ROUTERS:
        assert not np.isnan(ood_scores[name]).any()
        ood_res[name] = {
            "frontier": frontier_stats(ood_scores[name], data_test, n_boot, seed, budget),
            "deployed": {k: op_stats(ood_route[name][k], data_test, n_boot, seed) for k in TARGETS},
            "mean_fold_val_apgr": float(np.mean(ood_val_apgr[name])),
        }
        log(f"  OOD {name}: APGR {fmt(ood_res[name]['frontier']['apgr'])}")
    ood_res[HEUR] = {"frontier": id_res[HEUR]["frontier"],
                     "deployed": {k: op_stats(heur_route[k], data_test, n_boot, seed) for k in TARGETS}}
    ood_res[HEADLINE]["cost_alpha_per_fold"] = ood_alphas
    res["ood"] = ood_res

    # ---- 6. per-family breakdown for the headline router (ID vs OOD on the same rows)
    t95 = id_res[HEADLINE]["thresholds"]["rel95"]
    id_route95 = M.route_at_threshold(test_scores[HEADLINE], t95)
    per_family = []
    for fam in sorted(set(fam_test)):
        j = np.flatnonzero(fam_test == fam)
        d = data_test.subset(j)
        row = {"task_family": fam, "n_test": int(len(j)),
               "cheap": float(d.cheap_score.mean()), "strong": float(d.strong_score.mean()),
               "apgr_id": M.apgr(test_scores[HEADLINE][j], d),
               "apgr_ood": M.apgr(ood_scores[HEADLINE][j], d),
               "apgr_plain_id": M.apgr(test_scores[PLAIN][j], d),
               "apgr_plain_ood": M.apgr(ood_scores[PLAIN][j], d),
               "apgr_heuristic": M.apgr(test_scores[HEUR][j], d)}
        for tag, route in (("id", id_route95), ("ood", ood_route[HEADLINE]["rel95"])):
            op = M.operating_point(route[j], d)
            row[f"router_{tag}"] = op["quality"]
            row[f"frac_strong_{tag}"] = op["frac_strong"]
            row[f"cost_saving_{tag}"] = op["cost_saving"]
        per_family.append(row)
    res["per_family"] = per_family
    # how much of the signal is within a task family (difficulty) rather than between families (dataset)?
    big = [r for r in per_family if r["n_test"] >= 150 and abs(r["strong"] - r["cheap"]) > 0.02]
    w = np.array([r["n_test"] for r in big], dtype=float)
    res["within_family_apgr"] = {
        "families": [r["task_family"] for r in big],
        "plain_id": float(np.average([r["apgr_plain_id"] for r in big], weights=w)),
        "plain_ood": float(np.average([r["apgr_plain_ood"] for r in big], weights=w)),
        "heuristic": float(np.average([r["apgr_heuristic"] for r in big], weights=w)),
    }

    # ---- 7. calibration of P(needs_strong)
    y_test = df["needs_strong"].to_numpy()[te]
    cal = {}
    for tag, s in (("id", test_scores[PLAIN]), ("ood", ood_scores[PLAIN])):  # the plain LR is the probability
        cal[tag] = {"ece": expected_calibration_error(s, y_test, 10),
                    "base_rate": float(y_test.mean()), "mean_pred": float(np.mean(s))}
    res["calibration"] = cal

    # ---- 8. serving artifact + routing overhead
    front = M.frontier(test_scores[HEADLINE], data_test, 101)
    meta = {
        "kind": "linear", "score": "cost-adjusted P(needs strong); see router/models/cost_aware.py",
        "cost_alpha": float(fitted[HEADLINE].alpha), "embedder": cfg["features"]["embedder"],
        "max_seq_length": cfg["features"]["max_seq_length"], "use_handcrafted": True,
        "thresholds": {k: float(v) for k, v in id_res[HEADLINE]["thresholds"].items()},
        "frontier": [{"frac_strong": float(a), "rel_quality": float(b), "rel_cost": float(c)}
                     for a, b, c in zip(front["frac_strong"], front["rel_quality"], front["rel_cost"])],
        "trained_on": f"RouterBench 0-shot, ID train split (n={len(tr)}), seed {cfg['split']['seed']}",
        "cheap_model": cheap_name, "strong_model": strong_name,
        "created": dt.date.today().isoformat(),
    }
    art = LinearRouterArtifact.from_router(fitted[HEADLINE], meta)
    art.save(art_dir / "router_lr")
    np.testing.assert_allclose(art.predict_proba(feats["EH"][te[:200]]), test_scores[HEADLINE][:200], atol=1e-5)
    log("saved artifacts/router_lr")

    from router.serve.engine import RouterEngine
    engine = RouterEngine(str(art_dir / "router_lr"))
    rng = np.random.default_rng(seed)
    lat_idx = rng.choice(te, size=300, replace=False)
    engine_p = np.array([engine.predict(prompts[i])[0] for i in lat_idx[:50]])
    offline_p = test_scores[HEADLINE][[pos_in_test[int(i)] for i in lat_idx[:50]]]
    res["latency"] = measure_overhead(lambda p: engine.predict(p), [prompts[i] for i in lat_idx], n=300, warmup=10)
    res["latency"]["max_abs_diff_engine_vs_offline"] = float(np.max(np.abs(engine_p - offline_p)))
    res["latency"]["note"] = "single-query embed + features + classify, CPU, in-process"
    log(f"routing overhead p50={res['latency']['p50_ms']:.1f} ms p95={res['latency']['p95_ms']:.1f} ms; "
        f"engine/offline max |dp|={res['latency']['max_abs_diff_engine_vs_offline']:.2e}")

    # ---- 9. robustness to surface perturbations (subset of the ID test set)
    if not args.quick:
        log("robustness: typos / lowercase / filler on 1,500 test prompts")
        sub = np.sort(rng.choice(len(te), size=min(1500, len(te)), replace=False))
        d_sub = data_test.subset(sub)
        clean = [prompts[te[j]] for j in sub]
        s_clean = test_scores[HEADLINE][sub]
        op_c = M.operating_point(M.route_at_threshold(s_clean, t95), d_sub)
        rob = {"n": int(len(sub)), "clean_apgr": M.apgr(s_clean, d_sub), "perturbations": {}}
        for pname, fn in (("typos_5pct", lambda x: perturb_typos(x, 0.05, seed)),
                          ("lowercase", lambda x: perturb_case(x, seed)),
                          ("polite_filler", lambda x: perturb_filler(x, seed))):
            pert = fn(clean)
            x = combine_features(cached_embeddings(routing_views(pert), embedder, cache_dir),
                                 handcrafted_features(pert))
            s = fitted[HEADLINE].score(x)
            op_p = M.operating_point(M.route_at_threshold(s, t95), d_sub)
            rob["perturbations"][pname] = {
                "apgr": M.apgr(s, d_sub),
                "route_flip_rate_at_rel95": float(np.mean((s >= t95) != (s_clean >= t95))),
                "mean_abs_dp": float(np.mean(np.abs(s - s_clean))),
                "rel_quality_clean": op_c["rel_quality"], "rel_quality_perturbed": op_p["rel_quality"],
                "cost_saving_clean": op_c["cost_saving"], "cost_saving_perturbed": op_p["cost_saving"],
            }
        res["robustness"] = rob

    # ---- 10. stretch: budget-constrained online policy (simulated on the ID test stream)
    from router.serve.policy import BudgetPolicy
    stream = np.random.default_rng(seed).permutation(len(te))
    bud = []
    for target in (0.2, 0.4, 0.6):
        pol = BudgetPolicy(target_frac=target, window=500,
                           init_threshold=float(np.quantile(val_scores[HEADLINE], 1 - target)))
        route = np.zeros(len(te), dtype=bool)
        for j in stream:
            route[j] = pol.decide(float(test_scores[HEADLINE][j]))
        op = M.operating_point(route, data_test)
        k = int(round(op["frac_strong"] * len(te)))
        hind = np.zeros(len(te), dtype=bool)
        hind[np.argsort(-test_scores[HEADLINE], kind="stable")[:k]] = True
        op_h = M.operating_point(hind, data_test)
        bud.append({"target_frac_strong": target, "realised_frac_strong": op["frac_strong"],
                    "rel_quality": op["rel_quality"], "cost_saving": op["cost_saving"],
                    "lift_vs_random": op["lift_vs_random"],
                    "hindsight_rel_quality_same_frac": op_h["rel_quality"]})
    res["budget_policy"] = bud

    # ---- 11. other model pairs (same prompts, features and splits; different labels)
    alt = {}
    raw_path = ROOT / cfg["data"]["raw_dir"] / "routerbench_0shot.parquet"
    for c2, s2 in (("mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"), ("claude-instant-v1", "claude-v2")):
        d2 = add_labels(build_pair_table(raw_path, c2, s2), cfg["labels"]["epsilon"])
        d2 = d2.set_index("sample_id").loc[df["sample_id"]].reset_index()  # align rows to the main table
        da = M.EvalData.from_frame(d2)
        g2 = d2["gap"].to_numpy()
        x2 = (d2["strong_cost"] - d2["cheap_cost"]).to_numpy()
        r2 = fit_cost_aware(feats["EH"][tr], g2[tr], x2[tr], feats["EH"][va], da.subset(va))
        e2 = evaluate_scores(r2.score(feats["EH"][va]), r2.score(feats["EH"][te]), da.subset(va), da.subset(te),
                             n_boot, seed, budget)
        alt[f"{c2} -> {s2}"] = {"always_cheap": B.always_cheap(da.subset(te)),
                                "always_strong": B.always_strong(da.subset(te)), "cost_alpha": r2.alpha, **e2}
        log(f"alt pair {c2} -> {s2}: AUC {fmt(e2['frontier']['auc_cost_quality'])}")
    res["alt_pairs"] = alt

    # ---- 12. figures
    log("figures")
    from router.eval.plots import plot_frontier, plot_per_task, plot_reliability

    def panel(title, router_scores, op_ci, op_route):
        label = (f"threshold frozen on validation (95% target):\n"
                 f"{op_ci['cost_saving']['point'] * 100:.0f}% cheaper than always-strong\n"
                 f"at {op_ci['rel_quality']['point'] * 100:.1f}% of its quality\n"
                 f"({op_ci['frac_cheap']['point'] * 100:.0f}% of traffic to the cheap model)")
        return {"title": title, "data": data_test, "router_name": HEADLINE,
                "router": M.frontier(router_scores, data_test, 101),
                "band": frontier_band(np.asarray(router_scores), data_test, 200, seed),
                "oracle": M.frontier(test_scores[ORACLE], data_test, 101),
                "heuristic": M.frontier(test_scores[HEUR], data_test, 101),
                "cheap": cheap_name, "strong": strong_name,
                "op": M.operating_point(op_route, data_test), "op_label": label}

    headline_figure([
        panel(f"In-distribution: random held-out test split (n={len(te):,})", test_scores[HEADLINE],
              id_res[HEADLINE]["deployed"]["rel95"], id_route95),
        panel("Out-of-distribution: each task family held out of training\n(same test prompts)",
              ood_scores[HEADLINE], ood_res[HEADLINE]["deployed"]["rel95"], ood_route[HEADLINE]["rel95"]),
    ], assets / "frontier.png")

    curves = {n: M.frontier(s, data_test, 101) for n, s in test_scores.items()
              if n not in ("Random", NO_BOTH_FAIL, ORACLE_GAP)}
    curves["Oracle"] = curves.pop(ORACLE)
    plot_frontier(curves, data_test, assets / "frontier_all_routers.png",
                  "All routers, in-distribution test split", x="cost")
    plot_reliability(reliability_curve(test_scores[PLAIN], y_test, 10), cal["id"]["ece"],
                     assets / "reliability_id.png", "Calibration of P(needs strong), ID test")
    plot_reliability(reliability_curve(np.clip(ood_scores[PLAIN], 0, 1), y_test, 10), cal["ood"]["ece"],
                     assets / "reliability_ood.png", "Calibration of P(needs strong), OOD (LOFO)")
    pt = pd.DataFrame(per_family)
    plot_per_task(pt.rename(columns={"router_id": "router", "frac_strong_id": "frac_strong"})[
        ["task_family", "cheap", "strong", "router", "frac_strong"]], assets / "per_task.png")

    # ---- 13. write results
    (results_dir / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    tables = render_tables(res)
    (results_dir / "tables.md").write_text(tables["all"], encoding="utf-8")
    if not args.no_readme and not args.quick:
        inject_readme(tables["readme"])
    log("done: results/results.json, results/tables.md, assets/frontier.png")
    print("\n" + tables["headline"])


# ----------------------------------------------------------------------------- reporting
def render_tables(res: dict) -> dict:
    m = res["meta"]
    idr, ood = res["id"], res["ood"]
    h_id, h_ood = idr[HEADLINE], ood[HEADLINE]
    ds = res["dataset"]["test"]

    def op_row(label, split_label, o):
        return [label, split_label, fmt(o["rel_quality"], True), fmt(o["cost_saving"], True),
                fmt(o["frac_cheap"], True), fmt_pts(o["lift_vs_random"])]

    head = md_table(
        ["Target (set on validation)", "Split", "Quality retained vs always-strong",
         "Cost saving vs always-strong", "Traffic to cheap model",
         "Quality lift vs random routing at the same cost (points)"],
        [op_row("95% of strong quality", "ID test", h_id["deployed"]["rel95"]),
         op_row("95% of strong quality", "OOD (leave-one-family-out)", h_ood["deployed"]["rel95"]),
         op_row("90% of strong quality", "ID test", h_id["deployed"]["rel90"]),
         op_row("90% of strong quality", "OOD (leave-one-family-out)", h_ood["deployed"]["rel90"])])
    context = (
        f"Benchmark: {m['benchmark']}, pair `{m['cheap_model']}` (cheap) vs `{m['strong_model']}` (strong); "
        f"{m['n_total']:,} prompts, split {m['n_train']:,}/{m['n_val']:,}/{m['n_test']:,} train/val/test. "
        f"On the test split always-cheap scores {ds['always_cheap']['quality']:.3f} and always-strong "
        f"{ds['always_strong']['quality']:.3f} (cheap = {ds['cheap_rel_quality'] * 100:.1f}% of strong quality) "
        f"and the strong model costs {ds['cost_ratio_strong_over_cheap']:.1f}x more per query. "
        f"Brackets are 95% bootstrap CIs over test prompts ({m['n_boot']} resamples). "
        f"Router: {HEADLINE}. ID and OOD rows use the same {m['n_test']:,} test prompts."
    )

    def frontier_rows(block, names, split_label):
        rows = []
        for n in names:
            if n in block:
                f = block[n]["frontier"]
                rows.append([n, split_label, fmt(f["apgr"]), fmt(f["auc_cost_quality"]),
                             fmt(f["cpt50"], True), fmt(f["cpt80"], True),
                             fmt(f["sweep_cost_saving_at_rel95"], True),
                             fmt(f["sweep_cost_saving_at_rel90"], True),
                             fmt(f["rel_quality_at_half_budget"], True)])
        return rows

    fr_header = ["Router", "Split", "APGR (random = 0.5)", "AUC cost-quality", "CPT(50%)", "CPT(80%)",
                 "Cost saving @95% quality (test sweep)", "Cost saving @90% quality (test sweep)",
                 "Quality retained at 50% of strong-only cost"]
    order = ["Random", HEUR, "LR (handcrafted only)", "LR (embedding)", PLAIN, NO_BOTH_FAIL,
             "Gap regression, ridge (embedding + handcrafted)", "kNN, k=40 (embedding)",
             "MLP (embedding + handcrafted)", FINETUNED, COST_ONLY, HEADLINE, ORACLE_GAP, ORACLE]
    ablation = md_table(fr_header, frontier_rows(idr, order, "ID test"))
    ood_tab = md_table(fr_header, frontier_rows(ood, order, "OOD (LOFO)"))

    dep_rows = []
    for n in order:
        for block, lab in ((idr, "ID test"), (ood, "OOD (LOFO)")):
            if n in block and n not in ("Random", ORACLE, ORACLE_GAP):
                o = block[n]["deployed"]["rel95"]
                dep_rows.append([n, lab, fmt(o["rel_quality"], True), fmt(o["cost_saving"], True),
                                 fmt_pts(o["lift_vs_random"])])
    deployed = md_table(["Router", "Split", "Quality retained", "Cost saving",
                         "Lift vs random at same cost (points)"], dep_rows)

    fam = md_table(
        ["Task family", "n test", "Cheap", "Strong", "Within-family APGR, plain LR, ID",
         "Within-family APGR, plain LR, OOD", "Within-family APGR, heuristic",
         "Router quality ID", "To strong ID", "Router quality OOD", "To strong OOD"],
        [[r["task_family"], r["n_test"], f"{r['cheap']:.3f}", f"{r['strong']:.3f}", f"{r['apgr_plain_id']:.3f}",
          f"{r['apgr_plain_ood']:.3f}", f"{r['apgr_heuristic']:.3f}", f"{r['router_id']:.3f}",
          f"{r['frac_strong_id'] * 100:.0f}%", f"{r['router_ood']:.3f}", f"{r['frac_strong_ood'] * 100:.0f}%"]
         for r in res["per_family"]])

    wf = res["within_family_apgr"]
    fam += (f"\n\nWithin-family APGR ranks prompts inside one task family only (0.5 = random), so it measures "
            f"difficulty prediction rather than dataset identification. Test-size-weighted mean over "
            f"{', '.join(wf['families'])} (families with n >= 150 and a real quality gap), plain LR: "
            f"{wf['plain_id']:.3f} in distribution, {wf['plain_ood']:.3f} when the family was held out of "
            f"training; length/keyword heuristic: {wf['heuristic']:.3f}. The 'Router quality' and 'To strong' "
            f"columns are the headline cost-aware router at its 95% operating point.")
    cal, lat = res["calibration"], res["latency"]
    misc = "\n".join([
        f"- Calibration of P(needs strong), {PLAIN}: ECE {cal['id']['ece']:.3f} on ID test, "
        f"{cal['ood']['ece']:.3f} OOD (LOFO); base rate {cal['id']['base_rate']:.3f}.",
        f"- Routing overhead (embed + features + classify, single query, in-process, CPU "
        f"`{m['machine']}`): p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms over {lat['n']} "
        f"test prompts."])
    rob_tab = "_skipped (--quick)_"
    if "robustness" in res:
        rb = res["robustness"]
        rob_tab = (f"Clean APGR on the {rb['n']:,}-prompt subset: {rb['clean_apgr']:.3f}.\n\n" + md_table(
            ["Perturbation", "APGR", "Routes flipped at the 95% threshold", "Mean abs change in P",
             "Quality retained (clean -> perturbed)", "Cost saving (clean -> perturbed)"],
            [[k, f"{v['apgr']:.3f}", f"{v['route_flip_rate_at_rel95'] * 100:.1f}%", f"{v['mean_abs_dp']:.3f}",
              f"{v['rel_quality_clean'] * 100:.1f}% -> {v['rel_quality_perturbed'] * 100:.1f}%",
              f"{v['cost_saving_clean'] * 100:.1f}% -> {v['cost_saving_perturbed'] * 100:.1f}%"]
             for k, v in rb["perturbations"].items()]))
    bud = md_table(["Target share to strong", "Realised share", "Quality retained", "Cost saving",
                    "Lift vs random (points)", "Quality retained, hindsight threshold at same share"],
                   [[f"{b['target_frac_strong'] * 100:.0f}%", f"{b['realised_frac_strong'] * 100:.1f}%",
                     f"{b['rel_quality'] * 100:.1f}%", f"{b['cost_saving'] * 100:.1f}%",
                     f"{b['lift_vs_random'] * 100:+.2f}", f"{b['hindsight_rel_quality_same_frac'] * 100:.1f}%"]
                    for b in res["budget_policy"]])
    alt = md_table(["Pair (cheap -> strong)", "Cheap / strong quality (ID test)", "APGR (ID test)",
                    "Quality retained @95% target", "Cost saving @95% target", "Lift vs random (points)"],
                   [[k, f"{v['always_cheap']['quality']:.3f} / {v['always_strong']['quality']:.3f}",
                     fmt(v["frontier"]["apgr"]), fmt(v["deployed"]["rel95"]["rel_quality"], True),
                     fmt(v["deployed"]["rel95"]["cost_saving"], True),
                     fmt_pts(v["deployed"]["rel95"]["lift_vs_random"])]
                    for k, v in res["alt_pairs"].items()])

    o95, o95o = h_id["deployed"]["rel95"], h_ood["deployed"]["rel95"]
    bullets = (
        f"- Built a cost-aware LLM router (MiniLM sentence embedding + handcrafted features + logistic "
        f"regression with a predicted-cost term) retaining {o95['rel_quality']['point'] * 100:.1f}% of strong-model accuracy while routing "
        f"{o95['frac_cheap']['point'] * 100:.0f}% of traffic to a cheaper model, cutting cost "
        f"{o95['cost_saving']['point'] * 100:.0f}% on RouterBench ({m['cheap_model']} vs {m['strong_model']}, "
        f"in-distribution test split, n={m['n_test']:,}; {o95['lift_vs_random']['point'] * 100:+.1f} quality "
        f"points vs random routing at equal cost). Leave-one-task-family-out evaluation showed the gain does "
        f"not transfer to unseen task types ({o95o['rel_quality']['point'] * 100:.1f}% quality at a "
        f"{o95o['cost_saving']['point'] * 100:.0f}% cost cut, {o95o['lift_vs_random']['point'] * 100:+.1f} points "
        f"vs random), reported with bootstrap CIs.\n"
        f"- Deployed as an OpenAI-compatible FastAPI proxy with {lat['p95_ms']:.0f} ms p95 routing overhead "
        f"(CPU, in-process), request logging, and a cost/quality dashboard."
    )

    readme = "\n\n".join([
        f"_Generated by `python scripts/reproduce.py` on {m['generated']}; do not edit by hand._",
        context, "### Headline: deployed operating points", head,
        "### Routers and baselines: frontier metrics, in-distribution test split", ablation,
        "### Same metrics out of distribution (leave-one-task-family-out, pooled over the same test prompts)",
        ood_tab, "### Deployed 95% operating point for every router, ID vs OOD", deployed,
        "### Per task family (headline router, 95% target)", fam,
        "### Calibration and overhead", misc,
        "### Robustness to surface perturbations (ID test subset)", rob_tab,
        "### Stretch: budget-constrained online policy (simulated on the ID test stream)", bud,
        "### Other model pairs (same splits and features, ID test)", alt,
        "### Resume bullets (filled from the numbers above)", bullets,
    ])
    return {"headline": context + "\n\n" + head, "readme": readme, "all": "# Results\n\n" + readme + "\n"}


def inject_readme(block: str) -> None:
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in text and b in text:
        text = text[:text.index(a) + len(a)] + "\n" + block + "\n" + text[text.index(b):]
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
