"""Matplotlib figures (Agg backend, colourblind-safe palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from router.eval.baselines import always_cheap, always_strong  # noqa: E402
from router.eval.metrics import EvalData  # noqa: E402

# Okabe-Ito
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
DPI = 160


def _style():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.fontsize": 9, "legend.frameon": False,
    })


def _save(fig, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(curves: dict, data: EvalData, out_path, title: str, x: str = "cost",
                  bands: dict | None = None) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cheap, strong = always_cheap(data), always_strong(data)
    sx = 1000.0 if x == "cost" else 1.0

    def xv(df):
        return df["cost"].to_numpy() * sx if x == "cost" else df["frac_strong"].to_numpy()

    x0 = cheap["cost"] * sx if x == "cost" else 0.0
    x1 = strong["cost"] * sx if x == "cost" else 1.0
    ax.plot([x0, x1], [cheap["quality"], strong["quality"]], ls="--", color="grey",
            lw=1.5, label="Random mixing")
    for frac, lab in ((0.95, "95% of always-strong"), (0.90, "90% of always-strong")):
        y = frac * strong["quality"]
        ax.axhline(y, ls=":", color="grey", lw=1)
        ax.text(0.01, y, lab, fontsize=8, color="grey", va="bottom",
                transform=ax.get_yaxis_transform())

    ci = 0
    for name, df in curves.items():
        if name == "Oracle":
            kw = dict(color="black", ls="--", lw=1.8)
        else:
            kw = dict(color=PALETTE[ci % len(PALETTE)], lw=2.0)
            ci += 1
        ax.plot(xv(df), df["quality"], label=name, **kw)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(xv(df), lo, hi, color=kw["color"], alpha=0.15, lw=0)

    ax.scatter([x0], [cheap["quality"]], marker="s", s=60, color="#444444", zorder=5)
    ax.annotate("Always cheap", (x0, cheap["quality"]), textcoords="offset points",
                xytext=(8, -12), fontsize=9)
    ax.scatter([x1], [strong["quality"]], marker="*", s=130, color="#D55E00", zorder=5)
    ax.annotate("Always strong", (x1, strong["quality"]), textcoords="offset points",
                xytext=(-8, -16), fontsize=9, ha="right")

    ax.set_xlabel("Cost per 1k queries (USD)" if x == "cost" else "Fraction routed to strong model")
    ax.set_ylabel("Quality (mean score)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, out_path)


def plot_reliability(curve_df, ece: float, out_path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Perfect calibration")
    ax.plot(curve_df["mean_pred"], curve_df["frac_pos"], marker="o", color=PALETTE[0],
            lw=2, label=f"Router (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(needs strong)")
    ax.set_ylabel("Observed fraction needing strong")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, out_path)


def plot_per_task(table, out_path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    n = len(table)
    xs = np.arange(n)
    w = 0.26
    ax.bar(xs - w, table["cheap"], w, label="Always cheap", color=PALETTE[5])
    ax.bar(xs, table["router"], w, label="Router", color=PALETTE[0])
    ax.bar(xs + w, table["strong"], w, label="Always strong", color=PALETTE[1])
    labels = list(table["task_family"])
    if "frac_strong" in table.columns:
        labels = [f"{t}\n({f:.0%} strong)" for t, f in zip(table["task_family"], table["frac_strong"])]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Quality (mean score)")
    ax.set_title("Quality by task family")
    ax.legend(loc="lower right", ncol=3)
    _save(fig, out_path)
