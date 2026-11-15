"""Streamlit dashboard over the router's SQLite request log.

Run with::  streamlit run router/dashboards/app.py
DB path: env ROUTER_DB (default: serve.db_path from configs/default.yaml). Opened read-only.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from router.config import load_config  # noqa: E402


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_requests(db_path: Path) -> pd.DataFrame:
    """Read the log read-only; returns an empty frame if the DB or table is missing."""
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query("SELECT * FROM requests WHERE status = 'ok' ORDER BY id", conn)
        finally:
            conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df


def load_artifact_meta(stem: Path) -> dict:
    try:
        return json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def estimated_quality(frontier: list[dict], frac_strong: float) -> float | None:
    """Interpolate rel_quality at the observed strong share along the offline frontier."""
    if not frontier:
        return None
    pts = sorted(((p["frac_strong"], p["rel_quality"]) for p in frontier))
    xs, ys = zip(*pts)
    return float(np.interp(frac_strong, xs, ys))


def main() -> None:
    st.set_page_config(page_title="Cost-aware router", layout="wide")
    st.title("Cost-aware router dashboard")

    cfg = load_config()["serve"]
    db_path = _resolve(os.environ.get("ROUTER_DB", cfg["db_path"]))
    meta = load_artifact_meta(_resolve(os.environ.get("ROUTER_ARTIFACT", cfg["artifact"])))
    st.sidebar.caption(f"DB: {db_path}")
    if st.sidebar.button("Refresh"):
        st.rerun()

    df = load_requests(db_path)
    if df.empty:
        st.info("No requests logged yet (or the database does not exist). "
                "Send traffic to the proxy or run `python scripts/demo.py`, then refresh.")
        return

    n = len(df)
    frac_strong = float((df["route"] == "strong").mean())
    cost, always = float(df["cost_usd"].sum()), float(df["strong_cost_usd"].sum())
    saving = (1 - cost / always) if always > 0 else 0.0
    oh50, oh95 = np.percentile(df["overhead_ms"], [50, 95])

    c = st.columns(6)
    c[0].metric("Requests", f"{n:,}")
    c[1].metric("Routed to cheap", f"{(1 - frac_strong) * 100:.1f}%")
    c[2].metric("Actual cost", f"${cost:.4f}")
    c[3].metric("Always-strong cost", f"${always:.4f}")
    c[4].metric("Saving", f"{saving * 100:.1f}%")
    c[5].metric("Routing overhead p50 / p95", f"{oh50:.1f} / {oh95:.1f} ms")

    est = estimated_quality(meta.get("frontier", []), frac_strong)
    if est is not None:
        st.metric("Estimated quality (vs always-strong)", f"{est * 100:.1f}%")
        st.caption("Offline estimate from the benchmark frontier, not measured on live traffic.")

    st.subheader("Route mix over time")
    bucket = max(1, n // 20)
    mix = (
        df.assign(bucket=np.arange(n) // bucket, strong=(df["route"] == "strong").astype(float))
        .groupby("bucket")["strong"].mean().rename("share routed to strong")
    )
    st.line_chart(mix)
    st.caption(f"Each point averages {bucket} consecutive requests.")

    st.subheader("Cumulative cost: actual vs always-strong")
    st.line_chart(pd.DataFrame({
        "actual": df["cost_usd"].cumsum().to_numpy(),
        "always strong": df["strong_cost_usd"].cumsum().to_numpy(),
    }))

    left, right = st.columns(2)
    with left:
        st.subheader("Latency per route (ms)")
        stats = df.groupby("route")["latency_ms"].describe(percentiles=[0.5, 0.95])[["count", "mean", "50%", "95%"]]
        st.dataframe(stats)
        bins = np.histogram_bin_edges(df["latency_ms"], bins=20)
        hist = pd.DataFrame({
            r: np.histogram(g["latency_ms"], bins=bins)[0] for r, g in df.groupby("route")
        }, index=np.round(bins[:-1], 1))
        st.bar_chart(hist)
    with right:
        st.subheader("Histogram of p_strong")
        thr = float(df["threshold"].iloc[-1])
        counts, edges = np.histogram(df["p_strong"], bins=20, range=(0.0, 1.0))
        st.bar_chart(pd.DataFrame({"count": counts}, index=np.round(edges[:-1], 2)))
        st.caption(f"Current threshold: {thr:.3f} (route to strong when p_strong >= threshold). "
                   f"Requests above it: {(df['p_strong'] >= thr).mean() * 100:.1f}%.")


main()
