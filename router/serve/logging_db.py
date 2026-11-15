"""SQLite request log. Stores only a sha256 of the prompt, never the raw text."""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    query_hash TEXT,
    route TEXT,
    model TEXT,
    p_strong REAL,
    threshold REAL,
    policy TEXT,
    overhead_ms REAL,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    strong_cost_usd REAL,
    status TEXT
)
"""
COLUMNS = (
    "ts", "query_hash", "route", "model", "p_strong", "threshold", "policy", "overhead_ms",
    "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd", "strong_cost_usd", "status",
)


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pcts(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


class RequestLog:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(SCHEMA)
            self._conn.commit()

    def log(self, **row: Any) -> None:
        row.setdefault("ts", time.time())
        row.setdefault("status", "ok")
        unknown = set(row) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown request-log fields: {sorted(unknown)}")
        values = [row.get(c) for c in COLUMNS]
        with self._lock:
            self._conn.execute(
                f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                values,
            )
            self._conn.commit()

    def rows(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM requests ORDER BY id DESC"
        args: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            args = (int(limit),)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def summary(self) -> dict:
        with self._lock:
            rows = [dict(r) for r in self._conn.execute("SELECT * FROM requests WHERE status = 'ok'").fetchall()]
        n = len(rows)
        n_strong = sum(1 for r in rows if r["route"] == "strong")
        total = float(sum(r["cost_usd"] or 0.0 for r in rows))
        always = float(sum(r["strong_cost_usd"] or 0.0 for r in rows))
        oh50, oh95 = _pcts([r["overhead_ms"] for r in rows if r["overhead_ms"] is not None])
        la50, la95 = _pcts([r["latency_ms"] for r in rows if r["latency_ms"] is not None])
        return {
            "n_requests": n,
            "n_strong": n_strong,
            "frac_strong": (n_strong / n) if n else 0.0,
            "total_cost_usd": total,
            "always_strong_cost_usd": always,
            "cost_saving_frac": (1.0 - total / always) if always > 0 else 0.0,
            "overhead_ms_p50": oh50,
            "overhead_ms_p95": oh95,
            "latency_ms_p50": la50,
            "latency_ms_p95": la95,
            "total_prompt_tokens": int(sum(r["prompt_tokens"] or 0 for r in rows)),
            "total_completion_tokens": int(sum(r["completion_tokens"] or 0 for r in rows)),
            "total_tokens": int(sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows)),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
