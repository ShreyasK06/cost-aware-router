"""Download RouterBench from the Hugging Face Hub (no token needed) and convert it to parquet.

RouterBench ships as pandas pickles. Unpickling arbitrary files can execute code, so we load
with a restricted unpickler that only resolves classes from pandas / numpy plus a tiny
allowlist, then immediately re-save as parquet. Everything downstream reads the parquet only.
"""
from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import pandas as pd

REPO_ID = "withmartian/routerbench"
FILENAME = "routerbench_0shot.pkl"

_ALLOWED_PREFIXES = ("pandas.", "numpy.")
_ALLOWED_EXACT = {
    ("builtins", "slice"),
    ("builtins", "range"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "complex"),
    ("builtins", "object"),
    ("copyreg", "_reconstructor"),
    ("_codecs", "encode"),
    ("collections", "OrderedDict"),
    ("datetime", "datetime"),
    ("datetime", "date"),
    ("datetime", "timedelta"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses anything outside pandas/numpy and a tiny allowlist."""

    def find_class(self, module: str, name: str):
        if module in ("pandas", "numpy") or module.startswith(_ALLOWED_PREFIXES):
            return super().find_class(module, name)
        if (module, name) in _ALLOWED_EXACT:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global during unpickle: {module}.{name}")


def restricted_read_pickle(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        obj = RestrictedUnpickler(io.BytesIO(fh.read())).load()
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"expected a DataFrame in {path}, got {type(obj)!r}")
    return obj


def download_routerbench(raw_dir: str | Path = "data/raw", force: bool = False) -> Path:
    """Download the 0-shot RouterBench table and return the path of the parquet copy.

    Model score / cost columns are kept; the (large) per-model response text columns are dropped.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "routerbench_0shot.parquet"
    if out.exists() and not force:
        return out

    from huggingface_hub import hf_hub_download

    pkl_dir = raw_dir / "hf_untrusted"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset", local_dir=str(pkl_dir))
    df = restricted_read_pickle(pkl)

    keep = [c for c in df.columns if not str(c).endswith("|model_response")]
    df = df[keep].copy()
    for c in df.columns:
        if df[c].dtype == object:
            # scores are occasionally stored as python objects; keep text columns as str
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() > 0.99:
                df[c] = num
            else:
                df[c] = df[c].astype(str)
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    p = download_routerbench(a.raw_dir, a.force)
    d = pd.read_parquet(p)
    print(p, d.shape)
    print(list(d.columns))
