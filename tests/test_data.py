import io
import pickle

import numpy as np
import pandas as pd
import pytest

from router.data.dataset import (PAIR_COLUMNS, _prompt_text, add_labels, build_pair_table,
                                 lofo_folds, make_id_split, task_family)
from router.data.download import RestrictedUnpickler
from router.data.synthetic import make_synthetic_pair_table
from router.features.text import routing_view


def _raw_frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "prompt": [str([f"question {i}", "Answer:"]) for i in range(n)],
        "eval_name": ["mmlu-anatomy", "hellaswag", "grade-school-math", "chinese_zodiac"] * (n // 4),
        "cheap": rng.integers(0, 2, n).astype(float),
        "strong": rng.integers(0, 2, n).astype(float),
        "cheap|total_cost": rng.random(n) * 1e-4,
        "strong|total_cost": rng.random(n) * 1e-3,
    })


def test_task_family_mapping():
    assert task_family("mmlu-college-physics") == "mmlu"
    assert task_family("grade-school-math") == "gsm8k"
    assert task_family("Chinese_character_riddles") == "chinese"
    assert task_family("mtbench-math") == "mtbench"
    assert task_family("consensus_summary") == "misc"


def test_prompt_text_parses_message_list():
    assert _prompt_text("['a', 'b']") == "a\nb"
    assert _prompt_text("plain [not a list") == "plain [not a list"


def test_build_pair_table_schema(tmp_path):
    p = tmp_path / "raw.parquet"
    _raw_frame().to_parquet(p)
    df = build_pair_table(p, "cheap", "strong")
    assert list(df.columns) == PAIR_COLUMNS
    assert df.prompt.iloc[0] == "question 0\nAnswer:"
    assert set(df.task_family) == {"mmlu", "hellaswag", "gsm8k", "chinese"}
    with pytest.raises(KeyError):
        build_pair_table(p, "missing-model", "strong")


def test_labels():
    df = pd.DataFrame({"cheap_score": [0.0, 1.0, 0.0, 0.5], "strong_score": [1.0, 1.0, 0.0, 0.75]})
    out = add_labels(df, epsilon=0.3)
    assert out.needs_strong.tolist() == [1, 0, 0, 0]
    assert out.both_fail.tolist() == [False, False, True, False]
    assert add_labels(df, 0.0).needs_strong.tolist() == [1, 0, 0, 1]


def test_id_split_is_stratified_deterministic_and_leak_free():
    df = make_synthetic_pair_table(400, seed=1)
    dup = pd.concat([df, df.iloc[:50]], ignore_index=True)  # exact duplicate prompts
    s1 = make_id_split(dup, seed=3)
    s2 = make_id_split(dup, seed=3)
    assert (s1 == s2).all()
    assert set(s1) == {"train", "val", "test"}
    assert dup.assign(split=s1).groupby(["task_family", "prompt"]).split.nunique().max() == 1
    frac = (s1 == "train").mean()
    assert 0.55 < frac < 0.65
    for _, g in dup.assign(split=s1).groupby("task_family"):
        assert set(g.split) == {"train", "val", "test"}


def test_lofo_folds_hold_out_whole_family():
    df = make_synthetic_pair_table(400, seed=2)
    fams = df.task_family.to_numpy()
    seen = []
    for f, tr, va, te in lofo_folds(df):
        seen.append(f)
        assert f not in set(fams[tr]) and f not in set(fams[va])
        assert set(fams[te]) == {f}
        assert set(df.split.to_numpy()[te]) == {"test"}
    assert sorted(seen) == sorted(set(fams))
    pooled = np.concatenate([te for _, _, _, te in lofo_folds(df)])
    assert sorted(pooled) == sorted(np.flatnonzero(df.split.to_numpy() == "test"))


def test_restricted_unpickler_blocks_code_execution():
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned",))

    with pytest.raises(pickle.UnpicklingError):
        RestrictedUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    out = RestrictedUnpickler(io.BytesIO(pickle.dumps(df))).load()
    assert out.equals(df)


def test_routing_view_keeps_tail():
    long = "PREFIX " * 200 + "the real question?"
    v = routing_view(long)
    assert v.endswith("the real question?") and len(v) < len(long)
    assert routing_view("short") == "short"
