import time

import numpy as np

from router.features import (FEATURE_NAMES, Featurizer, HashingEmbedder, cached_embeddings,
                             combine_features, get_embedder, handcrafted_features)

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def test_handcrafted_shape_dtype_finite():
    X = handcrafted_features(["hello world", "What is 2+2?", "x" * 5000])
    assert X.shape == (3, len(FEATURE_NAMES))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_handcrafted_deterministic():
    p = ["Prove that sqrt(2) is irrational.", "def f(x):\n    return x"]
    assert np.array_equal(handcrafted_features(p), handcrafted_features(p))


def test_handcrafted_empty_inputs():
    X = handcrafted_features([""])
    assert X.shape == (1, len(FEATURE_NAMES)) and np.isfinite(X).all()
    assert handcrafted_features([]).shape == (0, len(FEATURE_NAMES))


def test_handcrafted_code_and_math_flags():
    code = handcrafted_features(["```python\ndef add(a, b):\n    return a + b\n```"])[0]
    plain = handcrafted_features(["tell me a joke"])[0]
    assert code[IDX["has_code_fence"]] == 1 and code[IDX["code_markers"]] > plain[IDX["code_markers"]]
    math_ = handcrafted_features(["Solve the equation 3x + 5 = 20 and prove it."])[0]
    assert math_[IDX["math_keyword"]] == 1 and math_[IDX["math_ops"]] >= 2
    assert plain[IDX["math_keyword"]] == 0


def test_handcrafted_mc_question_flags():
    x = handcrafted_features(["Why is the sky blue?\nA. rayleigh\nB. mie\nOptions above"])[0]
    assert x[IDX["multiple_choice"]] == 1 and x[IDX["q_why"]] == 1 and x[IDX["question_marks"]] == 1


def test_handcrafted_fast():
    prompts = ["Write a function to compute the LCS of two strings, step by step. " * 3] * 500
    t = time.perf_counter()
    handcrafted_features(prompts)
    assert (time.perf_counter() - t) / 500 < 1e-3  # generous bound; target is < 0.1 ms


def test_hashing_embedder_shape_norm_determinism():
    e = HashingEmbedder()
    a = e.encode(["hello world", "another prompt entirely"])
    assert a.shape == (2, 256) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, e.encode(["hello world", "another prompt entirely"]))
    assert e.name == "hashing" and e.dim == 256


def test_get_embedder_dispatch():
    from router.features import Embedder
    assert isinstance(get_embedder("hashing"), HashingEmbedder)
    e = get_embedder("some/model", 64)
    assert isinstance(e, Embedder) and e.name == "some/model" and e.max_seq_length == 64
    assert e._model is None  # lazy, no model loaded


def test_cached_embeddings_writes_then_reuses(tmp_path):
    class Counting:
        def __init__(self):
            self.inner, self.calls = HashingEmbedder(), 0
            self.name, self.max_seq_length = self.inner.name, 0

        def encode(self, prompts, batch_size=64, show_progress=False):
            self.calls += 1
            return self.inner.encode(prompts)

    e = Counting()
    prompts = ["a b c", "d e f", "g h i"]
    first = cached_embeddings(prompts, e, tmp_path / "cache")
    assert e.calls == 1 and len(list((tmp_path / "cache").glob("emb_*.npy"))) == 1
    second = cached_embeddings(prompts, e, tmp_path / "cache")
    assert e.calls == 1 and np.array_equal(first, second)
    cached_embeddings(prompts + ["new"], e, tmp_path / "cache")
    assert e.calls == 2


def test_combine_features_and_featurizer():
    emb = np.ones((4, 5), dtype=np.float32)
    hand = np.zeros((4, 3), dtype=np.float32)
    out = combine_features(emb, hand)
    assert out.shape == (4, 8) and out[:, :5].all() and not out[:, 5:].any()
    assert combine_features(emb, None).shape == (4, 5)
    assert combine_features(None, hand).shape == (4, 3)
    prompts = ["one", "two words", "three"]
    f = Featurizer(HashingEmbedder(), True)
    assert f.transform(prompts).shape == (3, 256 + len(FEATURE_NAMES))
    assert Featurizer(HashingEmbedder(), False).transform(prompts).shape == (3, 256)
