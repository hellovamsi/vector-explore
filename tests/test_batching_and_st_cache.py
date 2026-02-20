from __future__ import annotations

import sys
import types

from vector_explore.batching import batched_embed_texts
from vector_explore.embeddings.st_embed import load_st_model


class _StubEmbedder:
    def info(self):
        return type("Info", (), {"backend": "stub", "model": "stub", "dim": 1})

    def embed_texts(self, texts):
        return [[float(len(t))] for t in texts]


def test_batched_embed_texts_multiple_batches():
    embedder = _StubEmbedder()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = batched_embed_texts(embedder=embedder, texts=texts, batch_size=2, progress_enabled=False, description="x")
    assert len(out) == len(texts)
    assert out[0] == [1.0]
    assert out[-1] == [5.0]


def test_st_loader_cached(monkeypatch):
    calls = {"n": 0}

    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            calls["n"] += 1
            self.model_name = model_name
            self.device = device

        def encode(self, texts, **_kwargs):
            return []

    fake_mod = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    load_st_model.cache_clear()
    a = load_st_model("m1", None)
    b = load_st_model("m1", None)
    c = load_st_model("m2", None)
    assert a is b
    assert a is not c
    assert calls["n"] == 2
