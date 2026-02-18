from __future__ import annotations

import numpy as np

from vector_explore.chunkers.fixed import FixedChunkParams, chunk_fixed
from vector_explore.chunkers.sentences import SentenceChunkParams, chunk_sentences
from vector_explore.chunkers.semantic import SemanticChunkParams, chunk_semantic


class StubEmbedder:
    def info(self):
        return type("Info", (), {"backend": "stub", "model": "stub", "dim": 2})

    def embed_texts(self, texts):
        # Deterministic vector: [len, vowel_count]
        out = []
        for t in texts:
            v = float(len(t))
            vowel = float(sum(1 for ch in t.lower() if ch in "aeiou"))
            out.append([v, vowel])
        return out


def test_fixed_chunking_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    chunks = chunk_fixed(text, FixedChunkParams(target_chars=50, overlap_ratio=0.2))
    assert len(chunks) >= 2
    assert chunks[0]["overlap"] is None
    assert chunks[1]["overlap"]["type"] == "chars"


def test_sentence_chunking_basic():
    text = "Hello world. This is a test. Another sentence!"
    chunks = chunk_sentences(text, SentenceChunkParams(target_chars=20, overlap_sentences=1))
    assert len(chunks) >= 2
    assert chunks[1]["overlap"]["type"] == "sentences"


def test_semantic_chunking_runs():
    text = "One short sentence. Another short sentence. Totally different topic about whales. More whales."
    chunks = chunk_semantic(text, SemanticChunkParams(target_chars=60, overlap_sentences=1, similarity_drop_threshold=0.999), StubEmbedder())
    assert len(chunks) >= 2
    assert "boundary_reason" in chunks[0]["meta"]

