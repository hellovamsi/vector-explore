from __future__ import annotations

from pathlib import Path

import pytest

from vector_explore.paths import chunk_dir, embed_key, embedding_dir, index_key, novel_root, query_dir, store_index_dir


def test_new_layout_helpers():
    root = Path("/tmp/proj")
    nr = novel_root(root, "mobydick")
    assert nr == root / "runs" / "mobydick"
    ekey = embed_key("ollama", "qwen3-embedding:0.6b")
    assert ekey.startswith("ollama-")
    assert chunk_dir(root, novel_slug="mobydick", chunk_method="semantic") == root / "runs" / "mobydick" / "semantic"
    assert embedding_dir(root, novel_slug="mobydick", embed_key=ekey, chunk_method="semantic") == root / "runs" / "mobydick" / ekey / "semantic"
    idx = index_key("semantic", ekey)
    assert store_index_dir(root, novel_slug="mobydick", store_name="lancedb", index_key=idx) == root / "runs" / "mobydick" / "lancedb" / idx
    assert query_dir(root, novel_slug="mobydick", query_hash="abcd") == root / "runs" / "mobydick" / "queries" / "abcd"


def test_reserved_name_rejected():
    with pytest.raises(ValueError):
        novel_root(Path("/tmp/proj"), "src")
