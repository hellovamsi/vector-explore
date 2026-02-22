from __future__ import annotations

import os
import time
from pathlib import Path

from vector_explore.paths import write_json
from vector_explore.runs_inspect import discover_indexed_runs


def test_discover_indexed_runs_filters_and_sorts(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    base = time.time() - 2000

    older = _make_valid_run(
        runs_dir=runs_dir,
        novel_slug="frankenstein",
        chunk_method="fixed",
        embed_backend="st",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        store="lancedb",
        table="chunks",
        namespace=None,
        vector_dim=384,
        count=42,
        ts=base + 100,
    )
    _make_query_artifact(older, "q_old", ts=base + 110)

    newer = _make_valid_run(
        runs_dir=runs_dir,
        novel_slug="mobydick",
        chunk_method="semantic",
        embed_backend="ollama",
        embed_model="nomic-embed-text",
        store="lancedb",
        table="chunks",
        namespace=None,
        vector_dim=768,
        count=70,
        ts=base + 200,
    )
    _make_query_artifact(newer, "q_new", ts=base + 500)

    # Invalid: missing embed_params.json
    bad_no_embed = runs_dir / "pride" / "sentences" / "st-model-x" / "lancedb"
    bad_no_embed.mkdir(parents=True, exist_ok=True)
    write_json(bad_no_embed / "store_params.json", {"params": {"store": "lancedb", "table_or_collection": "chunks"}})
    write_json(bad_no_embed / "store_meta.json", {"vector_dim": 384, "count": 9})
    (bad_no_embed / "lancedb").mkdir(parents=True, exist_ok=True)

    # Invalid: missing store_meta.json
    bad_no_meta = runs_dir / "pride" / "fixed" / "ollama-test" / "lancedb"
    bad_no_meta.mkdir(parents=True, exist_ok=True)
    write_json(bad_no_meta / "store_params.json", {"params": {"store": "lancedb", "table_or_collection": "chunks"}})
    write_json(bad_no_meta.parent / "embed_params.json", {"params": {"backend": "ollama", "model": "test"}})
    (bad_no_meta / "lancedb").mkdir(parents=True, exist_ok=True)

    # Invalid: lancedb missing lancedb/ folder
    bad_lancedb_missing_dir = runs_dir / "pride" / "fixed" / "st-model-z" / "lancedb"
    bad_lancedb_missing_dir.mkdir(parents=True, exist_ok=True)
    write_json(bad_lancedb_missing_dir / "store_params.json", {"params": {"store": "lancedb", "table_or_collection": "chunks"}})
    write_json(bad_lancedb_missing_dir / "store_meta.json", {"vector_dim": 384, "count": 8})
    write_json(bad_lancedb_missing_dir.parent / "embed_params.json", {"params": {"backend": "st", "model": "model-z"}})

    found = discover_indexed_runs(runs_dir)
    assert len(found) == 2
    assert found[0].novel_slug == "mobydick"
    assert found[1].novel_slug == "frankenstein"
    assert found[0].queries_count == 1
    assert found[0].last_activity_ts > found[1].last_activity_ts


def test_discover_indexed_runs_limit(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    base = time.time() - 3000
    for i in range(12):
        store_dir = _make_valid_run(
            runs_dir=runs_dir,
            novel_slug=f"novel{i}",
            chunk_method="fixed",
            embed_backend="st",
            embed_model=f"model-{i}",
            store="lancedb",
            table="chunks",
            namespace=None,
            vector_dim=3,
            count=i,
            ts=base + i,
        )
        _make_query_artifact(store_dir, f"q{i}", ts=base + i + 0.5)

    found = discover_indexed_runs(runs_dir, limit=10)
    assert len(found) == 10
    assert found[0].novel_slug == "novel11"
    assert found[-1].novel_slug == "novel2"


def _make_valid_run(
    *,
    runs_dir: Path,
    novel_slug: str,
    chunk_method: str,
    embed_backend: str,
    embed_model: str,
    store: str,
    table: str,
    namespace: str | None,
    vector_dim: int,
    count: int,
    ts: float,
) -> Path:
    store_dir = runs_dir / novel_slug / chunk_method / f"{embed_backend}-{embed_model.replace('/', '_')}" / store
    store_dir.mkdir(parents=True, exist_ok=True)
    write_json(store_dir.parent / "embed_params.json", {"params": {"backend": embed_backend, "model": embed_model}})
    write_json(store_dir.parent.parent / "chunk_params.json", {"params": {"target_chars": 900, "overlap_sentences": 2}})
    write_json(store_dir / "store_params.json", {"params": {"store": store, "table_or_collection": table, "namespace": namespace}})
    write_json(store_dir / "store_meta.json", {"vector_dim": vector_dim, "count": count, "store": store})
    if store == "lancedb":
        (store_dir / "lancedb").mkdir(parents=True, exist_ok=True)

    for path in [
        store_dir.parent / "embed_params.json",
        store_dir.parent.parent / "chunk_params.json",
        store_dir / "store_params.json",
        store_dir / "store_meta.json",
    ]:
        os.utime(path, (ts, ts))
    return store_dir


def _make_query_artifact(store_dir: Path, query_id: str, *, ts: float) -> None:
    qdir = store_dir / "queries" / query_id
    qdir.mkdir(parents=True, exist_ok=True)
    answer_path = qdir / "answer.md"
    answer_path.write_text("answer\n", encoding="utf-8")
    os.utime(qdir, (ts, ts))
    os.utime(answer_path, (ts, ts))
