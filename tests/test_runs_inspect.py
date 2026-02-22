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
    bad_no_embed = runs_dir / "pride" / "lancedb" / "sentences__st-model-x"
    bad_no_embed.mkdir(parents=True, exist_ok=True)
    write_json(bad_no_embed / "store_params.json", {"params": {"store": "lancedb", "table_or_collection": "chunks"}})
    write_json(bad_no_embed / "store_meta.json", {"vector_dim": 384, "count": 9})
    (bad_no_embed / "chunks.lance").mkdir(parents=True, exist_ok=True)

    # Invalid: missing store_meta.json
    bad_no_meta = runs_dir / "pride" / "lancedb" / "fixed__ollama-test"
    bad_no_meta.mkdir(parents=True, exist_ok=True)
    write_json(bad_no_meta / "store_params.json", {"params": {"store": "lancedb", "table_or_collection": "chunks"}})
    bad_no_meta_embed = runs_dir / "pride" / "ollama-test" / "fixed" / "embed_params.json"
    bad_no_meta_embed.parent.mkdir(parents=True, exist_ok=True)
    write_json(bad_no_meta_embed, {"params": {"backend": "ollama", "model": "test"}})
    (bad_no_meta / "chunks.lance").mkdir(parents=True, exist_ok=True)

    # Invalid: lancedb missing chunks.lance/ folder
    bad_lancedb_missing_dir = runs_dir / "pride" / "lancedb" / "fixed__st-model-z"
    bad_lancedb_missing_dir.mkdir(parents=True, exist_ok=True)
    write_json(bad_lancedb_missing_dir / "store_params.json", {"params": {"store": "lancedb", "table_or_collection": "chunks"}})
    write_json(bad_lancedb_missing_dir / "store_meta.json", {"vector_dim": 384, "count": 8})
    bad_lancedb_embed = runs_dir / "pride" / "st-model-z" / "fixed" / "embed_params.json"
    bad_lancedb_embed.parent.mkdir(parents=True, exist_ok=True)
    write_json(bad_lancedb_embed, {"params": {"backend": "st", "model": "model-z"}})

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


def test_discover_indexed_runs_current_runs_layout(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    base = time.time() - 1000
    _make_valid_run(
        runs_dir=runs_dir,
        novel_slug="mobydick",
        chunk_method="semantic",
        embed_backend="ollama",
        embed_model="qwen3-embedding:0.6b",
        store="lancedb",
        table="chunks",
        namespace=None,
        vector_dim=384,
        count=1589,
        ts=base,
    )
    found = discover_indexed_runs(runs_dir)
    assert len(found) == 1
    assert found[0].novel_slug == "mobydick"
    assert found[0].chunk_method == "semantic"
    assert found[0].store == "lancedb"


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
    ekey = f"{embed_backend}-{embed_model.replace('/', '_').replace(':', '_')}"
    store_dir = runs_dir / novel_slug / store / f"{chunk_method}__{ekey}"
    embed_params_path = runs_dir / novel_slug / ekey / chunk_method / "embed_params.json"
    chunk_params_path = runs_dir / novel_slug / chunk_method / "chunk_params.json"
    store_dir.mkdir(parents=True, exist_ok=True)
    embed_params_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_params_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(embed_params_path, {"params": {"backend": embed_backend, "model": embed_model}})
    write_json(chunk_params_path, {"params": {"target_chars": 900, "overlap_sentences": 2}})
    write_json(store_dir / "store_params.json", {"params": {"store": store, "table_or_collection": table, "namespace": namespace}})
    write_json(store_dir / "store_meta.json", {"vector_dim": vector_dim, "count": count, "store": store})
    if store == "lancedb":
        (store_dir / "chunks.lance").mkdir(parents=True, exist_ok=True)

    for path in [
        embed_params_path,
        chunk_params_path,
        store_dir / "store_params.json",
        store_dir / "store_meta.json",
    ]:
        os.utime(path, (ts, ts))
    return store_dir


def _make_query_artifact(store_dir: Path, query_id: str, *, ts: float) -> None:
    novel_dir = store_dir.parent.parent
    qdir = novel_dir / "queries" / query_id
    qdir.mkdir(parents=True, exist_ok=True)
    answer_path = qdir / "answer.md"
    answer_path.write_text("answer\n", encoding="utf-8")
    os.utime(qdir, (ts, ts))
    os.utime(answer_path, (ts, ts))
