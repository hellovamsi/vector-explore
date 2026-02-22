from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import read_json


@dataclass(frozen=True)
class IndexedRunSummary:
    store_dir: Path
    novel_slug: str
    chunk_method: str
    chunk_params: dict[str, Any] | None
    embed_backend: str
    embed_model: str
    store: str
    table_or_collection: str | None
    namespace: str | None
    vector_dim: int | None
    count: int | None
    queries_count: int
    last_activity_ts: float
    display_id: str


def discover_indexed_runs(runs_dir: Path, *, limit: int = 10) -> list[IndexedRunSummary]:
    out: list[IndexedRunSummary] = []
    if limit <= 0 or not runs_dir.exists():
        return out

    for meta_path in runs_dir.rglob("store_meta.json"):
        store_dir = meta_path.parent
        store_params_path = store_dir / "store_params.json"
        if not store_params_path.exists():
            continue

        embed_dir = store_dir.parent
        chunk_dir = embed_dir.parent
        novel_dir = chunk_dir.parent
        if novel_dir == chunk_dir:
            continue

        embed_params_path = embed_dir / "embed_params.json"
        try:
            embed_params = read_json(embed_params_path)
        except Exception:
            continue

        embed_backend = _param_value(embed_params, "backend")
        embed_model = _param_value(embed_params, "model")
        if not embed_backend or not embed_model:
            continue

        try:
            store_params = read_json(store_params_path)
            store_meta = read_json(meta_path)
        except Exception:
            continue

        store_name = _param_value(store_params, "store") or store_dir.name
        if store_name == "lancedb" and not (store_dir / "lancedb").exists():
            continue

        chunk_params: dict[str, Any] | None = None
        chunk_params_path = chunk_dir / "chunk_params.json"
        if chunk_params_path.exists():
            try:
                raw_chunk_params = read_json(chunk_params_path)
                cparams = raw_chunk_params.get("params") if isinstance(raw_chunk_params, dict) else None
                if isinstance(cparams, dict):
                    chunk_params = cparams
            except Exception:
                chunk_params = None

        query_dir = store_dir / "queries"
        queries_count = 0
        if query_dir.exists():
            queries_count = len([p for p in query_dir.iterdir() if p.is_dir()])

        last_activity_ts = _compute_last_activity_ts(
            files=[
                meta_path,
                store_params_path,
                embed_params_path,
                chunk_params_path if chunk_params_path.exists() else None,
            ],
            query_dir=query_dir if query_dir.exists() else None,
        )
        try:
            display_id = str(store_dir.relative_to(runs_dir))
        except ValueError:
            display_id = str(store_dir)

        out.append(
            IndexedRunSummary(
                store_dir=store_dir,
                novel_slug=novel_dir.name,
                chunk_method=chunk_dir.name,
                chunk_params=chunk_params,
                embed_backend=embed_backend,
                embed_model=embed_model,
                store=store_name,
                table_or_collection=_param_value(store_params, "table_or_collection"),
                namespace=_param_value(store_params, "namespace"),
                vector_dim=_to_int_or_none(store_meta.get("vector_dim")),
                count=_to_int_or_none(store_meta.get("count")),
                queries_count=queries_count,
                last_activity_ts=last_activity_ts,
                display_id=display_id,
            )
        )

    out.sort(key=lambda r: r.last_activity_ts, reverse=True)
    return out[:limit]


def _compute_last_activity_ts(*, files: list[Path | None], query_dir: Path | None) -> float:
    ts: list[float] = []
    for path in files:
        if path is None:
            continue
        try:
            ts.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue

    if query_dir is not None:
        for p in query_dir.rglob("*"):
            try:
                ts.append(p.stat().st_mtime)
            except FileNotFoundError:
                continue

    return max(ts) if ts else 0.0


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _param_value(obj: Any, key: str) -> str | None:
    if not isinstance(obj, dict):
        return None
    nested = obj.get("params")
    if isinstance(nested, dict) and key in nested:
        value = nested.get(key)
    else:
        value = obj.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
