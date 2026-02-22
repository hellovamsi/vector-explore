from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl
from .paths import cache_ok, ensure_dir, index_key, params_fingerprint, sha256_file, store_index_dir, write_params, write_json
from .stores.lancedb_store import LanceDBStore
from .stores.pinecone_store import PineconeStore
from .stores.qdrant_store import QdrantStore
from .ui import progress


@dataclass(frozen=True)
class IndexParams:
    store: str  # lancedb | qdrant | pinecone
    table_or_collection: str
    namespace: str | None = None


def index_embeddings(
    *,
    project_root: Path,
    novel_slug: str,
    chunk_method: str,
    embed_key_value: str,
    store_name: str,
    embeddings_path: Path,
    params: IndexParams,
    env: Any,
    index_batch_size: int = 256,
    progress_enabled: bool = True,
    print_fn=print,
) -> Path:
    idx_key = index_key(chunk_method, embed_key_value)
    store_dir = store_index_dir(project_root, novel_slug=novel_slug, store_name=store_name, index_key=idx_key)
    input_sha = sha256_file(embeddings_path)
    params_dict = {"store": params.store, "table_or_collection": params.table_or_collection, "namespace": params.namespace}
    params_path = store_dir / "store_params.json"
    fingerprint = params_fingerprint(input_sha, params_dict)
    if cache_ok(params_path, fingerprint) and _store_exists(store_dir, params.store):
        print_fn(f"Reusing cached index: {store_dir}")
        return store_dir

    ensure_dir(store_dir)
    write_params(params_path, input_sha, params_dict)

    # Load embeddings
    rows = list(iter_jsonl(embeddings_path))
    if not rows:
        raise RuntimeError("No embeddings found to index.")
    dim = len(rows[0]["embedding"])

    items: list[dict] = []
    for r in rows:
        items.append(
            {
                "id": r["chunk_id"],
                "vector": r["embedding"],
                "text": r.get("text"),
                "metadata": {
                    "novel_slug": r.get("novel_slug"),
                    "method": r.get("method"),
                    "start_char": r.get("start_char"),
                    "end_char": r.get("end_char"),
                },
            }
        )

    store = _open_store(store_dir, params, env, vector_dim=dim)
    print_fn(f"Indexing into {store.info().name} at {store.info().location}")
    batch_size = max(1, int(index_batch_size))
    with progress(progress_enabled) as p:
        task = p.add_task(f"Upserting vectors into {params.store}", total=len(items))
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            store.upsert(batch)
            p.update(task, advance=len(batch))
    write_json(store_dir / "store_meta.json", {"vector_dim": dim, "count": len(items), "store": store.info().name})
    return store_dir


def _store_exists(store_dir: Path, store_name: str) -> bool:
    if store_name == "lancedb":
        return (store_dir / "chunks.lance").exists() and (store_dir / "store_meta.json").exists()
    if store_name == "qdrant":
        return (store_dir / "store_meta.json").exists()
    if store_name == "pinecone":
        return (store_dir / "store_meta.json").exists()
    return False


def _open_store(store_dir: Path, params: IndexParams, env: Any, *, vector_dim: int):
    if params.store == "lancedb":
        db_path = store_dir / "chunks.lance"
        return LanceDBStore(path=db_path, table_name=params.table_or_collection)
    if params.store == "qdrant":
        if not env.qdrant_url:
            raise RuntimeError("QDRANT_URL is required in .env for Qdrant.")
        collection = params.table_or_collection
        return QdrantStore(url=env.qdrant_url, api_key=env.qdrant_api_key, collection=collection, vector_dim=vector_dim)
    if params.store == "pinecone":
        if not env.pinecone_api_key or not env.pinecone_index:
            raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX are required in .env for Pinecone.")
        ns = params.namespace or "default"
        return PineconeStore(api_key=env.pinecone_api_key, index_name=env.pinecone_index, namespace=ns)
    raise ValueError(f"Unknown store: {params.store}")
