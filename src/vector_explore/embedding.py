from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from .batching import batched_embed_texts
from .embeddings.base import Embedder
from .io_utils import iter_jsonl, write_jsonl
from .paths import cache_ok, embed_key, embedding_dir, ensure_dir, params_fingerprint, sha256_file, sha256_bytes, write_params


@dataclass(frozen=True)
class EmbedParams:
    backend: str
    model: str


def embed_chunks(
    *,
    project_root: Path,
    novel_slug: str,
    chunk_method: str,
    chunks_path: Path,
    embedder: Embedder,
    embed_batch_size: int = 64,
    progress_enabled: bool = True,
    print_fn=print,
) -> Path:
    input_sha = sha256_file(chunks_path)
    info = embedder.info()
    params = {"backend": info.backend, "model": info.model}

    ekey = embed_key(info.backend, info.model)
    out_dir = embedding_dir(project_root, novel_slug=novel_slug, embed_key=ekey, chunk_method=chunk_method)
    ensure_dir(out_dir)
    params_path = out_dir / "embed_params.json"
    fingerprint = params_fingerprint(input_sha, params)
    if cache_ok(params_path, fingerprint) and (out_dir / "embeddings.jsonl").exists():
        print_fn(f"Reusing cached embeddings: {out_dir}")
        return out_dir

    rows = list(iter_jsonl(chunks_path))
    texts = [r["text"] for r in rows]
    vectors = batched_embed_texts(
        embedder=embedder,
        texts=texts,
        batch_size=embed_batch_size,
        progress_enabled=progress_enabled,
        description=f"Embedding chunks ({info.backend}/{info.model})",
    )
    if len(vectors) != len(rows):
        raise RuntimeError("Embedder returned wrong number of vectors.")
    created_at = dt.datetime.utcnow().isoformat() + "Z"
    out_rows = []
    for r, v in zip(rows, vectors, strict=True):
        out_rows.append(
            {
                "chunk_id": r["chunk_id"],
                "novel_slug": r.get("novel_slug", novel_slug),
                "method": r.get("method", chunk_method),
                "start_char": r.get("start_char"),
                "end_char": r.get("end_char"),
                "text": r["text"],
                "text_sha256": sha256_bytes(r["text"].encode("utf-8")),
                "embedding": v,
                "dim": len(v),
                "backend": info.backend,
                "model": info.model,
                "created_at": created_at,
            }
        )

    write_jsonl(out_dir / "embeddings.jsonl", out_rows)
    write_params(params_path, input_sha, params)
    print_fn(f"Embedded: {len(out_rows)} chunks")
    print_fn(f"Saved chunk+embeddings: {out_dir / 'embeddings.jsonl'}")
    return out_dir
