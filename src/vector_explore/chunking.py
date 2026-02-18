from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunkers.fixed import FixedChunkParams, chunk_fixed
from .chunkers.sentences import SentenceChunkParams, chunk_sentences
from .chunkers.semantic import SemanticChunkParams, chunk_semantic
from .embeddings.base import Embedder
from .io_utils import write_jsonl
from .paths import cache_ok, chunk_run_dir, ensure_dir, params_fingerprint, sha256_bytes, write_params
from .preprocess import strip_gutenberg_header_footer


def chunk_novel(
    *,
    project_root: Path,
    runs_dir: Path,
    novel_slug: str,
    novel_path: Path,
    method: str,
    params: dict[str, Any],
    semantic_embedder: Embedder | None = None,
    print_fn=print,
) -> Path:
    raw = novel_path.read_text(encoding="utf-8", errors="replace")
    clean = strip_gutenberg_header_footer(raw)
    input_sha = sha256_bytes(clean.encode("utf-8"))

    out_dir = chunk_run_dir(runs_dir, novel_slug=novel_slug, chunk_method=method)
    ensure_dir(out_dir)
    params_path = out_dir / "chunk_params.json"
    fingerprint = params_fingerprint(input_sha, {"method": method, **params})
    if cache_ok(params_path, fingerprint) and (out_dir / "chunks.jsonl").exists():
        print_fn(f"Reusing cached chunks: {out_dir}")
        return out_dir

    if method == "fixed":
        p = FixedChunkParams(**params)
        chunks = chunk_fixed(clean, p)
    elif method == "sentences":
        p = SentenceChunkParams(**params)
        chunks = chunk_sentences(clean, p)
    elif method == "semantic":
        if semantic_embedder is None:
            raise RuntimeError("Semantic chunking requires an embedder for boundary detection.")
        p = SemanticChunkParams(**params)
        chunks = chunk_semantic(clean, p, semantic_embedder)
    else:
        raise ValueError(f"Unknown chunking method: {method}")

    rows = []
    for i, c in enumerate(chunks):
        rows.append(
            {
                "chunk_id": f"{method}_{i:05d}",
                "novel_slug": novel_slug,
                "method": method,
                "start_char": c["start_char"],
                "end_char": c["end_char"],
                "text": c["text"],
                "overlap": c["overlap"],
                "meta": c["meta"],
            }
        )

    write_jsonl(out_dir / "chunks.jsonl", rows)
    write_params(params_path, input_sha, {"method": method, **params})
    print_fn(f"Chunked: {len(rows)} chunks")
    print_fn(f"Saved chunks: {out_dir / 'chunks.jsonl'}")
    return out_dir
