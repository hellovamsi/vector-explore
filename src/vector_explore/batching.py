from __future__ import annotations

from typing import Iterator

from .embeddings.base import Embedder
from .ui import progress


def batched(items: list[str], batch_size: int) -> Iterator[list[str]]:
    step = max(1, int(batch_size))
    for i in range(0, len(items), step):
        yield items[i : i + step]


def batched_embed_texts(
    *,
    embedder: Embedder,
    texts: list[str],
    batch_size: int,
    progress_enabled: bool,
    description: str,
) -> list[list[float]]:
    out: list[list[float]] = []
    with progress(progress_enabled) as p:
        task = p.add_task(description, total=len(texts))
        for chunk in batched(texts, batch_size):
            vecs = embedder.embed_texts(chunk)
            out.extend(vecs)
            p.update(task, advance=len(chunk))
    return out
