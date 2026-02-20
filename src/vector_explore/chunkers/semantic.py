from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sentences import split_sentences
from .overlap import apply_sentence_overlap
from ..embeddings.base import Embedder
from ..batching import batched_embed_texts


@dataclass(frozen=True)
class SemanticChunkParams:
    target_chars: int = 900
    overlap_sentences: int = 2
    similarity_drop_threshold: float = 0.65


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def chunk_semantic(
    text: str,
    params: SemanticChunkParams,
    embedder: Embedder,
    *,
    embed_batch_size: int = 64,
    progress_enabled: bool = True,
) -> list[dict]:
    sents = split_sentences(text)
    if not sents:
        return []

    target = max(10, int(params.target_chars))
    k = max(0, int(params.overlap_sentences))
    thr = float(params.similarity_drop_threshold)

    # Embed per-sentence to detect boundaries.
    sent_vecs = batched_embed_texts(
        embedder=embedder,
        texts=sents,
        batch_size=embed_batch_size,
        progress_enabled=progress_enabled,
        description="Embedding sentences for semantic boundaries",
    )
    sent_vecs_np = [np.asarray(v, dtype=np.float32) for v in sent_vecs]

    chunks: list[tuple[int, int, str, list[str], str]] = []
    buf: list[str] = []
    buf_len = 0

    for i, s in enumerate(sents):
        if buf:
            # similarity between current sentence and previous sentence
            sim = _cosine(sent_vecs_np[i - 1], sent_vecs_np[i])
            if sim < thr and buf_len > 0:
                chunk_text = " ".join(buf).strip()
                start = text.find(chunk_text) if chunk_text else 0
                end = start + len(chunk_text)
                chunks.append((start, end, chunk_text, buf[:], "similarity_drop"))
                buf = []
                buf_len = 0

        if buf and (buf_len + len(s) + 1) > target:
            chunk_text = " ".join(buf).strip()
            start = text.find(chunk_text) if chunk_text else 0
            end = start + len(chunk_text)
            chunks.append((start, end, chunk_text, buf[:], "size_limit"))
            buf = []
            buf_len = 0

        buf.append(s)
        buf_len += len(s) + 1

    if buf:
        chunk_text = " ".join(buf).strip()
        start = text.find(chunk_text) if chunk_text else 0
        end = start + len(chunk_text)
        chunks.append((start, end, chunk_text, buf[:], "end"))

    # Apply overlap by sentences; preserve boundary reason via meta.
    chunks_for_overlap: list[tuple[int, int, str, list[str]]] = [(a, b, c, d) for (a, b, c, d, _r) in chunks]
    overlapped = apply_sentence_overlap(chunks_for_overlap, overlap_sentences=k)

    out: list[dict] = []
    for idx, ((start, end, chunk_text, overlap), (_a, _b, _c, _d, reason)) in enumerate(zip(overlapped, chunks, strict=False)):
        out.append(
            {
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
                "overlap": None if overlap is None else {"type": overlap.kind, "size": overlap.size},
                "meta": {
                    "target_chars": target,
                    "overlap_sentences": k,
                    "similarity_drop_threshold": thr,
                    "boundary_reason": reason,
                },
            }
        )
    return out
