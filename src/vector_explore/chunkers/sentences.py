from __future__ import annotations

import re
from dataclasses import dataclass

from .overlap import apply_sentence_overlap


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class SentenceChunkParams:
    target_chars: int = 900
    overlap_sentences: int = 2


def chunk_sentences(text: str, params: SentenceChunkParams) -> list[dict]:
    sents = split_sentences(text)
    target = max(10, int(params.target_chars))
    k = max(0, int(params.overlap_sentences))

    chunks: list[tuple[int, int, str, list[str]]] = []
    buf: list[str] = []
    buf_len = 0
    for s in sents:
        if buf and (buf_len + len(s) + 1) > target:
            chunk_text = " ".join(buf).strip()
            start = text.find(chunk_text) if chunk_text else 0
            end = start + len(chunk_text)
            chunks.append((start, end, chunk_text, buf[:]))
            buf = []
            buf_len = 0
        buf.append(s)
        buf_len += len(s) + 1
    if buf:
        chunk_text = " ".join(buf).strip()
        start = text.find(chunk_text) if chunk_text else 0
        end = start + len(chunk_text)
        chunks.append((start, end, chunk_text, buf[:]))

    overlapped = apply_sentence_overlap(chunks, overlap_sentences=k)
    out: list[dict] = []
    for idx, (start, end, chunk_text, overlap) in enumerate(overlapped):
        out.append(
            {
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
                "overlap": None if overlap is None else {"type": overlap.kind, "size": overlap.size},
                "meta": {"target_chars": target, "overlap_sentences": k},
            }
        )
    return out
