from __future__ import annotations

from dataclasses import dataclass

from .overlap import apply_char_overlap


@dataclass(frozen=True)
class FixedChunkParams:
    target_chars: int = 900
    overlap_ratio: float = 0.15


def chunk_fixed(text: str, params: FixedChunkParams) -> list[dict]:
    target = max(50, int(params.target_chars))
    overlap_chars = max(0, int(target * float(params.overlap_ratio)))

    base: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        j = min(i + target, len(text))
        base.append((i, j, text[i:j]))
        i = j

    overlapped = apply_char_overlap(base, overlap_chars=overlap_chars)
    out: list[dict] = []
    for idx, (start, end, chunk_text, overlap) in enumerate(overlapped):
        out.append(
            {
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
                "overlap": None if overlap is None else {"type": overlap.kind, "size": overlap.size},
                "meta": {"target_chars": target, "overlap_ratio": params.overlap_ratio},
            }
        )
    return out

