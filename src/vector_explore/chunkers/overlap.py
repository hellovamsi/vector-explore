from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapSpec:
    kind: str  # "chars" | "sentences"
    size: int  # chars or sentence count


def apply_char_overlap(chunks: list[tuple[int, int, str]], *, overlap_chars: int) -> list[tuple[int, int, str, OverlapSpec | None]]:
    out: list[tuple[int, int, str, OverlapSpec | None]] = []
    prev_text = ""
    for idx, (start, end, text) in enumerate(chunks):
        if idx == 0 or overlap_chars <= 0:
            out.append((start, end, text, None))
            prev_text = text
            continue
        prefix = prev_text[-overlap_chars:]
        out.append((start, end, prefix + text, OverlapSpec("chars", overlap_chars)))
        prev_text = text
    return out


def apply_sentence_overlap(
    chunks: list[tuple[int, int, str, list[str]]],
    *,
    overlap_sentences: int,
) -> list[tuple[int, int, str, OverlapSpec | None]]:
    out: list[tuple[int, int, str, OverlapSpec | None]] = []
    prev_sents: list[str] = []
    for idx, (start, end, text, sentences) in enumerate(chunks):
        if idx == 0 or overlap_sentences <= 0 or not prev_sents:
            out.append((start, end, text, None))
            prev_sents = sentences
            continue
        prefix_sents = prev_sents[-overlap_sentences:]
        prefix = " ".join(prefix_sents).strip()
        joined = (prefix + " " + text).strip() if prefix else text
        out.append((start, end, joined, OverlapSpec("sentences", overlap_sentences)))
        prev_sents = sentences
    return out

