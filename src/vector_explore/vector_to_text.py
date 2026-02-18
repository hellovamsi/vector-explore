from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "was",
    "were",
    "are",
    "but",
    "not",
    "his",
    "her",
    "she",
    "him",
    "you",
    "your",
    "they",
    "their",
    "them",
    "this",
    "from",
    "have",
    "had",
    "has",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "which",
    "would",
    "could",
    "should",
    "into",
    "upon",
    "there",
    "then",
    "than",
    "will",
    "shall",
    "may",
    "might",
    "one",
    "all",
    "any",
    "some",
    "very",
    "much",
    "more",
    "most",
    "such",
    "been",
    "being",
    "over",
    "under",
    "out",
    "about",
    "because",
}


@dataclass(frozen=True)
class VectorToTextResult:
    neighbors: list[dict]
    keywords: list[str]
    blurb: str
    summary: str | None


def vector_to_text_lossy(
    retrieved: list[dict],
    *,
    top_keywords: int = 12,
    include_summary: bool = False,
    summarizer=None,
) -> VectorToTextResult:
    blurb = (
        "Vector→Text approximation (lossy): embeddings are not invertible. "
        "We reconstruct approximate meaning using nearest-neighbor texts in embedding space, plus extracted keywords."
    )
    texts = [(_safe_str(r.get("text")) or "") for r in retrieved]
    keywords = extract_keywords("\n".join(texts), top_n=top_keywords)
    summary = None
    if include_summary and summarizer is not None:
        summary = summarizer(texts, keywords)
    neighbors = []
    for r in retrieved:
        neighbors.append(
            {
                "id": r.get("id"),
                "score": r.get("score"),
                "text": r.get("text"),
            }
        )
    return VectorToTextResult(neighbors=neighbors, keywords=keywords, blurb=blurb, summary=summary)


def extract_keywords(text: str, *, top_n: int) -> list[str]:
    tokens = []
    for m in _WORD_RE.finditer(text.lower()):
        w = m.group(0)
        if w in _STOPWORDS:
            continue
        tokens.append(w)
    counts = Counter(tokens)
    return [w for (w, _c) in counts.most_common(top_n)]


def _safe_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)

