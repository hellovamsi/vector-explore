from __future__ import annotations

import re


def normalize_text(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.strip("\ufeff")
    return raw


_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def strip_gutenberg_header_footer(text: str) -> str:
    text = normalize_text(text)
    start = _START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = _END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()

