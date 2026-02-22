from __future__ import annotations


def build_rag_prompt_parts(*, question: str, keywords: list[str], sources: list[dict]) -> dict:
    return {
        "system": "You are a concise assistant. Answer the question using only the provided sources.",
        "citation_rule": "Cite sources by chunk_id in parentheses, e.g. (chunk_id=semantic_01234).",
        "question": question,
        "keywords": keywords,
        "sources": sources,
    }


def format_rag_prompt(parts: dict) -> str:
    lines: list[str] = []
    lines.append(parts["system"])
    lines.append(parts["citation_rule"])
    lines.append("")
    lines.append(f"Question: {parts['question']}")
    keywords = parts.get("keywords") or []
    if keywords:
        lines.append(f"Vector→Text keywords (lossy hint): {', '.join(keywords)}")
    lines.append("")
    lines.append("Sources:")
    for c in parts.get("sources") or []:
        cid = c.get("chunk_id")
        txt = (c.get("text") or "").strip()
        lines.append(f"[chunk_id={cid}] {txt}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def rag_prompt(*, question: str, keywords: list[str], sources: list[dict]) -> str:
    parts = build_rag_prompt_parts(question=question, keywords=keywords, sources=sources)
    return format_rag_prompt(parts)
