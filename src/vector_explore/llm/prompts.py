from __future__ import annotations


def rag_prompt(*, question: str, keywords: list[str], chunks: list[dict]) -> str:
    lines: list[str] = []
    lines.append("You are a concise assistant. Answer the question using only the provided sources.")
    lines.append("Cite sources by chunk_id in parentheses, e.g. (chunk_id=semantic_01234).")
    lines.append("")
    lines.append(f"Question: {question}")
    if keywords:
        lines.append(f"Vector→Text keywords (lossy hint): {', '.join(keywords)}")
    lines.append("")
    lines.append("Sources:")
    for c in chunks:
        cid = c.get("id")
        txt = (c.get("text") or "").strip()
        lines.append(f"[chunk_id={cid}] {txt}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

