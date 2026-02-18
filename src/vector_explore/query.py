from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Env
from .embeddings.base import Embedder
from .io_utils import write_jsonl
from .llm.ollama_generate import ollama_generate
from .llm.prompts import rag_prompt
from .paging import page_vector_preview
from .paths import ensure_dir, query_dir, query_hash, read_json, write_json
from .stores.lancedb_store import LanceDBStore
from .stores.pinecone_store import PineconeStore
from .stores.qdrant_store import QdrantStore
from .vector_to_text import vector_to_text_lossy


@dataclass(frozen=True)
class QueryParams:
    top_k: int = 5
    show_vectors: bool = True
    no_llm: bool = False
    vector_summary: bool = False


def run_query(
    *,
    store_dir: Path,
    question: str,
    embedder: Embedder,
    env: Env,
    params: QueryParams,
    input_fn=input,
    print_fn=print,
) -> Path:
    store_params_path = store_dir / "store_params.json"
    if not store_params_path.exists():
        raise RuntimeError(f"Missing store_params.json in {store_dir}. Run indexing first.")
    store_params = read_json(store_params_path)
    store_name = store_params["params"]["store"] if "params" in store_params else store_params.get("store")
    table_or_collection = store_params["params"]["table_or_collection"] if "params" in store_params else store_params.get("table_or_collection")
    namespace = store_params["params"].get("namespace") if "params" in store_params else store_params.get("namespace")

    qparams = {"top_k": params.top_k, "show_vectors": params.show_vectors}
    qh = query_hash(question, qparams)
    qdir = query_dir(store_dir, qh)
    ensure_dir(qdir)

    print_fn(f"Query: {question}")
    print_fn("Embedding query…")
    qvec = embedder.embed_texts([question])[0]
    if params.show_vectors:
        page_vector_preview(qvec, input_fn=input_fn, print_fn=print_fn, label="query_vector")

    store = _open_store(store_dir, store_name, table_or_collection, namespace, env)
    print_fn(f"Vector DB request: store={store_name} table/collection={table_or_collection} top_k={params.top_k}")
    retrieved = store.query(qvec, top_k=params.top_k)

    # Write query.json
    write_json(
        qdir / "query.json",
        {
            "question": question,
            "query_hash": qh,
            "top_k": params.top_k,
            "embed_backend": embedder.info().backend,
            "embed_model": embedder.info().model,
            "store": store_name,
        },
    )

    # Show + write retrieval.jsonl
    retrieval_rows = []
    print_fn(f"Vector DB response (top {len(retrieved)}):")
    for idx, r in enumerate(retrieved, 1):
        rid = r.get("id")
        score = r.get("score")
        txt = r.get("text") or ""
        vec = r.get("vector")
        print_fn(f"  rank={idx} score={score} id={rid}")
        if params.show_vectors and isinstance(vec, list):
            page_vector_preview(vec, input_fn=input_fn, print_fn=print_fn, label=f"retrieved_vector[{idx}]")
        print_fn(f"  text_preview={txt[:200].replace('\\n', ' ')}" + ("…" if len(txt) > 200 else ""))
        retrieval_rows.append(
            {
                "rank": idx,
                "score": score,
                "chunk_id": rid,
                "stored_vector": vec if params.show_vectors else None,
                "text_full": txt,
            }
        )
    write_jsonl(qdir / "retrieval.jsonl", retrieval_rows)

    # Vector→Text approximation (lossy)
    def summarizer(texts: list[str], keywords: list[str]) -> str:
        prompt = (
            "Summarize the following retrieved passages in 5-7 sentences. "
            "Focus on answering the user's question context. "
            f"Keywords: {', '.join(keywords)}\n\n"
            + "\n\n---\n\n".join(texts)
        )
        return ollama_generate(base_url=env.ollama_base_url, model=env.ollama_chat_model, prompt=prompt).response.strip()

    v2t = vector_to_text_lossy(retrieved, include_summary=params.vector_summary and not params.no_llm, summarizer=summarizer)
    print_fn(v2t.blurb)
    print_fn("Vector→Text keywords: " + (", ".join(v2t.keywords) if v2t.keywords else "(none)"))
    write_json(
        qdir / "vector_to_text.json",
        {"blurb": v2t.blurb, "keywords": v2t.keywords, "neighbors": v2t.neighbors, "summary": v2t.summary},
    )

    if params.no_llm:
        return qdir

    # Answer generation via Ollama
    context_prompt = rag_prompt(question=question, keywords=v2t.keywords, chunks=[{"id": r.get("id"), "text": r.get("text")} for r in retrieved])
    gen = ollama_generate(base_url=env.ollama_base_url, model=env.ollama_chat_model, prompt=context_prompt)
    answer = gen.response.strip()
    (qdir / "answer.md").write_text(answer + "\n", encoding="utf-8")
    print_fn(f"Saved answer: {qdir / 'answer.md'}")
    return qdir


def _open_store(store_dir: Path, store_name: str, table_or_collection: str, namespace: str | None, env: Env):
    if store_name == "lancedb":
        return LanceDBStore(path=store_dir / "lancedb", table_name=table_or_collection)
    if store_name == "qdrant":
        if not env.qdrant_url:
            raise RuntimeError("QDRANT_URL missing in .env.")
        meta_path = store_dir / "store_meta.json"
        vector_dim = 0
        if meta_path.exists():
            vector_dim = int(read_json(meta_path).get("vector_dim") or 0)
        return QdrantStore(url=env.qdrant_url, api_key=env.qdrant_api_key, collection=table_or_collection, vector_dim=vector_dim or 0)
    if store_name == "pinecone":
        if not env.pinecone_api_key or not env.pinecone_index:
            raise RuntimeError("PINECONE_API_KEY/PINECONE_INDEX missing in .env.")
        return PineconeStore(api_key=env.pinecone_api_key, index_name=env.pinecone_index, namespace=namespace or "default")
    raise ValueError(f"Unknown store: {store_name}")
