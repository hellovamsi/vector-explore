from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Env
from .embeddings.base import Embedder
from .io_utils import write_jsonl
from .llm.ollama_generate import ollama_generate
from .llm.prompts import build_rag_prompt_parts, rag_prompt
from .paging import page_vector_preview
from .paths import ensure_dir, query_dir, query_hash, read_json, store_index_dir, write_json
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
    project_root: Path,
    novel_slug: str,
    chunk_method: str,
    embed_key: str,
    store_name: str,
    question: str,
    embedder: Embedder,
    env: Env,
    params: QueryParams,
    input_fn=input,
    print_fn=print,
) -> Path:
    idx_key = f"{chunk_method}__{embed_key}"
    store_dir = store_index_dir(project_root, novel_slug=novel_slug, store_name=store_name, index_key=idx_key)
    store_params_path = store_dir / "store_params.json"
    if not store_params_path.exists():
        raise RuntimeError(f"Missing store_params.json in {store_dir}. Run indexing first.")
    store_params = read_json(store_params_path)
    store_name = store_params["params"]["store"] if "params" in store_params else store_params.get("store")
    table_or_collection = store_params["params"]["table_or_collection"] if "params" in store_params else store_params.get("table_or_collection")
    namespace = store_params["params"].get("namespace") if "params" in store_params else store_params.get("namespace")

    config = {
        "novel_slug": novel_slug,
        "chunk_method": chunk_method,
        "embed_key": embed_key,
        "store_name": store_name,
        "table_or_collection": table_or_collection,
        "namespace": namespace,
    }
    qparams = {"top_k": params.top_k, "show_vectors": params.show_vectors}
    qh = query_hash(question, qparams, config)
    qdir = query_dir(project_root, novel_slug=novel_slug, query_hash=qh)
    ensure_dir(qdir)

    print_fn(f"Query: {question}")
    print_fn("Embedding query...")
    qvec = embedder.embed_texts([question])[0]
    if params.show_vectors:
        page_vector_preview(qvec, input_fn=input_fn, print_fn=print_fn, label="query_vector")

    store = _open_store(store_dir, store_name, table_or_collection, namespace, env)
    print_fn(f"Vector DB request: store={store_name} table/collection={table_or_collection} top_k={params.top_k}")
    retrieved = store.query(qvec, top_k=params.top_k)

    write_json(
        qdir / "query.json",
        {
            "question": question,
            "query_hash": qh,
            "top_k": params.top_k,
            "show_vectors": params.show_vectors,
            "no_llm": params.no_llm,
            "vector_summary": params.vector_summary,
            "embed_backend": embedder.info().backend,
            "embed_model": embedder.info().model,
            **config,
        },
    )

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
        print_fn(f"  text_preview={txt[:200].replace('\\n', ' ')}" + ("..." if len(txt) > 200 else ""))
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

    def summarizer(texts: list[str], keywords: list[str]) -> str:
        instruction = (
            "Summarize the following retrieved passages in 5-7 sentences. "
            "Focus on answering the user's question context."
        )
        prompt = instruction + f" Keywords: {', '.join(keywords)}\n\n" + "\n\n---\n\n".join(texts)
        gen = ollama_generate(base_url=env.ollama_base_url, model=env.ollama_chat_model, prompt=prompt)
        input_log = build_llm_input_log(
            base_url=env.ollama_base_url,
            model=env.ollama_chat_model,
            timeout_s=120.0,
            request_body_raw=gen.request_body_raw,
            request_body=gen.request_body,
            prompt_text=prompt,
            prompt_parts={"instruction": instruction, "keywords": keywords, "passages": texts},
        )
        write_json(qdir / "llm_summary_input.json", input_log)
        (qdir / "llm_summary_output.json").write_text(gen.raw_response_json + "\n", encoding="utf-8")
        print_fn(f"Saved LLM input: {qdir / 'llm_summary_input.json'}")
        print_fn(f"Saved LLM output: {qdir / 'llm_summary_output.json'}")
        return gen.response.strip()

    v2t = vector_to_text_lossy(retrieved, include_summary=params.vector_summary and not params.no_llm, summarizer=summarizer)
    print_fn(v2t.blurb)
    print_fn("Vector->Text keywords: " + (", ".join(v2t.keywords) if v2t.keywords else "(none)"))
    write_json(
        qdir / "vector_to_text.json",
        {"blurb": v2t.blurb, "keywords": v2t.keywords, "neighbors": v2t.neighbors, "summary": v2t.summary},
    )

    if params.no_llm:
        return qdir

    sources = [{"chunk_id": r.get("id"), "text": r.get("text")} for r in retrieved]
    prompt_parts = build_rag_prompt_parts(question=question, keywords=v2t.keywords, sources=sources)
    context_prompt = rag_prompt(question=question, keywords=v2t.keywords, sources=sources)
    gen = ollama_generate(base_url=env.ollama_base_url, model=env.ollama_chat_model, prompt=context_prompt)
    answer_input_log = build_llm_input_log(
        base_url=env.ollama_base_url,
        model=env.ollama_chat_model,
        timeout_s=120.0,
        request_body_raw=gen.request_body_raw,
        request_body=gen.request_body,
        prompt_text=context_prompt,
        prompt_parts=prompt_parts,
    )
    write_json(qdir / "llm_answer_input.json", answer_input_log)
    (qdir / "llm_answer_output.json").write_text(gen.raw_response_json + "\n", encoding="utf-8")
    print_fn(f"Saved LLM input: {qdir / 'llm_answer_input.json'}")
    print_fn(f"Saved LLM output: {qdir / 'llm_answer_output.json'}")

    answer = gen.response.strip()
    (qdir / "answer.md").write_text(answer + "\n", encoding="utf-8")
    print_fn(f"Saved answer: {qdir / 'answer.md'}")
    return qdir


def build_llm_input_log(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    request_body_raw: str,
    request_body: dict[str, Any],
    prompt_text: str,
    prompt_parts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "provider": "ollama",
        "endpoint": "/api/generate",
        "url": f"{base_url.rstrip('/')}/api/generate",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "timeout_s": timeout_s,
        "model": model,
        "request_body_raw": request_body_raw,
        "request_body": request_body,
        "prompt_text": prompt_text,
        "prompt_parts": prompt_parts,
    }


def _open_store(store_dir: Path, store_name: str, table_or_collection: str, namespace: str | None, env: Env):
    if store_name == "lancedb":
        return LanceDBStore(path=store_dir / "chunks.lance", table_name=table_or_collection)
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
