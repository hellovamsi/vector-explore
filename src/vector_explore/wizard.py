from __future__ import annotations

from pathlib import Path

from .chunking import chunk_novel
from .config import load_env
from .download_gutenberg import NOVELS, download_if_missing
from .embedding import embed_chunks
from .embeddings.ollama_embed import OllamaEmbedder, ollama_healthcheck
from .embeddings.st_embed import SentenceTransformersEmbedder
from .indexing import IndexParams, index_embeddings
from .io_utils import iter_jsonl
from .paging import page_iterable, page_vector_preview, prompt_more
from .paths import embed_key
from .query import QueryParams, run_query


def run_wizard(
    project_root: Path,
    *,
    no_progress: bool = False,
    embed_batch_size: int = 64,
    index_batch_size: int = 256,
) -> None:
    progress_enabled = (not no_progress) and _is_tty()
    print("Read README.md Setup before running (recommended).")
    print("")
    print("Pick a free public-domain novel (downloaded from Project Gutenberg; each text includes its own license/terms).")
    for i, n in enumerate(NOVELS, 1):
        print(f"  {i}) {n.title}")
    choice = _prompt_choice("Select [1-3]: ", 1, len(NOVELS), default=1)
    novel = NOVELS[choice - 1]

    env = load_env(project_root)
    novel_path = download_if_missing(project_root, novel, progress_enabled=progress_enabled)

    print("")
    print("Chunking method (overlap is ON by default for all methods):")
    print("  1) Fixed-length (+overlap)")
    print("  2) Sentence-based (+overlap)")
    print("  3) Semantic: embedding-guided boundaries (+overlap)")
    method_choice = _prompt_choice("Select [1-3] [default: 3]: ", 1, 3, default=3)
    method = {1: "fixed", 2: "sentences", 3: "semantic"}[method_choice]

    semantic_embedder = None
    if method == "semantic":
        print(
            "Semantic chunking here means: we embed consecutive sentences and cut chunks when similarity drops "
            "(educational demo)."
        )
        semantic_embedder = _select_embedder(env, purpose="semantic chunking boundary detection")

    chunk_params = _prompt_chunk_params(method)
    chunk_out_dir = chunk_novel(
        project_root=project_root,
        novel_slug=novel.slug,
        novel_path=novel_path,
        method=method,
        params=chunk_params,
        semantic_embedder=semantic_embedder,
        embed_batch_size=embed_batch_size,
        progress_enabled=progress_enabled,
    )

    chunks_path = chunk_out_dir / "chunks.jsonl"
    if prompt_more(input, prompt="Preview chunks now? [Enter=yes, N+Enter=no]: "):
        _preview_chunks(chunks_path)

    print("")
    embedder = semantic_embedder
    if embedder is None:
        embedder = _select_embedder(env, purpose="chunk embeddings (index + query)")
    else:
        if not prompt_more(input, prompt="Reuse the same embedding backend/model for indexing? [Enter=yes, N+Enter=no]: "):
            embedder = _select_embedder(env, purpose="chunk embeddings (index + query)")

    embed_out_dir = embed_chunks(
        project_root=project_root,
        novel_slug=novel.slug,
        chunk_method=method,
        chunks_path=chunks_path,
        embedder=embedder,
        embed_batch_size=embed_batch_size,
        progress_enabled=progress_enabled,
    )

    embeddings_path = embed_out_dir / "embeddings.jsonl"
    if prompt_more(input, prompt="Preview chunk->vector pairs now? [Enter=yes, N+Enter=no]: "):
        _preview_embeddings(embeddings_path)

    print("")
    print("Vector store:")
    print("  1) LanceDB (default, local files)")
    print("  2) Qdrant (requires user-managed server + .env)")
    print("  3) Pinecone (cloud + .env)")
    store_choice = _prompt_choice("Select [1-3] [default: 1]: ", 1, 3, default=1)
    store = {1: "lancedb", 2: "qdrant", 3: "pinecone"}[store_choice]

    table_or_collection = "chunks"
    namespace = None
    if store == "qdrant":
        table_or_collection = f"{env.qdrant_collection_prefix}_{novel.slug}_{method}_{embedder.info().backend}"
        print("Qdrant: ensure QDRANT_URL (and optional QDRANT_API_KEY) are set in .env.")
    if store == "pinecone":
        namespace = f"{novel.slug}-{method}-{embedder.info().backend}"
        print("Pinecone: ensure PINECONE_API_KEY and PINECONE_INDEX are set in .env.")

    ekey = embed_key(embedder.info().backend, embedder.info().model)
    index_embeddings(
        project_root=project_root,
        novel_slug=novel.slug,
        chunk_method=method,
        embed_key_value=ekey,
        store_name=store,
        embeddings_path=embeddings_path,
        params=IndexParams(store=store, table_or_collection=table_or_collection, namespace=namespace),
        env=env,
        index_batch_size=index_batch_size,
        progress_enabled=progress_enabled,
    )

    print("")
    print("Suggested questions:")
    suggested = [
        "What motivates the protagonist, and what are the consequences?",
        "What are the major themes in this novel?",
        "Summarize the relationship between two central characters.",
    ]
    for i, q in enumerate(suggested, 1):
        print(f"  {i}) {q}")
    raw_q = input("Pick [1-3] or type your own: ").strip()
    if raw_q in {"1", "2", "3"}:
        question = suggested[int(raw_q) - 1]
    else:
        question = raw_q
    if not question:
        print("No question provided. Exiting.")
        return

    qparams = QueryParams(top_k=5, show_vectors=True, no_llm=False, vector_summary=False)
    run_query(
        project_root=project_root,
        novel_slug=novel.slug,
        chunk_method=method,
        embed_key=ekey,
        store_name=store,
        question=question,
        embedder=embedder,
        env=env,
        params=qparams,
        input_fn=input,
        print_fn=print,
    )


def _prompt_choice(prompt: str, lo: int, hi: int, *, default: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"Enter a number between {lo} and {hi}.")
            continue
        if lo <= v <= hi:
            return v
        print(f"Enter a number between {lo} and {hi}.")


def _prompt_chunk_params(method: str) -> dict:
    if method == "fixed":
        target = input("Target chunk size in chars [default 900]: ").strip() or "900"
        overlap = input("Overlap ratio (0-0.5) [default 0.15]: ").strip() or "0.15"
        return {"target_chars": int(target), "overlap_ratio": float(overlap)}
    if method == "sentences":
        target = input("Target chunk size in chars [default 900]: ").strip() or "900"
        k = input("Overlap sentences [default 2]: ").strip() or "2"
        return {"target_chars": int(target), "overlap_sentences": int(k)}
    if method == "semantic":
        target = input("Target chunk size in chars [default 900]: ").strip() or "900"
        k = input("Overlap sentences [default 2]: ").strip() or "2"
        thr = input("Similarity drop threshold (0-1) [default 0.65]: ").strip() or "0.65"
        return {"target_chars": int(target), "overlap_sentences": int(k), "similarity_drop_threshold": float(thr)}
    raise ValueError(method)


def _select_embedder(env, *, purpose: str):
    print("")
    print(f"Choose an embedding backend for {purpose}:")
    print("  1) Ollama embeddings (local; uses installed embedding models; no pulling)")
    print("  2) SentenceTransformers (local Python; downloads model weights if needed)")
    print("  3) Cloud (OpenAI/OpenRouter via .env)")
    choice = _prompt_choice("Select [1-3] [default: 1]: ", 1, 3, default=1)

    if choice == 1:
        base = env.ollama_base_url
        model = input(f"Ollama embedding model [default {env.ollama_embed_model}]: ").strip() or env.ollama_embed_model
        if not ollama_healthcheck(base):
            print(f"Warning: Ollama not reachable at {base}. Check README.md Setup.")
        return OllamaEmbedder(base_url=base, model=model)

    if choice == 2:
        model = input("SentenceTransformers model [default sentence-transformers/all-MiniLM-L6-v2]: ").strip() or (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        print("This will download model weights if missing (one-time).")
        if not prompt_more(input, prompt="Proceed? [Enter=yes, N+Enter=no]: "):
            raise RuntimeError("Cancelled SentenceTransformers selection.")
        return SentenceTransformersEmbedder(model_name=model)

    if not env.openai_base_url or not env.openai_api_key or not env.openai_embed_model:
        raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_EMBED_MODEL in .env.")
    try:
        from .embeddings.cloud_embed import OpenAICompatibleEmbedder
    except Exception as e:
        raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
    return OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=env.openai_embed_model)


def _preview_chunks(chunks_path: Path) -> None:
    lines: list[str] = []
    for r in iter_jsonl(chunks_path):
        cid = r["chunk_id"]
        txt = (r["text"] or "").strip().replace("\n", " ")
        lines.append(f"[{cid}] {txt[:400]}" + ("..." if len(txt) > 400 else ""))
    page_iterable(lines, page_size=3, input_fn=input, print_fn=print, header=f"Preview: {chunks_path}")


def _preview_embeddings(embeddings_path: Path) -> None:
    for r in iter_jsonl(embeddings_path):
        print(f"Chunk: {r['chunk_id']}")
        print((r["text"] or "")[:200].replace("\n", " ") + ("..." if len(r["text"] or "") > 200 else ""))
        vec = r["embedding"]
        page_vector_preview(vec, input_fn=input, print_fn=print, label="chunk_vector")
        if not prompt_more(input, prompt="Next chunk? [Enter=yes, N+Enter=no]: "):
            break


def _is_tty() -> bool:
    import sys

    return bool(getattr(sys.stdout, "isatty", lambda: False)())
