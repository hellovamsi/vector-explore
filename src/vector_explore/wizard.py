from __future__ import annotations

import datetime as dt
from pathlib import Path

from .chunking import chunk_novel
from .config import Env, load_env
from .download_gutenberg import NOVELS, download_if_missing
from .embedding import embed_chunks
from .embeddings.base import Embedder
from .embeddings.ollama_embed import OllamaEmbedder, ollama_healthcheck
from .embeddings.st_embed import SentenceTransformersEmbedder
from .indexing import IndexParams, index_embeddings
from .io_utils import iter_jsonl
from .paging import page_iterable, page_vector_preview, prompt_more
from .paths import default_paths, store_run_dir
from .query import QueryParams, run_query
from .runs_inspect import IndexedRunSummary, discover_indexed_runs


def run_wizard(project_root: Path) -> None:
    print("Read README.md Setup before running (recommended).")
    while True:
        print("")
        env = load_env(project_root)
        paths = default_paths(project_root)
        indexed_runs = discover_indexed_runs(paths.runs_dir, limit=10)

        print("Pick a free public-domain novel (downloaded from Project Gutenberg; each text includes its own license/terms).")
        run_offset = len(indexed_runs)
        if indexed_runs:
            print("Previous query-ready runs (most recent first):")
            for i, run in enumerate(indexed_runs, 1):
                print(f"  {i}) {_format_indexed_run(run)}")
            print("")
            print("Start a new run:")

        for i, n in enumerate(NOVELS, 1):
            print(f"  {run_offset + i}) {n.title}")

        if indexed_runs:
            default_choice = run_offset + 1
            choice = _prompt_choice(
                f"Select [1-{run_offset + len(NOVELS)}] [default: {default_choice}]: ",
                1,
                run_offset + len(NOVELS),
                default=default_choice,
            )
            if choice <= run_offset:
                run = indexed_runs[choice - 1]
                try:
                    embedder = _embedder_from_saved(run.embed_backend, run.embed_model, env)
                    run_query_stage(
                        store_dir=run.store_dir,
                        embedder=embedder,
                        env=env,
                        input_fn=input,
                        print_fn=print,
                        initial_query_params=QueryParams(top_k=5, show_vectors=True, no_llm=False, vector_summary=False),
                    )
                    return
                except Exception as e:
                    print(f"Unable to reuse run '{run.display_id}': {e}")
                    print("Please choose another run or start a new one.")
                    continue

            novel = NOVELS[choice - run_offset - 1]
        else:
            choice = _prompt_choice(f"Select [1-{len(NOVELS)}]: ", 1, len(NOVELS), default=1)
            novel = NOVELS[choice - 1]
        break

    novel_path = download_if_missing(project_root, novel)

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
        semantic_embedder = _select_embedder(project_root, env, purpose="semantic chunking boundary detection")

    chunk_params = _prompt_chunk_params(method)
    chunk_dir = chunk_novel(
        project_root=project_root,
        runs_dir=paths.runs_dir,
        novel_slug=novel.slug,
        novel_path=novel_path,
        method=method,
        params=chunk_params,
        semantic_embedder=semantic_embedder,
    )

    chunks_path = chunk_dir / "chunks.jsonl"
    if prompt_more(input, prompt="Preview chunks now? [Enter=yes, N+Enter=no]: "):
        _preview_chunks(chunks_path)

    print("")
    embedder = semantic_embedder
    if embedder is None:
        embedder = _select_embedder(project_root, env, purpose="chunk embeddings (index + query)")
    else:
        if not prompt_more(input, prompt="Reuse the same embedding backend/model for indexing? [Enter=yes, N+Enter=no]: "):
            embedder = _select_embedder(project_root, env, purpose="chunk embeddings (index + query)")

    embed_dir = embed_chunks(
        runs_dir=paths.runs_dir,
        novel_slug=novel.slug,
        chunk_method=method,
        chunks_path=chunks_path,
        embedder=embedder,
    )

    embeddings_path = embed_dir / "embeddings.jsonl"
    if prompt_more(input, prompt="Preview chunk→vector pairs now? [Enter=yes, N+Enter=no]: "):
        _preview_embeddings(embeddings_path)

    print("")
    print("Vector store:")
    print("  1) LanceDB (default, local files)")
    print("  2) Qdrant (requires user-managed server + .env)")
    print("  3) Pinecone (cloud + .env)")
    store_choice = _prompt_choice("Select [1-3] [default: 1]: ", 1, 3, default=1)
    store = {1: "lancedb", 2: "qdrant", 3: "pinecone"}[store_choice]
    store_name = store

    table_or_collection = "chunks"
    namespace = None
    if store == "qdrant":
        table_or_collection = f"{env.qdrant_collection_prefix}_{novel.slug}_{method}_{embedder.info().backend}"
        print("Qdrant: ensure QDRANT_URL (and optional QDRANT_API_KEY) are set in .env.")
    if store == "pinecone":
        namespace = f"{novel.slug}-{method}-{embedder.info().backend}"
        print("Pinecone: ensure PINECONE_API_KEY and PINECONE_INDEX are set in .env.")

    store_dir = store_run_dir(
        paths.runs_dir,
        novel_slug=novel.slug,
        chunk_method=method,
        embed_backend=embedder.info().backend,
        embed_model=embedder.info().model,
        store_name=store_name,
    )
    index_embeddings(
        embeddings_path=embeddings_path,
        store_dir=store_dir,
        params=IndexParams(store=store, table_or_collection=table_or_collection, namespace=namespace),
        env=env,
    )

    run_query_stage(
        store_dir=store_dir,
        embedder=embedder,
        env=env,
        input_fn=input,
        print_fn=print,
        initial_query_params=QueryParams(top_k=5, show_vectors=True, no_llm=False, vector_summary=False),
    )


def run_query_stage(
    *,
    store_dir: Path,
    embedder: Embedder,
    env: Env,
    input_fn=input,
    print_fn=print,
    initial_query_params: QueryParams,
) -> None:
    run_query_menu(
        store_dir=store_dir,
        embedder=embedder,
        env=env,
        input_fn=input_fn,
        print_fn=print_fn,
        initial_query_params=initial_query_params,
        run_query_fn=run_query,
    )


def run_query_menu(
    *,
    store_dir: Path,
    embedder: Embedder,
    env: Env,
    input_fn=input,
    print_fn=print,
    initial_query_params: QueryParams,
    run_query_fn=run_query,
) -> None:
    current = QueryParams(
        top_k=initial_query_params.top_k,
        show_vectors=initial_query_params.show_vectors,
        no_llm=initial_query_params.no_llm,
        vector_summary=initial_query_params.vector_summary,
    )
    last_question: str | None = None
    last_used_params: QueryParams | None = None

    while True:
        _display_main_menu(print_fn=print_fn)
        choice = input_fn("Select [1-4]: ").strip().lower()

        if choice in {"4", "q", "quit"}:
            return
        if choice == "1":
            question = _handle_ask_question(input_fn=input_fn, print_fn=print_fn)
            if question is None:
                continue
            try:
                run_query_fn(
                    store_dir=store_dir,
                    question=question,
                    embedder=embedder,
                    env=env,
                    params=current,
                    input_fn=input_fn,
                    print_fn=print_fn,
                )
                last_question = question
                last_used_params = current
            except Exception as e:
                print_fn(f"Query failed: {e}")
            continue
        if choice == "2":
            if last_question is None or last_used_params is None:
                print_fn("No previous question.")
                continue
            try:
                run_query_fn(
                    store_dir=store_dir,
                    question=last_question,
                    embedder=embedder,
                    env=env,
                    params=last_used_params,
                    input_fn=input_fn,
                    print_fn=print_fn,
                )
            except Exception as e:
                print_fn(f"Query failed: {e}")
            continue
        if choice == "3":
            current = _handle_query_settings(current, input_fn=input_fn, print_fn=print_fn)
            continue

        print_fn("Invalid selection. Enter 1, 2, 3, 4, q, or quit.")


def _display_main_menu(*, print_fn=print) -> None:
    print_fn("")
    print_fn("1) Ask a question")
    print_fn("2) Repeat last question")
    print_fn("3) Query settings")
    print_fn("4) Exit")


def _handle_ask_question(*, input_fn=input, print_fn=print) -> str | None:
    print_fn("")
    print_fn("Suggested questions:")
    suggested = [
        "What motivates the protagonist, and what are the consequences?",
        "What are the major themes in this novel?",
        "Summarize the relationship between two central characters.",
    ]
    for i, q in enumerate(suggested, 1):
        print_fn(f"  {i}) {q}")
    raw_q = input_fn("Pick [1-3] or type your own (blank=back): ").strip()
    if raw_q == "":
        return None
    if raw_q in {"1", "2", "3"}:
        return suggested[int(raw_q) - 1]
    return raw_q


def _handle_query_settings(current: QueryParams, *, input_fn=input, print_fn=print) -> QueryParams:
    updated = current
    while True:
        print_fn("")
        print_fn(
            "Current settings: "
            f"top_k={updated.top_k}, show_vectors={updated.show_vectors}, "
            f"vector_summary={updated.vector_summary}, no_llm={updated.no_llm}"
        )
        print_fn("Note: if no_llm=True, retrieval + vector->text still run, but LLM summary/answer are skipped.")
        print_fn("1) Set top_k")
        print_fn("2) Toggle show_vectors")
        print_fn("3) Toggle vector_summary")
        print_fn("4) Toggle no_llm")
        print_fn("5) Back")
        choice = input_fn("Select [1-5]: ").strip()

        if choice == "1":
            while True:
                raw = input_fn("Enter top_k [1-50]: ").strip()
                try:
                    top_k = int(raw)
                except ValueError:
                    print_fn("Enter an integer between 1 and 50.")
                    continue
                if 1 <= top_k <= 50:
                    updated = QueryParams(
                        top_k=top_k,
                        show_vectors=updated.show_vectors,
                        no_llm=updated.no_llm,
                        vector_summary=updated.vector_summary,
                    )
                    break
                print_fn("Enter an integer between 1 and 50.")
            continue
        if choice == "2":
            updated = QueryParams(
                top_k=updated.top_k,
                show_vectors=not updated.show_vectors,
                no_llm=updated.no_llm,
                vector_summary=updated.vector_summary,
            )
            continue
        if choice == "3":
            updated = QueryParams(
                top_k=updated.top_k,
                show_vectors=updated.show_vectors,
                no_llm=updated.no_llm,
                vector_summary=not updated.vector_summary,
            )
            continue
        if choice == "4":
            updated = QueryParams(
                top_k=updated.top_k,
                show_vectors=updated.show_vectors,
                no_llm=not updated.no_llm,
                vector_summary=updated.vector_summary,
            )
            continue
        if choice == "5":
            return updated
        print_fn("Invalid selection. Enter 1, 2, 3, 4, or 5.")


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


def _select_embedder(project_root: Path, env, *, purpose: str):
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

    # Cloud
    if not env.openai_base_url or not env.openai_api_key or not env.openai_embed_model:
        raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_EMBED_MODEL in .env.")
    try:
        from .embeddings.cloud_embed import OpenAICompatibleEmbedder
    except Exception as e:
        raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
    return OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=env.openai_embed_model)


def _embedder_from_saved(embed_backend: str, embed_model: str, env: Env) -> Embedder:
    if embed_backend == "ollama":
        if not env.ollama_base_url:
            raise RuntimeError("OLLAMA_BASE_URL missing in .env.")
        return OllamaEmbedder(base_url=env.ollama_base_url, model=embed_model)
    if embed_backend == "st":
        return SentenceTransformersEmbedder(model_name=embed_model)
    if embed_backend == "cloud":
        if not env.openai_base_url or not env.openai_api_key:
            raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL and OPENAI_API_KEY in .env.")
        try:
            from .embeddings.cloud_embed import OpenAICompatibleEmbedder
        except Exception as e:
            raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
        return OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=embed_model)
    raise RuntimeError(f"Unsupported embedding backend in saved run: {embed_backend}")


def _format_indexed_run(run: IndexedRunSummary) -> str:
    chunk_summary = _chunk_params_summary(run.chunk_params)
    table = run.table_or_collection or "-"
    ns = run.namespace or "-"
    count = "?" if run.count is None else str(run.count)
    dim = "?" if run.vector_dim is None else str(run.vector_dim)
    last = dt.datetime.fromtimestamp(run.last_activity_ts).strftime("%Y-%m-%d %H:%M") if run.last_activity_ts > 0 else "-"
    model = _truncate(run.embed_model, 44)
    return (
        f"{run.novel_slug}/{run.chunk_method} params={chunk_summary} | "
        f"embed={run.embed_backend}:{model} | "
        f"store={run.store} table={table} ns={ns} | "
        f"count={count} dim={dim} queries={run.queries_count} last={last}"
    )


def _chunk_params_summary(chunk_params: dict | None) -> str:
    if not chunk_params:
        return "?"
    keys = [k for k in ["target_chars", "overlap_ratio", "overlap_sentences", "similarity_drop_threshold"] if k in chunk_params]
    if not keys:
        return "?"
    return ",".join(f"{k}={chunk_params[k]}" for k in keys)


def _truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _preview_chunks(chunks_path: Path) -> None:
    lines: list[str] = []
    for r in iter_jsonl(chunks_path):
        cid = r["chunk_id"]
        txt = (r["text"] or "").strip().replace("\n", " ")
        lines.append(f"[{cid}] {txt[:400]}" + ("…" if len(txt) > 400 else ""))
    page_iterable(lines, page_size=3, input_fn=input, print_fn=print, header=f"Preview: {chunks_path}")


def _preview_embeddings(embeddings_path: Path) -> None:
    for r in iter_jsonl(embeddings_path):
        print(f"Chunk: {r['chunk_id']}")
        print((r["text"] or "")[:200].replace("\n", " ") + ("…" if len(r["text"] or '') > 200 else ""))
        vec = r["embedding"]
        page_vector_preview(vec, input_fn=input, print_fn=print, label="chunk_vector")
        if not prompt_more(input, prompt="Next chunk? [Enter=yes, N+Enter=no]: "):
            break
