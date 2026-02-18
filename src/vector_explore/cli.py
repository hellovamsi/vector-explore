from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import chunk_novel
from .config import load_env
from .download_gutenberg import get_novel, download_if_missing
from .embedding import embed_chunks
from .embeddings.ollama_embed import OllamaEmbedder
from .embeddings.st_embed import SentenceTransformersEmbedder
from .indexing import IndexParams, index_embeddings
from .paths import default_paths, store_run_dir
from .query import QueryParams, run_query
from .wizard import run_wizard


def main() -> None:
    parser = argparse.ArgumentParser(prog="vector-explore", description="Educational CLI for RAG building blocks.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("wizard", help="Interactive wizard (default)")

    p_dl = sub.add_parser("download", help="Download a Gutenberg novel (cached).")
    p_dl.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])

    p_chunk = sub.add_parser("chunk", help="Chunk the novel and write chunks.jsonl.")
    p_chunk.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_chunk.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_chunk.add_argument("--target-chars", type=int, default=900)
    p_chunk.add_argument("--overlap-ratio", type=float, default=0.15)
    p_chunk.add_argument("--overlap-sentences", type=int, default=2)
    p_chunk.add_argument("--similarity-drop-threshold", type=float, default=0.65)
    p_chunk.add_argument("--semantic-backend", choices=["ollama", "st"], default="ollama")
    p_chunk.add_argument("--semantic-model", default=None)

    p_embed = sub.add_parser("embed", help="Embed chunks.jsonl and write embeddings.jsonl.")
    p_embed.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_embed.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_embed.add_argument("--input", required=True, help="Path to chunks.jsonl")
    p_embed.add_argument("--backend", required=True, choices=["ollama", "st", "cloud"])
    p_embed.add_argument("--model", required=False)

    p_index = sub.add_parser("index", help="Index embeddings.jsonl into a vector store.")
    p_index.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_index.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_index.add_argument("--embed-backend", required=True, choices=["ollama", "st"])
    p_index.add_argument("--embed-model", required=True)
    p_index.add_argument("--input", required=True, help="Path to embeddings.jsonl")
    p_index.add_argument("--store", required=True, choices=["lancedb", "qdrant", "pinecone"])
    p_index.add_argument("--table-or-collection", default="chunks")
    p_index.add_argument("--namespace", default=None)

    p_q = sub.add_parser("query", help="Query a vector store and show internals.")
    p_q.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_q.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_q.add_argument("--embed-backend", required=True, choices=["ollama", "st", "cloud"])
    p_q.add_argument("--embed-model", required=True)
    p_q.add_argument("--store", required=True, choices=["lancedb", "qdrant", "pinecone"])
    p_q.add_argument("--question", required=True)
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.add_argument("--no-llm", action="store_true")
    p_q.add_argument("--no-vectors", action="store_true")
    p_q.add_argument("--vector-summary", action="store_true")

    p_cmp = sub.add_parser("compare", help="Compare retrieval across chunking methods (requires indexes exist).")
    p_cmp.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_cmp.add_argument("--methods", required=True, help="Comma-separated: fixed,sentences,semantic")
    p_cmp.add_argument("--embed-backend", required=True, choices=["ollama", "st", "cloud"])
    p_cmp.add_argument("--embed-model", required=True)
    p_cmp.add_argument("--store", required=True, choices=["lancedb", "qdrant", "pinecone"])
    p_cmp.add_argument("--question", required=True)
    p_cmp.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    print("Read README.md Setup before running (recommended).")
    cwd = Path.cwd()
    project_root = _find_project_root(cwd) or cwd

    env = load_env(project_root)
    paths = default_paths(project_root)

    if args.cmd in {None, "wizard"}:
        run_wizard(project_root)
        return

    if args.cmd == "download":
        novel = get_novel(args.novel)
        download_if_missing(project_root, novel)
        return

    if args.cmd == "chunk":
        novel = get_novel(args.novel)
        novel_path = download_if_missing(project_root, novel)
        semantic_embedder = None
        if args.method == "semantic":
            model = args.semantic_model or (env.ollama_embed_model if args.semantic_backend == "ollama" else "sentence-transformers/all-MiniLM-L6-v2")
            if args.semantic_backend == "ollama":
                semantic_embedder = OllamaEmbedder(base_url=env.ollama_base_url, model=model)
            else:
                semantic_embedder = SentenceTransformersEmbedder(model_name=model)
        params = {"target_chars": args.target_chars}
        if args.method == "fixed":
            params["overlap_ratio"] = args.overlap_ratio
        elif args.method == "sentences":
            params["overlap_sentences"] = args.overlap_sentences
        else:
            params["overlap_sentences"] = args.overlap_sentences
            params["similarity_drop_threshold"] = args.similarity_drop_threshold
        chunk_novel(
            project_root=project_root,
            runs_dir=paths.runs_dir,
            novel_slug=args.novel,
            novel_path=novel_path,
            method=args.method,
            params=params,
            semantic_embedder=semantic_embedder,
        )
        return

    if args.cmd == "embed":
        chunks_path = Path(args.input)
        if args.backend == "ollama":
            model = args.model or env.ollama_embed_model
            embedder = OllamaEmbedder(base_url=env.ollama_base_url, model=model)
        elif args.backend == "st":
            model = args.model or "sentence-transformers/all-MiniLM-L6-v2"
            embedder = SentenceTransformersEmbedder(model_name=model)
        else:
            if not env.openai_base_url or not env.openai_api_key:
                raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL and OPENAI_API_KEY in .env.")
            model = args.model or env.openai_embed_model
            if not model:
                raise RuntimeError("Cloud embeddings require --model or OPENAI_EMBED_MODEL in .env.")
            try:
                from .embeddings.cloud_embed import OpenAICompatibleEmbedder
            except Exception as e:
                raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
            embedder = OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=model)
        embed_chunks(runs_dir=paths.runs_dir, novel_slug=args.novel, chunk_method=args.method, chunks_path=chunks_path, embedder=embedder)
        return

    if args.cmd == "index":
        embeddings_path = Path(args.input)
        store_dir = store_run_dir(
            paths.runs_dir,
            novel_slug=args.novel,
            chunk_method=args.method,
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            store_name=args.store,
        )
        index_embeddings(
            embeddings_path=embeddings_path,
            store_dir=store_dir,
            params=IndexParams(store=args.store, table_or_collection=args.table_or_collection, namespace=args.namespace),
            env=env,
        )
        return

    if args.cmd == "query":
        if args.embed_backend == "ollama":
            embedder = OllamaEmbedder(base_url=env.ollama_base_url, model=args.embed_model)
        elif args.embed_backend == "st":
            embedder = SentenceTransformersEmbedder(model_name=args.embed_model)
        else:
            if not env.openai_base_url or not env.openai_api_key:
                raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL and OPENAI_API_KEY in .env.")
            try:
                from .embeddings.cloud_embed import OpenAICompatibleEmbedder
            except Exception as e:
                raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
            embedder = OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=args.embed_model)
        store_dir = store_run_dir(
            paths.runs_dir,
            novel_slug=args.novel,
            chunk_method=args.method,
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            store_name=args.store,
        )
        qparams = QueryParams(
            top_k=args.top_k,
            show_vectors=not args.no_vectors,
            no_llm=args.no_llm,
            vector_summary=args.vector_summary,
        )
        run_query(store_dir=store_dir, question=args.question, embedder=embedder, env=env, params=qparams)
        return

    if args.cmd == "compare":
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        if not methods:
            raise RuntimeError("--methods must be non-empty.")
        if args.embed_backend == "ollama":
            embedder = OllamaEmbedder(base_url=env.ollama_base_url, model=args.embed_model)
        elif args.embed_backend == "st":
            embedder = SentenceTransformersEmbedder(model_name=args.embed_model)
        else:
            if not env.openai_base_url or not env.openai_api_key:
                raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL and OPENAI_API_KEY in .env.")
            try:
                from .embeddings.cloud_embed import OpenAICompatibleEmbedder
            except Exception as e:
                raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
            embedder = OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=args.embed_model)

        def no_more(_p: str) -> str:
            return "n"

        for m in methods:
            store_dir = store_run_dir(
                paths.runs_dir,
                novel_slug=args.novel,
                chunk_method=m,
                embed_backend=args.embed_backend,
                embed_model=args.embed_model,
                store_name=args.store,
            )
            print("")
            print(f"=== method={m} store={args.store} ===")
            run_query(
                store_dir=store_dir,
                question=args.question,
                embedder=embedder,
                env=env,
                params=QueryParams(top_k=args.top_k, show_vectors=False, no_llm=True, vector_summary=False),
                input_fn=no_more,
                print_fn=print,
            )
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


def _find_project_root(start: Path) -> Path | None:
    cur = start.resolve()
    for _ in range(6):
        if (cur / "pyproject.toml").exists() and (cur / "src" / "vector_explore").exists():
            return cur
        cur = cur.parent
    return None
