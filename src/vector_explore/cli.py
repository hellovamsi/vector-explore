from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .chunking import chunk_novel
from .config import load_env
from .download_gutenberg import get_novel, download_if_missing
from .embedding import embed_chunks
from .embeddings.ollama_embed import OllamaEmbedder
from .embeddings.st_embed import SentenceTransformersEmbedder
from .indexing import IndexParams, index_embeddings
from .paths import embed_key
from .query import QueryParams, run_query
from .wizard import run_wizard


def main() -> None:
    parser = argparse.ArgumentParser(prog="vector-explore", description="Educational CLI for RAG building blocks.")
    sub = parser.add_subparsers(dest="cmd")

    p_wizard = sub.add_parser("wizard", help="Interactive wizard (default)")
    _add_progress_flags(p_wizard, with_index=True)

    p_dl = sub.add_parser("download", help="Download a Gutenberg novel (cached).")
    p_dl.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_dl.add_argument("--no-progress", action="store_true")

    p_chunk = sub.add_parser("chunk", help="Chunk the novel and write chunks.jsonl.")
    p_chunk.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_chunk.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_chunk.add_argument("--target-chars", type=int, default=900)
    p_chunk.add_argument("--overlap-ratio", type=float, default=0.15)
    p_chunk.add_argument("--overlap-sentences", type=int, default=2)
    p_chunk.add_argument("--similarity-drop-threshold", type=float, default=0.65)
    p_chunk.add_argument("--semantic-backend", choices=["ollama", "st"], default="ollama")
    p_chunk.add_argument("--semantic-model", default=None)
    _add_progress_flags(p_chunk, with_index=False)

    p_embed = sub.add_parser("embed", help="Embed chunks.jsonl and write embeddings.jsonl.")
    p_embed.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_embed.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_embed.add_argument("--input", required=True, help="Path to chunks.jsonl")
    p_embed.add_argument("--backend", required=True, choices=["ollama", "st", "cloud"])
    p_embed.add_argument("--model", required=False)
    _add_progress_flags(p_embed, with_index=False)

    p_index = sub.add_parser("index", help="Index embeddings.jsonl into a vector store.")
    p_index.add_argument("--novel", required=True, choices=["frankenstein", "mobydick", "pride"])
    p_index.add_argument("--method", required=True, choices=["fixed", "sentences", "semantic"])
    p_index.add_argument("--embed-backend", required=True, choices=["ollama", "st", "cloud"])
    p_index.add_argument("--embed-model", required=True)
    p_index.add_argument("--input", required=True, help="Path to embeddings.jsonl")
    p_index.add_argument("--store", required=True, choices=["lancedb", "qdrant", "pinecone"])
    p_index.add_argument("--table-or-collection", default="chunks")
    p_index.add_argument("--namespace", default=None)
    p_index.add_argument("--no-progress", action="store_true")
    p_index.add_argument("--index-batch-size", type=int, default=256)

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

    p_mig = sub.add_parser("migrate-layout", help="Migrate artifacts from runs/ to novel-rooted layout.")
    p_mig.add_argument("--mode", choices=["move", "copy"], default="move")
    p_mig.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p_mig.add_argument("--execute", dest="dry_run", action="store_false")

    args = parser.parse_args()
    print("Read README.md Setup before running (recommended).")
    cwd = Path.cwd()
    project_root = _find_project_root(cwd) or cwd

    env = load_env(project_root)

    if args.cmd in {None, "wizard"}:
        run_wizard(
            project_root,
            no_progress=getattr(args, "no_progress", False),
            embed_batch_size=getattr(args, "embed_batch_size", 64),
            index_batch_size=getattr(args, "index_batch_size", 256),
        )
        return

    if args.cmd == "download":
        novel = get_novel(args.novel)
        download_if_missing(project_root, novel, progress_enabled=_progress_enabled(args.no_progress))
        return

    if args.cmd == "chunk":
        novel = get_novel(args.novel)
        novel_path = download_if_missing(project_root, novel, progress_enabled=_progress_enabled(args.no_progress))
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
            novel_slug=args.novel,
            novel_path=novel_path,
            method=args.method,
            params=params,
            semantic_embedder=semantic_embedder,
            embed_batch_size=args.embed_batch_size,
            progress_enabled=_progress_enabled(args.no_progress),
        )
        return

    if args.cmd == "embed":
        chunks_path = Path(args.input)
        embedder = _select_embedder_from_args(
            backend=args.backend,
            model=args.model,
            env=env,
            default_st_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        embed_chunks(
            project_root=project_root,
            novel_slug=args.novel,
            chunk_method=args.method,
            chunks_path=chunks_path,
            embedder=embedder,
            embed_batch_size=args.embed_batch_size,
            progress_enabled=_progress_enabled(args.no_progress),
        )
        return

    if args.cmd == "index":
        embeddings_path = Path(args.input)
        ekey = embed_key(args.embed_backend, args.embed_model)
        index_embeddings(
            project_root=project_root,
            novel_slug=args.novel,
            chunk_method=args.method,
            embed_key_value=ekey,
            store_name=args.store,
            embeddings_path=embeddings_path,
            params=IndexParams(store=args.store, table_or_collection=args.table_or_collection, namespace=args.namespace),
            env=env,
            index_batch_size=args.index_batch_size,
            progress_enabled=_progress_enabled(args.no_progress),
        )
        return

    if args.cmd == "query":
        embedder = _select_embedder_from_args(backend=args.embed_backend, model=args.embed_model, env=env)
        ekey = embed_key(args.embed_backend, args.embed_model)
        qparams = QueryParams(
            top_k=args.top_k,
            show_vectors=not args.no_vectors,
            no_llm=args.no_llm,
            vector_summary=args.vector_summary,
        )
        run_query(
            project_root=project_root,
            novel_slug=args.novel,
            chunk_method=args.method,
            embed_key=ekey,
            store_name=args.store,
            question=args.question,
            embedder=embedder,
            env=env,
            params=qparams,
        )
        return

    if args.cmd == "compare":
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        if not methods:
            raise RuntimeError("--methods must be non-empty.")
        embedder = _select_embedder_from_args(backend=args.embed_backend, model=args.embed_model, env=env)
        ekey = embed_key(args.embed_backend, args.embed_model)

        def no_more(_p: str) -> str:
            return "n"

        for m in methods:
            print("")
            print(f"=== method={m} store={args.store} ===")
            run_query(
                project_root=project_root,
                novel_slug=args.novel,
                chunk_method=m,
                embed_key=ekey,
                store_name=args.store,
                question=args.question,
                embedder=embedder,
                env=env,
                params=QueryParams(top_k=args.top_k, show_vectors=False, no_llm=True, vector_summary=False),
                input_fn=no_more,
                print_fn=print,
            )
        return

    if args.cmd == "migrate-layout":
        migrate_layout(project_root=project_root, mode=args.mode, dry_run=args.dry_run, print_fn=print)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


def _add_progress_flags(p: argparse.ArgumentParser, *, with_index: bool) -> None:
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=64)
    if with_index:
        p.add_argument("--index-batch-size", type=int, default=256)


def _progress_enabled(no_progress: bool) -> bool:
    import sys

    return (not no_progress) and bool(getattr(sys.stdout, "isatty", lambda: False)())


def _select_embedder_from_args(*, backend: str, model: str | None, env, default_st_model: str | None = None):
    if backend == "ollama":
        return OllamaEmbedder(base_url=env.ollama_base_url, model=model or env.ollama_embed_model)
    if backend == "st":
        return SentenceTransformersEmbedder(model_name=model or default_st_model or "sentence-transformers/all-MiniLM-L6-v2")
    if not env.openai_base_url or not env.openai_api_key:
        raise RuntimeError("Cloud embeddings require OPENAI_BASE_URL and OPENAI_API_KEY in .env.")
    resolved_model = model or env.openai_embed_model
    if not resolved_model:
        raise RuntimeError("Cloud embeddings require --model or OPENAI_EMBED_MODEL in .env.")
    try:
        from .embeddings.cloud_embed import OpenAICompatibleEmbedder
    except Exception as e:
        raise RuntimeError("Cloud embeddings require httpx. Install with: pip install -e '.[cloud]'") from e
    return OpenAICompatibleEmbedder(base_url=env.openai_base_url, api_key=env.openai_api_key, model=resolved_model)


def migrate_layout(*, project_root: Path, mode: str, dry_run: bool, print_fn=print) -> None:
    runs_root = project_root / "runs"
    if not runs_root.exists():
        print_fn(f"No runs directory found at {runs_root}.")
        return

    op = _copy_path if mode == "copy" else _move_path
    print_fn(f"Migrating layout from {runs_root} mode={mode} dry_run={dry_run}")

    for novel_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        novel_slug = novel_dir.name
        for chunk_method_dir in sorted(p for p in novel_dir.iterdir() if p.is_dir()):
            chunk_method = chunk_method_dir.name
            new_chunk_dir = project_root / novel_slug / chunk_method
            for name in ["chunks.jsonl", "chunk_params.json"]:
                src = chunk_method_dir / name
                dst = new_chunk_dir / name
                if src.exists():
                    op(src, dst, dry_run=dry_run, print_fn=print_fn)

            for embed_dir in sorted(p for p in chunk_method_dir.iterdir() if p.is_dir() and "-" in p.name):
                ekey = embed_dir.name
                new_embed_dir = project_root / novel_slug / ekey / chunk_method
                for name in ["embeddings.jsonl", "embed_params.json"]:
                    src = embed_dir / name
                    dst = new_embed_dir / name
                    if src.exists():
                        op(src, dst, dry_run=dry_run, print_fn=print_fn)

                for store_dir in sorted(p for p in embed_dir.iterdir() if p.is_dir() and p.name in {"lancedb", "qdrant", "pinecone"}):
                    idx_dir = project_root / novel_slug / store_dir.name / f"{chunk_method}__{ekey}"
                    for name in ["store_params.json", "store_meta.json"]:
                        src = store_dir / name
                        dst = idx_dir / name
                        if src.exists():
                            op(src, dst, dry_run=dry_run, print_fn=print_fn)
                    if store_dir.name == "lancedb":
                        src_db = store_dir / "lancedb"
                        dst_db = idx_dir / "chunks.lance"
                        if src_db.exists():
                            op(src_db, dst_db, dry_run=dry_run, print_fn=print_fn)
                    old_qroot = store_dir / "queries"
                    if old_qroot.exists():
                        for qh_dir in sorted(p for p in old_qroot.iterdir() if p.is_dir()):
                            dst_qdir = project_root / novel_slug / "queries" / qh_dir.name
                            op(qh_dir, dst_qdir, dry_run=dry_run, print_fn=print_fn)


def _copy_path(src: Path, dst: Path, *, dry_run: bool, print_fn=print) -> None:
    print_fn(f"COPY {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _move_path(src: Path, dst: Path, *, dry_run: bool, print_fn=print) -> None:
    print_fn(f"MOVE {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
        else:
            shutil.copy2(src, dst)
            src.unlink()
        return
    shutil.move(str(src), str(dst))


def _find_project_root(start: Path) -> Path | None:
    cur = start.resolve()
    for _ in range(6):
        if (cur / "pyproject.toml").exists() and (cur / "src" / "vector_explore").exists():
            return cur
        cur = cur.parent
    return None
