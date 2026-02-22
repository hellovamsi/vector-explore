from __future__ import annotations

import argparse
from pathlib import Path

from .app import (
    ChunkCommand,
    CompareCommand,
    DownloadCommand,
    EmbedCommand,
    IndexCommand,
    QueryCommand,
    WizardCommand,
    build_context,
    execute_command,
)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    print("Read README.md Setup before running (recommended).")
    ctx = build_context(Path.cwd())
    execute_command(ctx, _command_from_args(args))


def _build_parser() -> argparse.ArgumentParser:
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

    return parser


def _command_from_args(args: argparse.Namespace):
    if args.cmd in {None, "wizard"}:
        return WizardCommand(
            no_progress=getattr(args, "no_progress", False),
            embed_batch_size=getattr(args, "embed_batch_size", 64),
            index_batch_size=getattr(args, "index_batch_size", 256),
        )
    if args.cmd == "download":
        return DownloadCommand(novel=args.novel, no_progress=args.no_progress)
    if args.cmd == "chunk":
        return ChunkCommand(
            novel=args.novel,
            method=args.method,
            target_chars=args.target_chars,
            overlap_ratio=args.overlap_ratio,
            overlap_sentences=args.overlap_sentences,
            similarity_drop_threshold=args.similarity_drop_threshold,
            semantic_backend=args.semantic_backend,
            semantic_model=args.semantic_model,
            no_progress=args.no_progress,
            embed_batch_size=args.embed_batch_size,
        )
    if args.cmd == "embed":
        return EmbedCommand(
            novel=args.novel,
            method=args.method,
            input_path=Path(args.input),
            backend=args.backend,
            model=args.model,
            no_progress=args.no_progress,
            embed_batch_size=args.embed_batch_size,
        )
    if args.cmd == "index":
        return IndexCommand(
            novel=args.novel,
            method=args.method,
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            input_path=Path(args.input),
            store=args.store,
            table_or_collection=args.table_or_collection,
            namespace=args.namespace,
            no_progress=args.no_progress,
            index_batch_size=args.index_batch_size,
        )
    if args.cmd == "query":
        return QueryCommand(
            novel=args.novel,
            method=args.method,
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            store=args.store,
            question=args.question,
            top_k=args.top_k,
            no_llm=args.no_llm,
            no_vectors=args.no_vectors,
            vector_summary=args.vector_summary,
        )
    if args.cmd == "compare":
        return CompareCommand(
            novel=args.novel,
            methods=[m.strip() for m in args.methods.split(",") if m.strip()],
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            store=args.store,
            question=args.question,
            top_k=args.top_k,
        )
    raise SystemExit(f"Unknown command: {args.cmd}")


def _add_progress_flags(p: argparse.ArgumentParser, *, with_index: bool) -> None:
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=64)
    if with_index:
        p.add_argument("--index-batch-size", type=int, default=256)

