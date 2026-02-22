from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .chunking import chunk_novel
from .config import Env, load_env
from .download_gutenberg import get_novel, download_if_missing
from .embedding import embed_chunks
from .embeddings.ollama_embed import OllamaEmbedder
from .embeddings.st_embed import SentenceTransformersEmbedder
from .indexing import IndexParams, index_embeddings
from .paths import embed_key
from .query import QueryParams, run_query
from .wizard import run_wizard


@dataclass(frozen=True)
class AppContext:
    project_root: Path
    env: Env


@dataclass(frozen=True)
class WizardCommand:
    no_progress: bool = False
    embed_batch_size: int = 64
    index_batch_size: int = 256


@dataclass(frozen=True)
class DownloadCommand:
    novel: str
    no_progress: bool = False


@dataclass(frozen=True)
class ChunkCommand:
    novel: str
    method: str
    target_chars: int = 900
    overlap_ratio: float = 0.15
    overlap_sentences: int = 2
    similarity_drop_threshold: float = 0.65
    semantic_backend: str = "ollama"
    semantic_model: str | None = None
    no_progress: bool = False
    embed_batch_size: int = 64


@dataclass(frozen=True)
class EmbedCommand:
    novel: str
    method: str
    input_path: Path
    backend: str
    model: str | None = None
    no_progress: bool = False
    embed_batch_size: int = 64


@dataclass(frozen=True)
class IndexCommand:
    novel: str
    method: str
    embed_backend: str
    embed_model: str
    input_path: Path
    store: str
    table_or_collection: str = "chunks"
    namespace: str | None = None
    no_progress: bool = False
    index_batch_size: int = 256


@dataclass(frozen=True)
class QueryCommand:
    novel: str
    method: str
    embed_backend: str
    embed_model: str
    store: str
    question: str
    top_k: int = 5
    no_llm: bool = False
    no_vectors: bool = False
    vector_summary: bool = False


@dataclass(frozen=True)
class CompareCommand:
    novel: str
    methods: list[str]
    embed_backend: str
    embed_model: str
    store: str
    question: str
    top_k: int = 3


Command = Union[WizardCommand, DownloadCommand, ChunkCommand, EmbedCommand, IndexCommand, QueryCommand, CompareCommand]


def build_context(cwd: Path) -> AppContext:
    project_root = find_project_root(cwd) or cwd
    return AppContext(project_root=project_root, env=load_env(project_root))


def execute_command(ctx: AppContext, cmd: Command) -> None:
    if isinstance(cmd, WizardCommand):
        run_wizard(
            ctx.project_root,
            no_progress=cmd.no_progress,
            embed_batch_size=cmd.embed_batch_size,
            index_batch_size=cmd.index_batch_size,
        )
        return

    if isinstance(cmd, DownloadCommand):
        novel = get_novel(cmd.novel)
        download_if_missing(ctx.project_root, novel, progress_enabled=progress_enabled(cmd.no_progress))
        return

    if isinstance(cmd, ChunkCommand):
        novel = get_novel(cmd.novel)
        novel_path = download_if_missing(ctx.project_root, novel, progress_enabled=progress_enabled(cmd.no_progress))
        semantic_embedder = None
        if cmd.method == "semantic":
            model = cmd.semantic_model or (ctx.env.ollama_embed_model if cmd.semantic_backend == "ollama" else "sentence-transformers/all-MiniLM-L6-v2")
            if cmd.semantic_backend == "ollama":
                semantic_embedder = OllamaEmbedder(base_url=ctx.env.ollama_base_url, model=model)
            else:
                semantic_embedder = SentenceTransformersEmbedder(model_name=model)
        params: dict[str, int | float] = {"target_chars": cmd.target_chars}
        if cmd.method == "fixed":
            params["overlap_ratio"] = cmd.overlap_ratio
        elif cmd.method == "sentences":
            params["overlap_sentences"] = cmd.overlap_sentences
        else:
            params["overlap_sentences"] = cmd.overlap_sentences
            params["similarity_drop_threshold"] = cmd.similarity_drop_threshold
        chunk_novel(
            project_root=ctx.project_root,
            novel_slug=cmd.novel,
            novel_path=novel_path,
            method=cmd.method,
            params=params,
            semantic_embedder=semantic_embedder,
            embed_batch_size=cmd.embed_batch_size,
            progress_enabled=progress_enabled(cmd.no_progress),
        )
        return

    if isinstance(cmd, EmbedCommand):
        embedder = select_embedder(
            backend=cmd.backend,
            model=cmd.model,
            env=ctx.env,
            default_st_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        embed_chunks(
            project_root=ctx.project_root,
            novel_slug=cmd.novel,
            chunk_method=cmd.method,
            chunks_path=cmd.input_path,
            embedder=embedder,
            embed_batch_size=cmd.embed_batch_size,
            progress_enabled=progress_enabled(cmd.no_progress),
        )
        return

    if isinstance(cmd, IndexCommand):
        ekey = embed_key(cmd.embed_backend, cmd.embed_model)
        index_embeddings(
            project_root=ctx.project_root,
            novel_slug=cmd.novel,
            chunk_method=cmd.method,
            embed_key_value=ekey,
            store_name=cmd.store,
            embeddings_path=cmd.input_path,
            params=IndexParams(store=cmd.store, table_or_collection=cmd.table_or_collection, namespace=cmd.namespace),
            env=ctx.env,
            index_batch_size=cmd.index_batch_size,
            progress_enabled=progress_enabled(cmd.no_progress),
        )
        return

    if isinstance(cmd, QueryCommand):
        embedder = select_embedder(backend=cmd.embed_backend, model=cmd.embed_model, env=ctx.env)
        ekey = embed_key(cmd.embed_backend, cmd.embed_model)
        qparams = QueryParams(
            top_k=cmd.top_k,
            show_vectors=not cmd.no_vectors,
            no_llm=cmd.no_llm,
            vector_summary=cmd.vector_summary,
        )
        run_query(
            project_root=ctx.project_root,
            novel_slug=cmd.novel,
            chunk_method=cmd.method,
            embed_key=ekey,
            store_name=cmd.store,
            question=cmd.question,
            embedder=embedder,
            env=ctx.env,
            params=qparams,
        )
        return

    if isinstance(cmd, CompareCommand):
        if not cmd.methods:
            raise RuntimeError("--methods must be non-empty.")
        embedder = select_embedder(backend=cmd.embed_backend, model=cmd.embed_model, env=ctx.env)
        ekey = embed_key(cmd.embed_backend, cmd.embed_model)

        def no_more(_p: str) -> str:
            return "n"

        for method in cmd.methods:
            print("")
            print(f"=== method={method} store={cmd.store} ===")
            run_query(
                project_root=ctx.project_root,
                novel_slug=cmd.novel,
                chunk_method=method,
                embed_key=ekey,
                store_name=cmd.store,
                question=cmd.question,
                embedder=embedder,
                env=ctx.env,
                params=QueryParams(top_k=cmd.top_k, show_vectors=False, no_llm=True, vector_summary=False),
                input_fn=no_more,
                print_fn=print,
            )
        return

    raise RuntimeError(f"Unsupported command type: {type(cmd)}")


def progress_enabled(no_progress: bool) -> bool:
    import sys

    return (not no_progress) and bool(getattr(sys.stdout, "isatty", lambda: False)())


def select_embedder(*, backend: str, model: str | None, env: Env, default_st_model: str | None = None):
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


def find_project_root(start: Path) -> Path | None:
    cur = start.resolve()
    for _ in range(6):
        if (cur / "pyproject.toml").exists() and (cur / "src" / "vector_explore").exists():
            return cur
        cur = cur.parent
    return None

