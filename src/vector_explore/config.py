from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Env:
    ollama_base_url: str
    ollama_embed_model: str
    ollama_chat_model: str

    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection_prefix: str

    pinecone_api_key: str | None
    pinecone_index: str | None

    openai_base_url: str | None
    openai_api_key: str | None
    openai_embed_model: str | None
    openai_chat_model: str | None


def load_env(project_root: Path) -> Env:
    load_dotenv(project_root / ".env")

    return Env(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "mistral:7b"),
        qdrant_url=_none_if_empty(os.getenv("QDRANT_URL")),
        qdrant_api_key=_none_if_empty(os.getenv("QDRANT_API_KEY")),
        qdrant_collection_prefix=os.getenv("QDRANT_COLLECTION_PREFIX", "vector_explore"),
        pinecone_api_key=_none_if_empty(os.getenv("PINECONE_API_KEY")),
        pinecone_index=_none_if_empty(os.getenv("PINECONE_INDEX")),
        openai_base_url=_none_if_empty(os.getenv("OPENAI_BASE_URL")),
        openai_api_key=_none_if_empty(os.getenv("OPENAI_API_KEY")),
        openai_embed_model=_none_if_empty(os.getenv("OPENAI_EMBED_MODEL")),
        openai_chat_model=_none_if_empty(os.getenv("OPENAI_CHAT_MODEL")),
    )


def _none_if_empty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None

