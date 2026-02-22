from __future__ import annotations

from pathlib import Path

from vector_explore.query import QueryParams
from vector_explore.wizard import run_query_menu


class _StubEmbedder:
    def info(self):
        return type("Info", (), {"backend": "stub", "model": "stub", "dim": 1})

    def embed_texts(self, texts):
        return [[1.0] for _ in texts]


class _StubEnv:
    ollama_base_url = "http://localhost:11434"
    ollama_embed_model = "x"
    ollama_chat_model = "x"
    qdrant_url = None
    qdrant_api_key = None
    qdrant_collection_prefix = "x"
    pinecone_api_key = None
    pinecone_index = None
    openai_base_url = None
    openai_api_key = None
    openai_embed_model = None
    openai_chat_model = None


def test_query_menu_basic_ask_then_exit():
    scripted = iter(["1", "3", "4"])
    calls: list[dict] = []
    out: list[str] = []

    def inp(_prompt: str) -> str:
        return next(scripted)

    def pr(s: str) -> None:
        out.append(s)

    def run_query_stub(**kwargs):
        calls.append(kwargs)
        return Path("/tmp/fake")

    run_query_menu(
        store_dir=Path("/tmp/store"),
        embedder=_StubEmbedder(),
        env=_StubEnv(),
        input_fn=inp,
        print_fn=pr,
        initial_query_params=QueryParams(),
        run_query_fn=run_query_stub,
    )

    assert len(calls) == 1
    assert calls[0]["question"] == "Summarize the relationship between two central characters."


def test_query_menu_repeat_last_question():
    scripted = iter(["1", "1", "2", "4"])
    calls: list[dict] = []

    def inp(_prompt: str) -> str:
        return next(scripted)

    def run_query_stub(**kwargs):
        calls.append(kwargs)
        return Path("/tmp/fake")

    run_query_menu(
        store_dir=Path("/tmp/store"),
        embedder=_StubEmbedder(),
        env=_StubEnv(),
        input_fn=inp,
        print_fn=lambda _s: None,
        initial_query_params=QueryParams(),
        run_query_fn=run_query_stub,
    )

    assert len(calls) == 2
    assert calls[0]["question"] == calls[1]["question"]


def test_query_menu_settings_modification():
    scripted = iter(["3", "1", "10", "5", "1", "custom question", "4"])
    calls: list[dict] = []

    def inp(_prompt: str) -> str:
        return next(scripted)

    def run_query_stub(**kwargs):
        calls.append(kwargs)
        return Path("/tmp/fake")

    run_query_menu(
        store_dir=Path("/tmp/store"),
        embedder=_StubEmbedder(),
        env=_StubEnv(),
        input_fn=inp,
        print_fn=lambda _s: None,
        initial_query_params=QueryParams(),
        run_query_fn=run_query_stub,
    )

    assert len(calls) == 1
    assert calls[0]["params"].top_k == 10


def test_query_menu_invalid_input_then_exit():
    scripted = iter(["9", "4"])
    calls: list[dict] = []
    out: list[str] = []

    def inp(_prompt: str) -> str:
        return next(scripted)

    def pr(s: str) -> None:
        out.append(s)

    def run_query_stub(**kwargs):
        calls.append(kwargs)
        return Path("/tmp/fake")

    run_query_menu(
        store_dir=Path("/tmp/store"),
        embedder=_StubEmbedder(),
        env=_StubEnv(),
        input_fn=inp,
        print_fn=pr,
        initial_query_params=QueryParams(),
        run_query_fn=run_query_stub,
    )

    assert calls == []
    assert any("Invalid selection" in line for line in out)
