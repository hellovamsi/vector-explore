from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbedderInfo:
    backend: str
    model: str
    dim: int | None


class Embedder(Protocol):
    def info(self) -> EmbedderInfo: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

