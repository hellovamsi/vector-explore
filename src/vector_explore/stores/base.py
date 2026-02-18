from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoreInfo:
    name: str
    location: str


class VectorStore(Protocol):
    def info(self) -> StoreInfo: ...

    def upsert(self, items: list[dict]) -> None: ...

    def query(self, vector: list[float], *, top_k: int) -> list[dict]: ...

