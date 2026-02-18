from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import StoreInfo, VectorStore


@dataclass
class LanceDBStore(VectorStore):
    path: Path
    table_name: str = "chunks"

    def info(self) -> StoreInfo:
        return StoreInfo(name="lancedb", location=str(self.path))

    def _open(self):
        try:
            import lancedb
        except Exception as e:
            raise RuntimeError("LanceDB not installed. Install with: pip install -e '.[lancedb]'") from e
        self.path.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self.path))

    def upsert(self, items: list[dict]) -> None:
        db = self._open()
        # Expected item fields: id, vector, text, metadata(dict)
        try:
            table = db.open_table(self.table_name)
        except Exception:
            table = db.create_table(self.table_name, data=items)
            return
        table.add(items)

    def query(self, vector: list[float], *, top_k: int) -> list[dict]:
        db = self._open()
        table = db.open_table(self.table_name)
        res = table.search(vector).limit(top_k).to_list()
        out: list[dict] = []
        for r in res:
            out.append(
                {
                    "id": r.get("id"),
                    "score": r.get("_distance") if "_distance" in r else r.get("score"),
                    "vector": r.get("vector"),
                    "text": r.get("text"),
                    "metadata": r.get("metadata"),
                }
            )
        return out

