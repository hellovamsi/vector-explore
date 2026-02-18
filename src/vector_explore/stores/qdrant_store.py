from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import StoreInfo, VectorStore


@dataclass
class QdrantStore(VectorStore):
    url: str
    api_key: str | None
    collection: str
    vector_dim: int

    def info(self) -> StoreInfo:
        return StoreInfo(name="qdrant", location=self.url)

    def _client(self):
        try:
            from qdrant_client import QdrantClient
        except Exception as e:
            raise RuntimeError("Qdrant client not installed. Install with: pip install -e '.[qdrant]'") from e
        return QdrantClient(url=self.url, api_key=self.api_key)

    def _ensure_collection(self) -> None:
        from qdrant_client.http import models as qm

        client = self._client()
        if client.collection_exists(self.collection):
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.vector_dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, items: list[dict]) -> None:
        from qdrant_client.http import models as qm

        self._ensure_collection()
        client = self._client()
        points: list[qm.PointStruct] = []
        for it in items:
            points.append(
                qm.PointStruct(
                    id=it["id"],
                    vector=it["vector"],
                    payload={"text": it.get("text"), "metadata": it.get("metadata")},
                )
            )
        client.upsert(collection_name=self.collection, points=points)

    def query(self, vector: list[float], *, top_k: int) -> list[dict]:
        client = self._client()
        res = client.search(collection_name=self.collection, query_vector=vector, limit=top_k, with_vectors=True, with_payload=True)
        out: list[dict] = []
        for r in res:
            out.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "vector": r.vector,
                    "text": (r.payload or {}).get("text"),
                    "metadata": (r.payload or {}).get("metadata"),
                }
            )
        return out

