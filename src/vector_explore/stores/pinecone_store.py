from __future__ import annotations

from dataclasses import dataclass

from .base import StoreInfo, VectorStore


@dataclass
class PineconeStore(VectorStore):
    api_key: str
    index_name: str
    namespace: str = "default"

    def info(self) -> StoreInfo:
        return StoreInfo(name="pinecone", location=self.index_name)

    def _index(self):
        try:
            from pinecone import Pinecone
        except Exception as e:
            raise RuntimeError("Pinecone client not installed. Install with: pip install -e '.[pinecone]'") from e
        pc = Pinecone(api_key=self.api_key)
        return pc.Index(self.index_name)

    def upsert(self, items: list[dict]) -> None:
        index = self._index()
        vectors = []
        for it in items:
            metadata = it.get("metadata") or {}
            metadata = {**metadata, "text": it.get("text")}
            vectors.append({"id": str(it["id"]), "values": it["vector"], "metadata": metadata})
        index.upsert(vectors=vectors, namespace=self.namespace)

    def query(self, vector: list[float], *, top_k: int) -> list[dict]:
        index = self._index()
        res = index.query(vector=vector, top_k=top_k, include_values=True, include_metadata=True, namespace=self.namespace)
        out: list[dict] = []
        for m in (res.get("matches") or []):
            md = m.get("metadata") or {}
            out.append(
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "vector": m.get("values"),
                    "text": md.get("text"),
                    "metadata": {k: v for k, v in md.items() if k != "text"},
                }
            )
        return out

