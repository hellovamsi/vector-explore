from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

from .base import Embedder, EmbedderInfo


@lru_cache(maxsize=8)
def load_st_model(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "SentenceTransformers not installed. Install with: pip install -e '.[st]'"
        ) from e
    return SentenceTransformer(model_name, device=device)


@dataclass(frozen=True)
class SentenceTransformersEmbedder(Embedder):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(backend="st", model=self.model_name, dim=None)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = load_st_model(self.model_name, self.device)
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row.tolist()] for row in embs]
