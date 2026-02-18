from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from .base import Embedder, EmbedderInfo


@dataclass(frozen=True)
class OpenAICompatibleEmbedder(Embedder):
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(backend="cloud", model=self.model, dim=None)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        out: list[list[float]] = []
        with httpx.Client(timeout=self.timeout_s) as client:
            for t in texts:
                resp = client.post(url, headers=headers, json={"model": self.model, "input": t})
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data") or []
                if not data or "embedding" not in data[0]:
                    raise RuntimeError(f"Unexpected embeddings response keys: {list(payload.keys())}")
                out.append([float(x) for x in data[0]["embedding"]])
        return out

