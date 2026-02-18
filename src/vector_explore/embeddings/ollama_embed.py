from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from .base import Embedder, EmbedderInfo


@dataclass(frozen=True)
class OllamaEmbedder(Embedder):
    base_url: str
    model: str
    timeout_s: float = 60.0

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(backend="ollama", model=self.model, dim=None)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Ollama embeddings endpoint: POST /api/embeddings with {model, prompt}
        out: list[list[float]] = []
        for t in texts:
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                method="POST",
                data=json.dumps({"model": self.model, "prompt": t}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            vec = payload.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError(f"Ollama embeddings returned unexpected payload keys: {list(payload.keys())}")
            out.append([float(x) for x in vec])
        return out


def ollama_healthcheck(base_url: str, *, timeout_s: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout_s) as resp:
            _ = resp.read()
        return True
    except Exception:
        return False

