from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaGenerateResult:
    model: str
    response: str


def ollama_generate(*, base_url: str, model: str, prompt: str, timeout_s: float = 120.0) -> OllamaGenerateResult:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        method="POST",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return OllamaGenerateResult(model=model, response=str(payload.get("response") or ""))

