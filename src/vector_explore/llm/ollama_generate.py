from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaGenerateResult:
    model: str
    response: str
    request_body_raw: str
    request_body: dict
    raw_response_json: str
    response_payload: dict
    http_status: int


def ollama_generate(*, base_url: str, model: str, prompt: str, timeout_s: float = 120.0) -> OllamaGenerateResult:
    request_body = {"model": model, "prompt": prompt, "stream": False}
    request_body_raw = json.dumps(request_body, ensure_ascii=False)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        method="POST",
        data=request_body_raw.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw_response_json = resp.read().decode("utf-8")
        http_status = int(getattr(resp, "status", 200))
    payload = json.loads(raw_response_json)
    return OllamaGenerateResult(
        model=model,
        response=str(payload.get("response") or ""),
        request_body_raw=request_body_raw,
        request_body=request_body,
        raw_response_json=raw_response_json,
        response_payload=payload,
        http_status=http_status,
    )
