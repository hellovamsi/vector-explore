from __future__ import annotations

import json

from vector_explore.llm.ollama_generate import ollama_generate


def test_ollama_generate_raw_capture(monkeypatch):
    payload = {"response": "hello", "done": True, "extra": {"a": 1}}
    raw = json.dumps(payload, separators=(",", ":"))

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return raw.encode("utf-8")

    def fake_urlopen(req, timeout):
        assert timeout == 120.0
        assert req.headers["Content-type"] == "application/json"
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = ollama_generate(base_url="http://localhost:11434", model="mistral", prompt="test")
    assert out.raw_response_json == raw
    assert out.response_payload["extra"] == {"a": 1}
    assert out.response == "hello"
    assert json.loads(out.request_body_raw) == {"model": "mistral", "prompt": "test", "stream": False}
    assert out.http_status == 200
