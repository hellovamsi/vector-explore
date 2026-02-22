from __future__ import annotations

from vector_explore.llm.prompts import build_rag_prompt_parts
from vector_explore.query import build_llm_input_log


def test_build_llm_input_log_schema_answer():
    parts = build_rag_prompt_parts(
        question="Q?",
        keywords=["k1", "k2"],
        sources=[{"chunk_id": "c1", "text": "t1"}],
    )
    out = build_llm_input_log(
        base_url="http://localhost:11434",
        model="mistral",
        timeout_s=60.0,
        request_body_raw='{"model":"m","prompt":"p","stream":false}',
        request_body={"model": "m", "prompt": "p", "stream": False},
        prompt_text="p",
        prompt_parts=parts,
    )
    for k in ["provider", "endpoint", "url", "method", "headers", "timeout_s", "request_body_raw", "request_body", "prompt_text", "prompt_parts"]:
        assert k in out
    assert out["provider"] == "ollama"
    assert out["prompt_parts"]["question"] == "Q?"
    assert out["prompt_parts"]["sources"][0]["chunk_id"] == "c1"


def test_build_llm_input_log_schema_summary():
    parts = {"instruction": "summarize", "keywords": ["k1"], "passages": ["p1", "p2"]}
    out = build_llm_input_log(
        base_url="http://localhost:11434",
        model="mistral",
        timeout_s=60.0,
        request_body_raw='{"model":"m","prompt":"p","stream":false}',
        request_body={"model": "m", "prompt": "p", "stream": False},
        prompt_text="p",
        prompt_parts=parts,
    )
    assert out["prompt_parts"]["instruction"] == "summarize"
    assert out["prompt_parts"]["passages"] == ["p1", "p2"]
