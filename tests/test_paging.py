from __future__ import annotations

from vector_explore.paging import page_vector_preview, prompt_more


def test_prompt_more_enter_yes():
    def inp(_p: str) -> str:
        return ""

    assert prompt_more(inp) is True


def test_prompt_more_n_no():
    def inp(_p: str) -> str:
        return "n"

    assert prompt_more(inp) is False


def test_page_vector_preview_stops():
    calls = {"n": 0}

    def inp(_p: str) -> str:
        calls["n"] += 1
        return "n"

    out = []

    def pr(s: str) -> None:
        out.append(s)

    page_vector_preview([1.0] * 100, input_fn=inp, print_fn=pr, initial=16, step=16)
    assert any("dim=100" in s for s in out)

