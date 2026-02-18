from __future__ import annotations

from collections.abc import Callable, Iterable


InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]


def prompt_more(input_fn: InputFn, prompt: str = "Enter=more, N+Enter=stop: ") -> bool:
    ans = input_fn(prompt).strip().lower()
    if ans == "":
        return True
    return ans not in {"n", "no", "x", "q", "quit", "stop"}


def page_iterable(
    items: list[str],
    *,
    page_size: int,
    input_fn: InputFn,
    print_fn: PrintFn,
    header: str | None = None,
) -> None:
    if header:
        print_fn(header)
    idx = 0
    while idx < len(items):
        end = min(idx + page_size, len(items))
        for line in items[idx:end]:
            print_fn(line)
        idx = end
        if idx >= len(items):
            break
        if not prompt_more(input_fn):
            break


def page_vector_preview(
    vector: list[float],
    *,
    initial: int = 16,
    step: int = 16,
    per_line: int = 8,
    input_fn: InputFn,
    print_fn: PrintFn,
    label: str = "vector",
) -> None:
    total = len(vector)
    shown = min(initial, total)
    while True:
        print_fn(f"{label} dim={total} showing[0:{shown}]")
        _print_floats(vector[:shown], per_line=per_line, print_fn=print_fn)
        if shown >= total:
            break
        if not prompt_more(input_fn, prompt="Enter=more numbers, N+Enter=stop: "):
            break
        shown = min(shown + step, total)


def _print_floats(values: Iterable[float], *, per_line: int, print_fn: PrintFn) -> None:
    row: list[str] = []
    for i, v in enumerate(values, 1):
        row.append(f"{v:.6g}")
        if i % per_line == 0:
            print_fn("[" + ", ".join(row) + "]")
            row = []
    if row:
        print_fn("[" + ", ".join(row) + "]")

