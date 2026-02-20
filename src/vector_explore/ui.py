from __future__ import annotations

import contextlib
from typing import Iterator


def get_console():
    try:
        from rich.console import Console
    except Exception:
        return None
    return Console()


@contextlib.contextmanager
def progress(enabled: bool) -> Iterator:
    if enabled:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as p:
                yield p
            return
        except Exception:
            pass
    yield _NoProgress()


@contextlib.contextmanager
def status(*, enabled: bool, message: str):
    if enabled:
        console = get_console()
        if console is not None:
            with console.status(message):
                yield
            return
    yield


class _NoProgress:
    def add_task(self, *_args, **_kwargs) -> int:
        return 0

    def update(self, *_args, **_kwargs) -> None:
        return None
