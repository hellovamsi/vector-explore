from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .paths import ensure_dir


@dataclass(frozen=True)
class Novel:
    slug: str
    title: str
    url: str


NOVELS: list[Novel] = [
    Novel("frankenstein", "Frankenstein", "https://www.gutenberg.org/ebooks/84.txt.utf-8"),
    Novel("mobydick", "Moby Dick", "https://www.gutenberg.org/ebooks/2701.txt.utf-8"),
    Novel("pride", "Pride and Prejudice", "https://www.gutenberg.org/ebooks/1342.txt.utf-8"),
]


def get_novel(slug: str) -> Novel:
    slug = slug.strip().lower()
    for n in NOVELS:
        if n.slug == slug:
            return n
    raise ValueError(f"Unknown novel slug: {slug}. Expected one of: {', '.join(n.slug for n in NOVELS)}")


def gutenberg_path(project_root: Path, novel_slug: str) -> Path:
    return project_root / "data" / "gutenberg" / f"{novel_slug}.txt"


def download_if_missing(project_root: Path, novel: Novel, *, progress_enabled: bool = True, print_fn=print) -> Path:
    dst = gutenberg_path(project_root, novel.slug)
    ensure_dir(dst.parent)
    if dst.exists():
        print_fn(f"Checking local cache: {dst} ...")
        print_fn("Found. Skipping download.")
        return dst

    print_fn(f"Downloading via curl from:\n  {novel.url}")
    _curl_download(novel.url, dst, show_progress=progress_enabled)
    print_fn(f"Saved: {dst}")
    return dst


def _curl_download(url: str, dst: Path, *, show_progress: bool) -> None:
    # -L follow redirects, -f fail on non-2xx
    cmd = ["curl", "-L", "-f"]
    if show_progress:
        cmd.append("--progress-bar")
    else:
        cmd.extend(["-sS"])
    cmd.extend([url, "-o", str(dst)])
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("curl is required but was not found. Install curl or use a system that provides it.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"curl failed (exit {e.returncode}). URL: {url}") from e
