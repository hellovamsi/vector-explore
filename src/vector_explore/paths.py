from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    data_dir: Path
    runs_dir: Path


def default_paths(project_root: Path) -> RunPaths:
    return RunPaths(
        project_root=project_root,
        data_dir=project_root / "data",
        runs_dir=project_root / "runs",
    )


def chunk_run_dir(runs_dir: Path, *, novel_slug: str, chunk_method: str) -> Path:
    return runs_dir / novel_slug / chunk_method


def embed_run_dir(runs_dir: Path, *, novel_slug: str, chunk_method: str, embed_backend: str, embed_model: str) -> Path:
    return runs_dir / novel_slug / chunk_method / f"{embed_backend}-{_sanitize(embed_model)}"


def store_run_dir(
    runs_dir: Path, *, novel_slug: str, chunk_method: str, embed_backend: str, embed_model: str, store_name: str
) -> Path:
    return embed_run_dir(runs_dir, novel_slug=novel_slug, chunk_method=chunk_method, embed_backend=embed_backend, embed_model=embed_model) / store_name


def query_dir(store_dir: Path, query_hash: str) -> Path:
    return store_dir / "queries" / query_hash


def query_hash(question: str, params: dict[str, Any]) -> str:
    payload = {"question": question, "params": params}
    return sha256_bytes(json_dumps_canonical(payload).encode("utf-8"))[:16]


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)[:120]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def params_fingerprint(input_sha256: str, params: dict[str, Any]) -> str:
    payload = {"input_sha256": input_sha256, "params": params}
    return sha256_bytes(json_dumps_canonical(payload).encode("utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_ok(params_path: Path, expected_fingerprint: str) -> bool:
    if not params_path.exists():
        return False
    try:
        meta = read_json(params_path)
    except Exception:
        return False
    return meta.get("fingerprint") == expected_fingerprint


def write_params(params_path: Path, input_sha256: str, params: dict[str, Any]) -> str:
    fingerprint = params_fingerprint(input_sha256, params)
    write_json(params_path, {"input_sha256": input_sha256, "params": params, "fingerprint": fingerprint})
    return fingerprint
