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


def default_paths(project_root: Path) -> RunPaths:
    return RunPaths(
        project_root=project_root,
        data_dir=project_root / "data",
    )


_RESERVED_NOVEL_DIRS = {"src", "tests", "docs", "data", "runs", ".venv"}


def novel_root(project_root: Path, novel_slug: str) -> Path:
    slug = novel_slug.strip()
    if not slug:
        raise ValueError("novel_slug must be non-empty.")
    if slug in _RESERVED_NOVEL_DIRS:
        raise ValueError(f"novel_slug '{slug}' is reserved.")
    return project_root / slug


def chunk_dir(project_root: Path, *, novel_slug: str, chunk_method: str) -> Path:
    return novel_root(project_root, novel_slug) / chunk_method


def embed_key(embed_backend: str, embed_model: str) -> str:
    return f"{embed_backend}-{_sanitize(embed_model)}"


def embedding_dir(project_root: Path, *, novel_slug: str, embed_key: str, chunk_method: str) -> Path:
    return novel_root(project_root, novel_slug) / embed_key / chunk_method


def index_key(chunk_method: str, embed_key: str) -> str:
    return f"{chunk_method}__{embed_key}"


def store_index_dir(project_root: Path, *, novel_slug: str, store_name: str, index_key: str) -> Path:
    return novel_root(project_root, novel_slug) / store_name / index_key


def queries_root(project_root: Path, *, novel_slug: str) -> Path:
    return novel_root(project_root, novel_slug) / "queries"


def query_dir(project_root: Path, *, novel_slug: str, query_hash: str) -> Path:
    return queries_root(project_root, novel_slug=novel_slug) / query_hash


def query_hash(question: str, params: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    payload = {"question": question, "params": params}
    if config is not None:
        payload["config"] = config
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
