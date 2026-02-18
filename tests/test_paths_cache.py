from __future__ import annotations

from pathlib import Path

from vector_explore.paths import cache_ok, ensure_dir, params_fingerprint, write_params


def test_cache_ok_roundtrip(tmp_path: Path):
    p = tmp_path / "params.json"
    fp = params_fingerprint("abc", {"x": 1})
    assert cache_ok(p, fp) is False
    write_params(p, "abc", {"x": 1})
    assert cache_ok(p, fp) is True

