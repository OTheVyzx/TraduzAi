from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _direct_requirement(name: str) -> str | None:
    prefix = name.casefold() + "=="
    for raw_line in (REPO_ROOT / "pipeline" / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if line.casefold().startswith(prefix):
            return line
    return None


def test_fonttools_is_directly_pinned_for_sidecar_runtime():
    assert _direct_requirement("fonttools") == "fonttools==4.62.1"
