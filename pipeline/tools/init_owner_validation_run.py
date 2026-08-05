"""Create an immutable validation root and atomically publish its context."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Sequence
import uuid

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.chapter_contract import canonical_source_tree_sha256


class ValidationContextError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationContext:
    validation_run_id: str
    validation_root: Path
    source_path: Path
    source_tree_sha256: str
    created_at: str
    off_out: Path
    off_audit: Path
    off_review: Path
    style_out: Path
    style_audit: Path
    style_review: Path
    matrix_out: Path

    def to_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ValidationContext":
        path_fields = {
            "validation_root", "source_path", "off_out", "off_audit",
            "off_review", "style_out", "style_audit", "style_review", "matrix_out",
        }
        return cls(**{
            key: Path(value).resolve() if key in path_fields else value
            for key, value in payload.items()
        })


def _ordered_images(source: Path) -> tuple[Path, ...]:
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    return tuple(sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in allowed))


def init_validation_context(base: Path, source: Path, context_path: Path) -> ValidationContext:
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValidationContextError("validation source must be a directory")
    images = _ordered_images(source)
    if not images:
        raise ValidationContextError("validation source contains no images")
    base = Path(base).resolve()
    run_id = f"validation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"
    root = base / run_id
    root.mkdir(parents=True, exist_ok=False)
    context = ValidationContext(
        validation_run_id=run_id,
        validation_root=root,
        source_path=source,
        source_tree_sha256=canonical_source_tree_sha256(images, source),
        created_at=datetime.now(timezone.utc).isoformat(),
        off_out=root / "off",
        off_audit=root / "off-audit.json",
        off_review=root / "off-review",
        style_out=root / "render",
        style_audit=root / "render-audit.json",
        style_review=root / "render-review",
        matrix_out=root / "matrix",
    )
    target = Path(context_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(context.to_dict(), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return context


def load_validation_context(path: Path) -> ValidationContext:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ValidationContext.from_dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--context", required=True)
    try:
        args = parser.parse_args(argv)
        init_validation_context(
            Path(args.base).resolve(),
            Path(args.source).resolve(),
            Path(args.context).resolve(),
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"validation context error: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
