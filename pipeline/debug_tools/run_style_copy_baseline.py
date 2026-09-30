"""Create an immutable, reproducible baseline from the tracked style atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import style_copy_score


SOURCE_MANIFEST_NAME = "style_copy_manifest.json"
BASELINE_MANIFEST_NAME = "baseline_manifest.json"
RUNTIME_LOCK_PATH = style_copy_score.DEFAULT_RUNTIME_LOCK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_component(value: str, *, label: str) -> str:
    if (
        not value
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a non-empty single path component")
    return value


def _source_artifacts(source_atlas_dir: Path) -> tuple[Path, Path]:
    source_manifest = source_atlas_dir / SOURCE_MANIFEST_NAME
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    image_name = manifest.get("image")
    if not isinstance(image_name, str):
        raise ValueError(f"{source_manifest} must contain a string image field")
    _single_component(image_name, label="atlas image name")
    if Path(image_name).suffix.lower() != ".png":
        raise ValueError(f"atlas image must be a PNG: {image_name}")

    source_atlas = source_atlas_dir / image_name
    if not source_atlas.is_file():
        raise FileNotFoundError(source_atlas)
    return source_atlas, source_manifest


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{text}\n".encode("utf-8")


def run_baseline(
    *,
    source_atlas_dir: Path,
    output_root: Path,
    run_id: str,
    seed: int,
    runtime_lock_path: Path = RUNTIME_LOCK_PATH,
) -> Path:
    source_atlas_dir = Path(source_atlas_dir)
    output_root = Path(output_root)
    run_id = _single_component(run_id, label="run_id")
    source_atlas, source_manifest = _source_artifacts(source_atlas_dir)
    runtime = dict(style_copy_score._runtime_metadata())
    style_copy_score.validate_runtime_contract(runtime, runtime_lock_path)

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    # The non-overwriting mkdir is the atomic claim that excludes concurrent writers.
    run_dir.mkdir(exist_ok=False)

    score = style_copy_score.score_synthetic(source_atlas_dir)
    copied_atlas = run_dir / source_atlas.name
    copied_source_manifest = run_dir / source_manifest.name
    shutil.copyfile(source_atlas, copied_atlas)
    shutil.copyfile(source_manifest, copied_source_manifest)

    baseline_manifest = {
        "artifacts": {
            "atlas": {
                "filename": copied_atlas.name,
                "sha256": _sha256(copied_atlas),
            },
            "source_manifest": {
                "filename": copied_source_manifest.name,
                "sha256": _sha256(copied_source_manifest),
            },
        },
        "runtime": runtime,
        "schema_version": 1,
        "score": score,
        "seed": int(seed),
    }
    manifest_path = run_dir / BASELINE_MANIFEST_NAME
    temporary_manifest = run_dir / f".{BASELINE_MANIFEST_NAME}.tmp"
    temporary_manifest.write_bytes(_canonical_json_bytes(baseline_manifest))
    os.replace(temporary_manifest, manifest_path)
    return run_dir


run_style_copy_baseline = run_baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-atlas-dir",
        "--atlas-dir",
        type=Path,
        default=style_copy_score.DEFAULT_ATLAS_DIR,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-lock", type=Path, default=RUNTIME_LOCK_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = run_baseline(
        source_atlas_dir=args.source_atlas_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        seed=args.seed,
        runtime_lock_path=args.runtime_lock,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
