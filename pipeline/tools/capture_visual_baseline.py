"""Capture recorder-owned canonical PNGs into a deterministic baseline bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _baseline_rel_path(source_rel_path: str, kind: str, identity: str) -> Path:
    source = Path(str(source_rel_path or ""))
    if "canonical_pages" in source.parts:
        index = source.parts.index("canonical_pages")
        return Path(*source.parts[index:])
    if "canonical_final_bands" in source.parts:
        index = source.parts.index("canonical_final_bands")
        return Path(*source.parts[index:])
    folder = "canonical_pages" if kind == "page" else "canonical_final_bands"
    return Path(folder) / f"{identity}.png"


def _contained_path(root: Path, relative: str | Path, *, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or value.drive:
        raise ValueError(f"{label} escapes root: {relative}")
    resolved_root = root.resolve()
    candidate = (resolved_root / value).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes root: {relative}") from exc
    return candidate


def capture_visual_baseline(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir)
    debug_root = run_root / "debug" / "e2e"
    source_manifest_path = debug_root / "00_run" / "canonical_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_entries = source_manifest.get("entries") or []
    if not isinstance(source_entries, list):
        raise ValueError("canonical manifest entries must be a list")
    declared_entry_count = source_manifest.get("entry_count")
    if declared_entry_count is None or int(declared_entry_count) != len(source_entries):
        raise ValueError("canonical manifest entry_count does not match entries")

    expected_page_ids = [
        str(value).strip()
        for value in source_manifest.get("expected_page_ids") or []
        if str(value).strip()
    ]
    expected_final_band_ids = [
        str(value).strip()
        for value in source_manifest.get("expected_final_band_ids") or []
        if str(value).strip()
    ]
    if not expected_page_ids:
        raise ValueError("canonical manifest must declare at least one expected page")
    if len(set(expected_page_ids)) != len(expected_page_ids):
        raise ValueError("canonical manifest expected_page_ids must be unique")
    if len(set(expected_final_band_ids)) != len(expected_final_band_ids):
        raise ValueError("canonical manifest expected_final_band_ids must be unique")

    text_metrics = source_manifest.get("text_metrics") or []
    if not isinstance(text_metrics, list) or not all(isinstance(item, dict) for item in text_metrics):
        raise ValueError("canonical manifest text_metrics must be a list of objects")
    declared_metric_count = source_manifest.get("text_metric_count")
    if declared_metric_count is None or int(declared_metric_count) != len(text_metrics):
        raise ValueError("canonical manifest text_metric_count does not match text_metrics")

    declared_keys = {
        str((entry or {}).get("key") or "").strip()
        for entry in source_entries
        if isinstance(entry, dict)
    }
    required_keys = {f"page:{page_id}:" for page_id in expected_page_ids}
    required_keys.update(
        f"final_band:{band_id.split('_band_', 1)[0] if '_band_' in band_id else 'page_unknown'}:{band_id}"
        for band_id in expected_final_band_ids
    )
    missing_required = sorted(required_keys - declared_keys)
    if missing_required:
        raise ValueError(f"canonical manifest missing required coverage: {missing_required}")

    requested_destination = Path(output_dir)
    # A captured baseline is immutable evidence; callers must claim a new path.
    requested_destination.parent.mkdir(parents=True, exist_ok=True)
    if requested_destination.exists():
        raise FileExistsError(requested_destination)
    staging_destination = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_destination.name}.staging-",
            dir=requested_destination.parent,
        )
    )

    try:
        captured_entries: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for raw_entry in sorted(source_entries, key=lambda item: str((item or {}).get("key") or "")):
            if not isinstance(raw_entry, dict):
                raise ValueError("canonical manifest entry must be an object")
            entry = dict(raw_entry)
            key = str(entry.get("key") or "").strip()
            if not key or key in seen_keys:
                raise ValueError(f"duplicate or empty canonical key: {key!r}")
            seen_keys.add(key)
            kind = str(entry.get("kind") or "").strip()
            identity = str(entry.get("page_id") if kind == "page" else entry.get("band_id") or "").strip()
            source_path = _contained_path(
                debug_root,
                str(entry.get("rel_path") or ""),
                label="canonical source path",
            )
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            png_bytes = source_path.read_bytes()
            actual_png_hash = _sha256_bytes(png_bytes)
            expected_png_hash = str(entry.get("png_sha256") or "").strip()
            if expected_png_hash and expected_png_hash != actual_png_hash:
                raise ValueError(f"canonical PNG hash mismatch for {key}")

            relative_target = _baseline_rel_path(str(entry.get("rel_path") or ""), kind, identity)
            target = _contained_path(staging_destination, relative_target, label="baseline destination path")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            entry["rel_path"] = relative_target.as_posix()
            entry["png_sha256"] = actual_png_hash
            captured_entries.append(entry)

        expected_page_ids = sorted(expected_page_ids)
        expected_final_band_ids = sorted(expected_final_band_ids)
        normalized_metrics = sorted(
            (dict(item) for item in text_metrics),
            key=lambda item: str(item.get("metric_id") or item.get("text_instance_id") or item.get("trace_id") or ""),
        )
        metric_keys = [
            str(item.get("metric_id") or item.get("text_instance_id") or item.get("trace_id") or "").strip()
            for item in normalized_metrics
        ]
        if any(not key for key in metric_keys) or len(set(metric_keys)) != len(metric_keys):
            raise ValueError("canonical text metrics require unique non-empty identities")

        content_payload = {
            "entries": captured_entries,
            "expected_page_ids": expected_page_ids,
            "expected_final_band_ids": expected_final_band_ids,
            "text_metrics": normalized_metrics,
        }
        content_hash = _sha256_bytes(_canonical_json(content_payload))
        result = {
            "schema_version": 1,
            "run_id": str(source_manifest.get("run_id") or run_root.name),
            "entry_count": len(captured_entries),
            "expected_page_ids": expected_page_ids,
            "expected_final_band_ids": expected_final_band_ids,
            "text_metric_count": len(normalized_metrics),
            "text_metrics": normalized_metrics,
            "content_sha256": content_hash,
            "entries": captured_entries,
        }
        (staging_destination / "visual_baseline_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_destination, requested_destination)
        return result
    except Exception:
        shutil.rmtree(staging_destination, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    result = capture_visual_baseline(args.run_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
