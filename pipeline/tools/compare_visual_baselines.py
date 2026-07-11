"""Compare canonical visual baseline manifests with an explicit change allowlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_manifest(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_dir():
        path = path / "visual_baseline_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _entries_by_key(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_entry in manifest.get("entries") or []:
        if not isinstance(raw_entry, dict):
            raise ValueError("visual baseline entry must be an object")
        key = str(raw_entry.get("key") or "").strip()
        if not key or key in result:
            raise ValueError(f"duplicate or empty visual baseline key: {key!r}")
        result[key] = dict(raw_entry)
    return result


def _content_hash(entry: Mapping[str, Any]) -> str:
    return str(entry.get("buffer_sha256") or entry.get("png_sha256") or "").strip()


def _text_metrics_by_key(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_metric in manifest.get("text_metrics") or []:
        if not isinstance(raw_metric, dict):
            raise ValueError("visual baseline text metric must be an object")
        key = str(
            raw_metric.get("metric_id")
            or raw_metric.get("text_instance_id")
            or raw_metric.get("trace_id")
            or ""
        ).strip()
        if not key or key in result:
            raise ValueError(f"duplicate or empty visual baseline trace_id: {key!r}")
        result[key] = dict(raw_metric)
    return result


def _expected_keys(manifest: Mapping[str, Any]) -> set[str]:
    expected = {
        f"page:{str(page_id).strip()}:"
        for page_id in manifest.get("expected_page_ids") or []
        if str(page_id).strip()
    }
    for band_id in manifest.get("expected_final_band_ids") or []:
        normalized = str(band_id).strip()
        if not normalized:
            continue
        page_id = normalized.split("_band_", 1)[0] if "_band_" in normalized else "page_unknown"
        expected.add(f"final_band:{page_id}:{normalized}")
    return expected


def _canonical_metric(metric: Mapping[str, Any]) -> str:
    return json.dumps(metric, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_visual_baselines(
    baseline: Mapping[str, Any] | str | Path,
    candidate: Mapping[str, Any] | str | Path,
    *,
    allowed_change_set: set[str] | None = None,
) -> dict[str, Any]:
    baseline_manifest = _load_manifest(baseline)
    candidate_manifest = _load_manifest(candidate)
    baseline_entries = _entries_by_key(baseline_manifest)
    candidate_entries = _entries_by_key(candidate_manifest)
    baseline_metrics = _text_metrics_by_key(baseline_manifest)
    candidate_metrics = _text_metrics_by_key(candidate_manifest)
    allowed = {str(key) for key in (allowed_change_set or set())}

    baseline_keys = set(baseline_entries)
    candidate_keys = set(candidate_entries)
    missing = sorted(baseline_keys - candidate_keys)
    added = sorted(candidate_keys - baseline_keys)
    changed = sorted(
        key
        for key in baseline_keys & candidate_keys
        if _content_hash(baseline_entries[key]) != _content_hash(candidate_entries[key])
    )
    allowed_changed = sorted(key for key in changed if key in allowed)
    unexpected_changed = sorted(key for key in changed if key not in allowed)
    unexpected_missing = sorted(key for key in missing if key not in allowed)
    unexpected_added = sorted(key for key in added if key not in allowed)

    expected_keys = _expected_keys(baseline_manifest)
    baseline_coverage_missing = sorted(expected_keys - baseline_keys)
    candidate_coverage_missing = sorted(expected_keys - candidate_keys)
    coverage_errors: list[str] = []
    if not baseline_entries:
        coverage_errors.append("baseline_has_no_entries")
    if not candidate_entries:
        coverage_errors.append("candidate_has_no_entries")

    baseline_metric_keys = set(baseline_metrics)
    candidate_metric_keys = set(candidate_metrics)
    changed_metrics = sorted(
        key
        for key in baseline_metric_keys & candidate_metric_keys
        if _canonical_metric(baseline_metrics[key]) != _canonical_metric(candidate_metrics[key])
    )
    missing_metrics = sorted(baseline_metric_keys - candidate_metric_keys)
    added_metrics = sorted(candidate_metric_keys - baseline_metric_keys)
    unexpected_changed_metrics = sorted(
        key for key in changed_metrics if f"text_metric:{key}" not in allowed
    )
    unexpected_missing_metrics = sorted(
        key for key in missing_metrics if f"text_metric:{key}" not in allowed
    )
    unexpected_added_metrics = sorted(
        key for key in added_metrics if f"text_metric:{key}" not in allowed
    )

    passed = not (
        unexpected_changed
        or unexpected_missing
        or unexpected_added
        or baseline_coverage_missing
        or candidate_coverage_missing
        or coverage_errors
        or unexpected_changed_metrics
        or unexpected_missing_metrics
        or unexpected_added_metrics
    )

    return {
        "schema_version": 1,
        "passed": passed,
        "baseline_entry_count": len(baseline_entries),
        "candidate_entry_count": len(candidate_entries),
        "changed_keys": changed,
        "allowed_changed_keys": allowed_changed,
        "unexpected_changed_keys": unexpected_changed,
        "missing_keys": missing,
        "unexpected_missing_keys": unexpected_missing,
        "added_keys": added,
        "unexpected_added_keys": unexpected_added,
        "coverage_errors": coverage_errors,
        "baseline_coverage_missing_keys": baseline_coverage_missing,
        "candidate_coverage_missing_keys": candidate_coverage_missing,
        "changed_text_metric_keys": changed_metrics,
        "unexpected_changed_text_metric_keys": unexpected_changed_metrics,
        "missing_text_metric_keys": missing_metrics,
        "unexpected_missing_text_metric_keys": unexpected_missing_metrics,
        "added_text_metric_keys": added_metrics,
        "unexpected_added_text_metric_keys": unexpected_added_metrics,
        "allowed_change_set": sorted(allowed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_visual_baselines(
        args.baseline,
        args.candidate,
        allowed_change_set=set(args.allow),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
