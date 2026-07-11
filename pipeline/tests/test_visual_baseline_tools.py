from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.capture_visual_baseline import capture_visual_baseline
from tools.compare_visual_baselines import compare_visual_baselines


def _write_canonical_run(run_dir: Path, entries: list[dict]) -> None:
    root = run_dir / "debug" / "e2e" / "00_run"
    root.mkdir(parents=True)
    for entry in entries:
        payload = bytes.fromhex(entry.pop("_png_hex"))
        target = run_dir / "debug" / "e2e" / entry["rel_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        entry["png_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "canonical_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "entry_count": len(entries),
                "expected_page_ids": sorted(
                    entry["page_id"] for entry in entries if entry["kind"] == "page"
                ),
                "expected_final_band_ids": sorted(
                    entry["band_id"] for entry in entries if entry["kind"] == "final_band"
                ),
                "text_metric_count": 1,
                "text_metrics": [
                    {
                        "trace_id": "ocr_001@page_001_band_000",
                        "page_id": "page_001",
                        "band_id": "page_001_band_000",
                        "font_size_final": 31,
                    }
                ],
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _entry(key: str, rel_path: str, buffer_hash: str, png_payload: bytes) -> dict:
    kind, page_id, band_id = key.split(":", 2)
    return {
        "schema_version": 1,
        "key": key,
        "kind": kind,
        "page_id": page_id,
        "band_id": band_id,
        "rel_path": rel_path,
        "shape": [4, 5, 3],
        "dtype": "uint8",
        "color_space": "bgr",
        "buffer_sha256": buffer_hash,
        "_png_hex": png_payload.hex(),
    }


def test_capture_visual_baseline_copies_lossless_artifacts_and_is_deterministic(tmp_path):
    run_dir = tmp_path / "run-a"
    entries = [
        _entry(
            "page:page_001:",
            "00_run/canonical_pages/page_001.png",
            "a" * 64,
            b"page-png",
        ),
        _entry(
            "final_band:page_001:page_001_band_000",
            "00_run/canonical_final_bands/page_001_band_000.png",
            "b" * 64,
            b"band-png",
        ),
    ]
    _write_canonical_run(run_dir, entries)

    first = capture_visual_baseline(run_dir, tmp_path / "baseline-a")
    second = capture_visual_baseline(run_dir, tmp_path / "baseline-b")

    assert first["entry_count"] == 2
    assert first["text_metric_count"] == 1
    assert first["content_sha256"] == second["content_sha256"]
    assert first["entries"] == second["entries"]
    assert (tmp_path / "baseline-a" / "canonical_pages" / "page_001.png").read_bytes() == b"page-png"
    assert (
        tmp_path / "baseline-a" / "canonical_final_bands" / "page_001_band_000.png"
    ).read_bytes() == b"band-png"


def test_compare_visual_baselines_blocks_unexpected_changes_and_accepts_allowlist(tmp_path):
    baseline_run = tmp_path / "baseline-run"
    candidate_run = tmp_path / "candidate-run"
    baseline_entries = [
        _entry("page:page_001:", "00_run/canonical_pages/page_001.png", "a" * 64, b"page-a"),
        _entry(
            "final_band:page_001:page_001_band_000",
            "00_run/canonical_final_bands/page_001_band_000.png",
            "b" * 64,
            b"band-a",
        ),
    ]
    candidate_entries = [
        _entry("page:page_001:", "00_run/canonical_pages/page_001.png", "c" * 64, b"page-b"),
        _entry(
            "final_band:page_001:page_001_band_000",
            "00_run/canonical_final_bands/page_001_band_000.png",
            "b" * 64,
            b"band-a",
        ),
    ]
    _write_canonical_run(baseline_run, baseline_entries)
    _write_canonical_run(candidate_run, candidate_entries)
    baseline = capture_visual_baseline(baseline_run, tmp_path / "baseline")
    candidate = capture_visual_baseline(candidate_run, tmp_path / "candidate")

    blocked = compare_visual_baselines(baseline, candidate)
    allowed = compare_visual_baselines(
        baseline,
        candidate,
        allowed_change_set={"page:page_001:"},
    )

    assert blocked["passed"] is False
    assert blocked["unexpected_changed_keys"] == ["page:page_001:"]
    assert allowed["passed"] is True
    assert allowed["allowed_changed_keys"] == ["page:page_001:"]


def test_compare_visual_baselines_reports_added_and_missing_entries(tmp_path):
    baseline = {
        "entries": [
            {"key": "page:page_001:", "buffer_sha256": "a" * 64},
            {"key": "page:page_002:", "buffer_sha256": "b" * 64},
        ]
    }
    candidate = {
        "entries": [
            {"key": "page:page_002:", "buffer_sha256": "b" * 64},
            {"key": "page:page_003:", "buffer_sha256": "c" * 64},
        ]
    }

    result = compare_visual_baselines(baseline, candidate)

    assert result["passed"] is False
    assert result["missing_keys"] == ["page:page_001:"]
    assert result["added_keys"] == ["page:page_003:"]


def test_compare_visual_baselines_rejects_empty_or_incomplete_coverage():
    empty = compare_visual_baselines({"entries": []}, {"entries": []})
    incomplete = compare_visual_baselines(
        {
            "expected_page_ids": ["page_001"],
            "expected_final_band_ids": ["page_001_band_000"],
            "entries": [{"key": "page:page_001:", "buffer_sha256": "a" * 64}],
        },
        {
            "entries": [{"key": "page:page_001:", "buffer_sha256": "a" * 64}],
        },
    )

    assert empty["passed"] is False
    assert empty["coverage_errors"] == ["baseline_has_no_entries", "candidate_has_no_entries"]
    assert incomplete["passed"] is False
    assert incomplete["baseline_coverage_missing_keys"] == [
        "final_band:page_001:page_001_band_000"
    ]
    assert incomplete["candidate_coverage_missing_keys"] == [
        "final_band:page_001:page_001_band_000"
    ]


def test_compare_visual_baselines_detects_per_text_metric_changes():
    baseline = {
        "entries": [{"key": "page:page_001:", "buffer_sha256": "a" * 64}],
        "text_metrics": [
            {"trace_id": "ocr_001@page_001_band_000", "font_size_final": 31}
        ],
    }
    candidate = {
        "entries": [{"key": "page:page_001:", "buffer_sha256": "a" * 64}],
        "text_metrics": [
            {"trace_id": "ocr_001@page_001_band_000", "font_size_final": 24}
        ],
    }

    result = compare_visual_baselines(baseline, candidate)

    assert result["passed"] is False
    assert result["unexpected_changed_text_metric_keys"] == [
        "ocr_001@page_001_band_000"
    ]


def test_capture_visual_baseline_rejects_manifest_path_escape(tmp_path):
    run_dir = tmp_path / "run"
    root = run_dir / "debug" / "e2e" / "00_run"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (root / "canonical_manifest.json").write_text(
        json.dumps(
            {
                "entry_count": 1,
                "expected_page_ids": ["page_001"],
                "expected_final_band_ids": [],
                "text_metric_count": 0,
                "text_metrics": [],
                "entries": [
                    {
                        "key": "page:page_001:",
                        "kind": "page",
                        "page_id": "page_001",
                        "band_id": "",
                        "rel_path": "../../../../outside.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        capture_visual_baseline(run_dir, tmp_path / "baseline")


def test_capture_visual_baseline_rejects_declared_count_or_required_coverage_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    root = run_dir / "debug" / "e2e" / "00_run"
    root.mkdir(parents=True)
    (root / "canonical_manifest.json").write_text(
        json.dumps(
            {
                "entry_count": 2,
                "expected_page_ids": ["page_001"],
                "expected_final_band_ids": ["page_001_band_000"],
                "text_metric_count": 0,
                "text_metrics": [],
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="entry_count"):
        capture_visual_baseline(run_dir, tmp_path / "baseline")


def test_capture_visual_baseline_refuses_to_overwrite_existing_bundle(tmp_path):
    run_dir = tmp_path / "run"
    entries = [
        _entry(
            "page:page_001:",
            "00_run/canonical_pages/page_001.png",
            "a" * 64,
            b"page-png",
        ),
    ]
    _write_canonical_run(run_dir, entries)
    output_dir = tmp_path / "baseline"
    output_dir.mkdir()
    marker = output_dir / "approved-baseline-marker.txt"
    marker.write_text("do-not-overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError):
        capture_visual_baseline(run_dir, output_dir)

    assert marker.read_text(encoding="utf-8") == "do-not-overwrite"


def test_capture_visual_baseline_leaves_no_partial_bundle_when_copy_fails(tmp_path):
    run_dir = tmp_path / "run"
    root = run_dir / "debug" / "e2e" / "00_run"
    root.mkdir(parents=True)
    (root / "canonical_manifest.json").write_text(
        json.dumps(
            {
                "entry_count": 1,
                "expected_page_ids": ["page_001"],
                "expected_final_band_ids": [],
                "text_metric_count": 0,
                "text_metrics": [],
                "entries": [
                    {
                        "key": "page:page_001:",
                        "kind": "page",
                        "page_id": "page_001",
                        "band_id": "",
                        "rel_path": "00_run/canonical_pages/missing.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "baseline"

    with pytest.raises(FileNotFoundError):
        capture_visual_baseline(run_dir, output_dir)

    assert not output_dir.exists()
