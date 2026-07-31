from __future__ import annotations

import json
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from debug_tools import style_benchmark_report
from debug_tools import generate_style_benchmark_v2, run_style_benchmark_v2


def test_score_reports_attribute_precision_top_k_round_trip_and_hard_negative_abstention():
    manifest = {
        "cases": [
            {
                "id": "positive",
                "level": "smoke",
                "text_a": "TEXT A",
                "text_b": "TEXT B",
                "font_name": "ComicNeue-Bold.ttf",
                "font_weight": 700,
                "font_width": 100,
                "font_size_px": 44,
                "alignment": "center",
                "fill": "#F4F4F4",
                "stroke": {"color": "#161616", "width_px": 3},
                "shadow": None,
                "gradient": None,
                "rotation_deg": 0,
                "container": {"width": 320, "height": 140},
            },
            {
                "id": "negative",
                "level": "hard-negative",
                "text_a": "NO TEXT A",
                "text_b": "NO TEXT B",
                "font_name": "ComicNeue-Bold.ttf",
                "fill": "#FFFFFF",
                "stroke": None,
                "shadow": None,
                "gradient": None,
                "rotation_deg": 0,
                "container": {"width": 280, "height": 120, "render_text": False},
            },
        ]
    }
    observations = [
        {
            "case_id": "positive",
            "variant": "a",
            "source_text": "TEXT A",
            "attributes": {
                "font_name": {"value": "Wrong.ttf", "top_k": ["Wrong.ttf", "ComicNeue-Bold.ttf"]},
                "font_weight": {"value": "unknown"},
                "font_width": {"value": "unknown"},
                "font_size_px": {"value": "unknown"},
                "alignment": {"value": "unknown"},
                "fill": {"value": "#F4F4F4", "confidence": 0.9},
                "stroke": {"value": {"color": "#161616", "width_px": 3}},
                "shadow": {"value": None},
                "gradient": {"value": None},
                "rotation_deg": {"value": "unknown"},
                "container": {"value": "unknown"},
            },
        },
        {
            "case_id": "positive",
            "variant": "b",
            "source_text": "TEXT B",
            "attributes": {
                "font_name": {"value": "ComicNeue-Bold.ttf"},
                "font_weight": {"value": "unknown"},
                "font_width": {"value": "unknown"},
                "font_size_px": {"value": "unknown"},
                "alignment": {"value": "unknown"},
                "fill": {"value": "#F4F4F4"},
                "stroke": {"value": {"color": "#161616", "width_px": 3}},
                "shadow": {"value": None},
                "gradient": {"value": None},
                "rotation_deg": {"value": "unknown"},
                "container": {"value": "unknown"},
            },
        },
        {
            "case_id": "negative",
            "variant": "a",
            "source_text": "NO TEXT A",
            "attributes": {"fill": {"value": "#FFFFFF", "confidence": 0.99}},
        },
        {
            "case_id": "negative",
            "variant": "b",
            "source_text": "NO TEXT B",
            "attributes": {"fill": {"value": "#FFFFFF", "confidence": 0.99}},
        },
    ]

    report = style_benchmark_report.score_benchmark(manifest, observations)

    assert report["attributes"]["fill"] == {
        "coverage": 1.0,
        "evaluated": 2,
        "known": 2,
        "precision": 1.0,
        "top_k_hits": 0,
        "unknown": 0,
    }
    assert report["attributes"]["font_name"]["top_k_hits"] == 1
    assert report["attributes"]["font_weight"]["unknown"] == 2
    assert report["attributes"]["font_width"]["coverage"] == 0.0
    assert report["attributes"]["font_size_px"]["precision"] == 0.0
    assert report["attributes"]["alignment"]["evaluated"] == 2
    assert report["attributes"]["rotation_deg"]["unknown"] == 2
    assert report["round_trip"] == {"evaluated": 1, "passed": 1, "rate": 1.0}
    assert report["hard_negative"] == {"evaluated": 2, "abstained": 0, "rate": 0.0}
    assert report["gates"]["hard_negative_abstention"] is False


def test_runner_writes_jsonl_summary_html_and_contact_sheet_inside_the_run(tmp_path: Path):
    spec_path = Path(__file__).resolve().parent / "fixtures" / "style_benchmark_v2" / "benchmark_spec.json"
    lock_path = tmp_path / "runtime.lock.json"
    lock_path.write_text(
        json.dumps({"schema_version": 1, "runtime": generate_style_benchmark_v2._runtime_metadata()}),
        encoding="utf-8",
    )

    run_dir = run_style_benchmark_v2.run_benchmark(
        spec_path=spec_path,
        level="smoke",
        output_root=tmp_path / "runs",
        run_id="report-smoke",
        seed=1729,
        runtime_lock_path=lock_path,
    )

    assert (run_dir / "style_benchmark_records.jsonl").is_file()
    assert (run_dir / "style_benchmark_summary.json").is_file()
    assert (run_dir / "index.html").is_file()
    assert (run_dir / "contact_sheets" / "contact_sheet.jpg").is_file()
    records = (run_dir / "style_benchmark_records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 6
    first_record = json.loads(records[0])
    assert first_record["style_evidence_v1"]["source"]
    assert first_record["style_evidence_v2"]["schema_version"] == 2
    assert first_record["style_evidence_v2_shadow_policy"]["apply_to_renderer"] is False


def test_v2_shadow_score_abstains_for_empty_hard_negative_cases(tmp_path: Path):
    spec_path = Path(__file__).resolve().parent / "fixtures" / "style_benchmark_v2" / "benchmark_spec.json"
    lock_path = tmp_path / "runtime.lock.json"
    lock_path.write_text(
        json.dumps({"schema_version": 1, "runtime": generate_style_benchmark_v2._runtime_metadata()}),
        encoding="utf-8",
    )

    run_dir = run_style_benchmark_v2.run_benchmark(
        spec_path=spec_path,
        level="all",
        output_root=tmp_path / "runs",
        run_id="report-all",
        seed=1729,
        runtime_lock_path=lock_path,
    )
    summary = json.loads((run_dir / "style_benchmark_summary.json").read_text(encoding="utf-8"))

    assert summary["score"]["gates"]["hard_negative_abstention"] is False
    assert summary["score_v2_shadow"]["gates"]["hard_negative_abstention"] is True
