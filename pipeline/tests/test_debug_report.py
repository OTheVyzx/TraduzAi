import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools.report import generate_debug_report


def test_debug_report_lists_top_issues_with_relative_artifact_links(tmp_path):
    e2e_root = tmp_path / "debug" / "e2e"
    artifact = e2e_root / "05_layout_geometry" / "page_001_bbox.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"png")
    gate_dir = e2e_root / "11_qa_export_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "export_gate.json").write_text(
        json.dumps(
            {
                "status": "BLOCK",
                "issues": [
                    {
                        "page": 1,
                        "layer": "t1",
                        "type": "p0_render_blocker",
                        "severity": "critical",
                        "flags": ["bbox_overreach_critical"],
                        "linked_artifacts": ["05_layout_geometry/page_001_bbox.png"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = generate_debug_report(e2e_root)

    report_dir = e2e_root / "13_report"
    md = (report_dir / "debug_report.md").read_text(encoding="utf-8")
    payload = json.loads((report_dir / "debug_report.json").read_text(encoding="utf-8"))

    assert result["issue_count"] == 1
    assert payload["top_issues"][0]["flags"] == ["bbox_overreach_critical"]
    assert payload["top_issues"][0]["artifact_links"][0]["href"] == "../05_layout_geometry/page_001_bbox.png"
    assert "[05_layout_geometry/page_001_bbox.png](../05_layout_geometry/page_001_bbox.png)" in md


def test_debug_report_preserves_issue_artifact_links_without_existing_files(tmp_path):
    e2e_root = tmp_path / "debug" / "e2e"
    gate_dir = e2e_root / "11_qa_export_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "export_gate.json").write_text(
        json.dumps(
            {
                "status": "BLOCK",
                "issues": [
                    {
                        "page": 7,
                        "page_id": "page_007",
                        "band_id": "page_007_band_003",
                        "trace_id": "ocr_001@page_007_band_003",
                        "text_id": "ocr_001",
                        "text_instance_id": "ocr_001@page_007_band_003",
                        "layer": "ocr_001",
                        "type": "p0_render_blocker",
                        "severity": "critical",
                        "flags": ["render_outside_balloon"],
                        "artifact_links": [
                            "translated/007.jpg",
                            "09_typeset/render_plan_final.jsonl",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = generate_debug_report(e2e_root)

    links = result["top_issues"][0]["artifact_links"]
    assert [link["label"] for link in links] == [
        "translated/007.jpg",
        "09_typeset/render_plan_final.jsonl",
    ]
    assert result["top_issues"][0]["trace_id"] == "ocr_001@page_007_band_003"
    assert result["top_issues"][0]["page_id"] == "page_007"
    assert result["top_issues"][0]["band_id"] == "page_007_band_003"
    assert result["top_issues"][0]["text_id"] == "ocr_001"
    assert result["top_issues"][0]["text_instance_id"] == "ocr_001@page_007_band_003"


def test_debug_report_tolerates_missing_artifacts(tmp_path):
    e2e_root = tmp_path / "debug" / "e2e"

    result = generate_debug_report(e2e_root)

    report_dir = e2e_root / "13_report"
    md = (report_dir / "debug_report.md").read_text(encoding="utf-8")
    payload = json.loads((report_dir / "debug_report.json").read_text(encoding="utf-8"))

    assert result["status"] == "UNKNOWN"
    assert payload["top_issues"] == []
    assert "Nenhuma issue encontrada" in md
