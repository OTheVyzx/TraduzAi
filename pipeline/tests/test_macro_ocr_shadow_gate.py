import json
from pathlib import Path

from pipeline.tools.run_macro_ocr_shadow_gate import evaluate_macro_ocr_shadow_gate


def _write_project(
    output_dir: Path,
    *,
    ocr_times: list[float],
    text_bboxes: list[list[int]],
) -> None:
    (output_dir / "translated").mkdir(parents=True)
    (output_dir / "translated" / "001.jpg").write_bytes(b"fake")
    entries = []
    for index, ocr_time in enumerate(ocr_times):
        entries.append(
            {
                "band_index": index,
                "y_top": index * 100,
                "y_bottom": index * 100 + 90,
                "durations_sec": {"ocr": ocr_time},
                "total_sec": ocr_time,
            }
        )
    project = {
        "paginas": [
            {
                "numero": 1,
                "page_profile": {
                    "y_in_strip_top": 0,
                    "y_in_strip_bottom": 1000,
                    "strip_perf_summary": {
                        "band_count": len(entries),
                        "text_count": len(text_bboxes),
                        "durations_sec": {"ocr": sum(ocr_times)},
                        "entries": entries,
                    },
                },
                "text_layers": [
                    {"text": f"T{idx}", "bbox": bbox, "confidence": 0.9}
                    for idx, bbox in enumerate(text_bboxes)
                ],
                "inpaint_blocks": [],
            }
        ],
        "estatisticas": {
            "total_paginas": 1,
            "total_textos": len(text_bboxes),
            "tempo_processamento_seg": 60.0,
        },
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_macro_ocr_shadow_gate_passes_when_savings_and_mapping_risk_are_good(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(
        output_dir,
        ocr_times=[5.0, 5.0, 5.0, 5.0],
        text_bboxes=[[10, 20, 80, 60], [10, 120, 80, 160]],
    )

    result = evaluate_macro_ocr_shadow_gate(
        output_dir,
        tmp_path / "gate",
        min_savings_seconds=10.0,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["estimated_savings_seconds"] == 15.0
    assert result["gate"]["fallback_rate"] == 0.0


def test_macro_ocr_shadow_gate_blocks_when_project_lacks_strip_summary(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "project.json").write_text(
        json.dumps({"paginas": [], "estatisticas": {}}),
        encoding="utf-8",
    )

    result = evaluate_macro_ocr_shadow_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "BLOCK"
    assert "strip_perf_summary" in result["gate"]["reasons"][0]


def test_macro_ocr_shadow_gate_fails_when_estimated_savings_are_too_low(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(
        output_dir,
        ocr_times=[2.0, 2.0],
        text_bboxes=[[10, 20, 80, 60]],
    )

    result = evaluate_macro_ocr_shadow_gate(
        output_dir,
        tmp_path / "gate",
        min_savings_seconds=10.0,
    )

    assert result["gate"]["status"] == "FAIL"
    assert "below threshold" in result["gate"]["reasons"][0]
