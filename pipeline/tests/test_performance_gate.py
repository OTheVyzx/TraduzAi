import json
from pathlib import Path

from pipeline.tools.run_performance_gate import evaluate_performance_gate


def _write_project(
    output_dir: Path,
    *,
    durations_sec: dict[str, float],
    total_seconds: float = 20.0,
) -> None:
    (output_dir / "translated").mkdir(parents=True)
    (output_dir / "translated" / "001.jpg").write_bytes(b"fake")
    project = {
        "paginas": [
            {
                "numero": 1,
                "page_profile": {
                    "strip_perf_summary": {
                        "band_count": 1,
                        "text_count": 1,
                        "durations_sec": durations_sec,
                        "entries": [
                            {
                                "band_index": 0,
                                "text_count": 1,
                                "remaining_inpaint_blocks": 1,
                                "durations_sec": durations_sec,
                                "total_sec": sum(durations_sec.values()),
                            }
                        ],
                    }
                },
                "text_layers": [{"id": "a"}],
                "inpaint_blocks": [{"bbox": [1, 2, 3, 4]}],
            }
        ],
        "estatisticas": {
            "total_paginas": 1,
            "total_textos": 1,
            "tempo_processamento_seg": total_seconds,
        },
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_performance_gate_passes_when_ocr_and_inpaint_are_visual_bottleneck(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(
        output_dir,
        durations_sec={"ocr": 8.0, "inpaint": 8.0, "translate": 2.0, "typeset": 2.0},
    )

    result = evaluate_performance_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["visual_bottleneck_share"] == 0.8
    assert (tmp_path / "gate" / "summary.json").exists()


def test_performance_gate_fails_when_visual_stages_are_not_the_bottleneck(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(
        output_dir,
        durations_sec={"ocr": 1.0, "inpaint": 1.0, "translate": 7.0, "typeset": 1.0},
    )

    result = evaluate_performance_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "FAIL"
    assert "visual bottleneck share" in result["gate"]["reasons"][0]


def test_performance_gate_blocks_when_required_artifacts_are_missing(tmp_path):
    result = evaluate_performance_gate(tmp_path / "missing", tmp_path / "gate")

    assert result["gate"]["status"] == "BLOCK"
    assert "project.json" in result["gate"]["reasons"][0]
    assert (tmp_path / "gate" / "summary.json").exists()
