import json
from pathlib import Path

from pipeline.tools.run_translation_batch_gate import evaluate_translation_batch_gate


def _write_project(output_dir: Path, durations_sec: dict[str, float]) -> None:
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
                                "durations_sec": durations_sec,
                                "total_sec": sum(durations_sec.values()),
                            }
                        ],
                    }
                },
                "text_layers": [{"text": "HELLO", "translated": "OLA"}],
                "inpaint_blocks": [{"bbox": [1, 2, 3, 4]}],
            }
        ],
        "estatisticas": {
            "total_paginas": 1,
            "total_textos": 1,
            "tempo_processamento_seg": sum(durations_sec.values()),
        },
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_translation_batch_gate_fails_when_translation_time_is_low_impact(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir, {"ocr": 40.0, "inpaint": 50.0, "translate": 0.5})

    result = evaluate_translation_batch_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["translation_seconds"] == 0.5
    assert "translation stage is below batching threshold" in result["gate"]["reasons"][0]
    assert (tmp_path / "gate" / "summary.json").exists()


def test_translation_batch_gate_passes_when_translation_time_is_material(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _write_project(output_dir, {"ocr": 20.0, "inpaint": 20.0, "translate": 8.0})

    result = evaluate_translation_batch_gate(
        output_dir,
        tmp_path / "gate",
        min_translation_seconds=5.0,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["translation_seconds"] == 8.0


def test_translation_batch_gate_blocks_when_project_is_missing(tmp_path):
    result = evaluate_translation_batch_gate(tmp_path / "missing", tmp_path / "gate")

    assert result["gate"]["status"] == "BLOCK"
    assert "missing project.json" in result["gate"]["reasons"][0]
