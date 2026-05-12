import json
from pathlib import Path

from PIL import Image

from pipeline.tools.run_scheduler_shadow_gate import evaluate_scheduler_shadow_gate


def _write_output(
    output_dir: Path,
    *,
    page_count: int = 2,
    text_count: int = 2,
    band_count: int = 4,
    include_strip_summary: bool = True,
) -> None:
    translated_dir = output_dir / "translated"
    translated_dir.mkdir(parents=True)
    pages = []
    for page_index in range(page_count):
        image_name = f"{page_index + 1:03d}.jpg"
        Image.new("RGB", (64, 96), color=(255, 255, 255)).save(translated_dir / image_name)
        text_layers = [
            {
                "id": f"p{page_index + 1}-t{text_index + 1}",
                "bbox": [10, 20 + text_index * 16, 50, 32 + text_index * 16],
                "original": f"TEXT {text_index + 1}",
                "translated": f"TEXTO {text_index + 1}",
                "tipo": "fala",
            }
            for text_index in range(text_count)
        ]
        page_profile = {"width": 64, "height": 96}
        if include_strip_summary and page_index == 0:
            page_profile["strip_perf_summary"] = {
                "band_count": band_count,
                "entries": [
                    {"band_index": index, "durations_sec": {"ocr": 0.1}}
                    for index in range(band_count)
                ],
            }
        pages.append(
            {
                "numero": page_index + 1,
                "arquivo_traduzido": f"translated/{image_name}",
                "page_profile": page_profile,
                "text_layers": text_layers,
                "inpaint_blocks": [
                    {"bbox": layer["bbox"], "confidence": 0.9}
                    for layer in text_layers
                ],
            }
        )
    project = {
        "paginas": pages,
        "estatisticas": {
            "total_paginas": page_count,
            "total_textos": page_count * text_count,
            "translated_regions": page_count * text_count,
        },
    }
    (output_dir / "project.json").write_text(
        json.dumps(project, ensure_ascii=False),
        encoding="utf-8",
    )


def test_scheduler_shadow_gate_passes_for_equivalent_outputs(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline)
    _write_output(candidate)

    result = evaluate_scheduler_shadow_gate(baseline, candidate, tmp_path / "gate")

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["scheduler_validation_status"] == "PASS"
    assert result["gate"]["output_compare_status"] == "PASS"
    assert result["gate"]["band_count"] == 4
    assert result["gate"]["task_count"] == 16
    assert result["gate"]["max_gpu_parallel"] == 1
    assert (tmp_path / "gate" / "summary.json").exists()


def test_scheduler_shadow_gate_fails_when_candidate_output_differs(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, text_count=2)
    _write_output(candidate, text_count=3)

    result = evaluate_scheduler_shadow_gate(baseline, candidate, tmp_path / "gate")

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["output_compare_status"] == "FAIL"
    assert "scheduler shadow output compare failed" in result["gate"]["reasons"]


def test_scheduler_shadow_gate_blocks_without_strip_perf_summary(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline)
    _write_output(candidate, include_strip_summary=False)

    result = evaluate_scheduler_shadow_gate(baseline, candidate, tmp_path / "gate")

    assert result["gate"]["status"] == "BLOCK"
    assert "candidate missing strip_perf_summary entries" in result["gate"]["reasons"]
