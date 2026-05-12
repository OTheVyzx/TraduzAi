import json
from pathlib import Path

from PIL import Image

from pipeline.tools.compare_pipeline_outputs import evaluate_pipeline_output_compare


def _write_output(
    output_dir: Path,
    *,
    text_count: int = 2,
    image_size: tuple[int, int] = (32, 48),
    page_count: int = 1,
    macro_shadow: bool = False,
    skip_reasons: list[str] | None = None,
) -> None:
    translated_dir = output_dir / "translated"
    translated_dir.mkdir(parents=True)
    pages = []
    skip_reasons = skip_reasons or []
    for page_index in range(page_count):
        image_name = f"{page_index + 1:03d}.jpg"
        Image.new("RGB", image_size, color=(255, 255, 255)).save(translated_dir / image_name)
        text_layers = []
        for text_index in range(text_count):
            skip_reason = skip_reasons[text_index] if text_index < len(skip_reasons) else None
            text_layers.append(
                {
                    "id": f"p{page_index + 1}-t{text_index + 1}",
                    "bbox": [10, 20 + text_index * 20, 80, 40 + text_index * 20],
                    "original": f"ORIGINAL {text_index + 1}",
                    "translated": f"TRADUZIDO {text_index + 1}",
                    "tipo": "fala",
                    "skip_processing": bool(skip_reason),
                    "skip_reason": skip_reason,
                }
            )
        page_profile = {"width": image_size[0], "height": image_size[1]}
        if macro_shadow:
            page_profile["macro_ocr_shadow"] = {"status": "PASS"}
        pages.append(
            {
                "numero": page_index + 1,
                "arquivo_traduzido": f"translated/{image_name}",
                "page_profile": page_profile,
                "text_layers": text_layers,
                "inpaint_blocks": [
                    {"bbox": layer["bbox"], "confidence": 0.9}
                    for layer in text_layers
                    if not layer["skip_processing"]
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


def test_output_compare_passes_when_contract_and_image_dimensions_match(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline)
    _write_output(candidate, macro_shadow=True)

    result = evaluate_pipeline_output_compare(baseline, candidate, tmp_path / "gate")

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["page_count"] == {"baseline": 1, "candidate": 1}
    assert result["gate"]["text_count"] == {"baseline": 2, "candidate": 2}
    assert result["gate"]["image_dimension_mismatch_count"] == 0
    assert (tmp_path / "gate" / "summary.json").exists()


def test_output_compare_fails_when_candidate_loses_text_without_audited_skip(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, text_count=2)
    _write_output(candidate, text_count=1)

    result = evaluate_pipeline_output_compare(baseline, candidate, tmp_path / "gate")

    assert result["gate"]["status"] == "FAIL"
    assert "text count changed without matching audited skips" in result["gate"]["reasons"]


def test_output_compare_blocks_when_required_artifacts_are_missing(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline)
    candidate.mkdir()

    result = evaluate_pipeline_output_compare(baseline, candidate, tmp_path / "gate")

    assert result["gate"]["status"] == "BLOCK"
    assert "candidate missing project.json" in result["gate"]["reasons"]
    assert (tmp_path / "gate" / "summary.json").exists()
