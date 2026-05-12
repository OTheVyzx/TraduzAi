import json
from pathlib import Path

from PIL import Image

from pipeline.tools.run_project_import_gate import evaluate_project_import_gate


def _write_image(path: Path, size: tuple[int, int] = (64, 96)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_importable_project(output_dir: Path) -> None:
    for rel in (
        "originals/001.jpg",
        "images/001.jpg",
        "translated/001.jpg",
        "layers/mask/001.png",
        "layers/brush/001.png",
        "layers/recovery/001.png",
    ):
        _write_image(output_dir / rel)

    project = {
        "versao": "2.0",
        "app": "traduzai",
        "obra": "Teste",
        "capitulo": 1,
        "paginas": [
            {
                "numero": 1,
                "arquivo_original": "originals/001.jpg",
                "arquivo_traduzido": "translated/001.jpg",
                "image_layers": {
                    "base": {"key": "base", "path": "originals/001.jpg", "visible": True, "locked": True},
                    "mask": {"key": "mask", "path": "layers/mask/001.png", "visible": False, "locked": False},
                    "inpaint": {"key": "inpaint", "path": "images/001.jpg", "visible": False, "locked": True},
                    "brush": {"key": "brush", "path": "layers/brush/001.png", "visible": False, "locked": False},
                    "recovery": {"key": "recovery", "path": "layers/recovery/001.png", "visible": False, "locked": False},
                    "rendered": {"key": "rendered", "path": "translated/001.jpg", "visible": True, "locked": True},
                },
                "inpaint_blocks": [{"bbox": [10, 20, 40, 60], "confidence": 0.9}],
                "text_layers": [
                    {
                        "id": "tl_001_001",
                        "bbox": [10, 20, 40, 60],
                        "layout_bbox": [10, 20, 40, 60],
                        "original": "HELLO",
                        "translated": "OLA",
                        "tipo": "fala",
                        "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 28},
                        "visible": True,
                    }
                ],
                "textos": [
                    {
                        "id": "tl_001_001",
                        "bbox": [10, 20, 40, 60],
                        "original": "HELLO",
                        "traduzido": "OLA",
                        "tipo": "fala",
                    }
                ],
            }
        ],
        "estatisticas": {"total_paginas": 1, "total_textos": 1},
    }
    (output_dir / "project.json").write_text(
        json.dumps(project, ensure_ascii=False),
        encoding="utf-8",
    )


def test_project_import_gate_passes_for_editor_compatible_project(tmp_path):
    output_dir = tmp_path / "run"
    _write_importable_project(output_dir)

    result = evaluate_project_import_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["page_count"] == 1
    assert result["gate"]["text_layer_count"] == 1
    assert result["gate"]["checked_image_count"] == 6
    assert (tmp_path / "gate" / "summary.json").exists()


def test_project_import_gate_fails_when_referenced_image_is_missing(tmp_path):
    output_dir = tmp_path / "run"
    _write_importable_project(output_dir)
    (output_dir / "translated" / "001.jpg").unlink()

    result = evaluate_project_import_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "FAIL"
    assert "missing referenced image" in result["gate"]["reasons"][0]


def test_project_import_gate_allows_placeholder_editing_layers(tmp_path):
    output_dir = tmp_path / "run"
    _write_importable_project(output_dir)
    for rel in ("layers/mask/001.png", "layers/brush/001.png", "layers/recovery/001.png"):
        _write_image(output_dir / rel, size=(1, 1))

    result = evaluate_project_import_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["image_dimension_mismatch_count"] == 0


def test_project_import_gate_fails_when_text_layer_bbox_cannot_be_hydrated(tmp_path):
    output_dir = tmp_path / "run"
    _write_importable_project(output_dir)
    project_path = output_dir / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["paginas"][0]["text_layers"][0]["bbox"] = [10, 20, 10, 60]
    project["paginas"][0]["text_layers"][0].pop("layout_bbox", None)
    project_path.write_text(json.dumps(project), encoding="utf-8")

    result = evaluate_project_import_gate(output_dir, tmp_path / "gate")

    assert result["gate"]["status"] == "FAIL"
    assert "invalid text layer bbox" in result["gate"]["reasons"][0]
