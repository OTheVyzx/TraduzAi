import json
from pathlib import Path

from PIL import Image

from pipeline.tools.render_vision_debug_sheet import render_vision_debug_sheet


def _write_project(output_dir: Path) -> None:
    for folder in ("originals", "images", "translated"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 60), color=(255, 255, 255)).save(output_dir / folder / "001.jpg")
    project = {
        "paginas": [
            {
                "numero": 1,
                "image_layers": {
                    "base": {"path": "originals/001.jpg"},
                    "inpaint": {"path": "images/001.jpg"},
                    "rendered": {"path": "translated/001.jpg"},
                },
                "text_layers": [
                    {
                        "id": "t1",
                        "original": "\ud558\ud558",
                        "translated": "\ud558\ud558",
                        "qa_flags": ["source_script_leak"],
                        "glossary_hits": [{"source": "x", "target": "y"}],
                    }
                ],
            }
        ]
    }
    (output_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")


def test_render_vision_debug_sheet_writes_html_for_p0_pages(tmp_path):
    output = tmp_path / "out"
    _write_project(output)

    result = render_vision_debug_sheet(output, tmp_path / "debug" / "sheet.html", filters=["P0"])

    html = (tmp_path / "debug" / "sheet.html").read_text(encoding="utf-8")
    assert result["status"] == "PASS"
    assert result["selected_pages"] == [1]
    assert "Pagina 1" in html
    assert "source_script_leak" in html
    assert (tmp_path / "debug" / "assets" / "translated_001.jpg").exists()


def test_render_vision_debug_sheet_blocks_without_project(tmp_path):
    result = render_vision_debug_sheet(tmp_path / "missing", tmp_path / "debug" / "sheet.html")

    assert result["status"] == "BLOCK"
    assert "missing project.json" in result["reasons"]
