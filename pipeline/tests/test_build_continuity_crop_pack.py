import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_continuity_crop_pack import build_continuity_crop_pack


def _write_project_fixture(run_dir: Path) -> None:
    (run_dir / "originals").mkdir(parents=True)
    (run_dir / "images").mkdir(parents=True)
    (run_dir / "layers" / "bubble-mask").mkdir(parents=True)

    original = Image.new("RGB", (220, 160), (236, 236, 236))
    draw = ImageDraw.Draw(original)
    draw.ellipse((30, 24, 190, 128), fill=(255, 255, 255), outline=(15, 15, 15), width=3)
    draw.rectangle((82, 66, 140, 82), fill=(10, 10, 10))
    original.save(run_dir / "originals" / "001.png")

    inpaint = original.copy()
    draw_inpaint = ImageDraw.Draw(inpaint)
    draw_inpaint.rectangle((78, 62, 146, 88), fill=(248, 248, 248))
    inpaint.save(run_dir / "images" / "001.png")

    bubble_mask = Image.new("L", (220, 160), 0)
    draw_mask = ImageDraw.Draw(bubble_mask)
    draw_mask.ellipse((30, 24, 190, 128), fill=7)
    bubble_mask.save(run_dir / "layers" / "bubble-mask" / "001.png")

    project = {
        "paginas": [
            {
                "numero": 1,
                "image_layers": {
                    "base": {"path": "originals/001.png"},
                    "inpaint": {"path": "images/001.png"},
                    "bubble_mask": {"path": "layers/bubble-mask/001.png"},
                },
                "text_layers": [
                    {
                        "id": "txt-1",
                        "bbox": [78, 60, 146, 90],
                        "source_bbox": [78, 60, 146, 90],
                        "text_pixel_bbox": [82, 66, 140, 82],
                        "balloon_bbox": [30, 24, 190, 128],
                        "bubble_mask_layer_path": "layers/bubble-mask/001.png",
                        "bubble_mask_value": 7,
                        "bubble_mask_source": "real_bubble_mask",
                        "balloon_type": "white",
                        "layout_profile": "white_balloon",
                        "line_polygons": [[[82, 66], [140, 66], [140, 82], [82, 82]]],
                        "confidence": 0.91,
                        "qa_flags": [],
                    }
                ],
                "inpaint_blocks": [],
            }
        ]
    }
    (run_dir / "project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")


def test_build_continuity_crop_pack_exports_masks_manifest_and_zip(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "pack"
    _write_project_fixture(run_dir)

    result = build_continuity_crop_pack(run_dir, out_dir, max_cases=10, write_zip=True)

    assert result["status"] == "PASS"
    assert result["selected_count"] == 1
    case_dir = out_dir / "candidates" / "case_000001"
    assert (case_dir / "original_crop.png").exists()
    assert (case_dir / "current_inpaint.png").exists()
    assert (case_dir / "text_mask.png").exists()
    assert (case_dir / "bubble_mask.png").exists()
    assert (case_dir / "overlay_debug.png").exists()
    assert (case_dir / "metadata.json").exists()

    text_mask = np.array(Image.open(case_dir / "text_mask.png"))
    bubble_mask = np.array(Image.open(case_dir / "bubble_mask.png"))
    assert int(np.count_nonzero(text_mask)) > 0
    assert int(np.count_nonzero(bubble_mask)) > int(np.count_nonzero(text_mask))

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_count"] == 1
    assert manifest["counts_by_type"]["balloon_fill"] == 1
    assert manifest["cases"][0]["files"]["original_crop"] == "candidates/case_000001/original_crop.png"

    archive = out_dir / "continuity_crop_pack_v001.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "candidates/case_000001/original_crop.png" in names


def test_build_continuity_crop_pack_routes_critical_flags_to_negative(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "pack"
    _write_project_fixture(run_dir)
    project_path = run_dir / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["paginas"][0]["text_layers"][0]["qa_flags"] = ["mask_outside_balloon_critical"]
    project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

    result = build_continuity_crop_pack(run_dir, out_dir, max_cases=10, write_zip=False)

    assert result["status"] == "PASS"
    metadata = json.loads((out_dir / "candidates" / "case_000001" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["estimated_type"] == "negative"
    assert metadata["bucket"] == "candidates_negative"
    assert "mask_outside_balloon_critical" in metadata["qa_flags"]
