import json
from pathlib import Path

from PIL import Image

from pipeline.tools.export_visual_review_sheet import export_visual_review_sheet


def _write_output(
    output_dir: Path,
    *,
    macro_reports: list[dict] | None = None,
    page_count: int = 2,
    color_offset: int = 0,
) -> None:
    translated_dir = output_dir / "translated"
    translated_dir.mkdir(parents=True)
    pages = []
    for page_number in range(1, page_count + 1):
        image_name = f"{page_number:03d}.jpg"
        Image.new(
            "RGB",
            (40, 50),
            color=(
                255 - page_number - color_offset,
                255 - page_number - color_offset,
                255 - page_number - color_offset,
            ),
        ).save(translated_dir / image_name)
        page_profile = {"width": 40, "height": 50}
        if page_number == 1 and macro_reports is not None:
            page_profile["macro_ocr_shadow"] = {
                "status": "PASS",
                "page_reports": macro_reports,
            }
        pages.append(
            {
                "numero": page_number,
                "arquivo_traduzido": f"translated/{image_name}",
                "page_profile": page_profile,
                "text_layers": [
                    {
                        "bbox": [5, 5, 20, 15],
                        "original": f"ORIGINAL {page_number}",
                        "translated": f"TRADUZIDO {page_number}",
                        "skip_processing": False,
                    }
                ],
                "inpaint_blocks": [{"bbox": [5, 5, 20, 15]}],
            }
        )
    (output_dir / "project.json").write_text(
        json.dumps({"paginas": pages, "estatisticas": {"total_textos": page_count}}),
        encoding="utf-8",
    )


def test_visual_review_sheet_prioritizes_macro_ocr_risk_pages(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline)
    _write_output(
        candidate,
        macro_reports=[
            {"page_number": 1, "different_text_rate": 0.1, "missing_text_rate": 0.0},
            {"page_number": 2, "different_text_rate": 0.8, "missing_text_rate": 0.0},
        ],
    )

    result = export_visual_review_sheet(
        baseline,
        candidate,
        tmp_path / "review" / "sheet.html",
        max_pages=1,
    )

    html_path = tmp_path / "review" / "sheet.html"
    html = html_path.read_text(encoding="utf-8")
    assert result["status"] == "PASS"
    assert result["selected_pages"] == [2]
    assert "Pagina 2" in html
    assert "different_text_rate" in html
    assert (tmp_path / "review" / "assets" / "baseline_002.jpg").exists()
    assert (tmp_path / "review" / "assets" / "candidate_002.jpg").exists()


def test_visual_review_sheet_reports_pixel_difference_rate(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, page_count=1)
    _write_output(candidate, page_count=1, color_offset=20)

    result = export_visual_review_sheet(
        baseline,
        candidate,
        tmp_path / "review" / "sheet.html",
        max_pages=1,
    )

    assert result["status"] == "PASS"
    assert result["page_reports"][0]["pixel_diff_rate"] > 0
    assert "pixel_diff_rate" in (tmp_path / "review" / "sheet.html").read_text(
        encoding="utf-8"
    )


def test_visual_review_sheet_exports_text_region_crops(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_output(baseline, page_count=1)
    _write_output(candidate, page_count=1)

    result = export_visual_review_sheet(
        baseline,
        candidate,
        tmp_path / "review" / "sheet.html",
        max_pages=1,
    )

    html = (tmp_path / "review" / "sheet.html").read_text(encoding="utf-8")
    assert result["page_reports"][0]["crop_count"] == 1
    assert result["asset_count"] == 4
    assert "Crops" in html
    assert (tmp_path / "review" / "assets" / "baseline_001_crop_001.jpg").exists()
    assert (tmp_path / "review" / "assets" / "candidate_001_crop_001.jpg").exists()


def test_visual_review_sheet_blocks_when_project_is_missing(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()

    result = export_visual_review_sheet(
        baseline,
        candidate,
        tmp_path / "review" / "sheet.html",
    )

    assert result["status"] == "BLOCK"
    assert "baseline missing project.json" in result["reasons"]
