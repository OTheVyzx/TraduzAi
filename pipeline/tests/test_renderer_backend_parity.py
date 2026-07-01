from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts import compare_render_backends
from typesetter import rust_backend


def _write_fixture(root: Path) -> Path:
    case_dir = root / "simple_balloon"
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "name": "simple_balloon",
                "width": 220,
                "height": 140,
                "background": "#ffffff",
                "texts": [
                    {
                        "id": "txt-1",
                        "translated": "SIM, NAO FUNCIONA",
                        "bbox": [30, 25, 190, 115],
                        "balloon_bbox": [30, 25, 190, 115],
                        "safe_text_box": [45, 42, 175, 92],
                        "estilo": {
                            "fonte": "ComicNeue-Bold.ttf",
                            "tamanho": 24,
                            "cor": "#000000",
                            "contorno": "#ffffff",
                            "contorno_px": 1,
                            "alinhamento": "center",
                            "sombra": False,
                            "glow": False,
                            "cor_gradiente": [],
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return case_dir


def test_compare_render_backends_writes_images_and_report(monkeypatch, tmp_path):
    fixture_dir = tmp_path / "fixtures"
    _write_fixture(fixture_dir)
    out_dir = tmp_path / "out"

    def fake_render_request_to_image(request, timeout=30):
        image = Image.new("RGBA", (request["image_width"], request["image_height"]), (0, 0, 0, 0))
        x1, y1, x2, y2 = request["blocks"][0]["box"]
        for x in range(x1 + 6, min(x2, x1 + 44)):
            for y in range(y1 + 6, min(y2, y1 + 24)):
                image.putpixel((x, y), (0, 0, 0, 255))
        return image

    monkeypatch.setattr(rust_backend, "render_request_to_image", fake_render_request_to_image)

    report = compare_render_backends.compare_fixture_dir(fixture_dir, out_dir)

    assert report["total"] == 1
    assert (out_dir / "simple_balloon" / "python.png").exists()
    assert (out_dir / "simple_balloon" / "koharu_rust.png").exists()
    assert (out_dir / "simple_balloon" / "diff.png").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "contact_sheet.png").exists()
    assert report["results"][0]["fallback_occurred"] is False
    assert report["results"][0]["metrics"]["pixel_diff_pct"] >= 0.0
    assert report["results"][0]["metrics"]["alpha_coverage_diff"] >= 0


def test_compare_render_backends_reports_rust_fallback(monkeypatch, tmp_path):
    fixture_dir = tmp_path / "fixtures"
    _write_fixture(fixture_dir)
    out_dir = tmp_path / "out"

    def fail_render_request_to_image(request, timeout=30):
        raise rust_backend.RustRendererError("boom")

    monkeypatch.setattr(rust_backend, "render_request_to_image", fail_render_request_to_image)

    report = compare_render_backends.compare_fixture_dir(fixture_dir, out_dir)

    assert report["results"][0]["fallback_occurred"] is True
    assert "boom" not in report["results"][0].get("error", "")
