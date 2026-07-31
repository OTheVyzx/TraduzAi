import json
from pathlib import Path

import cv2
import numpy as np

from debug_tools import style_audit_report


class _FakeEvidence:
    def to_dict(self) -> dict:
        return {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.95,
            "stroke_color": "#002244",
            "stroke_width_px": 3,
            "stroke_confidence": 0.9,
            "gradient": True,
            "gradient_colors": ["#FFFFFF", "#74D8FF"],
            "gradient_confidence": 0.88,
            "glow": True,
            "glow_color": "#BDEEFF",
            "glow_confidence": 0.86,
            "glow_px": 4,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.82,
        }


def test_style_audit_report_separates_detected_evidence_from_applied_style(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    originals_dir = run_dir / "originals"
    originals_dir.mkdir(parents=True)
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(originals_dir / "001.jpg"), image)

    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text": "ALCHEMY, THIS IS TRULY AN INSANE ABILITY.",
                        "tipo": "fala",
                        "bbox": [10, 10, 70, 50],
                        "content_class": "text",
                        "route_action": "translate_inpaint_render",
                        "style_origin": "auto",
                        "style_confidence": 0.95,
                        "estilo": {
                            "fonte": "ComicNeue-Bold.ttf",
                            "cor": "#000000",
                            "contorno": "",
                            "contorno_px": 0,
                            "cor_gradiente": [],
                            "glow": False,
                            "glow_cor": "",
                            "glow_px": 0,
                        },
                    }
                ]
            }
        ]
    }
    (run_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")
    monkeypatch.setattr(style_audit_report, "extract_text_style_evidence", lambda _crop: _FakeEvidence())

    records = style_audit_report._read_project_records(run_dir, originals_dir)

    assert len(records) == 1
    assert records[0]["gradient"] is True
    assert records[0]["glow"] is True
    assert records[0]["style_origin"] == "auto"
    assert records[0]["applied_font_name"] == "ComicNeue-Bold.ttf"
    assert records[0]["applied_gradient"] is False
    assert records[0]["applied_glow"] is False
    assert records[0]["applied_stroke_color"] == ""
    assert records[0]["style_evidence_v2"]["schema_version"] == 2
    assert records[0]["style_evidence_v2"]["attributes"]["fill"]["value"] == "#FFFFFF"
    assert records[0]["style_evidence_v2_shadow_policy"]["apply_to_renderer"] is False


def test_style_audit_report_skips_unapplied_sfx_visual_review_candidates(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    originals_dir = run_dir / "originals"
    originals_dir.mkdir(parents=True)
    image = np.full((120, 90, 3), 240, dtype=np.uint8)
    cv2.imwrite(str(originals_dir / "001.jpg"), image)

    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "sfx_visual_ornament",
                        "text": "",
                        "tipo": "sfx",
                        "bbox": [20, 10, 55, 110],
                        "content_class": "sfx",
                        "detector": "sfx_visual",
                        "route_action": "review_required",
                        "render_policy": "review_required",
                        "style_origin": "auto",
                        "estilo": {
                            "fonte": "ComicNeue-Bold.ttf",
                            "cor": "#000000",
                            "contorno": "",
                            "contorno_px": 0,
                            "cor_gradiente": [],
                            "glow": False,
                            "glow_cor": "",
                            "glow_px": 0,
                        },
                        "sfx": {"visual_promotion": True, "source_text": "", "adapted_text": ""},
                    }
                ]
            }
        ]
    }
    (run_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")

    def fail_extract(_crop):
        raise AssertionError("style extractor should not run for non-promoted SFX visuals")

    monkeypatch.setattr(style_audit_report, "extract_text_style_evidence", fail_extract)

    records = style_audit_report._read_project_records(run_dir, originals_dir)

    assert len(records) == 1
    assert records[0]["style_scan_skipped"] is True
    assert records[0]["style_scan_skip_reason"] == "not_style_copy_candidate"
    assert records[0].get("stroke_color") in (None, "")
    assert records[0].get("glow") is not True
    assert records[0].get("gradient") is not True


def test_style_audit_report_skips_low_confidence_primary_ocr_candidates(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    originals_dir = run_dir / "originals"
    originals_dir.mkdir(parents=True)
    image = np.full((90, 180, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(originals_dir / "001.jpg"), image)

    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "ocr_low_conf",
                        "text": "ALCHEMY, THIS IS TRULY AN INSANE ABILITY.",
                        "tipo": "fala",
                        "bbox": [10, 10, 160, 70],
                        "content_class": "text",
                        "route_action": "translate_inpaint_render",
                        "ocr_confidence": 0.52,
                        "style_origin": "auto",
                        "estilo": {
                            "fonte": "ComicNeue-Bold.ttf",
                            "cor": "#000000",
                            "contorno": "",
                            "contorno_px": 0,
                            "cor_gradiente": [],
                            "glow": False,
                            "glow_cor": "",
                            "glow_px": 0,
                        },
                    }
                ]
            }
        ]
    }
    (run_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")

    def fail_extract(_crop):
        raise AssertionError("style extractor should not run for low-confidence OCR text")

    monkeypatch.setattr(style_audit_report, "extract_text_style_evidence", fail_extract)

    records = style_audit_report._read_project_records(run_dir, originals_dir)

    assert len(records) == 1
    assert records[0]["style_scan_skipped"] is True
    assert records[0]["style_scan_skip_reason"] == "low_candidate_confidence"
