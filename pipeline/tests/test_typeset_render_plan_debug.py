import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_render_band_image_records_render_plan_and_preserves_debug_geometry(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((80, 120, 3), 245, dtype=np.uint8)
    page = {
        "_source_page_number": 3,
        "_band_index": 7,
        "texts": [
            {
                "id": "ocr_017",
                "text": "HELLO",
                "translated": "OLA",
                "bbox": [20, 20, 90, 55],
                "balloon_bbox": [10, 10, 100, 70],
                "balloon_type": "white",
                "tipo": "fala",
            }
        ],
    }
    block = dict(page["texts"][0])

    def fake_render(_img, render_block, **_kwargs):
        render_block["safe_text_box"] = [18, 18, 92, 62]
        render_block["_debug_safe_text_box"] = [18, 18, 92, 62]
        render_block["render_bbox"] = [20, 22, 86, 50]
        render_block["qa_flags"] = ["render_on_art_suspected"]
        render_block["_render_debug"] = {
            "font_size_final": 22,
            "line_height": 27,
            "wrapped_lines": ["OLA"],
            "fit_status": "PASS",
        }

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.object(renderer_mod, "build_render_blocks", return_value=[block]):
            with patch.object(renderer_mod, "render_text_block", side_effect=fake_render):
                renderer_mod.render_band_image(band, page)
    finally:
        bind_recorder(None)

    render_plan_path = tmp_path / "debug" / "e2e" / "09_typeset" / "render_plan_raw.jsonl"
    payload = json.loads(render_plan_path.read_text(encoding="utf-8").splitlines()[0])

    assert payload["stage"] == "typeset"
    assert payload["text_id"] == "ocr_017"
    assert payload["page_id"] == "page_003"
    assert payload["band_id"] == "page_003_band_007"
    assert payload["coordinate_space"] == "band"
    assert payload["safe_text_box"] == [18, 18, 92, 62]
    assert payload["render_bbox"] == [20, 22, 86, 50]
    assert payload["balloon_bbox"] == [10, 10, 100, 70]
    assert payload["fit_status"] == "PASS"
    assert payload["qa_flags"] == ["render_on_art_suspected"]
    assert page["texts"][0]["safe_text_box"] == [18, 18, 92, 62]
    assert page["texts"][0]["render_bbox"] == [20, 22, 86, 50]

    final_plan_path = tmp_path / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    final_payload = json.loads(final_plan_path.read_text(encoding="utf-8").splitlines()[0])
    assert final_payload["coordinate_space"] == "page"
    assert final_payload["text_id"] == "ocr_017"


def test_render_band_image_records_missing_balloon_bbox_audit(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((60, 100, 3), 245, dtype=np.uint8)
    page = {
        "_source_page_number": 2,
        "_band_index": 4,
        "texts": [
            {
                "id": "ocr_022",
                "text": "MISSING",
                "translated": "FALTANDO",
                "bbox": [10, 10, 60, 30],
                "tipo": "fala",
            }
        ],
    }

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.object(renderer_mod, "build_render_blocks", return_value=[dict(page["texts"][0])]):
            renderer_mod.render_band_image(band, page)
    finally:
        bind_recorder(None)

    audit_path = tmp_path / "debug" / "e2e" / "09_typeset" / "balloon_bbox_missing_audit.jsonl"
    payload = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])

    assert payload["text_id"] == "ocr_022"
    assert payload["band_id"] == "page_002_band_004"
    assert payload["fallback_used"] == "bbox_as_balloon_bbox"
    assert "sem balloon_bbox" in payload["warning_in_pipeline_log"]


def test_render_band_image_does_not_audit_skipped_raw_text_without_balloon_bbox(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((60, 100, 3), 245, dtype=np.uint8)
    page = {
        "_source_page_number": 2,
        "_band_index": 4,
        "texts": [
            {
                "id": "ocr_noise",
                "text": "NOISE",
                "translated": "NOISE",
                "bbox": [10, 10, 60, 30],
                "tipo": "noise",
                "skip_processing": True,
            }
        ],
    }

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.object(renderer_mod, "build_render_blocks", return_value=[]):
            renderer_mod.render_band_image(band, page)
    finally:
        bind_recorder(None)

    audit_path = tmp_path / "debug" / "e2e" / "09_typeset" / "balloon_bbox_missing_audit.jsonl"
    assert not audit_path.exists()
