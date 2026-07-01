import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_render_band_image_splits_raw_band_plan_from_deduped_final_page_plan(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((120, 160, 3), 245, dtype=np.uint8)
    page = {
        "_source_page_number": 1,
        "_band_index": 3,
        "texts": [
            {
                "id": "ocr_001",
                "text": "PLEASE",
                "translated": "POR FAVOR",
                "bbox": [30, 20, 110, 50],
                "balloon_bbox": [20, 10, 130, 80],
                "band_y_top": 2700,
                "band_height": 900,
                "tipo": "fala",
            }
        ],
    }

    render_calls = {"count": 0}

    def fake_render(_img, render_block):
        render_calls["count"] += 1
        render_block["safe_text_box"] = [28, 18, 112, 60]
        render_block["_debug_safe_text_box"] = [28, 18, 112, 60]
        render_block["render_bbox"] = [32, 22, 108, 48 + render_calls["count"]]
        render_block["_render_debug"] = {
            "target_bbox": [20, 10, 130, 80],
            "position_bbox": [24, 14, 126, 76],
            "capacity_bbox": [26, 16, 124, 74],
            "layout_safe_bbox": [22, 12, 128, 78],
            "fit_status": "PASS",
        }

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.object(renderer_mod, "build_render_blocks", side_effect=lambda texts: [dict(texts[0])]):
            with patch.object(renderer_mod, "render_text_block", side_effect=fake_render):
                renderer_mod.render_band_image(band, page)
                renderer_mod.render_band_image(band, page)
    finally:
        bind_recorder(None)

    root = tmp_path / "debug" / "e2e" / "09_typeset"
    raw = _jsonl(root / "render_plan_raw.jsonl")
    final = _jsonl(root / "render_plan_final.jsonl")

    assert len(raw) == 2
    assert all(entry["coordinate_space"] == "band" for entry in raw)
    assert all(entry["page_id"] == "page_001" and entry["band_id"] == "page_001_band_003" for entry in raw)
    assert all(entry["text_id"] == "ocr_001" and entry["band_y_top"] == 2700 for entry in raw)

    assert len(final) == 1
    assert final[0]["coordinate_space"] == "page"
    assert final[0]["text_id"] == "ocr_001"
    assert final[0]["render_bbox"] == [32, 2722, 108, 2750]
    assert final[0]["safe_text_box"] == [28, 2718, 112, 2760]
    assert final[0]["target_bbox"] == [20, 2710, 130, 2780]


def test_render_band_image_infers_trace_ids_from_text_band_id(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((80, 120, 3), 245, dtype=np.uint8)
    page = {
        "_source_page_number": None,
        "_band_index": None,
        "_band_id": None,
        "texts": [
            {
                "id": "ocr_001",
                "band_id": "page_002_band_019",
                "band_y_top": 48,
                "text": "HELLO",
                "translated": "OLA",
                "bbox": [20, 20, 80, 50],
                "balloon_bbox": [10, 10, 90, 60],
                "tipo": "fala",
            }
        ],
    }

    def fake_render(_img, render_block):
        render_block["render_bbox"] = [22, 24, 78, 46]

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.object(renderer_mod, "build_render_blocks", side_effect=lambda texts: [dict(texts[0])]):
            with patch.object(renderer_mod, "render_text_block", side_effect=fake_render):
                renderer_mod.render_band_image(band, page)
    finally:
        bind_recorder(None)

    final = _jsonl(tmp_path / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl")
    assert final[0]["page_id"] == "page_002"
    assert final[0]["band_id"] == "page_002_band_019"


def test_render_band_image_does_not_double_shift_page_space_inputs(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((120, 160, 3), 245, dtype=np.uint8)
    page = {
        "_coordinate_space": "page",
        "_source_page_number": 2,
        "_band_index": 3,
        "_band_y_top": 2700,
        "texts": [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_002_band_003",
                "band_id": "page_002_band_003",
                "band_y_top": 2700,
                "text": "PLEASE",
                "translated": "POR FAVOR",
                "bbox": [30, 2720, 110, 2750],
                "balloon_bbox": [20, 2710, 130, 2780],
                "tipo": "fala",
            }
        ],
    }

    def fake_render(_img, render_block):
        render_block["safe_text_box"] = [28, 2718, 112, 2760]
        render_block["render_bbox"] = [32, 2722, 108, 2749]
        render_block["_render_debug"] = {
            "target_bbox": [20, 2710, 130, 2780],
            "position_bbox": [24, 2714, 126, 2776],
            "capacity_bbox": [26, 2716, 124, 2774],
            "fit_status": "PASS",
        }

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.object(renderer_mod, "build_render_blocks", side_effect=lambda texts: [dict(texts[0])]):
            with patch.object(renderer_mod, "render_text_block", side_effect=fake_render):
                renderer_mod.render_band_image(band, page)
    finally:
        bind_recorder(None)

    final = _jsonl(tmp_path / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl")
    assert final[0]["coordinate_space"] == "page"
    assert final[0]["render_bbox"] == [32, 2722, 108, 2749]
    assert final[0]["safe_text_box"] == [28, 2718, 112, 2760]
    assert final[0]["target_bbox"] == [20, 2710, 130, 2780]


def test_render_band_image_copies_debug_by_trace_when_text_ids_repeat(tmp_path):
    from debug_tools import bind_recorder
    from typesetter import renderer as renderer_mod

    band = np.full((80, 120, 3), 245, dtype=np.uint8)
    first = {
        "id": "ocr_001",
        "text_id": "ocr_001",
        "trace_id": "ocr_001@page_001_band_001",
        "band_id": "page_001_band_001",
        "text": "FIRST",
        "translated": "PRIMEIRO",
        "bbox": [20, 20, 80, 50],
        "balloon_bbox": [10, 10, 90, 60],
        "tipo": "fala",
    }
    second = {
        "id": "ocr_001",
        "text_id": "ocr_001",
        "trace_id": "ocr_001@page_001_band_002",
        "band_id": "page_001_band_002",
        "text": "SECOND",
        "translated": "SEGUNDO",
        "bbox": [20, 20, 80, 50],
        "balloon_bbox": [10, 10, 90, 60],
        "tipo": "fala",
    }
    page = {"_band_id": "page_001_band_002", "texts": [first, second]}

    def fake_render(_img, render_block):
        render_block["render_bbox"] = [22, 24, 78, 46]

    bind_recorder(None)
    with patch.object(renderer_mod, "build_render_blocks", side_effect=lambda _texts: [dict(second)]):
        with patch.object(renderer_mod, "render_text_block", side_effect=fake_render):
            renderer_mod.render_band_image(band, page)

    assert first.get("render_bbox") is None
    assert second["render_bbox"] == [22, 24, 78, 46]


def test_project_render_plan_final_preserves_source_style_debug_fields(tmp_path):
    from debug_tools import DebugRecorder
    import main

    evidence = {
        "source": "pixel_analysis",
        "text_color": "#FFFFFF",
        "text_color_confidence": 0.82,
        "stroke_color": "#000000",
        "stroke_width_px": 3,
        "stroke_confidence": 0.78,
    }
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_001_band_002",
                        "band_id": "page_001_band_002",
                        "original": "BOOM",
                        "translated": "BUM",
                        "target_bbox": [20, 1010, 140, 1080],
                        "safe_text_box": [28, 1018, 132, 1070],
                        "render_bbox": [32, 1022, 128, 1064],
                        "tipo": "sfx",
                        "estilo": {
                            "fonte": "KOMIKAX_.ttf",
                            "cor": "#FFFFFF",
                            "contorno": "#000000",
                            "contorno_px": 3,
                        },
                        "style_origin": "source_detected",
                        "style_confidence": 0.82,
                        "style_source": "pixel_analysis",
                        "style_evidence": evidence,
                    }
                ],
            }
        ],
    }
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    audit = main._write_debug_render_plan_final_from_project(recorder, project)

    rows = _jsonl(tmp_path / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl")
    assert audit["summary"]["written_count"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "project_json_final"
    assert row["style_origin"] == "source_detected"
    assert row["style_confidence"] == 0.82
    assert row["style_source"] == "pixel_analysis"
    assert row["style_evidence"] == evidence
    assert row["estilo"]["cor"] == "#FFFFFF"
    assert row["estilo"]["contorno"] == "#000000"
    assert row["estilo"]["contorno_px"] == 3


def test_project_render_plan_final_skips_merged_into_primary_layers(tmp_path):
    from debug_tools import DebugRecorder
    import main

    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_001_band_001",
                        "band_id": "page_001_band_001",
                        "translated": "PRIMARIO",
                        "render_bbox": [20, 30, 120, 70],
                        "safe_text_box": [10, 20, 130, 80],
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                    },
                    {
                        "id": "ocr_001_fragment_2",
                        "text_id": "ocr_001_fragment_2",
                        "trace_id": "ocr_001@page_001_band_001#fragment_2",
                        "band_id": "page_001_band_001",
                        "translated": "FRAGMENTO",
                        "render_bbox": [40, 90, 150, 130],
                        "safe_text_box": [30, 80, 160, 140],
                        "visible": False,
                        "route_action": "merged_into_primary",
                        "render_policy": "normal",
                    },
                ],
            }
        ],
    }
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    audit = main._write_debug_render_plan_final_from_project(recorder, project)

    rows = _jsonl(tmp_path / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl")
    assert audit["summary"]["written_count"] == 1
    assert [row["text_id"] for row in rows] == ["ocr_001"]
