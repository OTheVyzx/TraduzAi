from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from qa.text_identity_audit import build_text_identity_audit, write_text_identity_audit
from schema.visual_scene import build_visual_text_instances
from main import _run_page_scene_identity_shadow
import qa.text_identity_audit as text_identity_audit


def _project(layers: list[dict]) -> dict:
    return {"paginas": [{"id": "page_003", "text_layers": layers}]}


def test_overlapping_band_observations_share_one_instance_with_exactly_one_owner():
    project = _project(
        [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "band_id": "page_003_band_035",
                "source_trace_ids": ["ocr_001@page_003_band_035"],
                "bbox": [100, 400, 300, 470],
                "translated": "WHO IS PAYING TODAY?",
            },
            {
                "id": "ocr_001_observer",
                "trace_id": "ocr_001@page_003_band_035",
                "band_id": "page_003_band_036",
                "source_trace_ids": ["ocr_001@page_003_band_035"],
                "bbox": [100, 400, 300, 470],
                "translated": "WHO IS PAYING TODAY?",
            },
        ]
    )

    instances = build_visual_text_instances(project)

    assert len(instances) == 1
    instance = instances[0]
    assert instance["owner_band_id"] == "page_003_band_035"
    assert instance["observer_band_ids"] == ["page_003_band_035", "page_003_band_036"]
    assert instance["source_trace_ids"] == ["ocr_001@page_003_band_035"]
    assert len(instance["layer_ids"]) == 2


def test_full_ocr_wins_over_spatially_identical_substring_without_creating_second_owner():
    project = _project(
        [
            {
                "id": "owner",
                "trace_id": "ocr_001@page_003_band_035",
                "band_id": "page_003_band_035",
                "bbox": [100, 400, 310, 480],
                "translated": "WHO IS PAYING TODAY?",
            },
            {
                "id": "partial_reocr",
                "trace_id": "reocr_001@page_003_band_036",
                "source_trace_ids": ["ocr_001@page_003_band_035"],
                "band_id": "page_003_band_036",
                "bbox": [105, 405, 305, 475],
                "translated": "PAYING TODAY?",
            },
        ]
    )

    instances = build_visual_text_instances(project)

    assert len(instances) == 1
    assert instances[0]["owner_layer_id"] == "owner"
    assert instances[0]["owner_text"] == "WHO IS PAYING TODAY?"


def test_owner_prefers_the_observation_with_full_band_coverage_and_edge_distance():
    project = _project(
        [
            {
                "id": "cropped_observer",
                "trace_id": "ocr_010@page_003_band_035",
                "band_id": "page_003_band_035",
                "source_trace_ids": ["ocr_010@page_003_band_035"],
                "band_bbox": [0, 400, 800, 450],
                "bbox": [100, 400, 300, 470],
                "translated": "FULL TEXT",
            },
            {
                "id": "interior_owner",
                "trace_id": "ocr_010@page_003_band_035",
                "band_id": "page_003_band_036",
                "source_trace_ids": ["ocr_010@page_003_band_035"],
                "band_bbox": [0, 350, 800, 550],
                "bbox": [100, 400, 300, 470],
                "translated": "FULL TEXT",
            },
        ]
    )

    instance = build_visual_text_instances(project)[0]

    assert instance["owner_layer_id"] == "interior_owner"
    assert instance["owner_selection"]["coverage_ratio"] == 1.0
    assert instance["owner_selection"]["edge_distance"] == 50


def test_two_distinct_reading_order_components_in_one_balloon_stay_separate():
    project = _project(
        [
            {
                "id": "upper",
                "trace_id": "ocr_upper@page_003_band_035",
                "band_id": "page_003_band_035",
                "balloon_bbox": [80, 350, 360, 610],
                "bbox": [110, 390, 320, 450],
                "translated": "WHO IS PAYING TODAY?",
                "reading_order": 1,
            },
            {
                "id": "lower",
                "trace_id": "ocr_lower@page_003_band_035",
                "band_id": "page_003_band_035",
                "balloon_bbox": [80, 350, 360, 610],
                "bbox": [110, 500, 320, 570],
                "translated": "I AM STARVING.",
                "reading_order": 2,
            },
        ]
    )

    instances = build_visual_text_instances(project)

    assert len(instances) == 2
    assert [item["reading_order"] for item in instances] == [1, 2]
    assert {item["container_id"] for item in instances} == {"page_003_container_001"}


def test_reused_local_ocr_ids_in_distinct_bands_do_not_form_one_identity():
    project = _project(
        [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "source_text_ids": ["ocr_001"],
                "band_id": "page_003_band_035",
                "bbox": [100, 400, 300, 470],
                "translated": "FIRST DIALOGUE.",
            },
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_036",
                "source_text_ids": ["ocr_001"],
                "band_id": "page_003_band_036",
                "bbox": [100, 720, 300, 790],
                "translated": "SECOND DIALOGUE.",
            },
        ]
    )

    instances = build_visual_text_instances(project)

    assert len(instances) == 2
    assert {item["owner_band_id"] for item in instances} == {
        "page_003_band_035",
        "page_003_band_036",
    }
    assert build_text_identity_audit(project)["summary"]["duplicate_owner_layer_ids"] == 0


def test_audit_requires_accepted_ocr_to_render_or_have_an_explicit_suppression_reason():
    project = _project(
        [
            {
                "id": "accepted_unrendered",
                "trace_id": "ocr_002@page_003_band_035",
                "band_id": "page_003_band_035",
                "bbox": [100, 400, 300, 470],
                "translated": "UNRENDERED",
                "ocr_accepted": True,
            }
        ]
    )

    audit = build_text_identity_audit(project)

    assert audit["summary"]["accepted_without_terminal_lifecycle"] == 1
    assert audit["summary"]["passed"] is False


def test_audit_marks_risky_multi_source_merge_for_review_without_changing_ownership():
    project = _project(
        [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "source_trace_ids": [
                    "ocr_001@page_003_band_035",
                    "ocr_002@page_003_band_035",
                ],
                "band_id": "page_003_band_035",
                "bbox": [100, 400, 500, 700],
                "translated": "COMBINED TEXT FROM DISTINCT BODIES.",
                "render_bbox": [80, 350, 700, 800],
                "qa_flags": ["same_balloon_fragment_merged", "TEXT_OVERFLOW", "render_outside_balloon"],
            }
        ]
    )

    audit = build_text_identity_audit(project)

    assert audit["summary"]["multi_source_review_required"] == 1
    review = audit["review_required"][0]
    assert review["text_instance_id"] == "page_003_text_001"
    assert review["source_trace_ids"] == [
        "ocr_001@page_003_band_035",
        "ocr_002@page_003_band_035",
    ]
    assert review["risk_flags"] == ["same_balloon_fragment_merged", "TEXT_OVERFLOW", "render_outside_balloon"]
    assert audit["instances"][0]["owner_layer_id"] == "ocr_001"


def test_audit_writer_creates_only_the_shadow_audit_artifact(tmp_path):
    output = write_text_identity_audit(
        tmp_path / "debug" / "e2e" / "05_layout_geometry",
        _project(
            [
                {
                    "id": "rendered",
                    "trace_id": "ocr_001@page_003_band_035",
                    "band_id": "page_003_band_035",
                    "bbox": [1, 2, 30, 40],
                    "translated": "RENDERED",
                    "render_bbox": [1, 2, 30, 40],
                }
            ]
        ),
    )

    assert output.is_file()
    assert output.name == "text_identity_audit.json"


def test_main_shadow_flag_writes_identity_audit_without_mutating_layers(tmp_path, monkeypatch: pytest.MonkeyPatch):
    project = _project(
        [
            {
                "id": "rendered",
                "trace_id": "ocr_001@page_003_band_035",
                "band_id": "page_003_band_035",
                "bbox": [1, 2, 30, 40],
                "translated": "RENDERED",
                "render_bbox": [1, 2, 30, 40],
            }
        ]
    )
    before = repr(project["paginas"][0]["text_layers"])
    monkeypatch.setenv("TRADUZAI_FLAG_PAGE_SCENE_IDENTITY_V2", "1")

    result = _run_page_scene_identity_shadow(project, tmp_path)

    assert result["enabled"] is True
    assert result["written"] is True
    assert Path(result["path"]).is_file()
    assert repr(project["paginas"][0]["text_layers"]) == before


def test_main_shadow_audit_write_failure_does_not_abort_the_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch):
    project = _project([])
    monkeypatch.setenv("TRADUZAI_FLAG_PAGE_SCENE_IDENTITY_V2", "1")
    monkeypatch.setattr(
        text_identity_audit,
        "write_text_identity_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    result = _run_page_scene_identity_shadow(project, tmp_path)

    assert result["enabled"] is True
    assert result["written"] is False
    assert "OSError" in result["error"]
