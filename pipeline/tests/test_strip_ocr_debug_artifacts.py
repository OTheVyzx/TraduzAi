import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import DebugRecorder, bind_recorder
from strip.bands import visual_card_edge_expansion
from strip.run import (
    _candidate_matches_band_text_bbox,
    _reconcile_overlapping_band_ocr_fragments_before_translation,
    run_chapter,
)
from strip.process_bands import (
    BandStageOutput,
    _band_to_page_dict,
    _merge_candidate_crop_recovery_into_ocr_page,
    _recover_empty_ocr_with_candidate_crops,
    _run_direct_paddle_candidate_crop_reocr,
    fuse_negative_dark_bubble_candidates,
    process_band,
)
from strip.types import Band, Balloon, BBox, OutputPage, VerticalStrip
from vision_stack.runtime import build_page_result


def test_owner_graph_snapshots_publish_component_and_observation_artifacts(tmp_path):
    from ownership.artifacts import OwnerArtifactPublisher

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-owner-strip")
    graph = {
        "page_id": "page_003",
        "components": [{"component_id": "component_3", "bbox_page": [1, 2, 30, 20]}],
        "observations": [],
        "owners": [],
        "component_dispositions": [
            {"component_id": "component_3", "decision": "preserve", "reason": "explicit_credit_policy"}
        ],
        "violations": [],
    }
    OwnerArtifactPublisher(recorder).publish(graphs={"page_003": graph})

    root = tmp_path / "debug" / "e2e"
    component = json.loads(
        (root / "02_strip_detect/page_owner_components.jsonl").read_text(encoding="utf-8")
    )
    assert component["page_id"] == "page_003"
    assert component["owner_id"] is None
    assert component["disposition"] == "preserve"


def test_debug_provenance_rejects_metadata_from_another_run(tmp_path):
    from inpainter import _debug_inpaint_metadata_matches_run, _write_strip_inpaint_debug

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-current")
    bind_recorder(recorder)
    image = np.full((12, 20, 3), [220, 40, 10], dtype=np.uint8)
    mask = np.zeros((12, 20), dtype=np.uint8)
    page = {"_band_id": "page_001_band_002", "_source_page_number": 1, "_band_index": 2, "texts": []}
    debug_root = tmp_path / "debug_inpaint"
    try:
        with patch.dict("os.environ", {"TRADUZAI_INPAINT_DEBUG_DIR": str(debug_root)}):
            _write_strip_inpaint_debug(
                page,
                original_rgb=image,
                working_rgb=image,
                cleaned_rgb=image.copy(),
                vision_blocks=[],
                used_real_inpaint=False,
                fast_fill_mask=mask,
                raw_mask=mask,
                expanded_mask=mask,
            )
    finally:
        bind_recorder(None)

    metadata = json.loads((debug_root / "page_001_band_002" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-current"
    assert metadata["band_id"] == "page_001_band_002"
    assert metadata["color_space"] == "RGB"
    assert metadata["dimensions"] == {"height": 12, "width": 20, "channels": 3}
    assert _debug_inpaint_metadata_matches_run(metadata, "run-current") is True
    assert _debug_inpaint_metadata_matches_run(metadata, "run-stale") is False


class FakeRuntime:
    def run_ocr_stage(self, _image_rgb, page_dict):
        band_id = page_dict["_band_id"]
        return {
            "image": band_id,
            "width": 120,
            "height": 80,
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "HELLO",
                    "bbox": [10, 12, 50, 40],
                    "confidence": 0.87,
                    "confidence_raw": 0.87,
                    "tipo": "dialogo",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [10, 12, 50, 40], "confidence": 0.87}],
        }


class FakeTranslator:
    def translate_pages(self, pages, **_kwargs):
        out = []
        for page in pages:
            page = dict(page)
            page["texts"] = [dict(text, translated=text["text"]) for text in page.get("texts", [])]
            out.append(page)
        return out


class FakeInpainter:
    def inpaint_band_image(self, image_rgb, _page):
        return np.array(image_rgb, copy=True)


class FakeTypesetter:
    def render_band_image(self, image_rgb, _page):
        return np.array(image_rgb, copy=True)


def test_band_to_page_dict_assigns_stable_band_id_and_block_trace_metadata():
    band = Band(
        y_top=100,
        y_bottom=200,
        balloons=[Balloon(BBox(10, 120, 60, 180), confidence=0.91)],
        strip_slice=np.full((100, 120, 3), 255, dtype=np.uint8),
    )

    page = _band_to_page_dict(band, page_idx=4, source_page_number=2)

    assert page["_band_id"] == "page_002_band_004"
    assert page["_vision_blocks"][0]["band_id"] == "page_002_band_004"


def test_translate_stage_metadata_merge_preserves_band_trace_ids():
    from strip.process_bands import _merge_translated_page_metadata

    merged = _merge_translated_page_metadata(
        {
            "numero": 2,
            "_source_page_number": 2,
            "_band_index": 4,
            "_band_id": "page_002_band_004",
            "_band_y_top": 1200,
            "texts": [{"id": "ocr_001", "bbox": [1, 2, 3, 4]}],
        },
        {
            "numero": 2,
            "_source_page_number": None,
            "_band_index": None,
            "_band_id": None,
            "_band_y_top": None,
            "texts": [{"id": "ocr_001", "translated": "ola"}],
        },
    )

    assert merged["_source_page_number"] == 2
    assert merged["_band_index"] == 4
    assert merged["_band_id"] == "page_002_band_004"
    assert merged["_band_y_top"] == 1200
    assert merged["texts"][0]["bbox"] == [1, 2, 3, 4]


def test_translate_stage_metadata_merge_preserves_owner_observations_append_only():
    from strip.process_bands import _merge_translated_page_metadata

    observation = {
        "observation_id": "observation_keep",
        "page_id": "page_002",
        "component_ids": [],
        "text": "SOURCE",
        "confidence": 0.88,
        "provider": "candidate_crop",
        "bbox_page": [10, 20, 80, 44],
        "polygons_page": [],
        "tile_provenance": ["band_a"],
        "coverage_score": None,
        "language_score": None,
        "rejection_reason": None,
        "legacy_selected": False,
    }

    merged = _merge_translated_page_metadata(
        {
            "texts": [{"id": "ocr_001", "text": "SOURCE"}],
            "owner_observations": [observation],
            "_owner_tile_projection": {
                "page_id": "page_002",
                "tile_id": "band_a",
                "offset_xy": [0, 100],
            },
        },
        {"texts": [{"id": "ocr_001", "translated": "FONTE"}]},
    )

    assert merged["owner_observations"] == [observation]
    assert merged["_owner_tile_projection"]["offset_xy"] == [0, 100]


def test_candidate_text_matching_rejects_edge_overlap_from_next_balloon():
    candidate_bbox = [29, 7109, 642, 7722]
    lower_balloon_text = {
        "id": "ocr_003",
        "text_pixel_bbox": [344, 7702, 540, 7761],
        "layout_bbox": [344, 7702, 540, 7761],
        "bbox": [344, 7702, 540, 7761],
    }
    top_balloon_text = {
        "id": "ocr_001",
        "text_pixel_bbox": [148, 7248, 310, 7268],
        "layout_bbox": [148, 7248, 310, 7268],
        "bbox": [148, 7248, 310, 7268],
    }

    assert _candidate_matches_band_text_bbox(candidate_bbox, top_balloon_text)
    assert not _candidate_matches_band_text_bbox(candidate_bbox, lower_balloon_text)


def test_reconcile_overlapping_band_ocr_fragments_keeps_full_text_for_translation():
    bands = [
        SimpleNamespace(y_top=92743, y_bottom=93423),
        SimpleNamespace(y_top=93211, y_bottom=93648),
    ]
    precomputed = {
        0: {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "THAT'S RIGHT! HOW",
                    "bbox": [351, 641, 601, 665],
                    "text_pixel_bbox": [351, 641, 601, 665],
                }
            ]
        },
        1: {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "THAT'S RIGHT! HOW DID HE DODGE KIM SIHYEOK'S SWORD STRIKE THOUGH...",
                    "bbox": [274, 172, 673, 267],
                    "text_pixel_bbox": [274, 172, 673, 267],
                }
            ]
        },
    }

    reconciled = _reconcile_overlapping_band_ocr_fragments_before_translation(bands, precomputed)

    assert reconciled == 1
    assert precomputed[0]["texts"] == []
    assert precomputed[1]["texts"][0]["text"].startswith("THAT'S RIGHT! HOW DID HE")
    assert precomputed[1]["texts"][0]["cross_band_fragment_trace_ids"] == ["band_000:ocr_001"]


def test_reconcile_overlapping_bands_suppresses_equivalent_full_duplicate():
    bands = [
        Band(y_top=0, y_bottom=200, balloons=[Balloon(BBox(20, 40, 180, 160), confidence=0.95)]),
        Band(y_top=100, y_bottom=300, balloons=[Balloon(BBox(20, 120, 180, 240), confidence=0.95)]),
    ]
    pages = {
        0: {"texts": [{"id": "edge", "text": "DID IT GO WELL?", "bbox": [30, 105, 170, 145], "confidence": 0.8}], "_vision_blocks": [{"id": "edge", "bbox": [30, 105, 170, 145]}]},
        1: {"texts": [{"id": "central", "text": "DID IT GO well?", "bbox": [30, 5, 170, 45], "confidence": 0.92}], "_vision_blocks": [{"id": "central", "bbox": [30, 5, 170, 45]}]},
    }

    reconciled = _reconcile_overlapping_band_ocr_fragments_before_translation(bands, pages)

    assert reconciled == 1
    assert pages[1]["texts"] == []
    assert pages[1]["_vision_blocks"] == []
    owner = pages[0]["texts"][0]
    assert owner["cross_band_owner_trace_id"] == "band_000:edge"
    assert owner["cross_band_suppressed_trace_ids"] == ["band_001:central"]


def test_reconcile_overlapping_bands_quarantines_unsupported_edge_fragment():
    band = Band(y_top=100, y_bottom=300, balloons=[Balloon(BBox(20, 160, 180, 260), confidence=0.9)])
    page = {
        "texts": [{"id": "edge", "text": "IMLAK...", "bbox": [30, 0, 100, 20], "confidence": 0.6}],
        "_vision_blocks": [{"id": "edge", "bbox": [30, 0, 100, 20]}],
    }

    reconciled = _reconcile_overlapping_band_ocr_fragments_before_translation([band], {0: page})

    assert reconciled == 0
    assert page["texts"] == []
    quarantined = page["_cross_band_quarantined_texts"][0]
    assert quarantined["route_action"] == "review_required"
    assert quarantined["route_reason"] == "cross_band_unsupported_edge_fragment"
    assert quarantined["skip_processing"] is True
    assert quarantined["render_completed"] is False
    assert page["_cross_band_unsupported_edge_fragment_quarantined"] is True
    assert page["_vision_blocks"] == []


def test_reconcile_overlapping_bands_keeps_repeated_text_in_distinct_geometry():
    bands = [
        Band(y_top=0, y_bottom=200, balloons=[Balloon(BBox(10, 20, 100, 80), confidence=0.9)]),
        Band(y_top=100, y_bottom=300, balloons=[Balloon(BBox(240, 160, 330, 220), confidence=0.9)]),
    ]
    pages = {
        0: {"texts": [{"id": "left", "text": "HELP!", "bbox": [10, 20, 100, 60]}], "_vision_blocks": [{"id": "left", "bbox": [10, 20, 100, 60]}]},
        1: {"texts": [{"id": "right", "text": "HELP!", "bbox": [240, 60, 330, 100]}], "_vision_blocks": [{"id": "right", "bbox": [240, 60, 330, 100]}]},
    }

    reconciled = _reconcile_overlapping_band_ocr_fragments_before_translation(bands, pages)

    assert reconciled == 0
    assert [text["id"] for text in pages[0]["texts"]] == ["left"]
    assert [text["id"] for text in pages[1]["texts"]] == ["right"]


def _visual_card_edge_page(*, bbox, retried=False, card=True):
    record = {"id": "card", "text": "TITLE", "bbox": bbox}
    if card:
        record["qa_flags"] = ["visual_card_ocr_recall"]
    return {"texts": [record], "_adaptive_edge_retry_done": retried}


def test_visual_card_edge_recall_expands_only_own_band_and_keeps_id():
    band = Band(y_top=200, y_bottom=400)

    expansion = visual_card_edge_expansion(
        band, page_y_top=100, page_y_bottom=900, ocr_result=_visual_card_edge_page(bbox=[10, 2, 180, 40])
    )

    assert expansion == {"y_top": 100, "y_bottom": 400, "reason": "visual_card_edge_top"}
    assert band.y_top == 200 and band.y_bottom == 400


def test_visual_card_retry_is_limited_to_once():
    expansion = visual_card_edge_expansion(
        Band(y_top=200, y_bottom=400),
        page_y_top=100,
        page_y_bottom=900,
        ocr_result=_visual_card_edge_page(bbox=[10, 2, 180, 40], retried=True),
    )

    assert expansion is None


def test_visual_card_expansion_stops_at_source_page_boundary():
    expansion = visual_card_edge_expansion(
        Band(y_top=120, y_bottom=280),
        page_y_top=100,
        page_y_bottom=300,
        ocr_result=_visual_card_edge_page(bbox=[10, 150, 180, 159]),
    )

    assert expansion == {"y_top": 120, "y_bottom": 300, "reason": "visual_card_edge_bottom"}


def test_normal_speech_band_does_not_receive_card_expansion():
    assert visual_card_edge_expansion(
        Band(y_top=200, y_bottom=400),
        page_y_top=100,
        page_y_bottom=900,
        ocr_result=_visual_card_edge_page(bbox=[10, 2, 180, 40], card=False),
    ) is None


def test_run_chapter_writes_bands_manifest_with_stable_ids(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        strip = VerticalStrip(
            image=np.full((420, 120, 3), 255, dtype=np.uint8),
            width=120,
            height=420,
            source_page_breaks=[0, 420],
            page_x_offsets=[0],
        )
        balloons = [
            Balloon(BBox(10, 20, 50, 70), confidence=0.87),
            Balloon(BBox(20, 260, 80, 320), confidence=0.91),
        ]

        def output_page():
                return OutputPage(
                    y_top=0,
                    y_bottom=420,
                    image=np.full((420, 120, 3), 255, dtype=np.uint8),
                )

        with (
            patch("strip.run.build_strip", return_value=strip),
            patch("strip.run.detect_strip_balloons", return_value=balloons),
            patch("strip.run.assemble_output_pages", side_effect=lambda *_args, **_kwargs: [output_page()]),
        ):
            run_chapter(
                [tmp_path / "001.jpg"],
                tmp_path / "translated",
                detector=object(),
                runtime=FakeRuntime(),
                translator=FakeTranslator(),
                inpainter=FakeInpainter(),
                typesetter=FakeTypesetter(),
            )

        manifest_path = tmp_path / "debug" / "e2e" / "02_strip_detect" / "bands_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["band_count"] == 2
        assert [band["band_id"] for band in manifest["bands"]] == [
            "page_001_band_000",
            "page_001_band_001",
        ]
        assert manifest["bands"][0]["balloon_ids"] == ["page_001_band_000_balloon_00"]
    finally:
        bind_recorder(None)


def test_run_chapter_writes_pr16_debug_artifacts(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        strip = VerticalStrip(
            image=np.full((180, 120, 3), 255, dtype=np.uint8),
            width=120,
            height=180,
            source_page_breaks=[0, 180],
            page_x_offsets=[0],
        )
        balloons = [Balloon(BBox(10, 20, 50, 70), confidence=0.87)]

        def output_page():
            return OutputPage(
                y_top=0,
                y_bottom=180,
                image=np.full((180, 120, 3), 255, dtype=np.uint8),
            )

        with (
            patch("strip.run.build_strip", return_value=strip),
            patch("strip.run.detect_strip_balloons", return_value=balloons),
            patch("strip.run.assemble_output_pages", side_effect=lambda *_args, **_kwargs: [output_page()]),
        ):
            run_chapter(
                [tmp_path / "001.jpg"],
                tmp_path / "translated",
                detector=object(),
                runtime=FakeRuntime(),
                translator=FakeTranslator(),
                inpainter=FakeInpainter(),
                typesetter=FakeTypesetter(),
            )

        root = tmp_path / "debug" / "e2e"
        assert (root / "01_input_extract" / "input_manifest.json").exists()
        assert (root / "08_inpaint" / "inpaint_blocks.jsonl").exists()
        assert (root / "10_copyback_reassemble" / "copyback_decisions.jsonl").exists()
        assert (root / "10_copyback_reassemble" / "reassemble_manifest.json").exists()
        assert (root / "10_copyback_reassemble" / "page_cleanup_breakdown.json").exists()
        assert (root / "09_typeset" / "rendered_bands" / "page_001_band_000.jpg").exists()
        assert (root / "10_copyback_reassemble" / "final_bands" / "page_001_band_000.jpg").exists()
        assert (root / "10_copyback_reassemble" / "final_band_crops.jsonl").exists()
        assert (root / "12_contact_sheets" / "translated_comparison.jpg").exists()
        assert (root / "12_contact_sheets" / "problem_bands.jpg").exists()
        assert (root / "02_strip_detect" / "candidate_text_matching.jsonl").exists()
        assert (root / "02_strip_detect" / "page_owner_components.jsonl").exists()
        assert (root / "03_ocr" / "page_owner_observations.jsonl").exists()
        assert (root / "04_text_normalization_router" / "page_owner_graph.json").exists()
        assert (root / "09_typeset" / "owner_render_plan.jsonl").exists()
        assert (root / "11_qa_export_gate" / "owner_invariant_report.json").exists()

        breakdown = json.loads(
            (root / "10_copyback_reassemble" / "page_cleanup_breakdown.json").read_text(encoding="utf-8")
        )
        assert "cleanup_total" in breakdown["durations_sec"]
        detect_candidate = json.loads(
            (root / "02_strip_detect" / "detect_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        match_candidate = json.loads(
            (root / "02_strip_detect" / "candidate_text_matching.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert detect_candidate["matched_trace_ids"] == ["ocr_001@page_001_band_000"]
        assert detect_candidate["matched_text_ids"] == ["ocr_001"]
        assert detect_candidate["match_method"] == "same_band_bbox_overlap"
        assert detect_candidate["has_inner_dark_text"] is False
        assert detect_candidate["inner_dark_component_count"] == 0
        assert detect_candidate["inner_dark_area"] == 0
        assert detect_candidate["significant_component_count"] == 0
        assert detect_candidate["significant_area"] == 0
        assert detect_candidate["bright_pixel_ratio"] == 1.0
        assert detect_candidate["dark_pixel_ratio"] == 0.0
        assert match_candidate["match_method"] == "same_band_bbox_overlap"
        decision = json.loads(
            (root / "10_copyback_reassemble" / "copyback_decisions.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert decision["band_id"] == "page_001_band_000"
        assert decision["page_id"] == "page_001"
        assert decision["text_ids"] == ["ocr_001"]
        assert decision["trace_ids"] == ["ocr_001@page_001_band_000"]
        assert decision["trace_ids_in_band"] == ["ocr_001@page_001_band_000"]
        final_crop = json.loads(
            (root / "10_copyback_reassemble" / "final_band_crops.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert final_crop["band_id"] == "page_001_band_000"
        assert final_crop["translated_output_page"] == "001.jpg"
        assert final_crop["crop_bbox_in_translated_page"] == [0, 0, 120, 180]
        assert final_crop["final_crop_path"] == "10_copyback_reassemble/final_bands/page_001_band_000.jpg"
    finally:
        bind_recorder(None)


def test_process_band_writes_post_typeset_and_copyback_visual_debug(tmp_path):
    class MarkingTypesetter:
        def render_band_image(self, image_rgb, _page):
            out = np.array(image_rgb, copy=True)
            out[12:40, 10:50] = [0, 0, 0]
            return out

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        band = Band(
            y_top=100,
            y_bottom=180,
            balloons=[Balloon(BBox(10, 112, 50, 140), confidence=0.87)],
            strip_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
            original_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
        )

        process_band(
            band,
            runtime=FakeRuntime(),
            translator=FakeTranslator(),
            inpainter=FakeInpainter(),
            typesetter=MarkingTypesetter(),
            page_idx=0,
            source_page_number=1,
        )

        root = tmp_path / "debug" / "e2e"
        assert (root / "09_typeset" / "page_001_band_000" / "post_typeset.jpg").exists()
        assert (root / "10_copyback_reassemble" / "page_001_band_000" / "post_copyback.jpg").exists()
        manifest = json.loads(
            (root / "10_copyback_reassemble" / "page_001_band_000" / "band_crop_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["band_id"] == "page_001_band_000"
        assert manifest["post_typeset"] == "09_typeset/page_001_band_000/post_typeset.jpg"
        assert manifest["post_copyback"] == "10_copyback_reassemble/page_001_band_000/post_copyback.jpg"
    finally:
        bind_recorder(None)


def test_run_chapter_can_skip_page_cleanup_rerender_for_skip_inpaint(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        strip = VerticalStrip(
            image=np.full((180, 120, 3), 255, dtype=np.uint8),
            width=120,
            height=180,
            source_page_breaks=[0, 180],
            page_x_offsets=[0],
        )
        balloons = [Balloon(BBox(10, 20, 50, 70), confidence=0.87)]

        def output_page():
            return OutputPage(
                y_top=0,
                y_bottom=180,
                image=np.full((180, 120, 3), 255, dtype=np.uint8),
            )

        with (
            patch("strip.run.build_strip", return_value=strip),
            patch("strip.run.detect_strip_balloons", return_value=balloons),
            patch("strip.run.assemble_output_pages", side_effect=lambda *_args, **_kwargs: [output_page()]),
        ):
            run_chapter(
                [tmp_path / "001.jpg"],
                tmp_path / "translated",
                detector=object(),
                runtime=FakeRuntime(),
                translator=FakeTranslator(),
                inpainter=FakeInpainter(),
                typesetter=FakeTypesetter(),
                skip_page_cleanup_rerender=True,
            )

        breakdown = json.loads(
            (
                tmp_path
                / "debug"
                / "e2e"
                / "10_copyback_reassemble"
                / "page_cleanup_breakdown.json"
            ).read_text(encoding="utf-8")
        )
        assert breakdown["cleanup_skipped"] is True
        assert breakdown["durations_sec"]["cleanup_inpaint"] == 0.0
    finally:
        bind_recorder(None)


def test_ocr_confidence_audit_counts_only_lost_available_confidence():
    import strip.run as strip_run

    assert hasattr(strip_run, "_build_ocr_confidence_audit")

    page = OutputPage(
        y_top=0,
        y_bottom=100,
        image=np.full((100, 100, 3), 255, dtype=np.uint8),
        text_layers={
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "band_id": "page_001_band_000",
                    "confidence_raw": 0.87,
                    "confidence": 0.87,
                },
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "band_id": "page_001_band_001",
                    "confidence_raw": 0.66,
                    "confidence": 0.0,
                },
                {
                    "id": "ocr_003",
                    "text_id": "ocr_003",
                    "band_id": "page_001_band_002",
                    "confidence": 0.0,
                },
            ]
        },
    )

    audit = strip_run._build_ocr_confidence_audit([page])

    assert audit["summary"]["total_blocks"] == 3
    assert audit["summary"]["blocks_with_available_confidence"] == 2
    assert audit["summary"]["blocks_with_confidence_zero"] == 1
    assert audit["by_band"][0]["text_id"] == "ocr_002"


def test_process_band_writes_ocr_raw_blocks_jsonl_with_confidence_and_trace(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        band = Band(
            y_top=100,
            y_bottom=180,
            balloons=[Balloon(BBox(10, 112, 50, 140), confidence=0.87)],
            strip_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
            original_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
        )

        process_band(
            band,
            runtime=FakeRuntime(),
            translator=FakeTranslator(),
            inpainter=FakeInpainter(),
            typesetter=FakeTypesetter(),
            page_idx=0,
            source_page_number=1,
        )

        raw_path = tmp_path / "debug" / "e2e" / "03_ocr" / "ocr_raw_blocks.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        payload = rows[0]

        assert payload["text_id"] == "ocr_001"
        assert payload["page_id"] == "page_001"
        assert payload["band_id"] == "page_001_band_000"
        assert payload["trace_id"] == "ocr_001@page_001_band_000"
        assert payload["confidence_raw"] == 0.87
        assert payload["bbox_band"] == [10, 12, 50, 40]
        assert payload["bbox_page"] == [10, 112, 50, 140]
        assert band.ocr_result["texts"][0]["trace_id"] == "ocr_001@page_001_band_000"

        copyback_path = tmp_path / "debug" / "e2e" / "10_copyback_reassemble" / "copyback_decisions.jsonl"
        decision = json.loads(copyback_path.read_text(encoding="utf-8").splitlines()[0])
        assert decision["page_id"] == "page_001"
        assert decision["text_id"] == "ocr_001"
        assert decision["text_ids"] == ["ocr_001"]
        assert decision["trace_ids"] == ["ocr_001@page_001_band_000"]
        assert decision["trace_ids_in_band"] == ["ocr_001@page_001_band_000"]
    finally:
        bind_recorder(None)


def test_process_band_attaches_page_space_owner_observations_and_debug_manifest(tmp_path):
    class ObservationRuntime:
        def run_ocr_stage(self, _image_rgb, page_dict):
            assert page_dict["_owner_page_id"] == "page_001"
            assert page_dict["_owner_tile_id"] == "tile_page_001_a"
            assert page_dict["_owner_tile_offset_xy"] == [-5, 0]
            return {
                "image": page_dict["_band_id"],
                "width": 120,
                "height": 80,
                "texts": [
                    {
                        "id": "ocr_001",
                        "text": "WHOLE BODY",
                        "bbox": [10, 12, 50, 40],
                        "line_polygons": [
                            [[10, 12], [50, 12], [50, 40], [10, 40]],
                        ],
                        "confidence": 0.93,
                        "ocr_source": "paddle_full_page",
                    }
                ],
                "_vision_blocks": [{"bbox": [10, 12, 50, 40], "confidence": 0.93}],
            }

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="owner-observations")
    bind_recorder(recorder)
    try:
        balloon = Balloon(BBox(10, 112, 50, 140), confidence=0.93)
        balloon.metadata = {
            "bbox_page": [5, 12, 45, 40],
            "page_id": "page_001",
            "region_id": "region_must_not_become_semantic_owner",
        }
        band = Band(
            y_top=100,
            y_bottom=180,
            balloons=[balloon],
            strip_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
            original_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
            tile_id="tile_page_001_a",
        )

        process_band(
            band,
            runtime=ObservationRuntime(),
            translator=FakeTranslator(),
            inpainter=FakeInpainter(),
            typesetter=FakeTypesetter(),
            page_idx=0,
            source_page_number=1,
        )

        observations = band.ocr_result["owner_observations"]
        assert len(observations) == 1
        assert observations[0]["bbox_page"] == [5, 12, 45, 40]
        assert observations[0]["polygons_page"] == [
            [[5, 12], [45, 12], [45, 40], [5, 40]],
        ]
        assert observations[0]["provider"] == "paddle_full_page"
        assert observations[0]["component_ids"] == []
        assert "owner_id" not in observations[0]
        assert "semantic_owner" not in observations[0]

        manifest_path = tmp_path / "debug" / "e2e" / "03_ocr" / "owner_observations.jsonl"
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["observation_id"] == observations[0]["observation_id"]
        assert rows[0]["bbox_page"] == [5, 12, 45, 40]
        assert rows[0]["run_id"] == "owner-observations"
    finally:
        bind_recorder(None)


def test_candidate_crop_observation_is_projected_before_legacy_filters():
    class CropRuntime:
        def run_ocr_stage(self, image_rgb, _page_dict):
            height, width = image_rgb.shape[:2]
            return {
                "width": width,
                "height": height,
                "texts": [
                    {
                        "text": "COMPLETE CANDIDATE BODY",
                        "bbox": [2, 3, min(width, 32), min(height, 17)],
                        "line_polygons": [
                            [[2, 3], [min(width, 32), 3], [min(width, 32), min(height, 17)], [2, min(height, 17)]],
                        ],
                        "confidence": 0.88,
                        "ocr_source": "adaptive_crop",
                    }
                ],
                "_vision_blocks": [{"bbox": [2, 3, min(width, 32), min(height, 17)]}],
            }

    balloon = Balloon(BBox(20, 130, 90, 175), confidence=0.50)
    balloon.metadata = {
        "bbox_page": [12, 30, 82, 75],
        "page_id": "page_002",
        "region_id": "region_not_an_owner",
    }
    band = Band(
        y_top=100,
        y_bottom=200,
        balloons=[balloon],
        strip_slice=np.full((100, 140, 3), 255, dtype=np.uint8),
        original_slice=np.full((100, 140, 3), 255, dtype=np.uint8),
        tile_id="tile_page_002_a",
    )
    page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=2)
    strong_evidence = {
        "has_inner_dark_text": True,
        "significant_component_count": 3,
        "significant_area": 360,
        "inner_light_component_count": 0,
        "inner_light_area": 0,
        "bright_pixel_ratio": 0.5,
        "dark_pixel_ratio": 0.05,
    }

    with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=strong_evidence):
        recovered = _recover_empty_ocr_with_candidate_crops(
            band,
            runtime=CropRuntime(),
            page_dict=page_dict,
            band_id=page_dict["_band_id"],
        ).to_page_dict()

    mapped_bbox = recovered["texts"][0]["bbox"]
    observation = recovered["owner_observations"][0]
    assert observation["bbox_page"] == [
        mapped_bbox[0] - 8,
        mapped_bbox[1],
        mapped_bbox[2] - 8,
        mapped_bbox[3],
    ]
    assert observation["provider"] == "adaptive_crop"
    assert observation["component_ids"] == []


def test_direct_candidate_crop_retains_every_provider_variant_before_best_selection():
    calls = 0

    def recognize(_image, **_kwargs):
        nonlocal calls
        calls += 1
        return [
            {
                "text": f"VARIANT {calls}",
                "bbox_pts": [[2, 3], [22, 3], [22, 13], [2, 13]],
                "confidence": 0.50 + calls * 0.01,
            }
        ]

    with (
        patch("ocr_legacy.recognizer_paddle.is_paddle_available", return_value=True),
        patch("ocr_legacy.recognizer_paddle.run_paddle_primary_recognition", side_effect=recognize),
    ):
        result = _run_direct_paddle_candidate_crop_reocr(
            np.full((30, 50, 3), 255, dtype=np.uint8),
            idioma_origem="en",
        )

    records_by_provider = result["_owner_observation_records_by_provider"]
    assert calls == 6
    assert list(records_by_provider) == [
        "candidate_crop_direct_paddle_native",
        "candidate_crop_direct_paddle_native_x2",
        "candidate_crop_direct_paddle",
        "candidate_crop_direct_paddle_x2",
        "candidate_crop_direct_paddle_native_inverted",
        "candidate_crop_direct_paddle_inverted",
    ]
    assert [records[0]["text"] for records in records_by_provider.values()] == [
        "VARIANT 1",
        "VARIANT 2",
        "VARIANT 3",
        "VARIANT 4",
        "VARIANT 5",
        "VARIANT 6",
    ]


def test_candidate_crop_merge_preserves_observation_when_legacy_overlap_rejects_text():
    base_page = {
        "texts": [{"text": "BASE BODY", "bbox": [10, 10, 80, 36]}],
        "_vision_blocks": [{"bbox": [10, 10, 80, 36]}],
        "owner_observations": [
            {
                "observation_id": "observation_base",
                "page_id": "page_001",
                "component_ids": [],
                "text": "BASE BODY",
                "confidence": 0.9,
                "provider": "full_page",
                "bbox_page": [10, 110, 80, 136],
                "polygons_page": [],
                "tile_provenance": ["band_a"],
                "coverage_score": None,
                "language_score": None,
                "rejection_reason": None,
                "legacy_selected": True,
            }
        ],
    }
    recovered_page = {
        "texts": [{"text": "PARTIAL", "bbox": [12, 11, 76, 34]}],
        "_vision_blocks": [{"bbox": [12, 11, 76, 34]}],
        "owner_observations": [
            {
                "observation_id": "observation_candidate",
                "page_id": "page_001",
                "component_ids": [],
                "text": "PARTIAL",
                "confidence": 0.82,
                "provider": "candidate_crop",
                "bbox_page": [12, 111, 76, 134],
                "polygons_page": [],
                "tile_provenance": ["band_a:candidate_0"],
                "coverage_score": None,
                "language_score": None,
                "rejection_reason": None,
                "legacy_selected": False,
            }
        ],
    }

    merged = _merge_candidate_crop_recovery_into_ocr_page(base_page, recovered_page)

    assert merged == 0
    assert [row["observation_id"] for row in base_page["owner_observations"]] == [
        "observation_base",
        "observation_candidate",
    ]


def test_empty_primary_ocr_observations_survive_candidate_recovery_replacement():
    failed_observation = {
        "observation_id": "observation_primary_failed",
        "page_id": "page_001",
        "component_ids": [],
        "text": "",
        "confidence": 0.2,
        "provider": "full_page",
        "bbox_page": [10, 20, 70, 40],
        "polygons_page": [],
        "tile_provenance": ["band_a"],
        "coverage_score": None,
        "language_score": None,
        "rejection_reason": "empty_text",
        "legacy_selected": False,
    }
    candidate_observation = {
        **failed_observation,
        "observation_id": "observation_candidate_recovered",
        "text": "RECOVERED BODY",
        "provider": "candidate_crop",
        "rejection_reason": None,
    }

    class EmptyRuntime:
        def run_ocr_stage(self, _image_rgb, _page_dict):
            return {
                "texts": [],
                "_vision_blocks": [],
                "owner_observations": [failed_observation],
            }

    band = Band(
        y_top=0,
        y_bottom=80,
        balloons=[Balloon(BBox(10, 12, 70, 40), confidence=0.9)],
        strip_slice=np.full((80, 100, 3), 255, dtype=np.uint8),
        original_slice=np.full((80, 100, 3), 255, dtype=np.uint8),
    )
    recovered_page = {
        "texts": [{"id": "ocr_001", "text": "RECOVERED BODY", "bbox": [10, 12, 70, 40]}],
        "_vision_blocks": [{"bbox": [10, 12, 70, 40], "confidence": 0.9}],
        "owner_observations": [candidate_observation],
    }

    with patch(
        "strip.process_bands._recover_empty_ocr_with_candidate_crops",
        return_value=BandStageOutput("ocr_candidate_recovery", recovered_page),
    ):
        process_band(
            band,
            runtime=EmptyRuntime(),
            translator=FakeTranslator(),
            inpainter=FakeInpainter(),
            typesetter=FakeTypesetter(),
            page_idx=0,
            source_page_number=1,
        )

    assert [
        row["observation_id"] for row in band.ocr_result["owner_observations"]
    ] == [
        "observation_primary_failed",
        "observation_candidate_recovered",
    ]


def test_negative_candidate_rejection_is_retained_in_page_space():
    page = {
        "texts": [],
        "_vision_blocks": [],
        "_owner_tile_projection": {
            "page_id": "page_002",
            "tile_id": "band_negative",
            "offset_xy": [-5, 80],
        },
    }
    negative_evidence = {
        "texts": [{"text": "HIDDEN BODY", "bbox": [10, 20, 30, 40], "confidence": 0.10}],
        "blocks": [{"bbox": [10, 20, 30, 40], "confidence": 0.10}],
    }

    promoted = fuse_negative_dark_bubble_candidates(
        page,
        negative_evidence,
        np.full((80, 80, 3), 20, dtype=np.uint8),
    )

    assert promoted == 0
    assert len(page["owner_observations"]) == 1
    observation = page["owner_observations"][0]
    assert observation["provider"] == "negative_detect_ocr"
    assert observation["bbox_page"] == [5, 100, 25, 120]
    assert observation["rejection_reason"] == "legacy_negative_low_confidence"
    assert observation["component_ids"] == []


def test_write_inpaint_blocks_debug_includes_trace_ids(tmp_path):
    from strip.run import _write_inpaint_blocks_debug

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        page = OutputPage(
            y_top=0,
            y_bottom=80,
            image=np.full((80, 120, 3), 255, dtype=np.uint8),
            ocr_result={
                "texts": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "page_id": "page_001",
                        "band_id": "page_001_band_000",
                        "trace_id": "ocr_001@page_001_band_000",
                    },
                    {
                        "id": "ocr_002",
                        "text_id": "ocr_002",
                        "page_id": "page_001",
                        "band_id": "page_001_band_000",
                        "trace_id": "ocr_002@page_001_band_000",
                    },
                ],
                "_vision_blocks": [
                    {
                        "bbox": [10, 12, 50, 40],
                        "text_id": "ocr_001",
                        "page_id": "page_001",
                        "band_id": "page_001_band_000",
                        "trace_id": "ocr_001@page_001_band_000",
                    }
                ],
            },
            inpaint_blocks=[{"bbox": [10, 12, 50, 40], "confidence": 0.87}],
        )

        _write_inpaint_blocks_debug([page])
    finally:
        bind_recorder(None)

    rows = [
        json.loads(line)
        for line in (tmp_path / "debug" / "e2e" / "08_inpaint" / "inpaint_blocks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    payload = rows[0]
    assert payload["page_id"] == "page_001"
    assert payload["band_id"] == "page_001_band_000"
    assert payload["text_id"] == "ocr_001"
    assert payload["trace_id"] == "ocr_001@page_001_band_000"
    assert payload["trace_ids"] == ["ocr_001@page_001_band_000"]
    assert payload["trace_ids_in_band"] == [
        "ocr_001@page_001_band_000",
        "ocr_002@page_001_band_000",
    ]


def test_write_inpaint_blocks_debug_falls_back_to_text_layer_overlap(tmp_path):
    from strip.run import _write_inpaint_blocks_debug

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        page = OutputPage(
            y_top=0,
            y_bottom=120,
            image=np.full((120, 160, 3), 255, dtype=np.uint8),
            ocr_result={
                "_vision_blocks": [
                    {
                        "bbox": [0, 0, 20, 20],
                    }
                ],
            },
            text_layers=[
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "page_id": "page_002",
                    "band_id": "page_002_band_019",
                    "trace_id": "ocr_001@page_002_band_019",
                    "source_bbox": [30, 40, 120, 90],
                    "text_pixel_bbox": [35, 45, 115, 85],
                }
            ],
            inpaint_blocks=[{"bbox": [30, 40, 120, 90], "confidence": 0.91}],
        )

        _write_inpaint_blocks_debug([page])
    finally:
        bind_recorder(None)

    rows = [
        json.loads(line)
        for line in (tmp_path / "debug" / "e2e" / "08_inpaint" / "inpaint_blocks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    payload = rows[0]
    assert payload["page_id"] == "page_002"
    assert payload["band_id"] == "page_002_band_019"
    assert payload["text_id"] == "ocr_001"
    assert payload["trace_id"] == "ocr_001@page_002_band_019"
    assert payload["trace_ids"] == ["ocr_001@page_002_band_019"]
    assert payload["trace_ids_in_band"] == ["ocr_001@page_002_band_019"]


def test_inpaint_block_enrichment_copies_trace_identity_from_text_layer():
    from strip.run import _enrich_inpaint_block_from_text_layers

    enriched = _enrich_inpaint_block_from_text_layers(
        {"bbox": [30, 40, 120, 90], "confidence": 0.91},
        [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "page_id": "page_002",
                "band_id": "page_002_band_019",
                "trace_id": "ocr_001@page_002_band_019",
                "source_bbox": [30, 40, 120, 90],
                "text_pixel_bbox": [35, 45, 115, 85],
                "balloon_bbox": [24, 34, 126, 96],
            }
        ],
    )

    assert enriched["page_id"] == "page_002"
    assert enriched["band_id"] == "page_002_band_019"
    assert enriched["text_id"] == "ocr_001"
    assert enriched["trace_id"] == "ocr_001@page_002_band_019"


def test_build_page_result_preserves_raw_confidence_and_text_id_for_debug():
    block = SimpleNamespace(xyxy=(10, 12, 50, 40), mask=None, confidence=0.86)

    page = build_page_result(
        image_path="band_001",
        image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
        blocks=[block],
        texts=["HELLO"],
    )

    assert page["texts"][0]["text_id"] == "ocr_001"
    assert page["texts"][0]["confidence_raw"] == 0.86
    assert page["_vision_blocks"][0]["text_id"] == "ocr_001"
    assert page["_vision_blocks"][0]["confidence_raw"] == 0.86
