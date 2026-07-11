import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import DebugRecorder, bind_recorder
from strip.run import (
    _candidate_matches_band_text_bbox,
    _stitch_output_band_crop,
    _write_lossless_visual_baseline,
    _write_final_band_crop_debug,
    _write_output_pages_after_lossless_debug,
    run_chapter,
)
from strip.process_bands import _band_to_page_dict, process_band
from strip.types import Band, Balloon, BBox, OutputPage, VerticalStrip
from vision_stack.runtime import build_page_result


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
        assert final_crop["crop_bbox_in_translated_page"] == [0, 0, 120, 166]
        assert final_crop["final_crop_path"] == "10_copyback_reassemble/final_bands/page_001_band_000.jpg"
    finally:
        bind_recorder(None)


def test_final_band_debug_writes_lossless_canonical_page_and_band(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        page_image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page_image[20:40, 10:50, :] = 17
        page = OutputPage(y_top=0, y_bottom=80, image=page_image)
        band = Band(
            y_top=0,
            y_bottom=80,
            rendered_slice=page_image.copy(),
            ocr_result={"_band_id": "page_001_band_000"},
        )

        _write_lossless_visual_baseline([page], [band])
        recorder.finalize()

        root = tmp_path / "debug" / "e2e" / "00_run"
        canonical = json.loads((root / "canonical_manifest.json").read_text(encoding="utf-8"))
        assert [entry["key"] for entry in canonical["entries"]] == [
            "final_band:page_001:page_001_band_000",
            "page:page_001:",
        ]
        assert (root / "canonical_pages" / "page_001.png").exists()
        assert (root / "canonical_final_bands" / "page_001_band_000.png").exists()
    finally:
        bind_recorder(None)


def test_lossless_visual_baseline_records_failure_without_interrupting_pipeline(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        page = OutputPage(
            y_top=0,
            y_bottom=20,
            image=np.zeros((20, 12, 3), dtype=np.uint8),
        )
        band = Band(y_top=0, y_bottom=20, ocr_result={"_band_id": "page_001_band_000"})
        monkeypatch.setattr(
            recorder,
            "write_canonical_image",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk unavailable")),
        )

        _write_lossless_visual_baseline([page], [band])

        events = [
            json.loads(line)
            for line in (tmp_path / "debug" / "e2e" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        failure = next(event for event in events if event["action"] == "lossless_visual_baseline_failed")
        assert failure["error_type"] == "RuntimeError"
        assert failure["error"] == "disk unavailable"
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


def test_output_jpeg_write_runs_after_lossless_final_band_debug(monkeypatch, tmp_path):
    from strip import run

    calls: list[str] = []
    monkeypatch.setattr(run, "_write_lossless_visual_baseline", lambda _pages, _bands: calls.append("lossless"))
    monkeypatch.setattr(
        run,
        "_write_output_pages_jpegs",
        lambda _pages, _output_dir: calls.append("jpeg") or 0.25,
    )

    elapsed = _write_output_pages_after_lossless_debug([], [], tmp_path)

    assert elapsed == 0.25
    assert calls == ["lossless", "jpeg"]


def test_lossless_band_crop_stitches_all_cross_page_segments():
    upper = OutputPage(
        y_top=0,
        y_bottom=60,
        image=np.full((60, 8, 3), 11, dtype=np.uint8),
    )
    lower = OutputPage(
        y_top=60,
        y_bottom=120,
        image=np.full((60, 8, 3), 29, dtype=np.uint8),
    )
    band = Band(y_top=40, y_bottom=80, ocr_result={"_band_id": "page_001_band_000"})

    stitched = _stitch_output_band_crop([upper, lower], band)

    assert stitched is not None
    assert stitched.shape == (40, 8, 3)
    assert np.all(stitched[:20] == 11)
    assert np.all(stitched[20:] == 29)


def test_lossless_band_crop_clamps_nominal_band_to_visible_page_union():
    page = OutputPage(
        y_top=0,
        y_bottom=50,
        image=np.full((50, 8, 3), 17, dtype=np.uint8),
    )
    band = Band(y_top=-12, y_bottom=70, ocr_result={"_band_id": "page_001_band_000"})

    stitched = _stitch_output_band_crop([page], band)

    assert stitched is not None
    assert stitched.shape == (50, 8, 3)
    assert np.all(stitched == 17)


def test_lossless_band_crop_rejects_internal_page_gap():
    upper = OutputPage(y_top=0, y_bottom=40, image=np.full((40, 8, 3), 11, dtype=np.uint8))
    lower = OutputPage(y_top=50, y_bottom=100, image=np.full((50, 8, 3), 29, dtype=np.uint8))
    band = Band(y_top=20, y_bottom=80, ocr_result={"_band_id": "page_001_band_000"})

    assert _stitch_output_band_crop([upper, lower], band) is None


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
