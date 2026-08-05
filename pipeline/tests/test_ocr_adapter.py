"""Append-only OCR observation adapter contracts."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.ocr_adapter import (  # noqa: E402
    TileProjection,
    attach_observation_manifest,
    collect_page_observations,
    record_to_observation,
    ocr_record_to_observation,
)
from ownership.hash_contract import sha256_text  # noqa: E402
from ownership.ocr_contract import OCRObservationRecord  # noqa: E402
from ownership.model import OWNER_GRAPH_SCHEMA_VERSION, OwnerGraph  # noqa: E402


def _projection() -> TileProjection:
    return TileProjection(
        page_id="page_003",
        tile_id="band_014",
        offset_xy=(17, 240),
    )


def test_atomic_ocr_record_identity_is_preserved_without_regeneration() -> None:
    payload = "CURRENT ENGLISH BODY"
    record = OCRObservationRecord(
        observation_id="atomic-observation-1",
        attempt_id="atomic-attempt-1",
        run_id="run-atomic",
        origin_execution_id="execution-atomic",
        page_id="page_003",
        page_source_sha256="a" * 64,
        root_input_pixel_sha256="b" * 64,
        input_pixel_sha256="c" * 64,
        invocation_id="invocation-atomic",
        provider_family="paddleocr",
        variant_id="inverted",
        payload_sha256=sha256_text(payload),
        text=payload,
        confidence=0.96,
        bbox_page=(10, 20, 180, 70),
        polygon_page=((10, 20), (180, 20), (180, 70), (10, 70)),
        source="paddle_full_page",
    )

    observation = ocr_record_to_observation(
        record,
        TileProjection(page_id="page_003", tile_id="full-page", coordinate_space="page"),
    )

    assert observation.observation_id == record.observation_id
    assert observation.run_id == record.run_id
    assert observation.origin_execution_id == record.origin_execution_id
    assert observation.invocation_id == record.invocation_id
    assert observation.attempt_id == record.attempt_id
    assert observation.provider_family == record.provider_family
    assert observation.provider_variant == record.variant_id
    assert observation.page_source_sha256 == record.page_source_sha256
    assert observation.root_input_pixel_sha256 == record.root_input_pixel_sha256
    assert observation.input_pixel_sha256 == record.input_pixel_sha256
    assert observation.payload_sha256 == record.payload_sha256


def test_full_page_crop_negative_rotated_and_recovery_become_observations() -> None:
    records_by_provider = {
        "full_page": [{"text": "FULL", "bbox": [10, 20, 80, 44]}],
        "candidate_crop": [{"text": "CROP", "bbox": [20, 50, 90, 76]}],
        "negative": [{"text": "NEGATIVE", "bbox": [30, 80, 120, 108]}],
        "rotated": [{"text": "ROTATED", "bbox": [40, 110, 140, 138]}],
        "recovery": [{"text": "RECOVERY", "bbox": [50, 140, 160, 170]}],
    }

    observations = collect_page_observations(records_by_provider, _projection())

    assert [item.provider for item in observations] == list(records_by_provider)
    assert [item.text for item in observations] == [
        "FULL",
        "CROP",
        "NEGATIVE",
        "ROTATED",
        "RECOVERY",
    ]
    assert len({item.observation_id for item in observations}) == 5


def test_adapter_preserves_rejected_candidates_with_reason() -> None:
    observation = record_to_observation(
        {
            "text": "LOW CONFIDENCE SOURCE",
            "bbox": [4, 6, 20, 18],
            "confidence": 0.21,
            "rejection_reason": "legacy_low_confidence_drop",
            "provider": "adaptive_crop",
        },
        _projection(),
    )

    assert observation.text == "LOW CONFIDENCE SOURCE"
    assert observation.confidence == 0.21
    assert observation.rejection_reason is None
    assert observation.legacy_rejection_reason == "legacy_low_confidence_drop"
    assert observation.provider == "adaptive_crop"


def test_partial_and_full_readings_survive_until_owner_reconcile() -> None:
    observations = collect_page_observations(
        {
            "crop": [
                {
                    "text": "PARTIAL",
                    "bbox": [20, 20, 100, 48],
                    "legacy_selected": False,
                }
            ],
            "full_page": [
                {
                    "text": "PARTIAL AND COMPLETE",
                    "bbox": [20, 20, 220, 48],
                    "legacy_selected": True,
                }
            ],
        },
        _projection(),
    )

    assert len(observations) == 2
    assert {item.text for item in observations} == {
        "PARTIAL",
        "PARTIAL AND COMPLETE",
    }
    assert {item.legacy_selected for item in observations} == {False, True}
    assert all(item.rejection_reason is None for item in observations)


def test_ocr_adapter_maps_every_polygon_to_page_space() -> None:
    observation = record_to_observation(
        {
            "text": "TWO LINES",
            "bbox": [3, 5, 43, 35],
            "line_polygons": [
                [[3, 5], [23, 5], [23, 15], [3, 15]],
                [[4, 20], [43, 20], [43, 35], [4, 35]],
            ],
            "provider": "paddle_full_page",
        },
        _projection(),
    )

    assert observation.bbox_page == (20, 245, 60, 275)
    assert observation.polygons_page == (
        ((20, 245), (40, 245), (40, 255), (20, 255)),
        ((21, 260), (60, 260), (60, 275), (21, 275)),
    )


def test_ocr_adapter_preserves_layout_container_in_page_space() -> None:
    observation = record_to_observation(
        {
            "text": "LONG BODY",
            "bbox": [20, 20, 100, 48],
            "bubble_inner_bbox": [5, 8, 180, 90],
            "provider": "paddle_full_page",
        },
        _projection(),
    )

    assert observation.layout_bbox_page == (22, 248, 197, 330)


def test_ocr_adapter_prefers_visual_container_over_tight_text_safe_box() -> None:
    observation = record_to_observation(
        {
            "text": "CARD BODY",
            "bbox": [40, 30, 120, 58],
            "safe_text_box": [36, 26, 124, 62],
            "bubble_inner_bbox": [34, 24, 126, 64],
            "balloon_bbox": [10, 8, 180, 100],
            "card_panel_bbox": [5, 4, 190, 108],
            "provider": "visual_card_full_page",
        },
        _projection(),
    )

    assert observation.layout_bbox_page == (22, 244, 207, 348)


def test_ocr_adapter_does_not_assign_semantic_owner_from_band() -> None:
    observation = record_to_observation(
        {
            "text": "TITLE",
            "bbox": [10, 12, 80, 34],
            "provider": "crop",
            "owner_id": "band_decided_owner",
            "semantic_owner": "title",
            "type": "title",
        },
        _projection(),
    )

    assert observation.component_ids == ()
    assert not hasattr(observation, "owner_id")
    assert not hasattr(observation, "semantic_owner")

    legacy_page = {"texts": [{"text": "TITLE", "bbox": [10, 12, 80, 34]}]}
    attached = attach_observation_manifest(legacy_page, [observation])
    assert attached is not legacy_page
    assert attached["texts"] == legacy_page["texts"]
    assert attached["owner_observations"][0]["observation_id"] == observation.observation_id


def test_provider_variant_and_attempt_are_part_of_observation_identity() -> None:
    record = {
        "text": "SAME GEOMETRY",
        "bbox": [10, 20, 80, 44],
        "provider": "paddleocr",
    }

    full_page = record_to_observation(
        record,
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(17, 240),
            provider_variant="full_page",
            attempt_id="attempt_001",
        ),
    )
    retry = record_to_observation(
        record,
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(17, 240),
            provider_variant="full_page",
            attempt_id="attempt_002",
        ),
    )
    crop = record_to_observation(
        record,
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(17, 240),
            provider_variant="candidate_crop",
            attempt_id="attempt_001",
        ),
    )

    assert len({full_page.observation_id, retry.observation_id, crop.observation_id}) == 3
    assert full_page.provider_variant == "full_page"
    assert retry.attempt_id == "attempt_002"


def test_collection_provider_keys_distinguish_paths_for_one_backend() -> None:
    shared = {
        "text": "SAME GEOMETRY",
        "bbox": [10, 20, 80, 44],
        "provider": "paddleocr",
    }

    observations = collect_page_observations(
        {
            "full_page": [shared],
            "candidate_crop": [shared],
        },
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(17, 240),
            attempt_id="attempt_001",
        ),
    )

    assert [item.provider for item in observations] == ["paddleocr", "paddleocr"]
    assert [item.provider_variant for item in observations] == [
        "full_page",
        "candidate_crop",
    ]
    assert observations[0].observation_id != observations[1].observation_id


def test_sink_variant_field_populates_provider_variant() -> None:
    observations = collect_page_observations(
        {
            "paddleocr": [
                {
                    "text": "SINK RECORD",
                    "bbox": [10, 20, 80, 44],
                    "provider": "paddleocr",
                    "variant": "full_page_raw",
                },
                {
                    "text": "EXPLICIT RECORD",
                    "bbox": [10, 50, 80, 74],
                    "provider": "paddleocr",
                    "provider_variant": "explicit_variant",
                    "variant": "ignored_sink_variant",
                },
            ]
        },
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            provider_variant="projection_default",
        ),
    )

    assert [item.provider_variant for item in observations] == [
        "full_page_raw",
        "explicit_variant",
    ]


def test_page_space_projection_is_explicit_and_does_not_apply_tile_offset() -> None:
    observation = record_to_observation(
        {
            "text": "PAGE RECORD",
            "bbox": [20, 245, 60, 275],
            "line_polygons": [
                [[20, 245], [60, 245], [60, 275], [20, 275]],
            ],
            "provider": "paddleocr",
        },
        TileProjection(
            page_id="page_003",
            tile_id="page_003_full",
            offset_xy=(17, 240),
            coordinate_space="page",
            projection_id="projection_page_003_full",
            page_size=(300, 500),
        ),
    )

    assert observation.bbox_page == (20, 245, 60, 275)
    assert observation.polygons_page == (
        ((20, 245), (60, 245), (60, 275), (20, 275)),
    )
    assert observation.projection_ids == ("projection_page_003_full",)


def test_adapter_preserves_raw_text_geometry_confidence_and_rotation() -> None:
    observation = record_to_observation(
        {
            "text": "NORMALIZED TEXT",
            "raw_ocr": "  Raw OCR text  ",
            "bbox": [3, 5, 43, 35],
            "source_bbox": [4, 6, 42, 34],
            "text_pixel_bbox": [5, 7, 41, 33],
            "line_texts": ["Raw OCR", "text"],
            "confidence_raw": 0.87,
            "rotation_deg": -12.5,
            "rotation_source": "line_polygons",
            "provider": "paddleocr",
            "provider_record_id": "line_0042",
        },
        _projection(),
    )

    assert observation.text == "NORMALIZED TEXT"
    assert observation.raw_text == "  Raw OCR text  "
    assert observation.confidence == 0.87
    assert observation.source_bbox_page == (21, 246, 59, 274)
    assert observation.text_pixel_bbox_page == (22, 247, 58, 273)
    assert observation.line_texts == ("Raw OCR", "text")
    assert observation.rotation_deg == -12.5
    assert observation.rotation_source == "line_polygons"
    assert observation.provider_record_id == "line_0042"


def test_missing_geometry_is_retained_as_rejected_observation() -> None:
    observation = record_to_observation(
        {
            "text": "PROVIDER RETURNED TEXT WITHOUT GEOMETRY",
            "provider": "paddleocr",
        },
        _projection(),
    )

    assert observation.text == "PROVIDER RETURNED TEXT WITHOUT GEOMETRY"
    assert observation.bbox_page == (17, 240, 17, 240)
    assert observation.rejection_reason == "missing_bbox"


def test_malformed_geometry_does_not_abort_observation_capture() -> None:
    observation = record_to_observation(
        {
            "text": "MALFORMED GEOMETRY",
            "bbox": ["not-a-number", 2, 10, 12],
            "line_polygons": [[["bad", 2], [10, 2], [10, 12], [0, 12]]],
            "provider": "paddleocr",
        },
        TileProjection(
            page_id="page_003",
            tile_id="page_003_full",
            coordinate_space="page",
        ),
    )

    assert observation.bbox_page == (0, 0, 0, 0)
    assert observation.polygons_page == ()
    assert observation.rejection_reason == "invalid_bbox"


def test_reversed_bbox_is_canonicalized_but_marked_rejected() -> None:
    observation = record_to_observation(
        {
            "text": "REVERSED",
            "bbox": [43, 35, 3, 5],
            "provider": "paddleocr",
        },
        _projection(),
    )

    assert observation.bbox_page == (20, 245, 60, 275)
    assert observation.rejection_reason == "invalid_bbox_order"


def test_tile_extent_validation_keeps_outside_candidate_with_reason() -> None:
    observation = record_to_observation(
        {
            "text": "OUTSIDE TILE",
            "bbox": [60, 0, 70, 10],
            "provider": "paddleocr",
        },
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(10, 20),
            tile_size=(50, 50),
            page_size=(100, 100),
        ),
    )

    assert observation.bbox_page == (70, 20, 80, 30)
    assert observation.rejection_reason == "bbox_outside_tile"


def test_manifest_attachment_is_idempotent_and_unions_projection_provenance() -> None:
    first = record_to_observation(
        {
            "text": "SHARED EVIDENCE",
            "bbox": [10, 20, 80, 44],
            "provider": "paddleocr",
        },
        TileProjection(
            page_id="page_003",
            tile_id="tile_a",
            projection_id="projection_a",
            provider_variant="full_page",
            attempt_id="attempt_001",
        ),
    )
    second = record_to_observation(
        {
            "text": "SHARED EVIDENCE",
            "bbox": [0, 20, 70, 44],
            "provider": "paddleocr",
        },
        TileProjection(
            page_id="page_003",
            tile_id="tile_b",
            offset_xy=(10, 0),
            projection_id="projection_b",
            provider_variant="full_page",
            attempt_id="attempt_001",
        ),
    )
    assert first.observation_id == second.observation_id

    page = {"texts": [{"text": "SHARED EVIDENCE"}], "owner_observations": []}
    attached_once = attach_observation_manifest(page, [first])
    attached_twice = attach_observation_manifest(attached_once, [first, second])
    attached_again = attach_observation_manifest(attached_twice, [first, second])

    assert page["owner_observations"] == []
    assert len(attached_twice["owner_observations"]) == 1
    manifest = attached_twice["owner_observations"][0]
    assert manifest["tile_provenance"] == ["tile_a", "tile_b"]
    assert manifest["projection_ids"] == ["projection_a", "projection_b"]
    assert attached_again == attached_twice


def test_manifest_rejects_same_id_with_conflicting_identity_fields() -> None:
    observation = record_to_observation(
        {
            "text": "SOURCE",
            "bbox": [10, 20, 80, 44],
            "provider": "paddleocr",
        },
        _projection(),
    )
    conflicting = {
        **attach_observation_manifest({}, [observation])["owner_observations"][0],
        "bbox_page": [999, 999, 1000, 1000],
    }

    with pytest.raises(ValueError, match="observation_id collision"):
        attach_observation_manifest(
            {"owner_observations": [conflicting]},
            [observation],
        )


def test_enriched_observation_round_trips_through_owner_graph_serialization() -> None:
    observation = record_to_observation(
        {
            "text": "NORMALIZED",
            "raw_ocr": "Raw",
            "bbox": [3, 5, 43, 35],
            "source_bbox": [4, 6, 42, 34],
            "text_pixel_bbox": [5, 7, 41, 33],
            "line_polygons": [[[3, 5], [43, 5], [43, 35], [3, 35]]],
            "line_texts": ["Raw"],
            "confidence": 0.91,
            "rotation_deg": 90,
            "rotation_source": "rotated_page_ocr",
            "provider": "paddleocr",
            "provider_record_id": "record_0042",
        },
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(17, 240),
            projection_id="projection_014",
            provider_variant="rotated",
            attempt_id="attempt_002",
        ),
    )
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_003",
        run_id="run-ocr-adapter",
        origin_execution_id="execution-ocr-adapter",
        page_source_sha256="a" * 64,
        components=[],
        observations=[observation],
        owners=[],
        projections=[],
    )

    serialized = graph.to_dict()
    payload = serialized["observations"][0]

    assert payload["provider_variant"] == "rotated"
    assert payload["attempt_id"] == "attempt_002"
    assert payload["projection_ids"] == ["projection_014"]
    assert payload["raw_text"] == "Raw"
    assert payload["source_bbox_page"] == [21, 246, 59, 274]
    assert payload["text_pixel_bbox_page"] == [22, 247, 58, 273]
    assert payload["rotation_deg"] == 90.0
    assert OwnerGraph.from_dict(serialized).to_dict() == serialized


def test_legacy_observation_payload_uses_text_as_raw_text_fallback() -> None:
    legacy = {
        "schema_version": 1,
        "page_id": "page_003",
        "components": [],
        "observations": [
            {
                "observation_id": "legacy_observation",
                "page_id": "page_003",
                "component_ids": [],
                "text": "LEGACY SOURCE",
                "confidence": 0.8,
                "provider": "legacy_ocr",
                "bbox_page": [10, 20, 80, 44],
                "tile_provenance": ["legacy_tile"],
            }
        ],
        "owners": [],
        "projections": [],
        "component_dispositions": [],
        "violations": [],
    }

    graph = OwnerGraph.from_dict(legacy)

    assert graph.observations[0].raw_text == "LEGACY SOURCE"
    assert graph.to_dict()["observations"][0]["raw_text"] == "LEGACY SOURCE"


def test_malformed_numeric_metadata_does_not_abort_capture() -> None:
    observation = record_to_observation(
        {
            "text": "VALID TEXT",
            "bbox": [10, 20, 80, 44],
            "confidence": "not-a-score",
            "coverage_score": "bad-coverage",
            "language_score": "bad-language",
            "rotation_deg": "bad-rotation",
            "provider": "paddleocr",
        },
        _projection(),
    )

    assert observation.confidence == 0.0
    assert observation.coverage_score is None
    assert observation.language_score is None
    assert observation.rotation_deg is None
    assert observation.rejection_reason == "invalid_confidence"


def test_blank_provider_reading_is_retained_with_empty_text_reason() -> None:
    observation = record_to_observation(
        {
            "text": "",
            "bbox": [10, 20, 80, 44],
            "confidence": 0.0,
            "provider": "paddleocr",
        },
        _projection(),
    )

    assert observation.text == ""
    assert observation.raw_text == ""
    assert observation.rejection_reason == "empty_text"


def test_projection_rejects_ambiguous_coordinate_space() -> None:
    with pytest.raises(ValueError, match="coordinate_space"):
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            coordinate_space="strip",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_size", (0, 500)),
        ("tile_size", (300, -1)),
    ],
)
def test_projection_rejects_non_positive_extents(field: str, value: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=field):
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            **{field: value},
        )


def test_projection_rejects_invalid_offset_shape() -> None:
    with pytest.raises(ValueError, match="offset_xy"):
        TileProjection(
            page_id="page_003",
            tile_id="band_014",
            offset_xy=(10,),  # type: ignore[arg-type]
        )


def test_malformed_polygons_are_retained_as_rejected_geometry() -> None:
    observation = record_to_observation(
        {
            "text": "VALID TEXT",
            "bbox": [10, 20, 80, 44],
            "line_polygons": [[["bad", 20], [80, 20], [80, 44], [10, 44]]],
            "provider": "paddleocr",
        },
        _projection(),
    )

    assert observation.bbox_page == (27, 260, 97, 284)
    assert observation.polygons_page == ()
    assert observation.rejection_reason == "invalid_polygons"


def test_malformed_secondary_bbox_is_not_silently_accepted() -> None:
    observation = record_to_observation(
        {
            "text": "VALID TEXT",
            "bbox": [10, 20, 80, 44],
            "source_bbox": ["bad", 20, 80, 44],
            "provider": "paddleocr",
        },
        _projection(),
    )

    assert observation.bbox_page == (27, 260, 97, 284)
    assert observation.source_bbox_page == (17, 240, 17, 240)
    assert observation.rejection_reason == "invalid_source_bbox"


def test_manifest_rejects_same_id_when_text_is_later_normalized() -> None:
    raw = record_to_observation(
        {
            "text": "raw  spacing",
            "bbox": [10, 20, 80, 44],
            "provider": "paddleocr",
        },
        _projection(),
    )
    legacy_selected = replace(
        raw,
        text="RAW SPACING",
        legacy_selected=True,
    )

    with pytest.raises(ValueError, match="observation_id collision.*text"):
        attach_observation_manifest(
            attach_observation_manifest({}, [raw]),
            [legacy_selected],
        )
