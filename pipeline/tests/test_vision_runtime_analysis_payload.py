from __future__ import annotations

import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64


def _identity() -> dict[str, object]:
    return {
        "source_sha256": SHA_A,
        "authenticated_neighbor_sha256s": [],
        "region": {"bbox": [83, 422, 407, 650], "coordinate_space": "logical_page"},
        "coordinate_space": "logical_page",
        "transform_sha256": SHA_B,
        "source_language": "en",
        "analysis_config_sha256": SHA_A,
        "provider_family": "vision-paddleocr",
        "provider_name": "PaddleOCR",
        "provider_model": "en_PP-OCRv3_det+en_PP-OCRv4_rec",
        "provider_version": "2.9.1",
        "capability_version": "vision-runtime-v1",
    }


def _references() -> dict[str, object]:
    ref = {"artifact_ref": "analysis/ocr-observations.json", "sha256": SHA_A, "count": 6}
    return {
        "artifact_refs": [{"kind": "ocr_observations", "sha256": SHA_A}],
        "transform_ref": {"kind": "crop", "sha256": SHA_B, "inverse_sha256": SHA_B},
        "ocr_observations": ref,
        "selected_observation_id": "ocr_selection_001",
        "selection_provenance": {"kind": "variant_consensus_v1", "sha256": SHA_A},
        "logical_units": {"artifact_ref": "analysis/logical-units.json", "sha256": SHA_A, "count": 1},
        "physical_subblocks": {"artifact_ref": "analysis/subblocks.json", "sha256": SHA_A, "count": 6},
        "relations": {"artifact_ref": "analysis/relations.json", "sha256": SHA_A, "count": 0},
        "reading_order": {"artifact_ref": "analysis/reading-order.json", "sha256": SHA_A, "count": 6},
        "container_contour_ref": None,
        "writing_body_ref": None,
        "tail_ref": None,
        "dependency_hashes": {"provider": SHA_A, "analysis_config": SHA_A},
    }


def test_analysis_payload_covers_every_mask_channel_without_zero_mask_aliases() -> None:
    from vision_runtime.analysis_payload import build_analysis_payload

    payload = build_analysis_payload(identity=_identity(), references=_references())

    assert payload["status"] == "complete"
    assert set(payload["mask_channels"]) == {
        "glyph", "outline", "shadow", "glow", "ignore_or_uncertain"
    }
    assert payload["mask_channels"]["glyph"] == {
        "state": "unknown", "reason_code": "provider_not_run"
    }
    assert payload["mask_channels"]["glow"] == {
        "state": "unavailable", "reason_code": "capability_unavailable"
    }


def test_confirmed_empty_requires_hash_bound_negative_evidence() -> None:
    from vision_runtime.analysis_payload import build_analysis_payload

    with pytest.raises(ValueError, match="evidence_sha256"):
        build_analysis_payload(
            identity=_identity(), references=_references(),
            mask_channels={"outline": {"state": "confirmed_empty"}},
        )


def test_confirmed_present_requires_artifact_reference_and_sha256() -> None:
    from vision_runtime.analysis_payload import build_analysis_payload

    payload = build_analysis_payload(
        identity=_identity(), references=_references(),
        mask_channels={
            "glyph": {
                "state": "confirmed_present",
                "artifact_ref": "analysis/masks/glyph.png",
                "sha256": SHA_B,
            }
        },
    )

    assert payload["mask_channels"]["glyph"]["sha256"] == SHA_B
