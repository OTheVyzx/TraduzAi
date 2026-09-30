from __future__ import annotations

import json

from integration_v1.invalidation_probe import run_invalidation_probe
from ownership.hash_contract import canonical_json_sha256


def test_typography_reuses_visual_revision_and_provider_change_is_scoped(tmp_path):
    payload = {
        "status": "complete",
        "source_sha256": "a" * 64,
        "authenticated_neighbor_sha256s": ["b" * 64],
        "region": {"bbox": [1, 2, 30, 40], "coordinate_space": "logical_page"},
        "coordinate_space": "logical_page",
        "transform_sha256": "c" * 64,
        "source_language": "en",
        "analysis_config_sha256": "d" * 64,
        "provider_family": "vision-paddleocr",
        "provider_name": "PaddleOCR",
        "provider_model": "v4",
        "provider_version": "2.9.1",
        "capability_version": "vision-runtime-v1",
        "selected_observation_id": "observation-001",
    }
    record = {**payload, "analysis_record_sha256": canonical_json_sha256(payload)}
    path = tmp_path / "analysis-record.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    trace = run_invalidation_probe(path)

    assert trace["typography_only"]["provider_calls_delta"] == {
        "detection": 0,
        "ocr": 0,
    }
    assert trace["typography_only"]["visual_revision_unchanged"] is True
    assert trace["ocr_provider_change"]["provider_calls_delta"] == {
        "detection": 0,
        "ocr": 1,
    }
    assert trace["ocr_provider_change"]["detection_identity_unchanged"] is True
    assert trace["ocr_provider_change"]["ocr_identity_changed"] is True
