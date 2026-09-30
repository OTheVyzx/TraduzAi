from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _write_json(path: Path, value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _handoff(tmp_path: Path) -> Path:
    observations = [
        {"observation_id": "obs-a", "text": "HELLO", "bbox_page": [1, 1, 20, 10]},
        {"observation_id": "obs-b", "text": "WORLD", "bbox_page": [21, 1, 40, 10]},
        {"observation_id": "obs-sfx", "text": "BAM", "bbox_page": [1, 20, 20, 30]},
    ]
    structure = {
        "schema": "traduzai.vision.structural-analysis.v1",
        "logical_units": [
            {
                "logical_unit_id": "unit-dialogue",
                "classification": "connected_balloon",
                "observation_ids": ["obs-a", "obs-b"],
                "text": "HELLO WORLD",
                "uncertainty_reasons": [],
            },
            {
                "logical_unit_id": "unit-sfx",
                "classification": "unknown_text",
                "observation_ids": ["obs-sfx"],
                "text": "BAM",
                "uncertainty_reasons": ["container_not_observed"],
            },
        ],
        "physical_subblocks": [
            {
                "physical_subblock_id": "block-a",
                "logical_unit_id": "unit-dialogue",
                "observation_ids": ["obs-a"],
                "order": 0,
            },
            {
                "physical_subblock_id": "block-b",
                "logical_unit_id": "unit-dialogue",
                "observation_ids": ["obs-b"],
                "order": 1,
            },
        ],
        "relations": [],
        "reading_order": ["unit-dialogue", "unit-sfx"],
    }
    containers = {"schema": "traduzai.vision.white-containers.v1", "containers": []}
    observation_sha = _write_json(tmp_path / "ocr-observations.json", observations)
    structure_sha = _write_json(tmp_path / "structure.json", structure)
    container_sha = _write_json(tmp_path / "containers.json", containers)
    handoff = {
        "schema": "traduzai.vision.structural-handoff.v1",
        "status": "complete",
        "case": "transfer-test",
        "proof_scope": "vision_transfer_only",
        "vision_commit": "a" * 40,
        "source_member": "page.webp",
        "source_sha256": "b" * 64,
        "provider_evidence": {
            "artifact_ref": "ocr-observations.json",
            "sha256": observation_sha,
            "provider_called": True,
            "cache_hit": False,
        },
        "containers": {"artifact_ref": "containers.json", "sha256": container_sha},
        "structure": {"artifact_ref": "structure.json", "sha256": structure_sha},
    }
    _write_json(tmp_path / "handoff.json", handoff)
    return tmp_path / "handoff.json"


def test_structural_handoff_consumer_translates_logical_unit_once_and_preserves_unknown(tmp_path):
    from integration_v1.structural_consumer import consume_structural_handoff

    calls: list[dict[str, object]] = []

    def translate(**kwargs):
        calls.append(kwargs)
        return {
            "target": "OLÁ MUNDO",
            "provenance": "fresh_complete_unit_translation",
            "backend": "test-physical",
            "variant": "cold_complete_unit",
            "provider_model": "test",
            "provider_called": True,
            "cache_hit": False,
            "provider_metadata_sha256": "c" * 64,
        }

    result = consume_structural_handoff(
        _handoff(tmp_path), tmp_path / "out", translator=translate,
    )

    assert [call["source"] for call in calls] == ["HELLO WORLD"]
    translated, preserved = result["units"]
    assert translated["destination_ids"] == ["block-a", "block-b"]
    assert translated["status"] == "translation_ready"
    assert preserved["status"] == "review_required"
    assert preserved["source_preserved"] is True
    assert result["stages"]["restore"]["status"] == "review_required"
    assert result["answer_by_id"] is False


def test_structural_handoff_consumer_rejects_tampered_artifact(tmp_path):
    from integration_v1.structural_consumer import consume_structural_handoff

    handoff = _handoff(tmp_path)
    (tmp_path / "structure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        consume_structural_handoff(handoff, tmp_path / "out", translator=lambda **_: {})
