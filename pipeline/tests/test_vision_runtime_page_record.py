from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_chapter_index_binds_each_owner_page_to_its_source(tmp_path: Path, monkeypatch) -> None:
    from vision_runtime import page_record

    calls = []

    def write_record(work_dir, root, page, config, *, source_sha256, project_id):
        calls.append((root, source_sha256, page.owner_page_result.page_id))
        record_path = root / "analysis_record.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("{}", encoding="utf-8")
        return {
            "path": str(record_path), "sha256": _sha(b"{}"),
            "analysis_record_sha256": _sha(page.owner_page_result.page_id.encode()),
            "coverage_ledger_sha256": _sha(b"ledger"),
            "page_id": page.owner_page_result.page_id,
        }

    monkeypatch.setattr(page_record, "write_page_analysis_record", write_record)
    pages = [
        SimpleNamespace(owner_page_result=SimpleNamespace(
            page_id=f"page_{number:03}",
            request=SimpleNamespace(original_page=SimpleNamespace(source_file_sha256=_sha(f"source-{number}".encode()))),
        ))
        for number in (1, 2)
    ]
    receipt = page_record.write_chapter_analysis_records(
        tmp_path, pages, {}, source_manifest_sha256=_sha(b"manifest")
    )
    index = json.loads(Path(receipt["path"]).read_text(encoding="utf-8"))

    assert receipt["page_count"] == 2
    assert [call[1] for call in calls] == [_sha(b"source-1"), _sha(b"source-2")]
    assert [entry["page_id"] for entry in index["pages"]] == ["page_001", "page_002"]
    assert [entry["analysis_record_ref"] for entry in index["pages"]] == [
        "vision/pages/page_001/analysis_record.json",
        "vision/pages/page_002/analysis_record.json",
    ]
    assert index["publishable"] is False


def test_chapter_index_rejects_missing_owner_result(tmp_path: Path) -> None:
    from vision_runtime.page_record import write_chapter_analysis_records

    with pytest.raises(RuntimeError, match="owner page result is missing"):
        write_chapter_analysis_records(
            tmp_path, [SimpleNamespace(owner_page_result=None)], {},
            source_manifest_sha256=_sha(b"manifest"),
        )
    assert not (tmp_path / "vision" / "index.json").exists()


def test_visual_identity_ignores_typography_and_output_but_tracks_ocr() -> None:
    from vision_runtime.cache_key import visual_config_sha256

    base = {
        "idioma_origem": "en", "engine_preset_id": "max",
        "work_dir": "first", "font_path": "first.ttf", "text_position": [1, 2],
    }
    same_vision = {**base, "work_dir": "second", "font_path": "second.ttf", "text_position": [3, 4]}
    different_vision = {**base, "engine_preset_id": "normal"}
    assert visual_config_sha256(base) == visual_config_sha256(same_vision)
    assert visual_config_sha256(base) != visual_config_sha256(different_vision)


def test_visual_identity_tracks_detector_model_content(tmp_path: Path) -> None:
    from vision_runtime.cache_key import visual_config_sha256

    model = tmp_path / "comic-text-detector.pt"
    model.write_bytes(b"first detector weights")
    config = {"idioma_origem": "en", "models_dir": str(tmp_path)}
    initial = visual_config_sha256(config)
    model.write_bytes(b"replaced detector weights")
    assert visual_config_sha256(config) != initial


def test_verified_reader_rejects_old_index_without_stable_source_identity(tmp_path: Path) -> None:
    from vision_runtime.cache_reader import read_verified_chapter_analysis

    root = tmp_path / "vision"
    root.mkdir()
    (root / "index.json").write_text(json.dumps({
        "schema": "traduzai.vision-chapter-analysis-index.v1",
        "pages": [], "source_manifest_sha256": _sha(b"run-specific"),
    }), encoding="utf-8")
    result = read_verified_chapter_analysis(
        tmp_path, source_tree_sha256=_sha(b"stable-source"),
        page_source_sha256s={}, visual_config_sha256=_sha(b"visual-settings"),
    )
    assert result == {"status": "miss", "reason": "source_tree_changed", "pages": {}}


def test_cached_content_lineage_survives_multiple_render_executions() -> None:
    from vision_runtime.cache_reader import cached_content_lineage

    pages = {
        "page_001": {
            "record": {"run_id": "original-run", "execution_id": "second-render"},
            "coverage": SimpleNamespace(run_id="original-run", origin_execution_id="original-analysis"),
        }
    }
    assert cached_content_lineage(pages) == ("original-run", "original-analysis")
    pages["page_001"]["coverage"].run_id = "different-run"
    assert cached_content_lineage(pages) is None


def test_pre_ocr_stage_roundtrip_preserves_mask_and_band_ocr(tmp_path: Path) -> None:
    from strip.types import BBox, Balloon, Band
    from vision_runtime.stage_cache import load_pre_ocr_stage, save_pre_ocr_stage

    balloon = Balloon(BBox(1, 2, 6, 8), 0.9, metadata={"region_id": "r1"}, mask=np.ones((3, 4), dtype=np.uint8))
    band = Band(0, 12, balloons=[balloon], tile_id="tile-1")
    evidence = SimpleNamespace(
        ocr_page={"texts": [{"text": "hello", "bbox": [1, 2, 6, 8]}]},
        perf={"durations_sec": {"ocr": 0.5}}, terminal_reason=None,
    )
    receipt = save_pre_ocr_stage(
        tmp_path, page_id="page_001", source_file_sha256=_sha(b"source"),
        page_pixel_sha256=_sha(b"pixels"), visual_config_sha256=_sha(b"config"),
        balloons=[balloon], bands=[band], band_evidence_by_index={0: evidence},
    )
    restored = load_pre_ocr_stage(
        receipt["json"], receipt["npz"], page_id="page_001",
        source_file_sha256=_sha(b"source"), page_pixel_sha256=_sha(b"pixels"),
        visual_config_sha256=_sha(b"config"),
    )
    assert restored["bands"][0]["ocr_page"] == evidence.ocr_page
    assert np.array_equal(restored["balloons"][0].mask, balloon.mask)
    assert restored["bands"][0]["band"].balloons[0].strip_bbox == balloon.strip_bbox
    with pytest.raises(ValueError, match="visual_config_changed"):
        load_pre_ocr_stage(
            receipt["json"], receipt["npz"], page_id="page_001",
            source_file_sha256=_sha(b"source"), page_pixel_sha256=_sha(b"pixels"),
            visual_config_sha256=_sha(b"changed"),
        )
    receipt["npz"].write_bytes(receipt["npz"].read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="arrays_changed"):
        load_pre_ocr_stage(
            receipt["json"], receipt["npz"], page_id="page_001",
            source_file_sha256=_sha(b"source"), page_pixel_sha256=_sha(b"pixels"),
            visual_config_sha256=_sha(b"config"),
        )
