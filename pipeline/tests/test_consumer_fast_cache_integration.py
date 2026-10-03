from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_visual_cache_is_disabled_by_default(monkeypatch) -> None:
    from consumer_fast.vision_cache import select_cached_pages

    monkeypatch.delenv("TRADUZAI_EXPERIMENTAL_VISION_CACHE", raising=False)
    assert select_cached_pages(None, None, {})["status"] == "disabled"


def test_visual_cache_identity_changes_with_crop_flag_and_mayo_model(tmp_path: Path, monkeypatch) -> None:
    from vision_runtime.cache_key import visual_config_sha256

    config = {"models_dir": str(tmp_path), "engine_preset_id": "max"}
    monkeypatch.delenv("TRADUZAI_OCR_CROP_FIRST", raising=False)
    monkeypatch.delenv("TRADUZAI_EXPERIMENTAL_MAYO", raising=False)
    baseline = visual_config_sha256(config)
    monkeypatch.setenv("TRADUZAI_OCR_CROP_FIRST", "1")
    assert visual_config_sha256(config) != baseline
    model = tmp_path / "mayo"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"model-one")
    (model / "yolov8m-seg-local.yaml").write_bytes(b"model-config")
    monkeypatch.setenv("TRADUZAI_EXPERIMENTAL_MAYO", "1")
    monkeypatch.setenv("TRADUZAI_EXPERIMENTAL_MAYO_MODEL_DIR", str(model))
    first_model = visual_config_sha256(config)
    (model / "model.safetensors").write_bytes(b"model-two")
    assert visual_config_sha256(config) != first_model


def test_verified_reader_hit_and_source_config_artifact_invalidation(tmp_path: Path, monkeypatch) -> None:
    from integration_v1.contracts import AnalysisRecord
    from ownership.coverage import PageCoverageResult
    from vision_runtime import cache_reader, stage_cache

    root = tmp_path
    vision = root / "vision"
    page = vision / "pages" / "page_001"
    stage = vision / "pre_ocr"
    page.mkdir(parents=True)
    stage.mkdir(parents=True)
    source_sha, tree_sha, visual_sha = (_sha(value) for value in (b"source", b"tree", b"visual"))
    coverage_bytes = b"coverage"
    (page / "coverage_result.json").write_bytes(coverage_bytes)
    (stage / "page_001.json").write_bytes(b"stage-json")
    (stage / "page_001.npz").write_bytes(b"stage-npz")
    record_sha = _sha(b"record identity")
    record = {
        "page_id": "page_001", "source_sha256": source_sha,
        "analysis_config_sha256": visual_sha, "status": "complete", "publishable": False,
        "run_id": "run-original", "analysis_record_sha256": record_sha,
        "artifact_refs": [{"kind": "vision_coverage_result", "path": "vision/pages/page_001/coverage_result.json",
                           "sha256": _sha(coverage_bytes)}],
    }
    record_bytes = json.dumps(record).encode()
    (page / "analysis_record.json").write_bytes(record_bytes)
    ledger_sha = _sha(b"ledger")
    index = {
        "schema": "traduzai.vision-chapter-analysis-index.v1", "source_tree_sha256": tree_sha,
        "pages": [{
            "page_id": "page_001", "analysis_record_ref": "vision/pages/page_001/analysis_record.json",
            "analysis_record_file_sha256": _sha(record_bytes), "analysis_record_sha256": record_sha,
            "coverage_ledger_sha256": ledger_sha,
            "pre_ocr_json_ref": "vision/pre_ocr/page_001.json", "pre_ocr_json_sha256": _sha(b"stage-json"),
            "pre_ocr_npz_ref": "vision/pre_ocr/page_001.npz", "pre_ocr_npz_sha256": _sha(b"stage-npz"),
        }],
    }
    (vision / "index.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(AnalysisRecord, "build", staticmethod(lambda payload: SimpleNamespace(
        analysis_record_sha256=record_sha, to_dict=lambda: payload)))
    monkeypatch.setattr(PageCoverageResult, "from_canonical_json_bytes", staticmethod(
        lambda _data: SimpleNamespace(ledger=SimpleNamespace(sha256=ledger_sha),
                                      page_source_sha256=_sha(b"pixels"), run_id="run-original",
                                      origin_execution_id="execution-original")))
    monkeypatch.setattr(stage_cache, "load_pre_ocr_stage", lambda *_args, **_kwargs: {"bands": []})

    def read(*, source=source_sha, tree=tree_sha, visual=visual_sha):
        return cache_reader.read_verified_chapter_analysis(
            root, source_tree_sha256=tree, page_source_sha256s={"page_001": source},
            visual_config_sha256=visual)

    assert read()["status"] == "hit"
    assert read(tree=_sha(b"other"))["reason"] == "source_tree_changed"
    assert read(source=_sha(b"other"))["reason"] == "page_source_changed"
    assert read(visual=_sha(b"other"))["reason"] == "visual_config_changed"
    (stage / "page_001.npz").write_bytes(b"tampered")
    assert read()["reason"] == "pre_ocr_stage_changed"


@pytest.mark.parametrize("cache_hit", [False, True])
def test_physical_executor_routes_verified_cache_to_real_strip_entry(
    tmp_path: Path, monkeypatch, cache_hit: bool,
) -> None:
    from consumer_fast import physical_executor, vision_cache
    from extractor import extractor
    from ownership.chapter_contract import ChapterSourceManifest
    from strip import run as strip_run

    source = tmp_path / "source.png"
    source.write_bytes(b"source-image")
    work_dir = tmp_path / "work"
    config = {
        "runtime_id": "consumer-fast-v1", "legacy_translation_fallback_allowed": False,
        "translation_provider_policy": "consumer-fast-bounded-owner-v1",
        "consumer_fast_execution_plan": {"schema": "traduzai.consumer-fast-plan.v1"},
        "source_path": str(source), "work_dir": str(work_dir), "models_dir": str(tmp_path),
        "engine_preset_id": "max",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(extractor, "extract", lambda *_args: ([source], tmp_path))
    monkeypatch.setattr(physical_executor, "_vision_adapters", lambda *_args: (object(), object()))
    monkeypatch.setattr(physical_executor, "_build_inpainter", lambda *_args: object())
    manifest_calls = []

    def manifest(_files, _root, **kwargs):
        manifest_calls.append(kwargs)
        return SimpleNamespace(
            pages=[SimpleNamespace(page_id="page_001", source_file_sha256=_sha(b"source-image"))],
            source_tree_sha256=_sha(b"tree"), sha256=_sha(b"manifest"),
        )

    monkeypatch.setattr(ChapterSourceManifest, "from_extracted_pages", staticmethod(manifest))
    cached_page = {"page_001": {"stage": {"bands": []}}}
    monkeypatch.setattr(vision_cache, "select_cached_pages", lambda *_args: {
        "status": "hit" if cache_hit else "miss", "reason": None if cache_hit else "index_missing",
        "pages": cached_page if cache_hit else {},
        "lineage": ("run-original", "execution-original") if cache_hit else None,
        "visual_config_sha256": _sha(b"visual"),
    })
    captured = {}

    class StopAfterRouting(Exception):
        pass

    def stop_run(**kwargs):
        captured.update(kwargs)
        raise StopAfterRouting

    monkeypatch.setattr(strip_run, "run_chapter", stop_run)
    with pytest.raises(StopAfterRouting):
        physical_executor.execute_config(config_path)
    assert captured["vision_stage_cache"] == (cached_page if cache_hit else None)
    assert captured["vision_config_sha256"] == _sha(b"visual")
    assert captured["replay_of_execution_id"] == ("execution-original" if cache_hit else None)
    assert captured["run_id"] == ("run-original" if cache_hit else manifest_calls[0]["run_id"])
    assert len(manifest_calls) == (2 if cache_hit else 1)


def test_cache_publish_requires_complete_stage_set(tmp_path: Path, monkeypatch) -> None:
    from consumer_fast.vision_cache import publish_cached_pages
    from vision_runtime import page_record

    monkeypatch.setenv("TRADUZAI_EXPERIMENTAL_VISION_CACHE", "1")

    artifact_root = tmp_path / "execution"
    stage = artifact_root / "vision" / "pre_ocr"
    stage.mkdir(parents=True)
    manifest = SimpleNamespace(
        pages=[SimpleNamespace(page_id="page_001")], sha256=_sha(b"manifest"),
        source_tree_sha256=_sha(b"tree"),
    )
    pages = [SimpleNamespace()]
    cache_root = tmp_path / "cache"
    assert publish_cached_pages(cache_root, artifact_root, pages, {}, manifest)["reason"] == "pre_ocr_stage_missing"
    assert not (cache_root / "vision" / "index.json").exists()
    (stage / "page_001.json").write_bytes(b"json")
    (stage / "page_001.npz").write_bytes(b"npz")
    monkeypatch.setattr(page_record, "write_chapter_analysis_records", lambda *_args, **_kwargs: {
        "path": str(cache_root / "vision" / "index.json"), "page_count": 1})
    result = publish_cached_pages(cache_root, artifact_root, pages, {}, manifest)
    assert result["status"] == "written"
    assert (cache_root / "vision" / "pre_ocr" / "page_001.npz").read_bytes() == b"npz"
    unsafe = SimpleNamespace(pages=[SimpleNamespace(page_id="../bad")],
                             sha256=manifest.sha256, source_tree_sha256=manifest.source_tree_sha256)
    assert publish_cached_pages(cache_root, artifact_root, pages, {}, unsafe)["reason"] == "unsafe_page_id"


def test_consumer_fast_delivery_report_persists_review(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _apply_translation_delivery_review

    project = {"paginas": [{"numero": 1, "text_layers": [
        {"page_id": "page_001", "id": "owner-1", "source_payload": "HELLO", "visible": False},
    ]}]}
    gate = _apply_translation_delivery_review(project, {"status": "PASS", "issues": []})
    project["qa"]["export_gate"] = gate
    path = tmp_path / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    reopened = json.loads(path.read_text(encoding="utf-8"))
    report = reopened["qa"]["translation_delivery"]
    assert report["not_applied_count"] == 1
    assert report["quality_approved"] is False
    assert report["items"][0]["pending_work"]
    assert reopened["needs_review"] is True


def test_physical_artifact_binding_preserves_webp_original_and_png_render(tmp_path: Path) -> None:
    from consumer_fast.physical_executor import _bind_physical_page_artifacts

    original = tmp_path / "originals" / "001.webp"
    rendered = tmp_path / "translated" / "001.png"
    inpainted = tmp_path / "images" / "001.png"
    for path in (original, rendered, inpainted):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    project = {"paginas": [{"image_layers": {name: {"path": "wrong"} for name in (
        "base", "inpaint", "rendered")}}]}
    _bind_physical_page_artifacts(project, [SimpleNamespace(path=rendered)], [original], tmp_path)
    page = project["paginas"][0]
    assert page["arquivo_original"] == "originals/001.webp"
    assert page["arquivo_traduzido"] == "translated/001.png"
    assert page["image_layers"]["base"]["path"] == "originals/001.webp"
    assert page["image_layers"]["inpaint"]["path"] == "images/001.png"
    original.unlink()
    with pytest.raises(RuntimeError, match="base artifact is missing"):
        _bind_physical_page_artifacts(project, [SimpleNamespace(path=rendered)], [original], tmp_path)
