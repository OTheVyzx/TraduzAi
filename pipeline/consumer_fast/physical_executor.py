"""Direct Consumer Fast physical executor over the canonical owner stage engine."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    from consumer_fast_core import atomic_json

    atomic_json(path, value)


def _vision_adapters(config: dict[str, Any], models_dir: Path):
    from vision_stack.runtime import (
        _get_detector,
        _profile_to_detection_threshold,
        _run_koharu_cjk_http_detect_ocr_batch,
        _run_koharu_cjk_http_detect_ocr,
        run_final_pixel_ocr_probe,
        run_ocr_stage,
    )

    class Detector:
        def detect(self, image, conf_threshold=None):
            threshold = conf_threshold
            if threshold is None:
                threshold = _profile_to_detection_threshold("max")
            return _get_detector("max").detect(image, conf_threshold=threshold)

    class Runtime:
        def run_final_pixel_ocr_probe(self, image, **kwargs):
            return run_final_pixel_ocr_probe(image, **kwargs)

        def run_ocr_stage(self, image, page, work_title="", work_title_user_provided=False):
            return run_ocr_stage(
                image,
                page,
                profile="max",
                idioma_origem=config.get("idioma_origem", "en"),
                engine_preset_id=config.get("engine_preset_id", ""),
                work_title=work_title or config.get("obra", ""),
                work_title_user_provided=bool(
                    work_title_user_provided or config.get("work_title_user_provided")
                ),
            )

        def run_koharu_cjk_page(self, image, image_path, work_title="", work_title_user_provided=False):
            return _run_koharu_cjk_http_detect_ocr(
                image_rgb=image,
                image_label=str(image_path),
                models_dir=str(models_dir),
                profile="max",
                idioma_origem=config.get("idioma_origem", "en"),
                engine_preset_id=config.get("engine_preset_id", ""),
                work_title=work_title or config.get("obra", ""),
                work_title_user_provided=bool(
                    work_title_user_provided or config.get("work_title_user_provided")
                ),
            )

        def run_koharu_cjk_pages(self, jobs, models_dir="", idioma_origem="en", work_title="", work_title_user_provided=False):
            return _run_koharu_cjk_http_detect_ocr_batch(
                jobs,
                models_dir=str(models_dir or models_dir_path),
                profile="max",
                idioma_origem=idioma_origem or config.get("idioma_origem", "en"),
                engine_preset_id=config.get("engine_preset_id", ""),
                work_title=work_title or config.get("obra", ""),
                work_title_user_provided=bool(
                    work_title_user_provided or config.get("work_title_user_provided")
                ),
            )

    models_dir_path = models_dir
    return Detector(), Runtime()


def _build_inpainter(config: dict[str, Any]):
    from inpainter import inpaint_band_image

    if not config.get("skip_inpaint"):
        return SimpleNamespace(inpaint_band_image=inpaint_band_image)

    def skip_inpaint(band_rgb, ocr_page: dict):
        if isinstance(ocr_page, dict):
            ocr_page["_skip_inpaint_honored"] = True
            ocr_page["_strip_used_fast_white_fill"] = False
            ocr_page["_strip_used_fast_local_fill"] = False
            ocr_page["_strip_used_real_inpaint"] = False
            ocr_page["_strip_used_post_cleanup"] = False
        return band_rgb.copy() if hasattr(band_rgb, "copy") else band_rgb

    return SimpleNamespace(inpaint_band_image=skip_inpaint)


def _write_analysis_record(output_dir: Path, source_path: Path, config: dict[str, Any], output_pages) -> dict[str, Any]:
    """Persist a real, non-publishable Vision contract while lineage adapters remain incomplete."""
    from integration_v1.contracts import AnalysisRecord
    from vision_runtime.analysis_payload import build_analysis_payload

    source_sha = _sha256(source_path)
    config_sha = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    page = output_pages[0]
    image = getattr(page, "original_image", None)
    height, width = (image.shape[:2] if image is not None else (1, 1))
    root = output_dir / "consumer_fast" / "vision"
    root.mkdir(parents=True, exist_ok=True)
    observations_path = root / "ocr_observations.json"
    observations = [
        {
            "page": index + 1,
            "owner_ids": [str(item.get("owner_id") or item.get("id") or "") for item in (p.text_layers or {}).get("texts", []) if isinstance(item, dict)],
            "text_count": len((p.text_layers or {}).get("texts", [])),
        }
        for index, p in enumerate(output_pages)
    ]
    _atomic_json(observations_path, observations)
    observations_sha = _sha256(observations_path)
    payload = build_analysis_payload(
        status="building",
        identity={
            "source_sha256": source_sha,
            "authenticated_neighbor_sha256s": [],
            "region": {"bbox": [0, 0, int(width), int(height)], "coordinate_space": "logical_page"},
            "coordinate_space": "logical_page",
            "transform_sha256": hashlib.sha256(b"identity-transform-v1").hexdigest(),
            "source_language": str(config.get("idioma_origem", "en")),
            "analysis_config_sha256": config_sha,
            "provider_family": "vision_stack",
            "provider_name": "vision_stack.runtime",
            "provider_model": str(config.get("engine_preset_id") or "max"),
            "provider_version": "local-runtime",
            "capability_version": "consumer-fast-v1",
        },
        references={
            "artifact_refs": [{"kind": "ocr_observations", "path": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha}],
            "transform_ref": {"kind": "identity", "sha256": hashlib.sha256(b"identity-transform-v1").hexdigest(), "inverse_sha256": hashlib.sha256(b"identity-transform-v1").hexdigest()},
            "ocr_observations": {"artifact_ref": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha, "count": len(observations)},
            "selected_observation_id": None,
            "selection_provenance": {"kind": "owner_lineage_adapter_pending"},
            "logical_units": {"artifact_ref": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha, "count": 0},
            "physical_subblocks": {"artifact_ref": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha, "count": sum(row["text_count"] for row in observations)},
            "relations": {"artifact_ref": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha, "count": 0},
            "reading_order": {"artifact_ref": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha, "count": 0},
            "container_contour_ref": None,
            "writing_body_ref": None,
            "tail_ref": None,
            "dependency_hashes": {"source": source_sha, "analysis_config": config_sha, "ocr_observations": observations_sha},
        },
    )
    record = AnalysisRecord.build(payload).to_dict()
    record["status"] = "building"
    record["publishable"] = False
    path = root / "analysis_record.json"
    _atomic_json(path, record)
    return {"status": "building", "publishable": False, "path": str(path), "sha256": _sha256(path)}


def execute_config(config_path: Path) -> Path:
    """Run physical Vision/Consumer Fast/Renderer stages directly, without a main-pipeline fallback."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("runtime_id") != "consumer-fast-v1":
        raise ValueError("consumer-fast-v1 runtime binding is required")
    if config.get("legacy_translation_fallback_allowed") is not False:
        raise ValueError("legacy translation fallback must remain disabled")
    if config.get("translation_provider_policy") != "consumer-fast-bounded-owner-v1":
        raise ValueError("bounded Consumer Fast provider policy is missing")
    plan = config.get("consumer_fast_execution_plan") or {}
    if plan.get("schema") != "traduzai.consumer-fast-plan.v1":
        raise ValueError("Consumer Fast V1 execution plan is missing")

    started = time.perf_counter()
    from extractor.extractor import extract
    from ownership.chapter_contract import ChapterSourceManifest
    from strip.run import run_chapter
    from consumer_fast import provider_adapter
    from typesetter import renderer
    from vision_stack.engine_presets import resolve_engine_preset

    source = Path(str(config["source_path"])).resolve()
    work_dir = Path(config["work_dir"]).resolve()
    models_dir = Path(config["models_dir"]).resolve()
    originals_dir = work_dir / "originals"
    translated_dir = work_dir / "translated"
    images_dir = work_dir / "images"
    for folder in (originals_dir, translated_dir, images_dir):
        folder.mkdir(parents=True, exist_ok=True)

    image_files, _extraction_dir = extract(source, work_dir)
    if not image_files:
        raise RuntimeError("a fonte não contém páginas de imagem")
    image_files = [Path(item).resolve() for item in image_files]
    for index, image in enumerate(image_files, start=1):
        target_name = f"{index:03d}{image.suffix.lower()}"
        shutil.copy2(image, originals_dir / target_name)
    config["engine_preset_id"] = config.get("engine_preset_id") or resolve_engine_preset(
        config, idioma_origem=config.get("idioma_origem", "en")
    ).id
    config["work_title_user_provided"] = bool(config.get("work_title_user_provided"))

    detector, runtime = _vision_adapters(config, models_dir)
    execution_id = f"owner-execution-{uuid.uuid4().hex}"
    run_id = f"owner-run-{uuid.uuid4().hex}"
    extraction_root = Path(__import__("os").path.commonpath([str(item) for item in image_files]))
    if extraction_root.is_file():
        extraction_root = extraction_root.parent
    source_manifest = ChapterSourceManifest.from_extracted_pages(
        image_files, extraction_root, run_id=run_id, execution_id=execution_id
    )
    telemetry: dict[str, Any] = {}
    output_pages = run_chapter(
        image_files=image_files,
        output_dir=translated_dir,
        target_count=len(image_files),
        detector=detector,
        runtime=runtime,
        translator=provider_adapter,
        inpainter=_build_inpainter(config),
        typesetter=renderer,
        context=config.get("contexto") or {},
        glossario=config.get("glossario") or {},
        idioma_origem=config.get("idioma_origem", "en"),
        idioma_destino=config.get("idioma_destino", "pt-BR"),
        obra=config.get("obra", ""),
        work_title_user_provided=config["work_title_user_provided"],
        models_dir=str(models_dir),
        ollama_host=config.get("ollama_host", "http://localhost:11434"),
        ollama_model=config.get("ollama_model", "traduzai-translator"),
        chapter_telemetry=telemetry,
        skip_page_cleanup_rerender=bool(config.get("skip_inpaint")),
        owner_graph_mode="enforce",
        style_copy_mode="shadow",
        run_id=run_id,
        execution_id=execution_id,
        source_manifest=source_manifest,
        artifact_root=work_dir / ".owner-private" / execution_id,
        progress_callback=lambda *_args, **_kwargs: None,
    )
    if len(output_pages) != len(image_files):
        raise RuntimeError("owner executor page cardinality mismatch")
    for index, page in enumerate(output_pages, start=1):
        expected = originals_dir / f"{index:03d}{image_files[index - 1].suffix.lower()}"
        if not expected.is_file() or _sha256(expected) != _sha256(image_files[index - 1]):
            raise RuntimeError("original source bytes changed during physical execution")
        if not Path(page.path).is_file():
            raise RuntimeError(f"renderer output raster missing: {page.path}")
        if getattr(page, "inpainted_image", None) is not None:
            from PIL import Image

            Image.fromarray(page.inpainted_image).save(images_dir / Path(page.path).name)
        else:
            shutil.copy2(expected, images_dir / Path(page.path).name)

    from main import build_project_json
    from project_writer import write_project_json_atomic
    from qa.export_gate import append_qa_integrity_failure, evaluate_export_gate

    project = build_project_json(
        config,
        config.get("contexto") or {},
        [page.ocr_result for page in output_pages],
        [page.text_layers for page in output_pages],
        [Path(page.path) for page in output_pages],
        len(output_pages),
        time.perf_counter() - started,
        output_pages=output_pages,
    )
    project["runtime_id"] = "consumer-fast-v1"
    project["runtime_profile"] = "consumer_fast"
    project["verified"] = False
    gate = evaluate_export_gate(project)
    gate = append_qa_integrity_failure(gate, ["vision_analysis_record_building", "renderer_recipe_adapter_pending"])
    project.setdefault("qa", {})["export_gate"] = gate
    project["export_gate"] = gate
    project["consumer_fast_physical"] = {
        "execution_id": execution_id,
        "run_id": run_id,
        "execution_plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "vision_analysis_record": _write_analysis_record(work_dir, image_files[0], config, output_pages),
        "renderer": {"module": renderer.__name__, "recipe_contract_status": "adapter_pending", "recipe_count": 0},
        "stages": ["import", "analysis", "ocr", "logical_units", "translate", "restore", "layout", "rasterize", "review", "persist", "export_decision"],
        "translation_provider": provider_adapter.POLICY_ID,
        "legacy_fallback_allowed": False,
        "page_count": len(output_pages),
        "telemetry": telemetry,
    }

    # Do not materialize empty masks as if they were authoritative; project builder persists only observed masks.
    project_path = work_dir / "project.json"
    write_project_json_atomic(project_path, project)
    reopened = json.loads(project_path.read_text(encoding="utf-8"))
    if reopened != project:
        raise RuntimeError("project.json failed semantic persist/reopen equality")

    rasters = [
        {"path": Path(page.path).relative_to(work_dir).as_posix(), "sha256": _sha256(Path(page.path))}
        for page in output_pages
    ]
    evidence = {
        "runtime_id": "consumer-fast-v1",
        "status": "physical_short_run",
        "wall_time_internal_seconds": round(time.perf_counter() - started, 4),
        "project_path": str(project_path),
        "project_sha256_pre_finalize": _sha256(project_path),
        "source_manifest_sha256": source_manifest.source_tree_sha256,
        "execution_id": execution_id,
        "run_id": run_id,
        "page_count": len(output_pages),
        "rasters": rasters,
        "renderer_recipe": {"status": "not_emitted", "reason": "owner glyph-patch executor has no V1 RendererRecipe adapter"},
        "analysis_record": project["consumer_fast_physical"]["vision_analysis_record"],
        "qa": gate,
    }
    _atomic_json(work_dir / "consumer_fast_physical_execution.json", evidence)
    return project_path
