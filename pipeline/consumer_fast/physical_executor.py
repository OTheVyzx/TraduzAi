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
    """Persist the canonical Vision/page journal without upgrading mask authority."""
    from integration_v1.contracts import AnalysisRecord
    from vision_runtime.analysis_payload import build_analysis_payload
    from vision_runtime.structure import build_structural_analysis

    source_sha = _sha256(source_path)
    config_sha = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    page = output_pages[0]
    image = getattr(page, "original_image", None)
    height, width = (image.shape[:2] if image is not None else (1, 1))
    root = output_dir / "consumer_fast" / "vision"
    root.mkdir(parents=True, exist_ok=True)
    page_result = getattr(page, "owner_page_result", None)
    if page_result is None:
        raise RuntimeError("Vision PageExecutionResult is missing; AnalysisRecord cannot be complete")
    coverage = getattr(page_result, "coverage", None)
    if coverage is None or not getattr(coverage, "canonical_json_bytes", None):
        raise RuntimeError("Vision PageCoverageResult canonical bytes are missing")
    ledger = getattr(coverage, "ledger", None)
    if ledger is None or not getattr(ledger, "sha256", None):
        raise RuntimeError("Vision coverage ledger identity is missing")

    journal = page_result.to_canonical_dict()
    journal_path = root / "page_execution_evidence.json"
    _atomic_json(journal_path, journal)
    journal_sha = _sha256(journal_path)
    coverage_path = root / "coverage_result.json"
    coverage_path.write_bytes(coverage.canonical_json_bytes)
    coverage_sha = _sha256(coverage_path)
    observations = [
        {
            "observation_id": str(item.observation_id),
            "text": str(item.text),
            "bbox_page": list(item.bbox_page),
            "page_id": str(item.page_id),
            "provider": str(item.provider),
            "provider_family": str(item.provider_family),
            "run_id": str(item.run_id),
            "execution_id": str(item.origin_execution_id),
            "attempt_id": str(item.attempt_id),
            "invocation_id": str(item.invocation_id),
            "page_source_sha256": str(item.page_source_sha256),
            "payload_sha256": str(item.payload_sha256),
            "rejection_reason": item.rejection_reason,
        }
        for item in coverage.observations
    ]
    observations_path = root / "ocr_observations.json"
    _atomic_json(observations_path, observations)
    observations_sha = _sha256(observations_path)
    owner_graph_snapshot = getattr(page_result, "owner_graph", None)
    owner_graph = owner_graph_snapshot.read() if callable(getattr(owner_graph_snapshot, "read", None)) else None
    selected_observation_ids = sorted({
        str(observation_id)
        for owner in (getattr(owner_graph, "owners", ()) or ())
        for observation_id in (getattr(owner, "selected_observation_ids", ()) or ())
        if str(observation_id)
    })
    observation_ids = {str(item["observation_id"]) for item in observations}
    if not set(selected_observation_ids).issubset(observation_ids):
        raise RuntimeError("Vision owner selection references an observation outside the same page evidence")
    structural = build_structural_analysis(
        observations=[
            {
                "observation_id": item["observation_id"],
                "text": item["text"],
                "bbox_page": item["bbox_page"],
                "selection_state": "eligible" if item["observation_id"] in selected_observation_ids else "unresolved",
                "uncertainty_reasons": ([] if item["observation_id"] in selected_observation_ids else ["owner_selection_not_authorized_by_analysis_adapter"]),
            }
            for item in observations
        ],
        containers=[],
    )
    structural_payload = {
        "schema": "traduzai.vision-structural-analysis.v1",
        "logical_units": list(structural.logical_units),
        "physical_subblocks": list(structural.physical_subblocks),
        "relations": list(structural.relations),
        "reading_order": list(structural.reading_order),
        "source_observation_sha256": observations_sha,
    }
    structure_path = root / "structural_analysis.json"
    _atomic_json(structure_path, structural_payload)
    structure_sha = _sha256(structure_path)
    artifact_refs = [
        {"kind": kind, "path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path)}
        for kind, path in (
            ("vision_page_execution_evidence", journal_path),
            ("vision_coverage_result", coverage_path),
            ("vision_ocr_observations", observations_path),
            ("vision_structural_analysis", structure_path),
        )
    ]
    page_id = str(page_result.page_id)
    project_id = f"consumer-fast-{source_sha[:24]}"
    project_revision = 1
    payload = build_analysis_payload(
        status="complete",
        identity={
            "source_sha256": source_sha,
            "authenticated_neighbor_sha256s": [],
            "region": {"bbox": [0, 0, int(width), int(height)], "coordinate_space": "logical_page"},
            "coordinate_space": "logical_page",
            "transform_sha256": hashlib.sha256(b"identity-transform-v1").hexdigest(),
            "source_language": str(config.get("idioma_origem", "en")),
            "analysis_config_sha256": config_sha,
            "provider_family": "vision_v6",
            "provider_name": "vision_stack.runtime",
            "provider_model": str(config.get("engine_preset_id") or "max"),
            "provider_version": str(getattr(coverage.observations[0], "provider_family", "vision-v6") if coverage.observations else "vision-v6"),
            "capability_version": "consumer-fast-v1",
        },
        references={
            "artifact_refs": artifact_refs,
            "transform_ref": {"kind": "identity", "sha256": hashlib.sha256(b"identity-transform-v1").hexdigest(), "inverse_sha256": hashlib.sha256(b"identity-transform-v1").hexdigest()},
            "ocr_observations": {"artifact_ref": "consumer_fast/vision/ocr_observations.json", "sha256": observations_sha, "count": len(observations)},
            "selected_observation_id": selected_observation_ids[0] if len(selected_observation_ids) == 1 else None,
            "selection_provenance": {"kind": "vision_page_execution_and_owner_graph", "page_evidence_sha256": journal_sha, "coverage_sha256": coverage_sha, "coverage_ledger_sha256": ledger.sha256, "owner_graph_sha256": str(getattr(owner_graph_snapshot, "sha256", journal.get("owner_graph_sha256") or "")), "selected_observation_ids": selected_observation_ids, "selection_status": "selected" if selected_observation_ids else "unresolved"},
            "logical_units": {"artifact_ref": "consumer_fast/vision/structural_analysis.json", "sha256": structure_sha, "count": len(structural.logical_units)},
            "physical_subblocks": {"artifact_ref": "consumer_fast/vision/structural_analysis.json", "sha256": structure_sha, "count": len(structural.physical_subblocks)},
            "relations": {"artifact_ref": "consumer_fast/vision/structural_analysis.json", "sha256": structure_sha, "count": len(structural.relations)},
            "reading_order": {"artifact_ref": "consumer_fast/vision/structural_analysis.json", "sha256": structure_sha, "count": len(structural.reading_order)},
            "container_contour_ref": None,
            "writing_body_ref": None,
            "tail_ref": None,
            "dependency_hashes": {"source": source_sha, "analysis_config": config_sha, "ocr_observations": observations_sha, "vision_page_evidence": journal_sha, "coverage_result": coverage_sha, "coverage_ledger": ledger.sha256, "structural_analysis": structure_sha},
        },
    )
    payload.update({
        "project_id": project_id,
        "project_revision": project_revision,
        "run_id": str(page_result.request.run_id),
        "execution_id": str(page_result.request.execution_id),
        "page_id": page_id,
        "coverage_ledger_sha256": ledger.sha256,
    })
    record = AnalysisRecord.build(payload).to_dict()
    record["project_id"] = project_id
    record["project_revision"] = project_revision
    record["run_id"] = str(page_result.request.run_id)
    record["execution_id"] = str(page_result.request.execution_id)
    record["page_id"] = page_id
    record["coverage_ledger_sha256"] = ledger.sha256
    record["publishable"] = False
    path = root / "analysis_record.json"
    _atomic_json(path, record)
    return {
        "status": "complete", "publishable": False, "path": str(path), "sha256": _sha256(path),
        "project_id": project_id, "project_revision": project_revision, "run_id": record["run_id"],
        "execution_id": record["execution_id"], "page_id": page_id,
        "coverage_ledger_sha256": ledger.sha256,
        "analysis_record_sha256": record["analysis_record_sha256"],
    }


def _exact_line_plan(target_text: str, rendered_lines: list[str]):
    """Bind renderer-emitted lines to exact source substrings and whitespace."""
    from typesetter.recipe_contract import ExactLinePlan

    lines = [str(line) for line in rendered_lines]
    if not target_text or not lines:
        raise ValueError("RendererRecipe missing rendered_lines from typesetter output")
    separators: list[str] = []
    offset = 0
    for index, line in enumerate(lines):
        if not line or target_text[offset : offset + len(line)] != line:
            raise ValueError("RendererRecipe rendered_lines do not preserve exact target_text")
        offset += len(line)
        if index + 1 < len(lines):
            separator_start = offset
            while offset < len(target_text) and target_text[offset].isspace():
                offset += 1
            separator = target_text[separator_start:offset]
            if not separator:
                raise ValueError("RendererRecipe rendered_lines have no exact whitespace separator")
            separators.append(separator)
    if offset != len(target_text):
        raise ValueError("RendererRecipe rendered_lines do not consume exact target_text")
    return ExactLinePlan.build(target_text=target_text, lines=lines, separators=separators)


def _write_renderer_recipes(
    output_dir: Path,
    output_pages,
    *,
    analysis_record_sha256: str,
    project_id: str,
    project_revision: int,
    ledger_sha256: str,
    original_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Persist only recipes whose exact owner-render evidence came from the raster stage."""
    from typesetter.recipe_contract import RendererRecipe
    from typesetter.recipe_persistence import persist_renderer_recipe

    output_root = Path(output_dir).resolve()
    recipe_root = output_root / "consumer_fast"
    recipe_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    recipes: list[dict[str, Any]] = []
    no_patch_diagnostics: list[str] = []
    for page_index, page in enumerate(output_pages):
        page_result = getattr(page, "owner_page_result", None)
        if page_result is None:
            raise ValueError("RendererRecipe requires the same Vision PageExecutionResult")
        coverage = page_result.coverage
        page_ledger = getattr(coverage, "ledger", None)
        if page_ledger is None or page_ledger.sha256 != ledger_sha256:
            raise ValueError("RendererRecipe coverage ledger differs from AnalysisRecord")
        raster_path = Path(page.path).resolve()
        raster_bytes = raster_path.read_bytes()
        raster_sha = hashlib.sha256(raster_bytes).hexdigest()
        commits = list(getattr(page_result, "page_commits", ()) or ())
        owner_patches = [
            item.glyph_patch for item in commits
            if getattr(item, "glyph_patch", None) is not None
        ]
        if not owner_patches:
            graph_snapshot = getattr(page_result, "owner_graph", None)
            graph = graph_snapshot.read() if callable(getattr(graph_snapshot, "read", None)) else None
            owners = list(getattr(graph, "owners", ()) or ())
            owner_details = []
            unchanged_owners = []
            for owner in owners:
                owner_id = str(getattr(owner, "owner_id", ""))
                state = str(getattr(owner, "state", "unknown"))
                route = str(getattr(owner, "route_action", "unknown"))
                source = str(getattr(owner, "source_payload", "") or "")
                target = str(getattr(owner, "translated_payload", "") or "")
                owner_details.append(f"{owner_id}(state={state},route={route})")
                if source and source == target:
                    unchanged_owners.append(owner_id)
            no_patch_diagnostics.append(
                f"page_id={page_result.page_id} page_commits={len(commits)} "
                f"owners=[{','.join(owner_details) or 'none'}] "
                f"unchanged=[{','.join(unchanged_owners) or 'none'}]"
            )
            continue
        for patch in owner_patches:
            evidence = getattr(patch, "renderer_recipe_evidence", None)
            if not isinstance(evidence, dict):
                raise ValueError(
                    "RendererRecipe producer typesetter.renderer._render_owner_band_image "
                    "did not emit renderer_recipe_evidence"
                )
            required = (
                "target_text", "rendered_lines", "font_family", "font_path", "font_sha256",
                "font_size_px", "line_advance_px", "render_bbox", "effects", "anchors",
                "geometry", "policy_versions", "rasterizer_runtime_id",
                "rasterizer_runtime_sha256", "rasterizer_config_sha256", "source_pixel_sha256",
            )
            missing = [name for name in required if evidence.get(name) in (None, "", [], {})]
            if missing:
                raise ValueError(
                    "RendererRecipe producer typesetter.renderer._render_owner_band_image "
                    "missing fields: " + ", ".join(missing)
                )
            font_path = Path(str(evidence["font_path"])).resolve()
            if not font_path.is_file() or hashlib.sha256(font_path.read_bytes()).hexdigest() != evidence["font_sha256"]:
                raise ValueError("RendererRecipe font file identity/hash is unavailable or changed")
            try:
                font_relative_path = font_path.relative_to(repo_root).as_posix()
            except ValueError as exc:
                try:
                    font_relative_path = font_path.relative_to(output_root).as_posix()
                except ValueError:
                    raise ValueError("RendererRecipe font_path is outside the project/runtime roots") from exc
            source_pixel_sha = str(evidence["source_pixel_sha256"])
            if len(source_pixel_sha) != 64 or any(char not in "0123456789abcdef" for char in source_pixel_sha):
                raise ValueError("RendererRecipe source_pixel_sha256 is not a canonical SHA-256")
            original_path = Path(original_paths[page_index]).resolve() if original_paths else None
            original_sha = _sha256(original_path) if original_path and original_path.is_file() else ""
            lineage = {
                "project_id": project_id,
                "project_revision": int(project_revision),
                "coverage_ledger_sha256": ledger_sha256,
                "analysis_record_sha256": analysis_record_sha256,
                "run_id": str(page_result.request.run_id),
                "execution_id": str(page_result.request.execution_id),
                "page_id": str(page_result.page_id),
                "owner_id": str(patch.owner_id),
            }
            lineage_sha = hashlib.sha256(
                json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            recipe = RendererRecipe.build(
                owner_id=str(patch.owner_id),
                source_sha256=source_pixel_sha,
                output_sha256=raster_sha,
                target_text=str(evidence["target_text"]),
                font={
                    "family": str(evidence["font_family"]),
                    "relative_path": font_relative_path,
                    "sha256": str(evidence["font_sha256"]),
                },
                rasterizer={
                    "runtime_id": str(evidence["rasterizer_runtime_id"]),
                    "runtime_sha256": str(evidence["rasterizer_runtime_sha256"]),
                    "config_sha256": str(evidence["rasterizer_config_sha256"]),
                },
                line_plan=_exact_line_plan(str(evidence["target_text"]), list(evidence["rendered_lines"])),
                bbox=list(evidence["render_bbox"]),
                font_size_px=int(evidence["font_size_px"]),
                line_advance_px=int(evidence["line_advance_px"]),
                effects=dict(evidence["effects"]),
                anchors=dict(evidence["anchors"]),
                geometry=dict(evidence["geometry"]),
                policy_versions=dict(evidence["policy_versions"]),
                dependency_hashes={
                    "analysis_record": analysis_record_sha256,
                    "coverage_ledger": ledger_sha256,
                    "project_lineage": lineage_sha,
                    "renderer_runtime": str(evidence["rasterizer_runtime_sha256"]),
                    "raster_artifact": raster_sha,
                    "source_file": original_sha or source_pixel_sha,
                },
            )
            recipe.verify_output(raster_bytes)
            persisted_path = persist_renderer_recipe(recipe_root, recipe)
            loaded_path = persisted_path.resolve().relative_to(output_root).as_posix()
            recipes.append({
                "owner_id": recipe.owner_id,
                "page_id": str(page_result.page_id),
                "recipe_sha256": recipe.recipe_sha256,
                "output_sha256": recipe.output_sha256,
                "path": loaded_path,
                "project_id": project_id,
                "project_revision": int(project_revision),
                "coverage_ledger_sha256": ledger_sha256,
                "analysis_record_sha256": analysis_record_sha256,
                "lineage": lineage,
            })
    if not recipes:
        detail = "; ".join(no_patch_diagnostics) or "no_owner_execution_commits"
        raise ValueError(
            "RendererRecipe producer typesetter.renderer._render_owner_band_image was not invoked: "
            "no committed OwnerGlyphPatch; " + detail
        )
    lineage_receipt = {
        "schema": "traduzai.consumer-fast-renderer-lineage.v1",
        "project_id": project_id,
        "project_revision": int(project_revision),
        "coverage_ledger_sha256": ledger_sha256,
        "analysis_record_sha256": analysis_record_sha256,
        "recipes": recipes,
    }
    _atomic_json(recipe_root / "renderer_recipe_lineage.json", lineage_receipt)
    return {"status": "persisted", "recipe_count": len(recipes), "project_id": project_id,
            "project_revision": int(project_revision), "coverage_ledger_sha256": ledger_sha256,
            "recipes": recipes, "lineage_path": str(recipe_root / "renderer_recipe_lineage.json")}


def _loaded_module_evidence() -> list[dict[str, str]]:
    import sys

    names = (
        "vision_stack.runtime", "vision_runtime.analysis_payload", "vision_runtime.structure",
        "integration_v1.contracts", "consumer_fast.physical_executor", "consumer_fast.provider_adapter",
        "strip.run", "typesetter.renderer", "typesetter.recipe_contract", "typesetter.recipe_persistence",
    )
    rows = []
    for name in names:
        module = sys.modules.get(name)
        raw_path = getattr(module, "__file__", None) if module is not None else None
        if not raw_path:
            rows.append({"name": name, "status": "not_loaded"})
            continue
        path = Path(raw_path).resolve()
        rows.append({
            "name": name,
            "status": "loaded",
            "path": str(path),
            "sha256": _sha256(path),
        })
    return rows


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
    original_targets = [
        originals_dir / f"{index:03d}{image.suffix.lower()}"
        for index, image in enumerate(image_files, start=1)
    ]
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
        page.original_source_path = expected

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
    analysis_receipt = _write_analysis_record(work_dir, image_files[0], config, output_pages)
    ledger_sha = str(analysis_receipt["coverage_ledger_sha256"])
    project_id = str(analysis_receipt["project_id"])
    renderer_recipe_receipt: dict[str, Any]
    try:
        renderer_recipe_receipt = _write_renderer_recipes(
            work_dir,
            output_pages,
            analysis_record_sha256=str(analysis_receipt["analysis_record_sha256"]),
            project_id=project_id,
            project_revision=int(analysis_receipt["project_revision"]),
            ledger_sha256=ledger_sha,
            original_paths=original_targets,
        )
    except (ValueError, OSError) as exc:
        renderer_recipe_receipt = {
            "status": "not_emitted",
            "recipe_count": 0,
            "project_id": project_id,
            "project_revision": int(analysis_receipt["project_revision"]),
            "coverage_ledger_sha256": ledger_sha,
            "blocker": str(exc),
        }
    gate = evaluate_export_gate(project)
    integrity_failures = []
    if analysis_receipt.get("status") != "complete":
        integrity_failures.append("vision_analysis_record_incomplete")
    committed_patch_count = len([
        patch for page in output_pages
        for commit in (getattr(getattr(page, "owner_page_result", None), "page_commits", ()) or ())
        if (patch := getattr(commit, "glyph_patch", None)) is not None
    ])
    if not renderer_recipe_receipt.get("recipe_count") or renderer_recipe_receipt.get("recipe_count") != committed_patch_count:
        integrity_failures.append("renderer_recipe_missing_or_incomplete")
    if integrity_failures:
        gate = append_qa_integrity_failure(gate, integrity_failures)
    project["project_id"] = project_id
    project["project_revision"] = int(analysis_receipt["project_revision"])
    project["coverage_ledger_sha256"] = ledger_sha
    project.setdefault("qa", {})["export_gate"] = gate
    project["export_gate"] = gate
    project["consumer_fast_physical"] = {
        "execution_id": execution_id,
        "run_id": run_id,
        "execution_plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "project_id": project_id,
        "project_revision": int(analysis_receipt["project_revision"]),
        "coverage_ledger_sha256": ledger_sha,
        "vision_analysis_record": analysis_receipt,
        "renderer": {"module": renderer.__name__, "recipe_contract_status": renderer_recipe_receipt["status"], "recipe_count": renderer_recipe_receipt.get("recipe_count", 0), "recipe_lineage": renderer_recipe_receipt},
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
        "project_id": project_id,
        "project_revision": int(analysis_receipt["project_revision"]),
        "coverage_ledger_sha256": ledger_sha,
        "originals": [
            {"path": path.relative_to(work_dir).as_posix(), "sha256": _sha256(path)}
            for path in original_targets
        ],
        "rasters": rasters,
        "renderer_recipe": renderer_recipe_receipt,
        "analysis_record": project["consumer_fast_physical"]["vision_analysis_record"],
        "loaded_modules": _loaded_module_evidence(),
        "qa": gate,
    }
    _atomic_json(work_dir / "consumer_fast_physical_execution.json", evidence)
    return project_path
