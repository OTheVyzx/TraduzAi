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


def _apply_translation_delivery_review(project: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    """Persist owner-level delivery notices while final verification remains pending."""
    from qa.partial_delivery import apply_partial_delivery_policy

    reviewed_gate = apply_partial_delivery_policy(project, gate)
    project['qa']['translation_delivery']['quality_approved'] = False
    project['needs_review'] = True
    return reviewed_gate


def _bind_physical_page_artifacts(
    project: dict[str, Any], output_pages: list[Any], original_targets: list[Path], work_dir: Path,
) -> None:
    """Bind the original file's real extension and the renderer's output path."""
    pages = project.get("paginas")
    if not isinstance(pages, list) or len(pages) != len(output_pages) or len(pages) != len(original_targets):
        raise RuntimeError("Consumer Fast page artifact cardinality mismatch")
    root = work_dir.resolve()
    for page, output, original in zip(pages, output_pages, original_targets):
        rendered = Path(output.path).resolve()
        inpainted = (work_dir / "images" / rendered.name).resolve()
        paths = {"base": original.resolve(), "inpaint": inpainted, "rendered": rendered}
        layers = page.get("image_layers")
        if not isinstance(layers, dict) or any(not isinstance(layers.get(name), dict) for name in paths):
            raise RuntimeError("Consumer Fast image layer paths are missing")
        relative = {}
        for name, artifact in paths.items():
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise RuntimeError(f"Consumer Fast {name} artifact is missing or outside work_dir")
            relative[name] = artifact.relative_to(root).as_posix()
            layers[name]["path"] = relative[name]
        page["arquivo_original"] = relative["base"]
        page["arquivo_traduzido"] = relative["rendered"]


def _bind_verified_delivery_notices(project: dict[str, Any], output_pages: list[Any]) -> None:
    """Carry owner commit/translation evidence into the serialized project."""
    import cv2
    from ownership.hash_contract import canonical_page_sha256
    from qa.partial_delivery import notice_for_layer

    pages = project.get("paginas") or []
    if len(pages) != len(output_pages):
        raise RuntimeError("Consumer Fast delivery page count mismatch")
    for page, output in zip(pages, output_pages):
        result = getattr(output, "owner_page_result", None)
        if result is None or result.status != "final_verified" or result.terminal_proof is None:
            raise RuntimeError("Consumer Fast delivery lacks final owner proof")
        raster = cv2.imread(str(output.path), cv2.IMREAD_COLOR)
        if raster is None or canonical_page_sha256(cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)) != result.final_page.page_output_pixel_sha256:
            raise RuntimeError("Consumer Fast delivery raster differs from final owner proof")
        bindings = {binding.owner_id: binding for binding in result.translations}
        commits = tuple(result.page_commits)
        verdicts = {verdict.owner_id: verdict for verdict in result.final_replacement_verdicts
                    if verdict.status == "final_verified" and verdict.target_materialized}
        for key in ("text_layers", "textos"):
            layers = page.get(key) or []
            for layer in layers:
                owner_id = layer.get("owner_id")
                binding = bindings.get(owner_id)
                commit = next((item for item in commits if item.owner_id == owner_id
                               and item.translation_binding_sha256 == getattr(binding, "translation_binding_sha256", None)), None)
                verdict = verdicts.get(owner_id)
                if (binding is not None and commit is not None and verdict is not None
                        and verdict.translation_binding_sha256 == binding.translation_binding_sha256
                        and verdict.target_glyph_patch_sha256 == commit.target_glyph_patch_sha256
                        and str(layer.get("translated") or layer.get("traduzido") or layer.get("translated_payload") or "") == binding.target_text):
                    layer["visible"] = True
                layer["translation_delivery_notice"] = notice_for_layer(
                    layer, binding=binding, attempts=result.translation_attempts, commits=commits,
                )


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
    """Persist the Vision page record for the direct Consumer Fast entry point."""
    from vision_runtime.page_record import write_page_analysis_record

    source_sha = _sha256(source_path)
    return write_page_analysis_record(
        output_dir,
        output_dir / "consumer_fast" / "vision",
        output_pages[0],
        config,
        source_sha256=source_sha,
        project_id=f"consumer-fast-{source_sha[:24]}",
        capability_version="consumer-fast-v1",
    )


def _exact_line_plan(target_text: str, rendered_lines: list[str]):
    """Bind renderer-emitted lines to exact source substrings and whitespace."""
    from typesetter.recipe_contract import ExactLinePlan

    import re

    visual_lines = [str(line) for line in rendered_lines]
    if not target_text or not visual_lines:
        raise ValueError("RendererRecipe missing rendered_lines from typesetter output")
    source_words = list(re.finditer(r"\S+", target_text))
    if sum(len(line.split()) for line in visual_lines) != len(source_words):
        raise ValueError("RendererRecipe visual lines change target word count")
    lines = []
    separators: list[str] = []
    consumed = 0
    for index, visual_line in enumerate(visual_lines):
        count = len(visual_line.split())
        if count == 0:
            raise ValueError("RendererRecipe rendered line is empty")
        first = source_words[consumed]
        last = source_words[consumed + count - 1]
        source_line = target_text[first.start():last.end()]
        if source_line.casefold() != visual_line.casefold():
            raise ValueError("RendererRecipe visual line changes target text beyond casing")
        lines.append(source_line)
        consumed += count
        if index + 1 < len(visual_lines):
            next_word = source_words[consumed]
            separators.append(target_text[last.end():next_word.start()])
    if (source_words[0].start() != 0 or source_words[-1].end() != len(target_text)
            or any(not item or not item.isspace() for item in separators)):
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
                geometry={**dict(evidence["geometry"]), "visual_rendered_lines": list(evidence["rendered_lines"])},
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
        # A verified, byte-identical source page has no owner glyph rendering to bind.
        # Keep the strict recipe requirement for every changed or unverified page.
        source_preserved = all(
            getattr(result, "status", None) == "final_verified"
            and getattr(result, "terminal_proof", None) is not None
            and not (getattr(result, "page_commits", ()) or ())
            and getattr(getattr(getattr(result, "request", None), "original_page", None), "page_source_sha256", None)
            == getattr(getattr(result, "final_page", None), "page_output_pixel_sha256", None)
            for result in (getattr(page, "owner_page_result", None) for page in output_pages)
        )
        if source_preserved:
            return {
                "status": "not_applicable_source_preserved",
                "recipe_count": 0,
                "project_id": project_id,
                "project_revision": int(project_revision),
                "coverage_ledger_sha256": ledger_sha256,
                "reason": "no_committed_owner_glyph_patch; all final pages match source pixels",
                "page_diagnostics": no_patch_diagnostics,
            }
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
        "vision_stack.runtime", "vision_stack.ocr", "vision_runtime.analysis_payload",
        "vision_runtime.structure", "vision_runtime.cache_key", "vision_runtime.cache_reader",
        "vision_runtime.page_record", "vision_runtime.stage_cache", "consumer_fast.vision_cache",
        "ownership.coverage", "ownership.ocr_contract", "ownership.consensus_v2",
        "qa.partial_delivery", "integration_v1.contracts",
        "consumer_fast.physical_executor", "consumer_fast.provider_adapter",
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
    from consumer_fast.vision_cache import select_cached_pages, publish_cached_pages

    cache_root = work_dir / '.vision-cache'
    try:
        cache_selection = select_cached_pages(cache_root, source_manifest, config)
    except Exception as exc:
        cache_selection = {'status': 'miss', 'reason': f'cache_lookup_error:{type(exc).__name__}',
                           'pages': {}, 'lineage': None, 'visual_config_sha256': None}
    replay_of_execution_id = None
    if cache_selection['status'] == 'hit':
        run_id, replay_of_execution_id = cache_selection['lineage']
        source_manifest = ChapterSourceManifest.from_extracted_pages(
            image_files, extraction_root, run_id=run_id, execution_id=execution_id,
            replay_of_execution_id=replay_of_execution_id,
        )
    telemetry: dict[str, Any] = {'vision_cache': {
        'status': cache_selection['status'], 'reason': cache_selection['reason'],
    }}
    from main import _PipelineTiming
    performance_recorder = _PipelineTiming(work_dir=work_dir, run_id=run_id)
    telemetry["_performance_recorder"] = performance_recorder
    execution_artifact_root = work_dir / '.owner-private' / execution_id
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
        translation_context=(
            {
                "_translation_backend": "hy_mt2_gguf_local",
                "_hy_mt2_url": config.get("hy_mt2_url", "http://127.0.0.1:11438"),
                "_consumer_fast_bounded_hy": True,
            }
            if config.get("translation_backend") == "hy_mt2_gguf_local" else None
        ),
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
        replay_of_execution_id=replay_of_execution_id,
        source_manifest=source_manifest,
        artifact_root=execution_artifact_root,
        vision_stage_cache=cache_selection['pages'] if cache_selection['status'] == 'hit' else None,
        vision_config_sha256=cache_selection['visual_config_sha256'],
        progress_callback=lambda *_args, **_kwargs: None,
    )
    telemetry.pop("_performance_recorder", None)
    performance_recorder.finalize()
    telemetry["performance_timing_path"] = "performance_timing.json"
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

    try:
        telemetry['vision_cache']['publish'] = publish_cached_pages(
            cache_root, execution_artifact_root, output_pages, config, source_manifest,
        )
    except Exception as exc:
        telemetry['vision_cache']['publish'] = {
            'status': 'not_written', 'reason': f'cache_publish_error:{type(exc).__name__}',
        }

    from main import build_project_json, _observe_verified_owner_final_pages
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
    _bind_physical_page_artifacts(project, output_pages, original_targets, work_dir)
    _bind_verified_delivery_notices(project, output_pages)
    project["runtime_id"] = "consumer-fast-v1"
    project["runtime_profile"] = "consumer_fast"
    project["verified"] = False
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver
    try:
        observed_started = time.perf_counter()
        project.setdefault("qa", {})["final_pixel_reports"] = _observe_verified_owner_final_pages(
            project_data=project, output_pages=output_pages,
            observer=DetectorOcrFinalPixelObserver(detector=detector, runtime=runtime),
            source_language=config.get("idioma_origem", "en"),
        )
        telemetry["final_pixel_observer_seconds"] = round(time.perf_counter()-observed_started, 4)
    except Exception as exc:
        # The export gate still sees the missing report and blocks the page.
        project.setdefault("qa", {})["final_pixel_observer_error"] = f"{type(exc).__name__}: {exc}"
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
    recipe_count = renderer_recipe_receipt.get("recipe_count")
    source_preserved_without_patch = (
        committed_patch_count == 0
        and recipe_count == 0
        and renderer_recipe_receipt.get("status") == "not_applicable_source_preserved"
    )
    if not source_preserved_without_patch and (
        recipe_count != committed_patch_count or renderer_recipe_receipt.get("status") != "persisted"
    ):
        integrity_failures.append("renderer_recipe_missing_or_incomplete")
    if integrity_failures:
        gate = append_qa_integrity_failure(gate, integrity_failures)
    gate = _apply_translation_delivery_review(project, gate)
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
