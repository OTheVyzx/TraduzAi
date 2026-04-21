"""
TraduzAi Pipeline - Entry point
Receives a config JSON path as argument, runs the full pipeline,
and outputs JSON progress messages to stdout for the Tauri sidecar to consume.
"""

import os
import json
import shutil
import sys
import time
import faulthandler
from pathlib import Path

# Adiciona o diretório da pipeline ao path para resolver imports locais no Pyright/Linter
pipeline_root = Path(__file__).parent.absolute()
if str(pipeline_root) not in sys.path:
    sys.path.insert(0, str(pipeline_root))

from corpus.runtime import extract_expected_terms, load_corpus_bundle, merge_corpus_into_context
from extractor.extractor import cleanup, extract
from inpainter.lama import run_inpainting
from layout.balloon_layout import enrich_page_layout
from ocr.contextual_reviewer import contextual_review_page
from ocr.detector import run_ocr
from translator.context import fetch_context, merge_context
from translator.translate import list_supported_google_languages, translate_pages
from typesetter.renderer import run_typesetting
from vision_stack.runtime import warmup_visual_stack

_EMIT_STDOUT_FAILED = False


def _report_emit_failure(exc: OSError) -> None:
    global _EMIT_STDOUT_FAILED
    if _EMIT_STDOUT_FAILED:
        return

    _EMIT_STDOUT_FAILED = True
    try:
        sys.stderr.write(f"Falha ao emitir evento JSON no stdout: {exc}\n")
        sys.stderr.flush()
    except OSError:
        pass


def emit(msg_type: str, **kwargs):
    """Emit a JSON message to stdout (consumed by Rust sidecar reader)."""
    payload = {"type": msg_type, **kwargs}
    try:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    except OSError as exc:
        _report_emit_failure(exc)


def emit_progress(
    step: str,
    step_progress: float,
    overall: float,
    page: int = 0,
    total: int = 0,
    message: str = "",
    eta: float = 0,
):
    emit(
        "progress",
        step=step,
        step_progress=step_progress,
        overall_progress=overall,
        current_page=page,
        total_pages=total,
        message=message,
        eta_seconds=eta,
    )


def wait_if_paused(config: dict):
    """Block cooperatively while the Tauri pause marker exists."""
    pause_file = config.get("pause_file")
    if not pause_file:
        return

    pause_path = Path(pause_file)
    while pause_path.exists():
        time.sleep(0.25)


def main():
    faulthandler.enable()
    if len(sys.argv) < 2:
        emit("error", message="Nenhum arquivo de configuracao fornecido")
        sys.exit(1)

    if sys.argv[1] == "--warmup-visual":
        models_dir = ""
        profile = "normal"
        args = sys.argv[2:]
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--models-dir" and index + 1 < len(args):
                models_dir = args[index + 1]
                index += 2
                continue
            if arg == "--profile" and index + 1 < len(args):
                profile = args[index + 1]
                index += 2
                continue
            index += 1

        warmup_visual_stack(models_dir=models_dir, profile=profile)
        emit("complete", output_path="")
        return

    if sys.argv[1] == "--list-supported-languages":
        print(json.dumps(list_supported_google_languages(), ensure_ascii=False), flush=True)
        return

    if sys.argv[1] == "--retypeset" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_retypeset(project_json_path, page_idx)
        return

    if (sys.argv[1] == "--process-block") and len(sys.argv) >= 6:
        mode = sys.argv[2]  # "ocr" or "translate"
        project_json_path = Path(sys.argv[3])
        page_idx = int(sys.argv[4])
        block_id = sys.argv[5]
        _run_process_block(project_json_path, page_idx, block_id, mode)
        return

    if sys.argv[1] == "--hardware-info":
        from utils.hardware import get_hardware_facts
        print(json.dumps(get_hardware_facts(), ensure_ascii=False), flush=True)
        return

    config_path = sys.argv[1]
    _log_env_info()
    try:
        _run_pipeline(config_path)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        # Garantir que o erro vá para o stdout (Tauri) e stderr (Log)
        err_msg = f"FALHA CATASTROFICA NO PIPELINE: {exc}\n{tb}"
        emit("error", message=err_msg)
        sys.stderr.write(f"\n--- CRASH DUMP ---\n{err_msg}\n------------------\n")
        sys.stderr.flush()
        sys.exit(1)

def _log_env_info():
    try:
        import platform
        import cv2
        import numpy as np
        import matplotlib
        import PIL
        info = (
            f"OS: {platform.platform()} | Python: {sys.version}\n"
            f"CWD: {os.getcwd()}\n"
            f"CV2: {cv2.__version__} | NumPy: {np.__version__} | "
            f"Matplotlib: {matplotlib.__version__} | PIL: {PIL.__version__}\n"
            f"Backend Matplotlib: {matplotlib.get_backend()}"
        )
        sys.stderr.write(f"\n--- AMBIENTE DE EXECUCAO ---\n{info}\n----------------------------\n")
        sys.stderr.flush()
    except Exception:
        pass

def _run_pipeline(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    wait_if_paused(config)
    work_dir = Path(config["work_dir"])
    models_dir = Path(config["models_dir"])

    raw_source = config["source_path"].strip()
    if raw_source.startswith("file:///"):
        raw_source = raw_source[8:]
    elif raw_source.startswith("file://"):
        raw_source = raw_source[7:]
    source_path = Path(raw_source)

    images_dir = work_dir / "images"
    originals_dir = work_dir / "originals"
    translated_dir = work_dir / "translated"
    images_dir.mkdir(parents=True, exist_ok=True)
    originals_dir.mkdir(parents=True, exist_ok=True)
    translated_dir.mkdir(parents=True, exist_ok=True)

    corpus_bundle = load_corpus_bundle(
        config.get("obra", ""),
        models_root=models_dir / "corpus",
        fallback_root=Path(__file__).resolve().parent / "models" / "corpus",
    )
    corpus_expected_terms = extract_expected_terms(corpus_bundle)

    emit_progress("extract", 0, 0, message="Extraindo arquivos...")
    try:
        image_files, tmp_dir = extract(source_path, work_dir)
    except Exception as e:
        emit("error", message=str(e))
        sys.exit(1)

    for img_path in image_files:
        shutil.copy2(img_path, originals_dir / img_path.name)

    total_pages = len(image_files)
    mode = config.get("mode", "auto")
    ocr_results = []
    page_text_layers = []
    context = merge_corpus_into_context(config.get("contexto", {}), corpus_bundle)
    start_time = time.time()

    if mode == "manual":
        emit_progress("extract", 100, 95, message="Modo Manual: Preparando projeto...")
        for img_path in image_files:
            # Em modo manual, o 'clean' (images) e o final (translated) sao apenas copias
            shutil.copy2(originals_dir / img_path.name, images_dir / img_path.name)
            shutil.copy2(originals_dir / img_path.name, translated_dir / img_path.name)
        inpainted_paths = [str(images_dir / f.name) for f in image_files]
        # Prepare empty lists for project building
        for _ in range(total_pages):
            ocr_results.append({"texts": [], "_vision_blocks": []})
            page_text_layers.append({"texts": []})
    else:
        # AUTOMATIC FLOW
        # Start AniList context fetch
        _context_future = None
        if not context.get("sinopse") and config.get("obra"):
            from concurrent.futures import ThreadPoolExecutor as _CtxTPE
            _ctx_pool = _CtxTPE(max_workers=1)
            _context_future = _ctx_pool.submit(fetch_context, config["obra"])

        emit_progress("ocr", 0, 10, total=total_pages, message="Iniciando OCR...")
        ocr_history = []

        for i, img_path in enumerate(image_files):
            def emit_ocr_page_progress(local_progress: float, message: str):
                wait_if_paused(config)
                prog = i + max(0.0, min(1.0, float(local_progress)))
                emit_progress("ocr", (prog/total_pages)*100, 10+(prog/total_pages)*20, page=i+1, total=total_pages, message=f"P{i+1}: {message}")

            page_result = run_ocr(
                str(img_path), models_dir=str(models_dir), profile="max",
                vision_worker_path=config.get("vision_worker_path", ""),
                progress_callback=lambda s, p, m: emit_ocr_page_progress(p, m),
                idioma_origem=config.get("idioma_origem", "en"),
            )
            page_result = contextual_review_page(page_result, ocr_history, corpus_expected_terms)
            page_result["_connected_balloon_reasoner"] = {
                "provider": config.get("connected_balloon_reasoner", "ollama"),
                "enabled": config.get("connected_balloon_reasoner_enabled", True),
                "host": config.get("connected_balloon_ollama_host", config.get("ollama_host", "http://localhost:11434")),
                "model": config.get("connected_balloon_ollama_model", "qwen2.5"),
                "use_image": config.get("connected_balloon_ollama_use_image", True),
            }
            page_result = enrich_page_layout(page_result)
            ocr_results.append(page_result)
            ocr_history.append(page_result)

        if _context_future:
            try:
                context = merge_context(context, _context_future.result(timeout=10))
            except: pass
            _ctx_pool.shutdown(wait=False)

        # Translation + Inpainting
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=1) as pool:
            def emit_tr_prog(p, t, m): emit_progress("translate", (p/t)*100, 35+(p/t)*12.5, page=p, total=t, message=m)
            tr_future = pool.submit(translate_pages, ocr_results, config["obra"], context, config.get("glossario",{}), config.get("idioma_destino","pt-BR"), config.get("idioma_origem","en"), "alta", config.get("ollama_host","http://localhost:11434"), config.get("ollama_model","traduzai-translator"), emit_tr_prog)
            
            def emit_in_prog(p, t, m): emit_progress("inpaint", (p/t)*100, 47.5+(p/t)*12.5, page=p, total=t, message=m)
            inpainted_paths = run_inpainting(image_files, ocr_results, str(images_dir), str(models_dir), context.get("corpus_visual_benchmark",{}), emit_in_prog)
            translated_results = tr_future.result()

        # Build Layers
        for page_index, (ocr_page, trans_page) in enumerate(zip(ocr_results, translated_results)):
            merged = []
            for idx, ocr_t in enumerate(ocr_page.get("texts", [])):
                tr = trans_page.get("texts", [])[idx].get("translated", "") if idx < len(trans_page.get("texts", [])) else ""
                merged.append(build_text_layer(page_number=page_index+1, layer_index=idx, ocr_text=ocr_t, translated=tr, corpus_visual_benchmark=context.get("corpus_visual_benchmark",{}), corpus_textual_benchmark=context.get("corpus_textual_benchmark",{})))
            page_text_layers.append({"texts": merged})

        def emit_ty_prog(p, t, m): emit_progress("typeset", (p/t)*100, 80+(p/t)*18, page=p, total=t, message=m)
        run_typesetting(inpainted_paths, page_text_layers, str(translated_dir), emit_ty_prog)



    # Wrap up
    emit_progress("typeset", 100, 98, message="Finalizando projeto...")
    project_data = build_project_json(config, context, ocr_results, page_text_layers, image_files, total_pages, time.time()-start_time)
    with open(work_dir/"project.json", "w", encoding="utf-8") as f:
        json.dump(project_data, f, ensure_ascii=False, indent=2)
    
    cleanup(tmp_dir)
    emit_progress("typeset", 100, 100, message="Concluido!")
    emit("complete", output_path=str(work_dir))


def _default_text_style() -> dict:
    return {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 28,
        "cor": "#FFFFFF",
        "cor_gradiente": [],
        "contorno": "#000000",
        "contorno_px": 2,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": False,
        "sombra_cor": "",
        "sombra_offset": [0, 0],
        "bold": False,
        "italico": False,
        "rotacao": 0,
        "alinhamento": "center",
        "force_upper": False,
    }


def _merge_style(style: dict | None) -> dict:
    merged = _default_text_style()
    if isinstance(style, dict):
        merged.update(style)
    return merged


def _bbox4(value, fallback=None) -> list[int]:
    source = value if isinstance(value, (list, tuple)) and len(value) >= 4 else fallback
    if not isinstance(source, (list, tuple)) or len(source) < 4:
        source = [0, 0, 32, 32]
    return [int(source[0]), int(source[1]), int(source[2]), int(source[3])]


def _preview_rel_path(page_number: int, layer_id: str) -> str:
    return f"layers/text-preview/{page_number:03}/{layer_id}.png"


def _bbox4_list(values) -> list[list[int]]:
    return [
        _bbox4(value)
        for value in (values or [])
        if isinstance(value, (list, tuple)) and len(value) >= 4
    ]


def build_text_layer(
    *,
    page_number: int,
    layer_index: int,
    ocr_text: dict,
    translated: str,
    corpus_visual_benchmark: dict,
    corpus_textual_benchmark: dict,
) -> dict:
    layer_id = ocr_text.get("id") or f"tl_{page_number:03}_{layer_index + 1:03}"
    source_bbox = _bbox4(ocr_text.get("bbox"))
    layout_bbox = _bbox4(ocr_text.get("balloon_bbox"), source_bbox)
    style = _merge_style(ocr_text.get("estilo"))
    balloon_subregions = _bbox4_list(ocr_text.get("balloon_subregions"))
    connected_lobe_bboxes = _bbox4_list(ocr_text.get("connected_lobe_bboxes"))
    connected_text_groups = _bbox4_list(ocr_text.get("connected_text_groups"))
    connected_position_bboxes = _bbox4_list(ocr_text.get("connected_position_bboxes"))
    connected_focus_bboxes = _bbox4_list(ocr_text.get("connected_focus_bboxes"))

    return {
        "id": layer_id,
        "kind": "text",
        "source_bbox": source_bbox,
        "layout_bbox": layout_bbox,
        "render_bbox": None,
        "bbox": layout_bbox,
        "tipo": ocr_text.get("tipo", "fala"),
        "original": ocr_text.get("text", ""),
        "translated": translated,
        "text": ocr_text.get("text", ""),
        "ocr_confidence": float(ocr_text.get("confidence", 0.0) or 0.0),
        "style": style,
        "estilo": style,
        "visible": True,
        "locked": False,
        "order": layer_index,
        "render_preview_path": _preview_rel_path(page_number, layer_id),
        "detector": ocr_text.get("detector"),
        "line_polygons": ocr_text.get("line_polygons"),
        "source_direction": ocr_text.get("source_direction"),
        "rendered_direction": ocr_text.get("rendered_direction"),
        "source_language": ocr_text.get("source_language"),
        "rotation_deg": float(ocr_text.get("rotation_deg", 0) or 0),
        "detected_font_size_px": ocr_text.get("detected_font_size_px"),
        "balloon_bbox": layout_bbox,
        "balloon_subregions": balloon_subregions,
        "layout_group_size": int(ocr_text.get("layout_group_size", 1) or 1),
        "connected_children": ocr_text.get("connected_children"),
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": ocr_text.get("connected_balloon_orientation"),
        "connected_detection_confidence": float(ocr_text.get("connected_detection_confidence", 0.0) or 0.0),
        "connected_group_confidence": float(ocr_text.get("connected_group_confidence", 0.0) or 0.0),
        "connected_position_confidence": float(ocr_text.get("connected_position_confidence", 0.0) or 0.0),
        "subregion_confidence": float(ocr_text.get("subregion_confidence", 0.0) or 0.0),
        "connected_position_reasoner": ocr_text.get("connected_position_reasoner"),
        "connected_reasoner_model": ocr_text.get("connected_reasoner_model"),
        "connected_reasoner_notes": ocr_text.get("connected_reasoner_notes"),
        "_connected_slot_index": ocr_text.get("_connected_slot_index"),
        "_connected_slot_count": ocr_text.get("_connected_slot_count"),
        "_connected_vertical_bias_ratio": ocr_text.get("_connected_vertical_bias_ratio"),
        "_is_lobe_subregion": bool(ocr_text.get("_is_lobe_subregion", False)),
        "corpus_visual_benchmark": corpus_visual_benchmark,
        "corpus_textual_benchmark": corpus_textual_benchmark,
    }


def _normalize_text_layer_for_renderer(raw_layer: dict, page_number: int, layer_index: int) -> dict:
    source_bbox = _bbox4(raw_layer.get("source_bbox"), raw_layer.get("bbox"))
    layout_bbox = _bbox4(raw_layer.get("layout_bbox"), raw_layer.get("balloon_bbox") or source_bbox)
    style = _merge_style(raw_layer.get("style") or raw_layer.get("estilo"))
    layer_id = raw_layer.get("id") or f"tl_{page_number:03}_{layer_index + 1:03}"
    translated = raw_layer.get("translated", raw_layer.get("traduzido", ""))
    original = raw_layer.get("original", raw_layer.get("text", ""))
    balloon_subregions = _bbox4_list(raw_layer.get("balloon_subregions"))
    connected_lobe_bboxes = _bbox4_list(raw_layer.get("connected_lobe_bboxes"))
    connected_text_groups = _bbox4_list(raw_layer.get("connected_text_groups"))
    connected_position_bboxes = _bbox4_list(raw_layer.get("connected_position_bboxes"))
    connected_focus_bboxes = _bbox4_list(raw_layer.get("connected_focus_bboxes"))

    return {
        "id": layer_id,
        "kind": "text",
        "source_bbox": source_bbox,
        "layout_bbox": layout_bbox,
        "render_bbox": raw_layer.get("render_bbox"),
        "bbox": layout_bbox,
        "tipo": raw_layer.get("tipo", "fala"),
        "original": original,
        "translated": translated,
        "text": original,
        "ocr_confidence": float(raw_layer.get("ocr_confidence", raw_layer.get("confianca_ocr", 0.0)) or 0.0),
        "style": style,
        "estilo": style,
        "visible": bool(raw_layer.get("visible", True)),
        "locked": bool(raw_layer.get("locked", False)),
        "order": int(raw_layer.get("order", layer_index) or layer_index),
        "render_preview_path": raw_layer.get("render_preview_path") or _preview_rel_path(page_number, layer_id),
        "detector": raw_layer.get("detector"),
        "line_polygons": raw_layer.get("line_polygons"),
        "source_direction": raw_layer.get("source_direction"),
        "rendered_direction": raw_layer.get("rendered_direction"),
        "source_language": raw_layer.get("source_language"),
        "rotation_deg": float(raw_layer.get("rotation_deg", 0) or 0),
        "detected_font_size_px": raw_layer.get("detected_font_size_px"),
        "balloon_bbox": _bbox4(raw_layer.get("balloon_bbox"), layout_bbox),
        "balloon_subregions": balloon_subregions,
        "layout_group_size": int(raw_layer.get("layout_group_size", 1) or 1),
        "connected_children": raw_layer.get("connected_children"),
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": raw_layer.get("connected_balloon_orientation"),
        "connected_detection_confidence": float(raw_layer.get("connected_detection_confidence", 0.0) or 0.0),
        "connected_group_confidence": float(raw_layer.get("connected_group_confidence", 0.0) or 0.0),
        "connected_position_confidence": float(raw_layer.get("connected_position_confidence", 0.0) or 0.0),
        "subregion_confidence": float(raw_layer.get("subregion_confidence", 0.0) or 0.0),
        "connected_position_reasoner": raw_layer.get("connected_position_reasoner"),
        "connected_reasoner_model": raw_layer.get("connected_reasoner_model"),
        "connected_reasoner_notes": raw_layer.get("connected_reasoner_notes"),
        "_connected_slot_index": raw_layer.get("_connected_slot_index"),
        "_connected_slot_count": raw_layer.get("_connected_slot_count"),
        "_connected_vertical_bias_ratio": raw_layer.get("_connected_vertical_bias_ratio"),
        "_is_lobe_subregion": bool(raw_layer.get("_is_lobe_subregion", False)),
        "corpus_visual_benchmark": raw_layer.get("corpus_visual_benchmark", {}),
        "corpus_textual_benchmark": raw_layer.get("corpus_textual_benchmark", {}),
    }


def _page_text_layers_for_renderer(page: dict, page_index: int) -> list[dict]:
    page_number = int(page.get("numero", page_index + 1) or page_index + 1)
    raw_layers = page.get("text_layers")
    if isinstance(raw_layers, list) and raw_layers:
        return [
            _normalize_text_layer_for_renderer(layer, page_number, idx)
            for idx, layer in enumerate(raw_layers)
            if isinstance(layer, dict)
        ]

    return [
        _normalize_text_layer_for_renderer(layer, page_number, idx)
        for idx, layer in enumerate(page.get("textos", []))
        if isinstance(layer, dict)
    ]


def _sync_page_legacy_aliases(page: dict) -> None:
    image_layers = page.setdefault("image_layers", {})
    page["arquivo_original"] = ((image_layers.get("base") or {}).get("path")) or page.get("arquivo_original")
    page["arquivo_traduzido"] = ((image_layers.get("rendered") or {}).get("path")) or page.get("arquivo_traduzido")

    text_layers = page.get("text_layers") or []
    page["textos"] = [
        {
            "id": layer.get("id"),
            "bbox": _bbox4(layer.get("layout_bbox"), layer.get("source_bbox")),
            "tipo": layer.get("tipo", "fala"),
            "original": layer.get("original", ""),
            "traduzido": layer.get("translated", ""),
            "confianca_ocr": float(layer.get("ocr_confidence", 0.0) or 0.0),
            "estilo": _merge_style(layer.get("style") or layer.get("estilo")),
        }
        for layer in text_layers
        if isinstance(layer, dict)
    ]


def _save_project_json(project_json_path: Path, project: dict) -> None:
    with open(project_json_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)


def _resolve_image_layer_path(page: dict, layer_key: str, fallback: str) -> str:
    image_layers = page.get("image_layers") or {}
    layer = image_layers.get(layer_key) or {}
    return layer.get("path") or fallback


def _ensure_image_layer(page: dict, layer_key: str, path: str, *, visible: bool, locked: bool) -> None:
    image_layers = page.setdefault("image_layers", {})
    layer = image_layers.setdefault(layer_key, {})
    layer["key"] = layer_key
    layer["path"] = path
    layer["visible"] = bool(layer.get("visible", visible))
    layer["locked"] = bool(layer.get("locked", locked))


def _run_retypeset(project_json_path: Path, page_idx: int):
    with open(project_json_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_json_path.parent
    pages = project.get("paginas", [])
    if page_idx < 0 or page_idx >= len(pages):
        emit("error", message="Indice de pagina invalido")
        return

    page = pages[page_idx]
    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    inpaint_rel = _resolve_image_layer_path(page, "inpaint", f"images/{Path(original_rel).name}")
    rendered_rel = _resolve_image_layer_path(page, "rendered", f"translated/{Path(original_rel).name}")

    img_name = Path(original_rel).name
    inpainted_path = work_dir / inpaint_rel
    output_dir = work_dir / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inpainted_path.exists():
        inpainted_path = work_dir / "originals" / img_name

    if not inpainted_path.exists():
        emit("error", message=f"Imagem base nao encontrada: {img_name}")
        return

    trans_texts = _page_text_layers_for_renderer(page, page_idx)
    trans_page_dict = {"texts": trans_texts}

    try:
        from typesetter.renderer import _typeset_single_page
        _typeset_single_page((str(inpainted_path), trans_page_dict, str(output_dir)))
        page["text_layers"] = trans_texts
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=False, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "rendered", rendered_rel, visible=True, locked=True)
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        emit("complete", output_path=str(output_dir / img_name))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no retypeset: {e}\n{tb}")


def _run_reinpaint(project_json_path: Path, page_idx: int):
    with open(project_json_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_json_path.parent
    pages = project.get("paginas", [])
    if page_idx < 0 or page_idx >= len(pages):
        emit("error", message="Indice de pagina invalido")
        return

    page = pages[page_idx]
    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    inpaint_rel = _resolve_image_layer_path(page, "inpaint", f"images/{Path(original_rel).name}")
    img_name = Path(original_rel).name
    original_path = work_dir / original_rel
    if not original_path.exists():
        original_path = work_dir / "originals" / img_name
    output_dir = (work_dir / inpaint_rel).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            original_path = candidate

    if not original_path.exists():
        emit("error", message=f"Imagem original nao encontrada: {img_name}")
        return

    page_texts = _page_text_layers_for_renderer(page, page_idx)

    inpaint_blocks = page.get("inpaint_blocks") or [
        {
            "bbox": t.get("source_bbox", t.get("layout_bbox", [0, 0, 0, 0])),
            "confidence": float(t.get("ocr_confidence", t.get("confianca_ocr", 0.0)) or 0.0),
        }
        for t in page_texts
    ]

    try:
        ocr_data = {
            "image": str(original_path),
            "width": 0,
            "height": 0,
            "texts": page_texts,
            "_vision_blocks": inpaint_blocks,
        }
        outputs = run_inpainting(
            image_files=[original_path],
            ocr_results=[ocr_data],
            output_dir=str(output_dir),
            models_dir=str(Path("D:/traduzai_data/models")),
        )
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=False, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        if isinstance(page.get("text_layers"), list):
            page["text_layers"] = page_texts
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        emit("complete", output_path=str(outputs[0]) if outputs else str(output_dir / img_name))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no reinpaint: {e}\n{tb}")


def build_project_json(config, context, ocr_results, page_text_layers, image_files, total_pages, elapsed):
    """Build the project.json structure."""
    pages = []
    for i, (img, ocr, text_page) in enumerate(zip(image_files, ocr_results, page_text_layers)):
        text_layers = text_page.get("texts", [])
        page = {
            "numero": i + 1,
            "image_layers": {
                "base": {
                    "key": "base",
                    "path": f"originals/{img.name}",
                    "visible": True,
                    "locked": True,
                },
                "mask": {
                    "key": "mask",
                    "path": f"layers/mask/{i + 1:03}.png",
                    "visible": False,
                    "locked": False,
                },
                "inpaint": {
                    "key": "inpaint",
                    "path": f"images/{img.name}",
                    "visible": False,
                    "locked": True,
                },
                "brush": {
                    "key": "brush",
                    "path": f"layers/brush/{i + 1:03}.png",
                    "visible": False,
                    "locked": False,
                },
                "rendered": {
                    "key": "rendered",
                    "path": f"translated/{img.name}",
                    "visible": True,
                    "locked": True,
                },
            },
            "inpaint_blocks": [
                {
                    "bbox": block.get("bbox", [0, 0, 0, 0]),
                    "confidence": block.get("confidence", 0.0),
                }
                for block in ocr.get("_vision_blocks", [])
            ],
            "text_layers": text_layers,
        }
        _sync_page_legacy_aliases(page)
        pages.append(page)

    return {
        "versao": "2.0",
        "app": "traduzai",
        "obra": config.get("obra", ""),
        "capitulo": config.get("capitulo", 1),
        "idioma_origem": config.get("idioma_origem", "en"),
        "idioma_destino": config.get("idioma_destino", "pt-BR"),
        "contexto": context,
        "paginas": pages,
        "estatisticas": {
            "total_paginas": total_pages,
            "total_textos": sum(len(p["text_layers"]) for p in pages),
            "tempo_processamento_seg": round(elapsed, 1),
            "data_criacao": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        emit("error", message=f"{e}\n--- traceback ---\n{tb}")
        sys.exit(1)
def _run_process_block(project_path: Path, page_idx: int, block_id: str, mode: str):
    """Refazer processo para um unico bloco."""
    from ocr.detector import run_ocr_on_block
    from translator.translate import translate_single_block
    from typesetter.renderer import render_page_image
    
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)
        
    page = project["paginas"][page_idx]
    
    found_block = None
    # Support both "text_layers" and "textos" keys
    layers = page.get("text_layers", page.get("textos", []))
    for layer in layers:
        if layer["id"] == block_id:
            found_block = layer
            break
            
    if not found_block:
        emit("error", message=f"Bloco {block_id} nao encontrado na pagina {page_idx}")
        return
        
    if mode == "ocr":
        emit_progress("ocr", 10, 50, message="Redetectando texto...")
        # Get original image path
        orig_img = project_path.parent / page["arquivo_original"]
        text, conf = run_ocr_on_block(str(orig_img), found_block["bbox"])
        found_block["original"] = text
        found_block["ocr_confidence"] = conf
        found_block["confianca_ocr"] = conf
    
    if mode == "translate":
        emit_progress("translate", 50, 80, message="Traduzindo bloco...")
        translate_single_block(found_block, project)

    if mode == "inpaint" or mode == "ocr":
        emit_progress("inpaint", 30, 70, message="Limpando balão...")
        # Run local inpaint for this block
        orig_img = project_path.parent / page["arquivo_original"]
        out_img = project_path.parent / page["arquivo_traduzido"]
        
        # We use a temporary run_inpainting call for the single page
        from inpainter.lama import run_inpainting
        ocr_data = {
            "image": str(orig_img),
            "texts": [found_block],
            "_vision_blocks": [{"bbox": found_block["bbox"], "confidence": 1.0}]
        }
        # Render into images/ (inpainted version)
        inpaint_out_dir = project_path.parent / "images"
        inpaint_out_dir.mkdir(parents=True, exist_ok=True)
        run_inpainting([orig_img], [ocr_data], str(inpaint_out_dir))
        
    # Save project
    with open(project_path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)
        
    # Always re-render the page image to reflect changes
    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = project_path.parent / page["arquivo_traduzido"]
    render_page_image(project, page_idx, str(out_img))
    
    emit("complete", output_path=str(out_img))
