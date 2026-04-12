"""
TraduzAi Pipeline - Entry point
Receives a config JSON path as argument, runs the full pipeline,
and outputs JSON progress messages to stdout for the Tauri sidecar to consume.
"""

import json
import shutil
import sys
import time
from pathlib import Path

from corpus.runtime import extract_expected_terms, load_corpus_bundle, merge_corpus_into_context
from extractor.extractor import cleanup, extract
from inpainter.lama import run_inpainting
from layout.balloon_layout import enrich_page_layout
from ocr.contextual_reviewer import contextual_review_page
from ocr.detector import run_ocr
from translator.context import fetch_context, merge_context
from translator.translate import translate_pages
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

    if sys.argv[1] == "--retypeset" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_retypeset(project_json_path, page_idx)
        return

    if sys.argv[1] == "--reinpaint-page" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_reinpaint(project_json_path, page_idx)
        return

    config_path = sys.argv[1]
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

    emit(
        "log",
        source_path=str(source_path),
        exists=source_path.exists(),
        suffix=source_path.suffix.lower(),
    )

    emit_progress("extract", 0, 0, message="Extraindo arquivos...")
    wait_if_paused(config)
    try:
        image_files, tmp_dir = extract(source_path, work_dir)
    except (FileNotFoundError, ValueError) as e:
        emit("error", message=str(e))
        sys.exit(1)

    emit_progress(
        "extract",
        100,
        5,
        total=len(image_files),
        message=f"{len(image_files)} pagina(s) encontrada(s)",
    )

    total_pages = len(image_files)

    for img_path in image_files:
        wait_if_paused(config)
        shutil.copy2(img_path, originals_dir / img_path.name)

    # Start AniList context fetch in background during OCR (hides network latency)
    _context_future = None
    context_base = merge_corpus_into_context(config.get("contexto", {}), corpus_bundle)
    if not context_base.get("sinopse") and config.get("obra"):
        from concurrent.futures import ThreadPoolExecutor as _CtxTPE
        _ctx_pool = _CtxTPE(max_workers=1, thread_name_prefix="context")
        _context_future = _ctx_pool.submit(fetch_context, config["obra"])

    emit_progress("ocr", 0, 10, total=total_pages, message="Iniciando OCR...")

    ocr_results = []
    ocr_history = []
    start_time = time.time()

    for i, img_path in enumerate(image_files):
        def emit_ocr_page_progress(local_progress: float, message: str):
            wait_if_paused(config)
            clamped = max(0.0, min(1.0, float(local_progress)))
            elapsed_now = time.time() - start_time
            completed_units = i + clamped
            eta_now = (
                (elapsed_now / completed_units) * (total_pages - completed_units)
                if completed_units > 0
                else 0
            )
            emit_progress(
                "ocr",
                (completed_units / total_pages) * 100,
                10 + (completed_units / total_pages) * 20,
                page=i + 1,
                total=total_pages,
                message=f"OCR pagina {i + 1}/{total_pages} - {message}",
                eta=eta_now,
            )

        emit_ocr_page_progress(0.02, "Preparando OCR")
        page_result = run_ocr(
            str(img_path),
            models_dir=str(models_dir),
            profile="max",
            vision_worker_path=config.get("vision_worker_path", ""),
            progress_callback=lambda stage, progress, message: emit_ocr_page_progress(
                min(0.92, max(0.02, float(progress) * 0.92)),
                message,
            ),
        )
        emit_ocr_page_progress(0.96, "Revisando coerencia textual")
        page_result = contextual_review_page(
            page_result,
            previous_pages=ocr_history,
            expected_terms=corpus_expected_terms,
        )
        emit_ocr_page_progress(0.99, "Ajustando layout dos baloes")
        page_result = enrich_page_layout(page_result)
        ocr_results.append(page_result)
        ocr_history.append(page_result)

        elapsed = time.time() - start_time
        pages_done = i + 1
        eta = (elapsed / pages_done) * (total_pages - pages_done) if pages_done > 0 else 0

        emit_progress(
            "ocr",
            (pages_done / total_pages) * 100,
            10 + (pages_done / total_pages) * 20,
            page=pages_done,
            total=total_pages,
            message=f"OCR pagina {pages_done}/{total_pages} - {len(page_result['texts'])} textos",
            eta=eta,
        )

    emit_progress("context", 0, 30, total=total_pages, message="Buscando contexto da obra...")

    wait_if_paused(config)
    context = context_base
    if _context_future is not None:
        try:
            fetched = _context_future.result(timeout=15)
            context = merge_context(context, fetched)
        except Exception as e:
            emit_progress("context", 50, 32, message=f"Aviso: contexto nao encontrado ({e})")
        finally:
            _ctx_pool.shutdown(wait=False)

    emit_progress("context", 100, 35, total=total_pages, message="Contexto carregado")

    emit_progress("translate", 0, 35, total=total_pages, message="Traducao + inpainting em paralelo...")

    # Translation is I/O-bound (HTTP calls); inpainting is CPU/GPU-bound.
    # Run them concurrently: translation in a thread, inpainting in main thread.
    from concurrent.futures import ThreadPoolExecutor as _TPE

    with _TPE(max_workers=1, thread_name_prefix="translate") as _translate_pool:
        def emit_translate_progress(page: int, total: int, msg: str):
            wait_if_paused(config)
            emit_progress(
                "translate",
                (page / total) * 100 if total else 0,
                35 + ((page / total) * 12.5 if total else 0),
                page=page,
                total=total,
                message=msg,
            )

        def emit_inpaint_progress(page: int, total: int, msg: str):
            wait_if_paused(config)
            emit_progress(
                "inpaint",
                (page / total) * 100,
                47.5 + (page / total) * 12.5,
                page=page,
                total=total,
                message=msg,
            )

        translate_future = _translate_pool.submit(
            translate_pages,
            ocr_results=ocr_results,
            obra=config["obra"],
            context=context,
            glossario=config.get("glossario", {}),
            idioma_destino=config.get("idioma_destino", "pt-BR"),
            qualidade="alta",
            ollama_host=config.get("ollama_host", "http://localhost:11434"),
            ollama_model=config.get("ollama_model", "traduzai-translator"),
            progress_callback=emit_translate_progress,
        )

        inpainted_paths = run_inpainting(
            image_files=image_files,
            ocr_results=ocr_results,
            output_dir=str(images_dir),
            models_dir=str(models_dir),
            corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
            progress_callback=emit_inpaint_progress,
        )

        translated_results = translate_future.result()

    emit_progress("inpaint", 100, 60, total=total_pages, message="Traducao e inpainting concluidos")

    emit_progress("typeset", 0, 80, total=total_pages, message="Aplicando texto traduzido...")

    wait_if_paused(config)

    def emit_typeset_progress(page: int, total: int, msg: str):
        wait_if_paused(config)
        emit_progress(
            "typeset",
            (page / total) * 100,
            80 + (page / total) * 18,
            page=page,
            total=total,
            message=msg,
        )

    merged_for_typeset = []
    for ocr_page, trans_page in zip(ocr_results, translated_results):
        merged_texts = []
        ocr_texts = ocr_page.get("texts", [])
        trans_texts = trans_page.get("texts", [])
        for idx, ocr_t in enumerate(ocr_texts):
            translated = (
                trans_texts[idx].get("translated", ocr_t.get("text", ""))
                if idx < len(trans_texts)
                else ocr_t.get("text", "")
            )
            merged_texts.append(
                {
                    **ocr_t,
                    "translated": translated,
                    "corpus_visual_benchmark": context.get("corpus_visual_benchmark", {}),
                    "corpus_textual_benchmark": context.get("corpus_textual_benchmark", {}),
                }
            )
        merged_for_typeset.append({"texts": merged_texts})

    run_typesetting(
        inpainted_paths=inpainted_paths,
        translated_results=merged_for_typeset,
        output_dir=str(translated_dir),
        progress_callback=emit_typeset_progress,
    )

    emit_progress("typeset", 100, 98, total=total_pages, message="Gerando project.json...")

    wait_if_paused(config)
    project_data = build_project_json(
        config=config,
        context=context,
        ocr_results=ocr_results,
        translated_results=translated_results,
        image_files=image_files,
        total_pages=total_pages,
        elapsed=time.time() - start_time,
    )

    project_json_path = work_dir / "project.json"
    with open(project_json_path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, ensure_ascii=False, indent=2)

    cleanup(tmp_dir)

    emit_progress("typeset", 100, 100, total=total_pages, message="Traducao concluida!")
    emit("complete", output_path=str(work_dir))


def _run_retypeset(project_json_path: Path, page_idx: int):
    with open(project_json_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_json_path.parent
    pages = project.get("paginas", [])
    if page_idx < 0 or page_idx >= len(pages):
        emit("error", message="Indice de pagina invalido")
        return

    page = pages[page_idx]

    img_name = Path(page["arquivo_original"]).name
    inpainted_path = work_dir / "images" / img_name
    output_dir = work_dir / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inpainted_path.exists():
        inpainted_path = work_dir / "originals" / img_name

    if not inpainted_path.exists():
        emit("error", message=f"Imagem base nao encontrada: {img_name}")
        return

    trans_texts = []
    for t in page.get("textos", []):
        trans_texts.append({
            "bbox": t.get("bbox", [0, 0, 0, 0]),
            "tipo": t.get("tipo", "fala"),
            "translated": t.get("traduzido", ""),
            "text": t.get("original", ""),
            "estilo": t.get("estilo", {}),
            "layout_group_size": 1,
        })

    trans_page_dict = {"texts": trans_texts}

    try:
        from typesetter.renderer import _typeset_single_page
        _typeset_single_page((str(inpainted_path), trans_page_dict, str(output_dir)))
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
    img_name = Path(page["arquivo_original"]).name
    original_path = work_dir / "originals" / img_name
    output_dir = work_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        candidate = Path(page["arquivo_original"])
        if candidate.exists():
            original_path = candidate

    if not original_path.exists():
        emit("error", message=f"Imagem original nao encontrada: {img_name}")
        return

    page_texts = []
    for t in page.get("textos", []):
        page_texts.append(
            {
                "bbox": t.get("bbox", [0, 0, 0, 0]),
                "tipo": t.get("tipo", "fala"),
                "text": t.get("original", ""),
                "translated": t.get("traduzido", ""),
                "estilo": t.get("estilo", {}),
            }
        )

    inpaint_blocks = page.get("inpaint_blocks") or [
        {
            "bbox": t.get("bbox", [0, 0, 0, 0]),
            "confidence": float(t.get("confianca_ocr", 0.0)),
        }
        for t in page.get("textos", [])
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
        emit("complete", output_path=str(outputs[0]) if outputs else str(output_dir / img_name))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no reinpaint: {e}\n{tb}")


def build_project_json(config, context, ocr_results, translated_results, image_files, total_pages, elapsed):
    """Build the project.json structure."""
    pages = []
    for i, (img, ocr, trans) in enumerate(zip(image_files, ocr_results, translated_results)):
        page_texts = []
        for t_idx, (ocr_text, trans_text) in enumerate(zip(ocr.get("texts", []), trans.get("texts", []))):
            page_texts.append(
                {
                    "id": f"t{i + 1}_{t_idx + 1}",
                    "bbox": ocr_text.get("bbox", [0, 0, 0, 0]),
                    "tipo": ocr_text.get("tipo", "fala"),
                    "original": ocr_text.get("text", ""),
                    "traduzido": trans_text.get("translated", ""),
                    "confianca_ocr": ocr_text.get("confidence", 0),
                    "estilo": ocr_text.get(
                        "estilo",
                        {
                            "fonte": "AnimeAce",
                            "tamanho": 16,
                            "cor": "#FFFFFF",
                            "contorno": "#000000",
                            "contorno_px": 2,
                            "bold": False,
                            "italico": False,
                            "rotacao": 0,
                            "alinhamento": "center",
                        },
                    ),
                }
            )

        pages.append(
            {
                "numero": i + 1,
                "arquivo_original": f"originals/{img.name}",
                "arquivo_traduzido": f"translated/{img.name}",
                "inpaint_blocks": [
                    {
                        "bbox": block.get("bbox", [0, 0, 0, 0]),
                        "confidence": block.get("confidence", 0.0),
                    }
                    for block in ocr.get("_vision_blocks", [])
                ],
                "textos": page_texts,
            }
        )

    return {
        "versao": "1.0",
        "app": "traduzai",
        "obra": config.get("obra", ""),
        "capitulo": config.get("capitulo", 1),
        "idioma_origem": config.get("idioma_origem", "en"),
        "idioma_destino": config.get("idioma_destino", "pt-BR"),
        "contexto": context,
        "paginas": pages,
        "estatisticas": {
            "total_paginas": total_pages,
            "total_textos": sum(len(p["textos"]) for p in pages),
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
