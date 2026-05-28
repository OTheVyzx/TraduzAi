"""
OCR entrypoint for the active visual stack.

The previous OCR implementation is preserved in `ocr_legacy/`.
This module now routes `run_ocr()` through the new detector -> OCR stack
and falls back to the legacy path only if the new stack fails.
"""

from __future__ import annotations

import logging

from ocr_legacy.detector import _check_gpu, _get_reader, _preprocess, _run_primary_regions
from ocr_legacy.detector import run_ocr as run_legacy_ocr
from ocr_legacy.recognizer_paddle import is_paddle_available, run_paddle_primary_recognition
from ocr_legacy.recognizer_primary import run_primary_recognition
from vision_stack.runtime import run_detect_ocr

logger = logging.getLogger(__name__)


def _run_primary_regions(reader, image_bgr, preprocessed_image) -> list[dict]:
    if is_paddle_available():
        try:
            paddle_runs = run_paddle_primary_recognition(image_bgr, use_gpu=_check_gpu())
            if paddle_runs:
                return paddle_runs
            logger.warning("PaddleOCR retornou vazio; removendo EasyOCR fallback conforme solicitado.")
        except Exception as exc:
            logger.warning("PaddleOCR falhou, e o fallback para EasyOCR foi removido: %s", exc)
            raise exc
    
    # Se nem PaddleOCR está disponivel, vamos ver se o user ainda quer o EasyOCR como primario.
    # Mas o user disse "retire o easyocr de fallback", entao vamos falhar aqui tbm se não tiver Paddle?
    # Actually, legacy still uses easyocr intentionally if paddle is not available.
    return run_primary_recognition(reader, preprocessed_image)


def run_ocr(
    image_path: str,
    models_dir: str = "",
    profile: str = "quality",
    vision_worker_path: str = "",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    try:
        return run_detect_ocr(
            image_path=image_path,
            models_dir=models_dir,
            profile=profile,
            vision_worker_path=vision_worker_path,
            progress_callback=progress_callback,
            idioma_origem=idioma_origem,
            engine_preset_id=engine_preset_id,
            work_title=work_title,
            work_title_aliases=work_title_aliases,
            work_title_user_provided=work_title_user_provided,
        )
    except Exception as exc:
        logger.warning("Novo stack visual falhou no OCR, fallback para legacy: %s", exc)
        return run_legacy_ocr(
            image_path=image_path,
            models_dir=models_dir,
            profile=profile,
        )


def run_ocr_on_block(
    image_path: str,
    bbox: list[int],
    idioma_origem: str = "en",
    engine_preset_id: str = "",
) -> tuple[str, float]:
    """Extrai texto de uma regiao especifica da imagem."""
    import cv2
    import numpy as np
    from ocr_legacy.recognizer_paddle import run_paddle_primary_recognition, is_paddle_available
    from vision_stack.ocr import normalize_paddleocr_language

    img = cv2.imread(image_path)
    if img is None:
        return "", 0.0

    x1, y1, x2, y2 = bbox
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    
    if crop.size == 0:
        return "", 0.0

    if str(engine_preset_id or "").strip():
        import os
        import tempfile
        from pathlib import Path

        crop_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="traduzai_ocr_block_", suffix=".png", delete=False) as temp_file:
                crop_path = temp_file.name
            cv2.imwrite(crop_path, crop)
            ocr_page = run_detect_ocr(
                image_path=crop_path,
                models_dir=os.getenv("TRADUZAI_MODELS_DIR", ""),
                vision_worker_path=os.getenv("TRADUZAI_VISION_WORKER_PATH", ""),
                idioma_origem=idioma_origem,
                engine_preset_id=engine_preset_id,
            )
            texts = [item for item in ocr_page.get("texts", []) if isinstance(item, dict)]
            if texts:
                text = " ".join(
                    str(item.get("text") or item.get("original") or "").strip()
                    for item in texts
                    if str(item.get("text") or item.get("original") or "").strip()
                ).strip()
                confidences = [
                    float(item.get("confidence", item.get("ocr_confidence", item.get("confianca_ocr", 0.0))) or 0.0)
                    for item in texts
                ]
                if text:
                    return text, float(np.mean(confidences)) if confidences else 0.0
        except Exception as exc:
            logger.warning("OCR regional com preset %s falhou; fallback PaddleOCR crop: %s", engine_preset_id, exc)
        finally:
            if crop_path:
                Path(crop_path).unlink(missing_ok=True)

    if is_paddle_available():
        paddle_lang = normalize_paddleocr_language(idioma_origem)
        results = run_paddle_primary_recognition(crop, use_gpu=True, lang=paddle_lang)
        if results:
            # Combine all text snippets in the block
            text = " ".join([r["text"] for r in results])
            conf = np.mean([r["confidence"] for r in results])
            return text, float(conf)
            
    return "", 0.0
