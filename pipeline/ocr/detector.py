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
            logger.warning("PaddleOCR retornou vazio; usando EasyOCR como fallback.")
        except Exception as exc:
            logger.warning("PaddleOCR falhou, fallback para EasyOCR: %s", exc)
    return run_primary_recognition(reader, preprocessed_image)


def run_ocr(
    image_path: str,
    models_dir: str = "",
    profile: str = "quality",
    vision_worker_path: str = "",
    progress_callback=None,
) -> dict:
    try:
        return run_detect_ocr(
            image_path=image_path,
            models_dir=models_dir,
            profile=profile,
            vision_worker_path=vision_worker_path,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.warning("Novo stack visual falhou no OCR, fallback para legacy: %s", exc)
        return run_legacy_ocr(
            image_path=image_path,
            models_dir=models_dir,
            profile=profile,
        )
