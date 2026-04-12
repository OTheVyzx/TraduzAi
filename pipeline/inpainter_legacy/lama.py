"""
Compatibility wrapper around the current inpainting backend.
The module path stays stable so the rest of the app does not break.
"""

from __future__ import annotations

import logging
from typing import Callable

from .classical import run_classical_inpainting
from .lama_onnx import is_lama_manga_available, run_lama_manga_inpainting

logger = logging.getLogger(__name__)


def run_inpainting(
    image_files,
    ocr_results,
    output_dir: str,
    models_dir: str = "",
    corpus_visual_benchmark: dict | None = None,
    progress_callback: Callable | None = None,
):
    if is_lama_manga_available():
        try:
            return run_lama_manga_inpainting(
                image_files=image_files,
                ocr_results=ocr_results,
                output_dir=output_dir,
                models_dir=models_dir,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.warning("LaMa manga ONNX falhou, fallback para backend classico: %s", exc)

    return run_classical_inpainting(
        image_files=image_files,
        ocr_results=ocr_results,
        output_dir=output_dir,
        corpus_visual_benchmark=corpus_visual_benchmark,
        progress_callback=progress_callback,
    )
