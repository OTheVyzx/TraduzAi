"""
Inpainting entrypoint for the active visual stack.

The previous inpainting implementation is preserved in `inpainter_legacy/`.
This module now routes `run_inpainting()` through the new detector-driven
LaMA stack and falls back to the legacy path only if the new stack fails.
"""

from __future__ import annotations

import logging
from typing import Callable

from inpainter_legacy.lama import run_inpainting as run_legacy_inpainting
from vision_stack.runtime import run_inpaint_pages

logger = logging.getLogger(__name__)


def run_inpainting(
    image_files,
    ocr_results,
    output_dir: str,
    models_dir: str = "",
    corpus_visual_benchmark: dict | None = None,
    progress_callback: Callable | None = None,
):
    profile = "quality"
    if ocr_results:
        profile = str(ocr_results[0].get("texts", [{}])[0].get("ocr_profile", "quality")) if ocr_results[0].get("texts") else "quality"

    try:
        return run_inpaint_pages(
            image_files=image_files,
            ocr_results=ocr_results,
            output_dir=output_dir,
            models_dir=models_dir,
            profile=profile,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.warning("Novo stack visual falhou no inpainting, fallback para legacy: %s", exc)
        return run_legacy_inpainting(
            image_files=image_files,
            ocr_results=ocr_results,
            output_dir=output_dir,
            models_dir=models_dir,
            corpus_visual_benchmark=corpus_visual_benchmark,
            progress_callback=progress_callback,
        )
