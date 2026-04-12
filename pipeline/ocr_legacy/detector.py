"""
OCR module for manga pages.
Current flow:
- page preprocessing
- primary EasyOCR full-page pass
- fallback rereads for low-confidence regions
- local reviewer chooses the best candidate per region
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .postprocess import (
    _find_hf_model,
    analyze_style,
    classify_text_type,
    fix_ocr_errors,
    is_non_english,
    is_watermark,
    normalize_bbox,
)
from .recognizer_fallback import run_fallback_recognition
from .recognizer_paddle import (
    choose_primary_ocr_engine,
    is_paddle_available,
    run_paddle_primary_recognition,
)
from .recognizer_primary import run_primary_recognition
from .reviewer import choose_best_candidate
from .semantic_reviewer import semantic_refine_text

logger = logging.getLogger(__name__)

_reader = None
_font_detector_legacy = None


def _get_font_detector_legacy():
    global _font_detector_legacy
    if _font_detector_legacy is not None:
        return _font_detector_legacy
    from pathlib import Path
    model_path = _find_hf_model(
        "fffonion/yuzumarker-font-detection",
        "yuzumarker-font-detection.safetensors",
    )
    if model_path is None:
        return None
    fonts_dir = Path(__file__).parent.parent.parent / "fonts"
    try:
        from typesetter.font_detector import FontDetector
        _font_detector_legacy = FontDetector(model_path, fonts_dir)
    except Exception as exc:
        logger.warning("FontDetector (legacy) não carregado: %s", exc)
        return None
    return _font_detector_legacy
QUALITY_TO_PROFILE = {
    "rapida": "compat",
    "normal": "quality",
    "alta": "max",
    "compat": "compat",
    "quality": "quality",
    "max": "max",
}


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        logger.info("Inicializando EasyOCR...")
        _reader = easyocr.Reader(["en", "ko"], gpu=_check_gpu(), verbose=False)
        logger.info("EasyOCR pronto.")
    return _reader


def _check_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Upscale + CLAHE + sharpening para melhorar deteccao de texto."""
    _, width = img.shape[:2]
    if width < 1000:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_channel)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)


def run_ocr(
    image_path: str,
    models_dir: str = "",
    profile: str = "quality",
) -> dict:
    """
    Roda OCR em uma pagina e devolve leituras ja revisadas.
    """
    del models_dir
    resolved_profile = QUALITY_TO_PROFILE.get(profile, "quality")

    reader = _get_reader()
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return {"image": image_path, "width": 0, "height": 0, "texts": []}

    orig_h, orig_w = img_cv.shape[:2]
    preprocessed = _preprocess(img_cv)
    scale = preprocessed.shape[0] / orig_h
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    primary_regions = _run_primary_regions(
        reader=reader,
        image_bgr=img_cv,
        preprocessed_image=preprocessed,
    )

    texts = []
    for region in primary_regions:
        confidence = float(region["confidence"])
        raw_text = region["text"].strip()
        if confidence < 0.20 or not raw_text:
            continue

        bbox_scale = 1.0 if str(region.get("source", "")).startswith("primary-paddle") else scale
        bbox = normalize_bbox(region["bbox_pts"], bbox_scale, orig_w, orig_h)
        primary_candidate = {
            "text": fix_ocr_errors(raw_text),
            "confidence": round(confidence, 3),
            "source": region.get("source", "primary"),
        }
        if not primary_candidate["text"]:
            continue

        if len(primary_candidate["text"]) <= 2 and primary_candidate["confidence"] < 0.6:
            continue
        if len(primary_candidate["text"]) <= 3 and not any(char.isalpha() for char in primary_candidate["text"]):
            continue

        provisional_type = classify_text_type(primary_candidate["text"], bbox, orig_w)
        fallback_candidates = run_fallback_recognition(
            reader=reader,
            image_bgr=img_cv,
            bbox=bbox,
            primary_text=primary_candidate["text"],
            primary_confidence=primary_candidate["confidence"],
            profile=resolved_profile,
        )
        chosen = choose_best_candidate(
            primary=primary_candidate,
            fallback_candidates=fallback_candidates,
            tipo=provisional_type,
        )

        final_text = semantic_refine_text(
            chosen["text"].strip(),
            tipo=provisional_type,
            confidence=float(chosen["confidence"]),
        )
        if not final_text or is_watermark(final_text):
            continue

        tipo = classify_text_type(final_text, bbox, orig_w)
        skip = is_non_english(final_text)
        estilo = analyze_style(img_rgb, bbox)
        if not skip:
            fd = _get_font_detector_legacy()
            if fd is not None:
                x1, y1, x2, y2 = bbox
                region = img_rgb[y1:y2, x1:x2]
                detected_font = fd.detect(region)
                estilo["fonte"] = detected_font
                if detected_font == "CCDaveGibbonsLower W00 Regular.ttf":
                    estilo["force_upper"] = True
        texts.append(
            {
                "text": final_text,
                "bbox": bbox,
                "confidence": round(float(chosen["confidence"]), 3),
                "tipo": tipo,
                "estilo": estilo,
                "ocr_source": chosen.get("source", "primary"),
                "ocr_reviewed": bool(fallback_candidates),
                "ocr_profile": resolved_profile,
                "ocr_semantic_reviewed": final_text != chosen["text"].strip(),
                "ocr_mode": ocr_mode,
                "skip_processing": skip,
            }
        )

    return {
        "image": image_path,
        "width": orig_w,
        "height": orig_h,
        "texts": texts,
    }


def _run_primary_regions(reader, image_bgr, preprocessed_image) -> list[dict]:
    engine = choose_primary_ocr_engine(is_paddle_available())
    if engine == "paddle":
        try:
            paddle_runs = run_paddle_primary_recognition(image_bgr, use_gpu=_check_gpu())
            if paddle_runs:
                return paddle_runs
            logger.warning("PaddleOCR retornou vazio; usando EasyOCR como fallback.")
        except Exception as exc:
            logger.warning("PaddleOCR falhou, fallback para EasyOCR: %s", exc)

    return run_primary_recognition(reader, preprocessed_image)
