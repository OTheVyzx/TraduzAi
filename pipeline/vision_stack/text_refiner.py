from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .ocr import _derive_text_pixel_bbox

logger = logging.getLogger(__name__)


def _coerce_bbox(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in raw_bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _block_bbox(block: Any) -> list[int] | None:
    if isinstance(block, dict):
        return _coerce_bbox(block.get("bbox") or block.get("xyxy"))
    return _coerce_bbox(getattr(block, "xyxy", None))


def _block_confidence(block: Any) -> float:
    try:
        if isinstance(block, dict):
            return float(block.get("confidence", 1.0) or 0.0)
        return float(getattr(block, "confidence", 1.0) or 0.0)
    except Exception:
        return 0.0


def _fallback_record(block: Any, *, detector: str) -> dict[str, Any]:
    bbox = _block_bbox(block) or [0, 0, 0, 0]
    return {
        "text": "",
        "bbox": list(bbox),
        "source_bbox": list(bbox),
        "line_polygons": [],
        "text_pixel_bbox": list(bbox),
        "confidence": _block_confidence(block),
        "text_refiner": detector,
    }


class PPOCRv5TextRefiner:
    """ROI text refiner backed by the existing PaddleOCR block recognizer."""

    name = "ppocrv5_text_refiner"

    def __init__(
        self,
        *,
        quality: str = "normal",
        lang: str = "en",
        ocr_engine: Any | None = None,
        crop_fallback_max: int | None = None,
    ) -> None:
        self.quality = "ultra" if str(quality).strip().lower() == "ultra" else "normal"
        self.lang = str(lang or "en")
        self.ocr_engine = ocr_engine
        self.crop_fallback_max = crop_fallback_max

    def refine(self, image_rgb: Any, blocks: list[Any], *, quality: str = "normal") -> list[dict[str, Any]]:
        if not blocks:
            return []
        if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            return [_fallback_record(block, detector=self.name) for block in blocks]

        active_quality = "ultra" if str(quality or self.quality).strip().lower() == "ultra" else "normal"
        try:
            records = self._recognize(image_rgb, blocks, active_quality)
        except Exception as exc:
            logger.warning("PP-OCRv5 text refiner falhou; mantendo geometrias legadas: %s", exc)
            return [_fallback_record(block, detector=self.name) for block in blocks]

        if not records:
            return [_fallback_record(block, detector=self.name) for block in blocks]

        refined: list[dict[str, Any]] = []
        for block, record in zip(blocks, list(records)):
            refined.append(self._normalize_record(image_rgb, block, record))
        if len(refined) < len(blocks):
            for block in blocks[len(refined) :]:
                refined.append(_fallback_record(block, detector=self.name))
        return refined

    def _recognize(self, image_rgb: np.ndarray, blocks: list[Any], quality: str) -> list[Any]:
        engine = self.ocr_engine
        if engine is None:
            from .ocr import OCREngine

            engine = OCREngine(model="paddleocr", lang=self.lang)
            self.ocr_engine = engine
        if not hasattr(engine, "recognize_blocks_from_page"):
            raise ValueError("OCR engine sem recognize_blocks_from_page")
        kwargs = {
            "allow_sparse_mapping": quality == "ultra",
            "crop_fallback_max": self.crop_fallback_max,
        }
        try:
            return list(engine.recognize_blocks_from_page(image_rgb, blocks, **kwargs))
        except TypeError:
            return list(engine.recognize_blocks_from_page(image_rgb, blocks))

    def _normalize_record(self, image_rgb: np.ndarray, block: Any, record: Any) -> dict[str, Any]:
        fallback = _fallback_record(block, detector=self.name)
        if isinstance(record, dict):
            raw = record
            text = str(raw.get("text") or raw.get("translated") or "")
            bbox = _coerce_bbox(raw.get("bbox")) or _block_bbox(block) or fallback["bbox"]
            source_bbox = _coerce_bbox(raw.get("source_bbox")) or bbox
            line_polygons = raw.get("line_polygons") if isinstance(raw.get("line_polygons"), list) else []
            text_pixel_bbox = _coerce_bbox(raw.get("text_pixel_bbox"))
            if text_pixel_bbox is None:
                text_pixel_bbox = _derive_text_pixel_bbox(image_rgb, source_bbox, line_polygons) or source_bbox
            confidence = raw.get("confidence", fallback["confidence"])
        else:
            text = str(record or "")
            bbox = _block_bbox(block) or fallback["bbox"]
            source_bbox = bbox
            line_polygons = []
            text_pixel_bbox = bbox
            confidence = fallback["confidence"]

        normalized = dict(fallback)
        normalized.update(
            {
                "text": text,
                "bbox": list(bbox),
                "source_bbox": list(source_bbox),
                "line_polygons": line_polygons,
                "text_pixel_bbox": list(text_pixel_bbox),
                "confidence": float(confidence or 0.0),
                "text_refiner": self.name,
            }
        )
        return normalized
