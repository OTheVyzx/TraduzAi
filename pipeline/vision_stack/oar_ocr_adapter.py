from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from .text_mask_evidence import normalize_bbox
except ImportError:
    from vision_stack.text_mask_evidence import normalize_bbox

logger = logging.getLogger(__name__)


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_boxes(value: Any, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, (list, tuple)):
        return []
    boxes: list[list[int]] = []
    for item in value:
        bbox = normalize_bbox(item, width, height)
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def parse_oar_ocr_payload(payload: dict[str, Any], *, width: int, height: int) -> list[dict[str, Any]]:
    regions = _first_present(payload, "text_regions", "textRegions", "regions", "texts") or []
    if isinstance(regions, dict):
        regions = list(regions.values())
    parsed: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = normalize_bbox(_first_present(region, "bbox", "box", "bounding_box", "boundingBox"), width, height)
        if bbox is None:
            continue
        parsed.append(
            {
                "text": str(_first_present(region, "text", "recognized_text", "recognizedText") or ""),
                "bbox": bbox,
                "word_boxes": _normalize_boxes(_first_present(region, "word_boxes", "wordBoxes"), width, height),
                "char_boxes": _normalize_boxes(_first_present(region, "char_boxes", "charBoxes"), width, height),
                "confidence": float(_first_present(region, "confidence", "score") or 0.0),
                "source": "oar-ocr",
            }
        )
    return parsed


def load_oar_ocr_regions(image_path: str | Path, *, width: int, height: int, timeout_s: int = 60) -> list[dict[str, Any]]:
    json_path = os.getenv("TRADUZAI_OAR_OCR_JSON", "").strip()
    if json_path:
        try:
            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            return parse_oar_ocr_payload(payload, width=width, height=height)
        except Exception as exc:
            logger.warning("Falha ao ler TRADUZAI_OAR_OCR_JSON=%s: %s", json_path, exc)
            return []

    bin_path = os.getenv("TRADUZAI_OAR_OCR_BIN", "").strip()
    if not bin_path:
        return []

    try:
        completed = subprocess.run(
            [bin_path, str(image_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        logger.warning("oar-ocr indisponivel: %s", exc)
        return []
    if completed.returncode != 0:
        logger.warning("oar-ocr retornou codigo %s: %s", completed.returncode, completed.stderr.strip())
        return []
    try:
        payload = json.loads(completed.stdout)
    except Exception as exc:
        logger.warning("oar-ocr retornou JSON invalido: %s", exc)
        return []
    return parse_oar_ocr_payload(payload, width=width, height=height)
