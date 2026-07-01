from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from .models import Box, collect_boxes, parse_box
except ImportError:  # pragma: no cover - supports direct script execution.
    from models import Box, collect_boxes, parse_box


DEFAULT_MODEL_CANDIDATES = (
    Path("pipeline/models/studio_lite/bubble_detector.onnx"),
    Path("models/studio_lite/bubble_detector.onnx"),
)


def model_status(request: dict[str, Any] | None = None) -> dict[str, Any]:
    request = request or {}
    manual_path = request.get("model_path") or os.environ.get("TRADUZAI_STUDIO_LITE_DETECTOR_ONNX")
    if manual_path:
        path = Path(str(manual_path))
        return {
            "status": "ready" if path.is_file() else "missing",
            "state": "ready" if path.is_file() else "missing",
            "source": "manual_path",
            "path": str(path),
            "downloads": False,
        }

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return {
                "status": "ready",
                "state": "ready",
                "source": "default_path",
                "path": str(candidate),
                "downloads": False,
            }

    return {
        "status": "missing",
        "state": "missing",
        "source": "default_path",
        "path": str(DEFAULT_MODEL_CANDIDATES[0]),
        "downloads": False,
    }


def detect_page(request: dict[str, Any]) -> dict[str, Any]:
    status = model_status(request)
    if status["state"] != "ready":
        return {
            "detections": [],
            "model_status": status,
            "warning": "modelo de deteccao indisponivel; nenhuma deteccao retornada",
        }

    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        return {
            "detections": [],
            "model_status": {
                **status,
                "state": "missing",
                "reason": "onnxruntime_unavailable",
                "detail": str(exc),
            },
            "warning": "onnxruntime indisponivel; nenhuma deteccao retornada",
        }

    # The Studio Lite slice keeps detection optional. Wiring real YOLO inference
    # can be added behind this stable response shape without changing callers.
    return {"detections": [], "model_status": status}


def build_mask(request: dict[str, Any]) -> dict[str, Any]:
    cv2, np = _load_cv2_np()
    width, height = _resolve_size(request, cv2)
    output_path = _require_path(request, "output_path")
    boxes = collect_boxes(request)

    mask = np.zeros((height, width), dtype=np.uint8)
    drawn_boxes: list[list[int]] = []
    for box in boxes:
        clipped = box.clip(width, height)
        if clipped is None:
            continue
        mask[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2] = 255
        drawn_boxes.append(clipped.to_list())

    _ensure_parent(output_path)
    if not cv2.imwrite(str(output_path), mask):
        raise RuntimeError(f"falha ao escrever mascara: {output_path}")

    return {
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "boxes": drawn_boxes,
        "mask_pixels": int(np.count_nonzero(mask)),
    }


def inpaint_region(request: dict[str, Any]) -> dict[str, Any]:
    cv2, np = _load_cv2_np()
    image_path = _require_path(request, "image_path")
    mask_path = _require_path(request, "mask_path")
    output_path = _require_path(request, "output_path")
    radius = float(request.get("radius") or 3.0)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"imagem nao encontrada ou invalida: {image_path}")
    alpha = None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].copy()
        image = image[:, :, :3].copy()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"mascara nao encontrada ou invalida: {mask_path}")

    height, width = image.shape[:2]
    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"mascara com tamanho diferente da imagem: mask={mask.shape[:2]} image={(height, width)}"
        )

    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    result = image.copy()
    roi_box = _optional_roi(request, width, height)
    bbox_was_supplied = request.get("bbox") is not None
    if roi_box is None and bbox_was_supplied:
        used_bbox = None
    elif roi_box is None:
        if np.count_nonzero(binary_mask) > 0:
            result = cv2.inpaint(result, binary_mask, radius, cv2.INPAINT_TELEA)
        used_bbox = [0, 0, width, height]
    else:
        roi_mask = binary_mask[roi_box.y1 : roi_box.y2, roi_box.x1 : roi_box.x2]
        if np.count_nonzero(roi_mask) > 0:
            roi_image = result[roi_box.y1 : roi_box.y2, roi_box.x1 : roi_box.x2]
            result[roi_box.y1 : roi_box.y2, roi_box.x1 : roi_box.x2] = cv2.inpaint(
                roi_image,
                roi_mask,
                radius,
                cv2.INPAINT_TELEA,
            )
        used_bbox = roi_box.to_list()

    if alpha is not None:
        result = np.dstack([result, alpha])

    _ensure_parent(output_path)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"falha ao escrever imagem: {output_path}")

    return {
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "bbox": used_bbox,
        "mask_pixels": int(np.count_nonzero(binary_mask)),
    }


def _normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("command"):
        return request
    action = request.get("action")
    if not action:
        return request
    config = request.get("config")
    normalized = dict(config) if isinstance(config, dict) else {}
    normalized["command"] = action
    for key in ("project_dir", "cache_dir", "output_path"):
        if request.get(key) is not None and normalized.get(key) is None:
            normalized[key] = request[key]
    return normalized


def _wrap_response(command: str, result: dict[str, Any]) -> dict[str, Any]:
    response = {"ok": True, "command": command, "result": result}
    for key in ("output_path", "mask_path", "inpaint_path"):
        if result.get(key) is not None:
            response[key] = result[key]
    return response


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    request = _normalize_request(request)
    command = request.get("command")
    if command == "model_status":
        return _wrap_response(command, model_status(request))
    if command == "detect_page":
        return _wrap_response(command, detect_page(request))
    if command == "build_mask":
        return _wrap_response(command, build_mask(request))
    if command == "inpaint_region":
        result = inpaint_region(request)
        if result.get("inpaint_path") is None and result.get("output_path") is not None:
            result = {**result, "inpaint_path": result["output_path"]}
        return _wrap_response(command, result)
    raise ValueError(f"comando Studio Lite desconhecido: {command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TraduzAI Studio Lite worker")
    parser.add_argument("--request", help="JSON request file. Defaults to stdin.")
    parser.add_argument("request_file", nargs="?", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        request_path = args.request or args.request_file
        request_text = Path(request_path).read_text(encoding="utf-8") if request_path else sys.stdin.read()
        request = json.loads(request_text)
        response = handle_request(request)
    except Exception as exc:
        response = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
        print(json.dumps(response, ensure_ascii=False), flush=True)
        return 1

    print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def _resolve_size(request: dict[str, Any], cv2: Any) -> tuple[int, int]:
    if request.get("width") is not None and request.get("height") is not None:
        return int(request["width"]), int(request["height"])
    image_path = _require_path(request, "image_path")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"imagem nao encontrada ou invalida: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _optional_roi(request: dict[str, Any], width: int, height: int) -> Box | None:
    if request.get("bbox") is None:
        return None
    bbox_format = str(request.get("bbox_format") or "xyxy")
    return parse_box(request["bbox"], bbox_format=bbox_format).clip(width, height)


def _require_path(request: dict[str, Any], key: str) -> Path:
    value = request.get(key)
    if not value:
        raise ValueError(f"campo obrigatorio ausente: {key}")
    return Path(str(value))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_cv2_np() -> tuple[Any, Any]:
    import cv2
    import numpy as np

    return cv2, np


if __name__ == "__main__":
    raise SystemExit(main())
