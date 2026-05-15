from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LABELS = {
    0: "bubble",
    1: "text_bubble",
    2: "text_free",
}


class ComicLayoutRTDetrDetector:
    """Optional RT-DETR comic-layout detector with legacy fallback."""

    name = "comic_layout_rtdetr"

    def __init__(
        self,
        *,
        models_dir: str | Path | None = None,
        quality: str = "normal",
        fallback: Any | None = None,
        model_path: str | Path | None = None,
        input_size: int = 1024,
    ) -> None:
        self.quality = "ultra" if str(quality).strip().lower() == "ultra" else "normal"
        self.fallback = fallback
        self.input_size = int(input_size)
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[1] / "models"
        self.model_path = Path(model_path) if model_path else self.models_dir / "detection" / "comic_layout_rtdetr" / "model.onnx"
        self.labels_path = self.model_path.with_name("labels.json")
        self.labels = self._load_labels()
        self._session = None
        self._load_error: str | None = None

    def detect(self, image_rgb: np.ndarray, conf_threshold: float | None = None) -> list[Any]:
        threshold = float(conf_threshold if conf_threshold is not None else (0.42 if self.quality == "ultra" else 0.5))
        try:
            session = self._get_session()
            if session is None:
                return self._fallback_detect(image_rgb, threshold)
            blocks = self._detect_with_session(session, image_rgb, threshold)
            if blocks:
                return blocks
            logger.info("comic_layout_rtdetr retornou 0 blocos; usando fallback legado")
        except Exception as exc:
            logger.warning("comic_layout_rtdetr falhou; usando fallback legado: %s", exc)
        return self._fallback_detect(image_rgb, threshold)

    def crop(self, image_rgb: np.ndarray, block: Any, padding: int = 4) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        xyxy = getattr(block, "xyxy", (0, 0, 0, 0))
        x1 = max(0, int(getattr(block, "x1", xyxy[0])) - padding)
        y1 = max(0, int(getattr(block, "y1", xyxy[1])) - padding)
        x2 = min(width, int(getattr(block, "x2", xyxy[2])) + padding)
        y2 = min(height, int(getattr(block, "y2", xyxy[3])) + padding)
        return image_rgb[y1:y2, x1:x2]

    def _fallback_detect(self, image_rgb: np.ndarray, threshold: float) -> list[Any]:
        if self.fallback is None:
            return []
        return self.fallback.detect(image_rgb, conf_threshold=threshold)

    def _get_session(self):
        if self._session is not None:
            return self._session
        if self._load_error is not None:
            return None
        if not self.model_path.exists():
            self._load_error = f"modelo ausente: {self.model_path}"
            logger.info("comic_layout_rtdetr indisponivel: %s", self._load_error)
            return None
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            logger.info("comic_layout_rtdetr carregado: %s", self.model_path)
            return self._session
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Nao foi possivel carregar comic_layout_rtdetr: %s", exc)
            return None

    def _detect_with_session(self, session: Any, image_rgb: np.ndarray, threshold: float) -> list[Any]:
        input_name = session.get_inputs()[0].name
        tensor, scale = self._preprocess(image_rgb)
        outputs = session.run(None, {input_name: tensor})
        rows = self._parse_outputs(outputs, threshold)

        from vision_stack.detector import TextBlock

        blocks = []
        for box, score, class_id in rows:
            x1, y1, x2, y2 = self._scale_box(box, scale, image_rgb.shape)
            if x2 <= x1 or y2 <= y1:
                continue
            block = TextBlock((x1, y1, x2, y2), confidence=float(score))
            block.detector = self.name
            block.region_type = self.labels.get(int(class_id), str(class_id))
            block.class_id = int(class_id)
            blocks.append(block)
        return blocks

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        height, width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        return tensor, (width / self.input_size, height / self.input_size)

    def _parse_outputs(self, outputs: list[Any], threshold: float) -> list[tuple[list[float], float, int]]:
        arrays = [np.asarray(output) for output in outputs]
        rows: list[tuple[list[float], float, int]] = []

        if len(arrays) >= 3 and arrays[0].shape[-1] == 4:
            boxes = arrays[0].reshape(-1, 4)
            scores = arrays[1].reshape(-1)
            labels = arrays[2].reshape(-1)
            for box, score, label in zip(boxes, scores, labels):
                if float(score) >= threshold:
                    rows.append((self._normalize_model_box(box.tolist()), float(score), int(label)))
            return rows

        candidates = [array for array in arrays if array.ndim >= 2 and array.shape[-1] >= 6]
        if not candidates:
            return rows
        detections = candidates[0].reshape(-1, candidates[0].shape[-1])
        for det in detections:
            score = float(det[4])
            if score < threshold:
                continue
            rows.append((self._normalize_model_box(det[:4].tolist()), score, int(det[5])))
        return rows

    def _normalize_model_box(self, box: list[float]) -> list[float]:
        if max(box) <= 1.5:
            box = [value * self.input_size for value in box]
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            cx, cy, width, height = box
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
        return [float(x1), float(y1), float(x2), float(y2)]

    def _scale_box(self, box: list[float], scale: tuple[float, float], image_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
        height, width = image_shape[:2]
        scale_x, scale_y = scale
        x1 = max(0.0, min(float(width), box[0] * scale_x))
        y1 = max(0.0, min(float(height), box[1] * scale_y))
        x2 = max(0.0, min(float(width), box[2] * scale_x))
        y2 = max(0.0, min(float(height), box[3] * scale_y))
        return x1, y1, x2, y2

    def _load_labels(self) -> dict[int, str]:
        if not self.labels_path.exists():
            return DEFAULT_LABELS.copy()
        try:
            data = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except Exception:
            return DEFAULT_LABELS.copy()
        if isinstance(data, dict):
            labels = {int(key): str(value) for key, value in data.items()}
            return labels or DEFAULT_LABELS.copy()
        return DEFAULT_LABELS.copy()
