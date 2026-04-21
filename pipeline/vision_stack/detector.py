"""
Text detector for the new visual stack.

Primary path: comic-text-detector weights.
Fallback path: PaddleOCR detection-only, preserving the detect -> OCR -> inpaint flow.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_URLS = {
    "comic-text-detector": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt",
    "comic-text-detector-cuda": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
}

def _default_models_dir() -> Path:
    env_dir = (os.getenv("TRADUZAI_MODELS_DIR") or os.getenv("MANGATL_MODELS_DIR") or "").strip()
    if env_dir:
        return Path(env_dir)

    default_dir = Path.home() / ".traduzai" / "models"
    legacy_dir = Path.home() / ".mangatl" / "models"
    if not default_dir.exists() and legacy_dir.exists():
        return legacy_dir
    return default_dir


MODELS_DIR = _default_models_dir()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PK_CTD_DIR = PROJECT_ROOT / "pk" / "huggingface" / "mayocream" / "comic-text-detector"


class _ComicTextDownBlock(nn.Module):
    def __init__(self, c3_cls, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = c3_cls(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class _ComicTextUpBlock(nn.Module):
    def __init__(self, c3_cls, skip_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            c3_cls(skip_channels + hidden_channels, hidden_channels),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ComicTextSegHead(nn.Module):
    def __init__(self, c3_cls):
        super().__init__()
        self.down_conv1 = _ComicTextDownBlock(c3_cls, 512, 512, stride=2)
        self.upconv0 = _ComicTextUpBlock(c3_cls, 0, 512, 256)
        self.upconv2 = _ComicTextUpBlock(c3_cls, 256, 512, 256)
        self.upconv3 = _ComicTextUpBlock(c3_cls, 0, 512, 256)
        self.upconv4 = _ComicTextUpBlock(c3_cls, 128, 256, 128)
        self.upconv5 = _ComicTextUpBlock(c3_cls, 64, 128, 64)
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f160: torch.Tensor,
        f80: torch.Tensor,
        f40: torch.Tensor,
        f20: torch.Tensor,
        f3: torch.Tensor,
    ) -> torch.Tensor:
        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(torch.cat([f20, u20], dim=1))
        u80 = self.upconv3(torch.cat([f40, u40], dim=1))
        u160 = self.upconv4(torch.cat([f80, u80], dim=1))
        u320 = self.upconv5(torch.cat([f160, u160], dim=1))
        return self.upconv6(u320)


@dataclass
class TextBlock:
    xyxy: tuple
    mask: Optional[np.ndarray] = None
    confidence: float = 1.0
    text: str = ""
    translated_text: str = ""
    font_size: int = 0
    is_vertical: bool = False
    color: tuple = (0, 0, 0)

    @property
    def x1(self):
        return int(self.xyxy[0])

    @property
    def y1(self):
        return int(self.xyxy[1])

    @property
    def x2(self):
        return int(self.xyxy[2])

    @property
    def y2(self):
        return int(self.xyxy[3])

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


class TextDetector:
    def __init__(
        self,
        model: str = "comic-text-detector",
        device: str = "cuda",
        half: bool = True,
        model_path: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.half = half and self.device.type == "cuda"
        self._model = None
        self._model_type = model
        self._model_path = model_path or self._get_model_path(model)
        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_model_path(self, model_name: str) -> Path:
        path = MODELS_DIR / f"{model_name}.pt"
        if not path.exists():
            self._download_model(model_name, path)
        return path

    def _download_model(self, model_name: str, dest: Path):
        import urllib.request

        url = MODEL_URLS.get(model_name)
        if not url:
            raise ValueError(f"Modelo '{model_name}' nao encontrado")

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando %s...", model_name)

        def progress(count, block_size, total_size):
            pct = count * block_size * 100 / total_size if total_size else 0
            print(f"\r  {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, progress)
        print()
        logger.info("Modelo salvo em %s", dest)

    def _load_model(self):
        if self._load_comic_text_detector_native():
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._model_path))
            self._model.to(self.device)
            if self.half:
                self._model.half()
            self._backend = "ultralytics"
            logger.info("Detector carregado via ultralytics (%s)", self.device)
            return
        except Exception as exc:
            logger.warning("comic-text-detector nao carregou via ultralytics: %s", exc)

        if self._load_paddle_detection_backend():
            return

        self._model = None
        self._backend = "contour-fallback"
        logger.warning("Detector visual caiu para contour-fallback local")

    def _load_paddle_detection_backend(self) -> bool:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            from paddleocr import PaddleOCR
            import paddle.base.libpaddle as libpaddle
            if hasattr(libpaddle, 'AnalysisConfig') and not hasattr(libpaddle.AnalysisConfig, 'set_optimization_level'):
                libpaddle.AnalysisConfig.set_optimization_level = lambda *args, **kwargs: None
        except Exception as exc:
            logger.warning("PaddleOCR indisponivel para deteccao; seguindo com fallback local: %s", exc)
            return False

        use_gpu = self.device.type == "cuda"
        self._model = PaddleOCR(
            use_angle_cls=False,
            lang="en",
        )
        self._backend = "paddle-det"
        logger.info("Detector carregado via paddle-det (gpu=%s)", use_gpu)
        return True

    def _load_comic_text_detector_native(self) -> bool:
        if self._model_type != "comic-text-detector":
            return False
        if self._model_path.suffix.lower() != ".pt" or not self._model_path.exists():
            return False

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                checkpoint = torch.load(str(self._model_path), map_location="cpu", weights_only=False)
            blk_det = checkpoint.get("blk_det") if isinstance(checkpoint, dict) else None
            if not isinstance(blk_det, dict):
                return False

            cfg = blk_det.get("cfg")
            weights = blk_det.get("weights")
            if not isinstance(cfg, dict) or weights is None:
                return False

            DetectionModel, non_max_suppression, letterbox, c3_cls = self._import_yolov5_runtime()
            safetensor_paths = self._get_comic_text_safetensor_paths()
            yolo_state = weights
            seg_state = None
            weight_source = "checkpoint"
            if safetensor_paths["yolo"].exists():
                try:
                    yolo_state = self._load_safetensor_state_dict(safetensor_paths["yolo"])
                    weight_source = "safetensors"
                except Exception as exc:
                    logger.warning("Nao foi possivel carregar yolo safetensor; usando checkpoint: %s", exc)
            if safetensor_paths["unet"].exists():
                try:
                    seg_state = self._load_safetensor_state_dict(safetensor_paths["unet"])
                except Exception as exc:
                    logger.warning("Nao foi possivel carregar unet safetensor; usando checkpoint: %s", exc)

            previous_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model = DetectionModel(cfg, ch=int(cfg.get("ch", 3)), nc=int(cfg.get("nc", 2)))
            finally:
                logging.disable(previous_disable)
            incompat = model.load_state_dict(yolo_state, strict=False)
            if incompat.missing_keys or incompat.unexpected_keys:
                logger.warning(
                    "comic-text-detector carregado com incompatibilidades: missing=%s unexpected=%s",
                    len(incompat.missing_keys),
                    len(incompat.unexpected_keys),
                )

            model.to(self.device)
            model.eval()
            if self.half:
                model.half()

            seg_head = self._make_comic_text_seg_head(c3_cls)
            if seg_state is None:
                seg_state = checkpoint.get("text_seg") if isinstance(checkpoint, dict) else None
            if isinstance(seg_state, dict):
                seg_incompat = seg_head.load_state_dict(seg_state, strict=True)
                if getattr(seg_incompat, "missing_keys", None) or getattr(seg_incompat, "unexpected_keys", None):
                    logger.warning(
                        "text_seg carregado com incompatibilidades: missing=%s unexpected=%s",
                        len(getattr(seg_incompat, "missing_keys", [])),
                        len(getattr(seg_incompat, "unexpected_keys", [])),
                    )
                seg_head.to(self.device)
                seg_head.eval()
                if self.half:
                    seg_head.half()
                self._ctd_seg_head = seg_head
            else:
                self._ctd_seg_head = None

            self._model = model
            self._backend = "comic-text-detector"
            self._ctd_nms = non_max_suppression
            self._ctd_letterbox = letterbox
            self._ctd_input_size = 1024
            self._ctd_feature_indices = (1, 4, 6, 8, 9)
            self._ctd_seg_threshold = 0.35
            self._ctd_attach_masks = False
            self._ctd_weight_source = weight_source
            logger.info("Detector carregado via comic-text-detector native (%s)", self.device)
            return True
        except Exception as exc:
            logger.warning("comic-text-detector native nao carregou: %s", exc)
            return False

    def _import_yolov5_runtime(self):
        hub_dir = Path(torch.hub.get_dir())
        candidates = sorted(hub_dir.glob("ultralytics_yolov5*"), reverse=True)

        if not candidates:
            try:
                torch.hub.load(
                    "ultralytics/yolov5",
                    "yolov5n",
                    pretrained=False,
                    autoshape=False,
                    trust_repo=True,
                )
            except Exception as exc:
                logger.warning("Nao foi possivel baixar runtime yolov5 para comic-text-detector: %s", exc)
            candidates = sorted(hub_dir.glob("ultralytics_yolov5*"), reverse=True)

        if not candidates:
            raise RuntimeError("runtime yolov5 nao encontrado para comic-text-detector")

        repo_path = str(candidates[0])
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        self._ensure_optional_yolov5_runtime_stubs()
        DetectionModel = importlib.import_module("models.yolo").Model
        non_max_suppression = importlib.import_module("utils.general").non_max_suppression
        letterbox = importlib.import_module("utils.augmentations").letterbox
        c3_cls = importlib.import_module("models.common").C3
        return DetectionModel, non_max_suppression, letterbox, c3_cls

    @staticmethod
    def _ensure_optional_yolov5_runtime_stubs():
        if "seaborn" in sys.modules or importlib.util.find_spec("seaborn") is not None:
            return

        seaborn_stub = types.ModuleType("seaborn")

        def _noop_color_palette(*args, **kwargs):
            del args, kwargs
            return [(0.0, 0.0, 0.0)]

        seaborn_stub.color_palette = _noop_color_palette
        sys.modules["seaborn"] = seaborn_stub

    def _get_comic_text_safetensor_paths(self) -> dict[str, Path]:
        return {
            "yolo": PK_CTD_DIR / "yolo-v5.safetensors",
            "unet": PK_CTD_DIR / "unet.safetensors",
        }

    @staticmethod
    def _load_safetensor_state_dict(path: Path) -> dict:
        from safetensors.torch import load_file

        return load_file(str(path))

    def _make_comic_text_seg_head(self, c3_cls):
        return _ComicTextSegHead(c3_cls)

    def detect(self, img_np: np.ndarray, conf_threshold: float = 0.5) -> list[TextBlock]:
        orig_h, orig_w = img_np.shape[:2]
        img_rgb = img_np if img_np.shape[2] == 3 else cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)

        if self._backend == "comic-text-detector":
            blocks = self._detect_comic_text_native(img_rgb, conf_threshold=conf_threshold)
        elif self._backend == "contour-fallback":
            blocks = self._detect_contour_fallback(img_rgb)
        else:
            target_size = self._get_inference_size(orig_h, orig_w)
            img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))

            if self._backend == "ultralytics":
                with torch.inference_mode():
                    results = self._model(img_resized, conf=0.45, verbose=False)
                blocks = self._parse_ultralytics(results, orig_h, orig_w, target_size)
            else:
                results = self._model.ocr(img_resized, det=True, rec=False, cls=False)
                blocks = self._parse_paddle_detection(results, orig_h, orig_w, target_size)

        blocks.sort(key=lambda b: (b.y1 // 100, -(b.x1)))
        return blocks

    def _detect_contour_fallback(self, img_rgb: np.ndarray) -> list[TextBlock]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0, sigmaY=3.0)
        dark_contrast = cv2.subtract(blur, gray)
        bright_contrast = cv2.subtract(gray, blur)

        dark_mask = (dark_contrast >= 18).astype(np.uint8) * 255
        bright_mask = (bright_contrast >= 18).astype(np.uint8) * 255

        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_join = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        mask = cv2.max(dark_mask, bright_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_join, iterations=1)

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blocks: list[TextBlock] = []
        for index in range(1, component_count):
            x, y, w, h, area = stats[index].tolist()
            if area < 32 or w < 8 or h < 8:
                continue
            if w > img_rgb.shape[1] * 0.9 or h > img_rgb.shape[0] * 0.5:
                continue
            fill_ratio = area / float(max(1, w * h))
            aspect_ratio = max(w / float(max(1, h)), h / float(max(1, w)))
            if fill_ratio < 0.06 or aspect_ratio > 18.0:
                continue
            blocks.append(
                TextBlock(
                    xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=0.55,
                )
            )

        return self._dedupe_blocks(blocks)

    def _detect_comic_text_native(self, img_rgb: np.ndarray, conf_threshold: float = 0.5) -> list[TextBlock]:
        orig_h, orig_w = img_rgb.shape[:2]
        target_size = self._get_inference_size(orig_h, orig_w)
        input_size = max(target_size)
        
        letterbox = self._ctd_letterbox
        nms = self._ctd_nms
        img_in, ratio, (dw, dh) = letterbox(img_rgb, new_shape=(input_size, input_size), auto=False, stride=64)
        tensor = img_in.transpose((2, 0, 1))[::-1]
        tensor = np.ascontiguousarray(tensor)
        tensor = torch.from_numpy(tensor).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor = tensor.unsqueeze(0) / 255.0

        pred, features = self._forward_comic_text_detector(tensor)

        det = nms(pred, conf_thres=min(conf_threshold, 0.25), iou_thres=0.35)[0]
        if det is None or len(det) == 0:
            return []

        scale_x = 1.0 / float(ratio[0])
        scale_y = 1.0 / float(ratio[1])
        blocks: list[TextBlock] = []
        for row in det.detach().cpu().numpy():
            x1, y1, x2, y2, conf, cls = row.tolist()
            del cls
            x1 = (x1 - float(dw)) * scale_x
            x2 = (x2 - float(dw)) * scale_x
            y1 = (y1 - float(dh)) * scale_y
            y2 = (y2 - float(dh)) * scale_y
            blocks.append(
                TextBlock(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(conf),
                )
            )

        blocks = self._dedupe_blocks(blocks)
        if bool(getattr(self, "_ctd_attach_masks", False)):
            seg_mask = self._predict_comic_text_mask(img_rgb, tensor, features=features)
            if seg_mask is not None and np.any(seg_mask):
                self._attach_segmentation_masks(blocks, seg_mask)
        return blocks

    def _forward_comic_text_detector(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        feature_indices = tuple(getattr(self, "_ctd_feature_indices", (1, 4, 6, 8, 9)))
        if not hasattr(self._model, "model"):
            with torch.inference_mode():
                raw = self._model(tensor)
            pred = raw[0] if isinstance(raw, tuple) else raw
            return pred, {}
        outputs = []
        captured: dict[int, torch.Tensor] = {}
        x = tensor
        with torch.inference_mode():
            for module in self._model.model:
                if module.f != -1:
                    x = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
                x = module(x)
                if module.i in feature_indices and isinstance(x, torch.Tensor):
                    captured[module.i] = x
                outputs.append(x if module.i in self._model.save else None)
        pred = x[0] if isinstance(x, tuple) else x
        return pred, captured

    def _predict_comic_text_mask(
        self,
        img_rgb: np.ndarray,
        tensor: torch.Tensor,
        features: Optional[dict[int, torch.Tensor]] = None,
    ) -> Optional[np.ndarray]:
        del tensor
        seg_head = getattr(self, "_ctd_seg_head", None)
        if seg_head is None or not features:
            return None

        required = tuple(getattr(self, "_ctd_feature_indices", (1, 4, 6, 8, 9)))
        if any(index not in features for index in required):
            return None

        with torch.inference_mode():
            seg_pred = seg_head(*(features[index] for index in required))

        seg_map = seg_pred.detach().float().cpu().squeeze().numpy()
        if seg_map.ndim != 2:
            return None
        seg_map = cv2.resize(seg_map, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        threshold = float(getattr(self, "_ctd_seg_threshold", 0.35))
        mask = (seg_map >= threshold).astype(np.uint8) * 255
        if not np.any(mask):
            return None
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        return mask

    @staticmethod
    def _attach_segmentation_masks(blocks: list[TextBlock], full_mask: np.ndarray):
        height, width = full_mask.shape[:2]
        for block in blocks:
            x1 = max(0, min(width, block.x1))
            x2 = max(0, min(width, block.x2))
            y1 = max(0, min(height, block.y1))
            y2 = max(0, min(height, block.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            local_mask = full_mask[y1:y2, x1:x2].copy()
            if not np.any(local_mask):
                continue
            block.mask = local_mask

    @staticmethod
    def _dedupe_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
        deduped: list[TextBlock] = []
        for block in sorted(blocks, key=lambda item: item.confidence, reverse=True):
            if block.width <= 2 or block.height <= 2:
                continue
            replaced = False
            for index, kept in enumerate(deduped):
                if TextDetector._bbox_iou(block.xyxy, kept.xyxy) >= 0.65:
                    if block.confidence > kept.confidence:
                        deduped[index] = block
                    replaced = True
                    break
            if not replaced:
                deduped.append(block)
        return deduped

    @staticmethod
    def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / max(1.0, area_a + area_b - inter)

    def _get_inference_size(self, h: int, w: int) -> tuple[int, int]:
        # Para strips verticais, aumentamos o limite para manter detalhes dos balões
        max_size = 1024
        if h > w * 2: # Strip vertical
            max_size = 2048 if h < 4000 else 3072
        
        scale = min(max_size / max(h, w), 1.0)
        new_h = int(h * scale / 32) * 32
        new_w = int(w * scale / 32) * 32
        return (max(new_h, 32), max(new_w, 32))

    def _parse_ultralytics(self, results, orig_h, orig_w, infer_size) -> list[TextBlock]:
        blocks = []
        ih, iw = infer_size
        sx, sy = orig_w / iw, orig_h / ih

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None

            for idx, (box, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = box
                block = TextBlock(
                    xyxy=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                    confidence=float(conf),
                )
                if masks is not None and idx < len(masks):
                    mask = cv2.resize(masks[idx], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    block.mask = (mask > 0.5).astype(np.uint8) * 255
                blocks.append(block)
        return blocks

    def _parse_paddle_detection(self, results, orig_h, orig_w, infer_size) -> list[TextBlock]:
        blocks = []
        ih, iw = infer_size
        sx, sy = orig_w / iw, orig_h / ih

        page = results[0] if results else []
        for box in page or []:
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            blocks.append(
                TextBlock(
                    xyxy=(min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy),
                    confidence=1.0,
                )
            )
        return blocks

    def crop(self, img_np: np.ndarray, block: TextBlock, padding: int = 4) -> np.ndarray:
        h, w = img_np.shape[:2]
        x1 = max(0, block.x1 - padding)
        y1 = max(0, block.y1 - padding)
        x2 = min(w, block.x2 + padding)
        y2 = min(h, block.y2 + padding)
        return img_np[y1:y2, x1:x2]

    def build_mask(self, img_np: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for block in blocks:
            if block.mask is not None and block.mask.shape == (h, w):
                mask = np.maximum(mask, block.mask)
            else:
                pad = 2
                x1 = max(0, block.x1 - pad)
                y1 = max(0, block.y1 - pad)
                x2 = min(w, block.x2 + pad)
                y2 = min(h, block.y2 + pad)
                mask[y1:y2, x1:x2] = 255

        if np.any(mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def unload(self):
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
