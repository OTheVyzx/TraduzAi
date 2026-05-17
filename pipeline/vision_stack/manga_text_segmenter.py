from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MANGA_TEXT_SEGMENTATION_REPO = "a-b-c-x-y-z/Manga-Text-Segmentation-2025"
MANGA_TEXT_SEGMENTATION_HF_DIR = "models--a-b-c-x-y-z--Manga-Text-Segmentation-2025"
MANGA_TEXT_SEGMENTATION_FILE = "model.pth"
MIN_VALID_MODEL_BYTES = 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


class MangaTextSegmentationUnavailable(RuntimeError):
    pass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except Exception:
        return default


def _is_valid_model_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= MIN_VALID_MODEL_BYTES
    except OSError:
        return False


def _model_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = str(os.getenv("TRADUZAI_MANGA_TEXT_SEGMENTATION_MODEL") or "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            MODELS_DIR / "huggingface" / MANGA_TEXT_SEGMENTATION_HF_DIR / MANGA_TEXT_SEGMENTATION_FILE,
            MODELS_DIR / MANGA_TEXT_SEGMENTATION_HF_DIR / MANGA_TEXT_SEGMENTATION_FILE,
            MODELS_DIR / "manga-text-segmentation-2025" / MANGA_TEXT_SEGMENTATION_FILE,
            PROJECT_ROOT / "models" / "huggingface" / MANGA_TEXT_SEGMENTATION_HF_DIR / MANGA_TEXT_SEGMENTATION_FILE,
            PROJECT_ROOT
            / "pk"
            / "huggingface"
            / "a-b-c-x-y-z"
            / "Manga-Text-Segmentation-2025"
            / MANGA_TEXT_SEGMENTATION_FILE,
        ]
    )
    return candidates


def resolve_manga_text_segmentation_model(download: bool = True) -> Path:
    for candidate in _model_candidates():
        if _is_valid_model_file(candidate):
            return candidate

    dest = _model_candidates()[0]
    if not download:
        raise MangaTextSegmentationUnavailable(f"Modelo {MANGA_TEXT_SEGMENTATION_FILE} nao encontrado")

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=MANGA_TEXT_SEGMENTATION_REPO,
            filename=MANGA_TEXT_SEGMENTATION_FILE,
            local_dir=str(dest.parent),
        )
        return Path(downloaded)
    except Exception as exc:
        raise MangaTextSegmentationUnavailable(
            f"Nao foi possivel baixar {MANGA_TEXT_SEGMENTATION_REPO}/{MANGA_TEXT_SEGMENTATION_FILE}: {exc}"
        ) from exc


def _load_state_dict(torch_module: Any, model_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch_module.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch_module.load(str(model_path), map_location="cpu")
    except Exception:
        checkpoint = torch_module.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise MangaTextSegmentationUnavailable(f"Checkpoint invalido: {model_path}")
    clean: dict[str, Any] = {}
    for key, value in checkpoint.items():
        clean[str(key).removeprefix("module.")] = value
    return clean


def _convert_batchnorm_to_groupnorm(module: Any, nn_module: Any) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn_module.BatchNorm2d):
            channels = int(child.num_features)
            groups = 8
            if channels < groups or channels % groups != 0:
                for candidate in range(min(channels, 8), 1, -1):
                    if channels % candidate == 0:
                        groups = candidate
                        break
                else:
                    groups = 1
            setattr(module, name, nn_module.GroupNorm(num_groups=groups, num_channels=channels))
        else:
            _convert_batchnorm_to_groupnorm(child, nn_module)


class MangaTextSegmenter:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "cuda",
        half: bool = True,
        threshold: float | None = None,
        max_side: int | None = None,
        tta_hflip: bool | None = None,
        tta_vflip: bool | None = None,
    ) -> None:
        try:
            import segmentation_models_pytorch as smp
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception as exc:
            raise MangaTextSegmentationUnavailable(
                "Manga-Text-Segmentation-2025 requer segmentation-models-pytorch, timm e torch"
            ) from exc

        self._torch = torch
        self._nn = nn
        self._functional = F
        self.threshold = float(threshold if threshold is not None else _env_float("TRADUZAI_MANGA_TEXT_SEG_THRESHOLD", 0.5))
        self.max_side = max(128, int(max_side if max_side is not None else _env_int("TRADUZAI_MANGA_TEXT_SEG_MAX_SIDE", 1536)))
        self.tta_hflip = bool(tta_hflip if tta_hflip is not None else _env_flag("TRADUZAI_MANGA_TEXT_SEG_TTA_HFLIP", False))
        self.tta_vflip = bool(tta_vflip if tta_vflip is not None else _env_flag("TRADUZAI_MANGA_TEXT_SEG_TTA_VFLIP", False))

        self.model_path = Path(model_path) if model_path else resolve_manga_text_segmentation_model(download=True)
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if half and self.device.type == "cuda" else torch.float32

        model = smp.UnetPlusPlus(
            encoder_name="tu-efficientnetv2_rw_m",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        )
        _convert_batchnorm_to_groupnorm(model.decoder, nn)
        incompat = model.load_state_dict(_load_state_dict(torch, self.model_path), strict=False)
        if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
            logger.warning(
                "Manga-Text-Segmentation-2025 carregado com incompatibilidades: missing=%s unexpected=%s",
                len(getattr(incompat, "missing_keys", [])),
                len(getattr(incompat, "unexpected_keys", [])),
            )
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        self.model = model

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        original_h, original_w = image_rgb.shape[:2]
        work = np.ascontiguousarray(image_rgb[:, :, :3])
        if work.dtype != np.uint8:
            work = np.clip(work, 0, 255).astype(np.uint8)

        scale = min(1.0, self.max_side / float(max(original_h, original_w)))
        if scale < 1.0:
            work = cv2.resize(
                work,
                (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        prob = self._predict_probability(work)
        if prob.shape[:2] != (original_h, original_w):
            prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return (prob > self.threshold).astype(np.uint8) * 255

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.segment(image_rgb)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.segment(image_rgb)

    def _predict_probability(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        F = self._functional
        h, w = image_rgb.shape[:2]
        arr = image_rgb.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        probs = []
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                probs.append(self.model(tensor).sigmoid())
                if self.tta_hflip:
                    flipped = torch.flip(tensor, [3])
                    probs.append(torch.flip(self.model(flipped).sigmoid(), [3]))
                if self.tta_vflip:
                    flipped = torch.flip(tensor, [2])
                    probs.append(torch.flip(self.model(flipped).sigmoid(), [2]))

        prob = torch.stack(probs, dim=0).mean(dim=0)[0, 0, :h, :w].detach().float().cpu().numpy()
        return prob
