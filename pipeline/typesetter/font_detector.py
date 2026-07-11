"""
Font detector usando YuzuMarker (ResNet50) como extrator de features visuais.

Não usa a cabeça de classificação (treinada em 6162 classes CJK).
Usa o backbone para extrair embeddings de 2048 dims e compara por
similaridade de cosseno com renders de amostra das fontes disponíveis.

Fonte padrão para todos os textos: ComicNeue-Bold (MAIÚSCULO).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SAMPLE_TEXT   = "ABCDEFGHabcdefgh123!?"
FINGERPRINT_SAMPLES = (
    ("ABCDEFGH123!?", "abcdefgh123!?", 36, 28),
    ("MANGA STYLE", "round narrow", 42, 25),
    ("WMWM IIl1", "AaBb 0987", 30, 32),
)
SIMILARITY_THRESHOLD = 0.72
TOP_K_FONT_CANDIDATES = 3
EXACT_FONT_MARGIN_THRESHOLD = 0.04
FULL_FONT_MARGIN = 0.08
DEFAULT_FONT  = "ComicNeue-Bold.ttf"
LEGACY_CANDIDATE_FONTS = [
    "KOMIKAX_.ttf",
    "Newrotic.ttf",
    "CCDaveGibbonsLower W00 Regular.ttf",
    "ComicNeue-Regular.ttf",
]


def _resolve_font_path(fonts_dir: Path, font_name: str) -> Path | None:
    for candidate in fonts_dir.rglob("*"):
        if candidate.name.lower() == font_name.lower():
            return candidate
    return None


def _load_detector_fonts_from_map(fonts_dir: Path) -> list[str]:
    font_map_path = fonts_dir / "font-map.json"
    if not font_map_path.exists():
        return []

    try:
        data = json.loads(font_map_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    fonts: list[str] = []
    for entry in data.get("available", []):
        if not isinstance(entry, dict) or not entry.get("detector", True):
            continue
        font_name = str(entry.get("arquivo") or "").strip()
        if font_name and _resolve_font_path(fonts_dir, font_name):
            fonts.append(font_name)
    return fonts


def _parse_env_google_font_specs() -> list["GoogleFontSpec"]:
    raw = os.environ.get("TRADUZAI_GOOGLE_FONT_FAMILIES", "")
    if not raw.strip():
        return []

    from typesetter.google_fonts import GoogleFontSpec

    specs: list[GoogleFontSpec] = []
    for family in raw.split(","):
        family = family.strip()
        if family:
            specs.append(GoogleFontSpec(family=family))
    return specs


def _draw_textpath_line(
    canvas: np.ndarray,
    text: str,
    *,
    font_path: Path | None,
    font_size: int,
    origin_x: int,
    top_y: int,
) -> None:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    if not text.strip():
        return

    font_props = (
        FontProperties(fname=str(font_path), size=font_size)
        if font_path is not None
        else FontProperties(size=font_size)
    )
    try:
        text_path = TextPath((0, 0), text, prop=font_props, usetex=False)
        bbox = text_path.get_extents()
    except AttributeError:
        # matplotlib bug: caracteres sem glyph geram codes=[] → [].flags explode
        return
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return

    transform = Affine2D().scale(1.0, -1.0).translate(
        float(origin_x) - float(bbox.x0),
        float(top_y) + float(bbox.y1),
    )
    transformed = text_path.transformed(transform)

    for polygon in transformed.to_polygons():
        if len(polygon) < 3:
            continue
        pts = np.round(polygon).astype(np.int32)
        cv2.fillPoly(canvas, [pts], color=(0, 0, 0))


def _render_font_sample_textpath(
    *,
    font_path: Path | None,
    upper_line: str,
    lower_line: str,
    canvas_size: int = 224,
    upper_font_size: int = 36,
    lower_font_size: int = 28,
) -> np.ndarray:
    oversample = 2
    scaled_size = int(canvas_size * oversample)
    canvas = np.full((scaled_size, scaled_size, 3), 255, dtype=np.uint8)

    _draw_textpath_line(
        canvas,
        upper_line,
        font_path=font_path,
        font_size=upper_font_size * oversample,
        origin_x=12 * oversample,
        top_y=60 * oversample,
    )
    _draw_textpath_line(
        canvas,
        lower_line,
        font_path=font_path,
        font_size=lower_font_size * oversample,
        origin_x=12 * oversample,
        top_y=110 * oversample,
    )

    return cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)


def _normalize_font_region(region_rgb: np.ndarray, *, canvas_size: int = 224) -> np.ndarray:
    """Center a glyph-like foreground crop without changing the legacy detector path."""
    if region_rgb is None or region_rgb.ndim < 3 or region_rgb.shape[2] < 3:
        return np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    rgb = region_rgb[:, :, :3].astype(np.uint8, copy=False)
    height, width = rgb.shape[:2]
    if height < 2 or width < 2:
        return cv2.resize(rgb, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    foreground = np.linalg.norm(rgb.astype(np.float32) - background, axis=2) > 28.0
    ys, xs = np.where(foreground)
    if len(xs) < 8:
        return cv2.resize(rgb, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = max(2, int(round(max(x2 - x1, y2 - y1) * 0.08)))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(rgb, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    crop_height, crop_width = crop.shape[:2]
    scale = min((canvas_size - 16) / max(1, crop_width), (canvas_size - 16) / max(1, crop_height))
    resized = cv2.resize(crop, (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    top = (canvas_size - resized.shape[0]) // 2
    left = (canvas_size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


class FontDetector:
    """Detecta estilo de fonte de uma região de texto de mangá.

    Compara features visuais (ResNet50 backbone) do crop original
    contra fingerprints pré-computados das fontes disponíveis.
    """

    def __init__(
        self,
        model_path: Path,
        fonts_dir: Path,
        *,
        enable_google_fonts: bool | None = None,
    ) -> None:
        self._model_path = model_path
        self._fonts_dir = fonts_dir
        self._model = None
        self._device = None
        self._fingerprints: dict[str, np.ndarray] = {}
        self._fingerprint_samples: dict[str, list[np.ndarray]] = {}
        self._candidate_fonts: list[str] = []
        self._enable_google_fonts = (
            os.environ.get("TRADUZAI_ENABLE_GOOGLE_FONTS", "0") == "1"
            if enable_google_fonts is None
            else enable_google_fonts
        )
        self._loaded = False

    def _load_model(self) -> None:
        try:
            import torch
            import torchvision.models as tv_models
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"FontDetector requer torch, torchvision e safetensors: {exc}"
            ) from exc

        state = load_file(str(self._model_path))
        prefix = "model._orig_mod.model."
        clean_state: dict[str, "torch.Tensor"] = {}
        for k, v in state.items():
            new_key = k[len(prefix):] if k.startswith(prefix) else k
            # Skip fc layer — it has 6162 CJK classes, incompatible with ResNet default
            if new_key.startswith("fc."):
                continue
            clean_state[new_key] = v

        resnet = tv_models.resnet50(weights=None)
        resnet.fc = torch.nn.Identity()  # set before load to avoid size mismatch
        resnet.load_state_dict(clean_state, strict=False)
        resnet = resnet.float()  # safetensors weights may be float64; ensure float32
        resnet.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = resnet.to(self._device)

    def _preprocess(self, img_rgb: np.ndarray) -> "torch.Tensor":
        import torch
        from PIL import Image as PilImage

        pil = PilImage.fromarray(img_rgb).convert("RGB").resize((224, 224))
        arr = np.array(pil, dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        arr = (arr - mean) / std
        # HWC → CHW → NCHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    def _extract_features(self, img_rgb: np.ndarray) -> np.ndarray:
        import torch

        tensor = self._preprocess(img_rgb)
        with torch.inference_mode():
            feats = self._model(tensor)
        vec = feats.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _render_font_sample(
        self,
        font_name: str,
        upper_line: str = SAMPLE_TEXT.upper(),
        lower_line: str = SAMPLE_TEXT.lower(),
        upper_font_size: int = 36,
        lower_font_size: int = 28,
    ) -> np.ndarray:
        font_path = _resolve_font_path(self._fonts_dir, font_name)
        try:
            return _render_font_sample_textpath(
                font_path=font_path,
                upper_line=upper_line,
                lower_line=lower_line,
                canvas_size=224,
                upper_font_size=upper_font_size,
                lower_font_size=lower_font_size,
            )
        except Exception:
            fallback = np.full((224, 224, 3), 255, dtype=np.uint8)
            cv2.putText(
                fallback,
                upper_line,
                (12, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            return fallback

    def _discover_candidate_fonts(self) -> list[str]:
        fonts: list[str] = []

        def add(font_name: str) -> None:
            if font_name != DEFAULT_FONT and font_name not in fonts:
                fonts.append(font_name)

        mapped_fonts = _load_detector_fonts_from_map(self._fonts_dir)
        for font_name in mapped_fonts:
            add(font_name)
        if not mapped_fonts:
            for font_name in LEGACY_CANDIDATE_FONTS:
                if _resolve_font_path(self._fonts_dir, font_name):
                    add(font_name)

        if self._enable_google_fonts:
            try:
                from typesetter.google_fonts import (
                    download_google_font_family,
                    specs_from_font_map,
                )

                specs = specs_from_font_map(self._fonts_dir / "font-map.json")
                specs.extend(_parse_env_google_font_specs())
                for spec in specs:
                    path = download_google_font_family(spec, self._fonts_dir)
                    add(path.name)
            except Exception:
                pass

        return fonts

    def _build_fingerprints(self) -> None:
        self._candidate_fonts = self._discover_candidate_fonts()
        all_fonts = [DEFAULT_FONT] + self._candidate_fonts
        for font_name in all_fonts:
            samples = [
                self._extract_features(
                    self._render_font_sample(
                        font_name,
                        upper_line=upper_line,
                        lower_line=lower_line,
                        upper_font_size=upper_font_size,
                        lower_font_size=lower_font_size,
                    )
                )
                for upper_line, lower_line, upper_font_size, lower_font_size in FINGERPRINT_SAMPLES
            ]
            aggregate = np.mean(np.stack(samples), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(aggregate))
            if norm > 0:
                aggregate /= norm
            self._fingerprint_samples[font_name] = samples
            self._fingerprints[font_name] = aggregate

    def _fonts_to_compare(self) -> list[str]:
        fonts = list(self._candidate_fonts)
        for font_name in self._fingerprints:
            if font_name != DEFAULT_FONT and font_name not in fonts:
                fonts.append(font_name)
        return fonts

    def _fallback_candidate(self) -> str:
        candidates = self._fonts_to_compare() or LEGACY_CANDIDATE_FONTS
        return candidates[0]

    def _best_match(self, region_feats: np.ndarray) -> tuple[str, float]:
        candidates = self._fonts_to_compare()
        if not candidates:
            return DEFAULT_FONT, 0.0

        best_font = candidates[0]
        best_sim = -1.0
        for font_name in candidates:
            fp = self._fingerprints.get(font_name)
            if fp is None:
                continue
            sim = float(np.dot(region_feats, fp))
            if sim > best_sim:
                best_sim = sim
                best_font = font_name

        confidence = min(1.0, max(0.0, best_sim))
        return best_font, confidence

    def _rank_candidates(self, region_feats: np.ndarray) -> list[tuple[str, float]]:
        ranked = []
        for font_name in self._fonts_to_compare():
            fingerprint = self._fingerprints.get(font_name)
            if fingerprint is None:
                continue
            ranked.append((font_name, float(np.dot(region_feats, fingerprint))))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def detect_with_evidence(self, region_rgb: np.ndarray) -> dict[str, object]:
        """Return calibrated top-k evidence for S3 shadow evaluation only."""
        if region_rgb is None or region_rgb.size < 8 * 8 * 3:
            return {
                "abstention_reason": "region_too_small",
                "confidence": 0.0,
                "margin": 0.0,
                "status": "unknown",
                "top_k": [],
                "value": "unknown",
            }
        if not self._loaded:
            try:
                self._load_model()
                self._build_fingerprints()
                self._loaded = True
            except Exception:
                return {
                    "abstention_reason": "model_unavailable",
                    "confidence": 0.0,
                    "margin": 0.0,
                    "status": "unknown",
                    "top_k": [],
                    "value": "unknown",
                }
        try:
            region_feats = self._extract_features(_normalize_font_region(region_rgb))
        except Exception:
            return {
                "abstention_reason": "feature_extraction_failed",
                "confidence": 0.0,
                "margin": 0.0,
                "status": "unknown",
                "top_k": [],
                "value": "unknown",
            }
        ranked = self._rank_candidates(region_feats)
        top_k = [
            {"font_name": font_name, "similarity": round(max(0.0, min(1.0, score)), 4)}
            for font_name, score in ranked[:TOP_K_FONT_CANDIDATES]
        ]
        if not ranked:
            return {
                "abstention_reason": "no_candidate_fonts",
                "confidence": 0.0,
                "margin": 0.0,
                "status": "unknown",
                "top_k": top_k,
                "value": "unknown",
            }
        best_font, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, best_score - runner_up)
        if best_score < SIMILARITY_THRESHOLD:
            return {
                "abstention_reason": "similarity_below_threshold",
                "confidence": 0.0,
                "margin": round(margin, 4),
                "status": "unknown",
                "top_k": top_k,
                "value": "unknown",
            }
        status = "exact" if margin >= EXACT_FONT_MARGIN_THRESHOLD else "family"
        confidence = max(0.0, min(1.0, best_score * min(1.0, margin / FULL_FONT_MARGIN)))
        return {
            "abstention_reason": "" if status == "exact" else "low_top_k_margin",
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "status": status,
            "top_k": top_k,
            "value": best_font,
        }

    def detect_with_score(
        self,
        region_rgb: np.ndarray,
        allow_default: bool = True,
    ) -> tuple[str, float]:
        """Return ``(font_name, confidence)`` for the best visual match."""
        if region_rgb is None or region_rgb.size < 8 * 8 * 3:
            return (DEFAULT_FONT, 0.0) if allow_default else (self._fallback_candidate(), 0.0)

        if not self._loaded:
            try:
                self._load_model()
                self._build_fingerprints()
                self._loaded = True
            except Exception:
                return (DEFAULT_FONT, 0.0) if allow_default else (self._fallback_candidate(), 0.0)

        try:
            region_feats = self._extract_features(region_rgb)
        except Exception:
            return (DEFAULT_FONT, 0.0) if allow_default else (self._fallback_candidate(), 0.0)

        from qa.runtime_fingerprint import record_engine_event

        record_engine_event(
            stage="font_detector",
            requested_engine="yuzumarker-font-detection",
            resolved_engine="yuzumarker-font-detection",
            backend=self._model if self._model is not None else self,
            execution_status="succeeded",
            result_status="accepted",
            fallback_used=False,
            model_path=self._model_path,
            execution_context="chapter",
        )

        best_font, confidence = self._best_match(region_feats)
        if allow_default and confidence < SIMILARITY_THRESHOLD:
            return DEFAULT_FONT, confidence
        return best_font, confidence

    def detect(self, region_rgb: np.ndarray, allow_default: bool = True) -> str:
        """Retorna o nome do arquivo de fonte mais adequado para a região.

        Sempre retorna DEFAULT_FONT se nenhuma candidata superar o threshold.
        """
        return self.detect_with_score(region_rgb, allow_default=allow_default)[0]
