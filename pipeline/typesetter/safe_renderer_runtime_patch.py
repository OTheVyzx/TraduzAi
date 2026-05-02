"""
Runtime guard for the legacy typesetter renderer.

Why this exists
---------------
Some exported pages still reach ``typesetter.renderer.render_text_block`` with
``bbox`` / ``layout_bbox`` set to the OCR text box, while ``balloon_bbox`` is
much larger.  When the translated Portuguese text is longer than the source,
the renderer may fit or clip against that small OCR box and cut letters at the
sides.

This module patches the renderer at import time and normalizes the layer before
rendering: when a long translated text clearly cannot fit in the OCR bbox and a
larger balloon_bbox exists, the render container is expanded to the balloon.
The original OCR/source boxes are preserved under private diagnostic keys.

The patch is intentionally conservative: short labels keep their original OCR
anchoring, so existing small-text placement behavior is not disturbed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
from typing import Any, MutableMapping

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "__traduzai_balloon_safe_patch_applied__"
_ORIGINAL_RENDER_ATTR = "__traduzai_original_render_text_block__"


def _normalize_box(box: Any) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [int(round(float(v))) for v in box]
    except Exception:
        return None
    x1, y1, x2, y2 = values
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left < 4 or bottom - top < 4:
        return None
    return [left, top, right, bottom]


def _box_area(box: list[int] | None) -> int:
    if not box:
        return 0
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_width(box: list[int] | None) -> int:
    return max(0, box[2] - box[0]) if box else 0


def _box_height(box: list[int] | None) -> int:
    return max(0, box[3] - box[1]) if box else 0


def _image_size(image: Any) -> tuple[int | None, int | None]:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        try:
            return int(size[0]), int(size[1])
        except Exception:
            pass
    shape = getattr(image, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        try:
            return int(shape[1]), int(shape[0])
        except Exception:
            pass
    return None, None


def _clamp_box_to_image(box: list[int], image_width: int | None, image_height: int | None) -> list[int]:
    x1, y1, x2, y2 = box
    if image_width is not None:
        x1 = max(0, min(image_width, x1))
        x2 = max(0, min(image_width, x2))
    if image_height is not None:
        y1 = max(0, min(image_height, y1))
        y2 = max(0, min(image_height, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return box
    return [x1, y1, x2, y2]


def _estimate_capacity_chars(box: list[int], font_size: int) -> float:
    """Rough capacity estimate for deciding whether to disturb OCR anchoring."""
    width = max(1, _box_width(box))
    height = max(1, _box_height(box))
    glyph_w = max(6.0, float(font_size) * 0.54)
    line_h = max(10.0, float(font_size) * 1.05)
    chars_per_line = max(1.0, width / glyph_w)
    lines = max(1.0, height / line_h)
    # Spaces and punctuation make this estimate optimistic; keep it conservative.
    return chars_per_line * lines * 0.82


def _is_likely_ocr_box_overflow(layer: MutableMapping[str, Any], base_box: list[int], text: str) -> bool:
    style = layer.get("estilo") or layer.get("style") or {}
    try:
        font_size = int(float(style.get("tamanho") or layer.get("detected_font_size_px") or 28))
    except Exception:
        font_size = 28
    font_size = max(10, min(96, font_size))
    collapsed_text = " ".join(str(text or "").split())
    if len(collapsed_text) >= 48:
        return True
    return len(collapsed_text) > _estimate_capacity_chars(base_box, font_size)


def _should_expand_to_balloon(layer: MutableMapping[str, Any], base_box: list[int], balloon_box: list[int], text: str) -> bool:
    base_area = _box_area(base_box)
    balloon_area = _box_area(balloon_box)
    if base_area <= 0 or balloon_area <= 0:
        return False
    if balloon_area < base_area * 1.20:
        return False
    if _box_width(balloon_box) < _box_width(base_box) * 1.12:
        return False
    return _is_likely_ocr_box_overflow(layer, base_box, text)


def _prepare_balloon_safe_text_container(image: Any, layer: MutableMapping[str, Any]) -> bool:
    """Expand a text layer render container to balloon_bbox when needed.

    Returns True when the layer was changed.
    """
    text = layer.get("translated") or layer.get("traduzido") or layer.get("text") or layer.get("original") or ""
    if not str(text).strip():
        return False

    balloon_box = _normalize_box(layer.get("balloon_bbox"))
    base_box = _normalize_box(layer.get("layout_bbox")) or _normalize_box(layer.get("bbox")) or _normalize_box(layer.get("source_bbox"))
    if not balloon_box or not base_box:
        return False

    if not _should_expand_to_balloon(layer, base_box, balloon_box, str(text)):
        return False

    image_width, image_height = _image_size(image)
    expanded_box = _clamp_box_to_image(balloon_box, image_width, image_height)

    layer.setdefault("_traduzai_original_bbox", layer.get("bbox"))
    layer.setdefault("_traduzai_original_layout_bbox", layer.get("layout_bbox"))
    layer.setdefault("_traduzai_original_text_pixel_bbox", layer.get("text_pixel_bbox"))
    layer["_traduzai_safe_container_reason"] = "balloon_bbox_over_ocr_bbox"

    # Force the existing renderer to lay out and render against the balloon rather
    # than the OCR box.  source_bbox is intentionally preserved for diagnostics.
    layer["bbox"] = list(expanded_box)
    layer["layout_bbox"] = list(expanded_box)
    layer["render_bbox"] = None

    # Some renderer paths prefer text_pixel_bbox as an anchor.  For overflow cases
    # that reproduces the clipping bug, so disable it for this render pass only.
    if "text_pixel_bbox" in layer:
        layer["text_pixel_bbox"] = None

    return True


def _patch_renderer_module(module: Any) -> None:
    if getattr(module, _PATCHED_ATTR, False):
        return
    original = getattr(module, "render_text_block", None)
    if not callable(original):
        return

    def render_text_block_patched(image: Any, text_data: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(text_data, MutableMapping):
            try:
                _prepare_balloon_safe_text_container(image, text_data)
            except Exception as exc:
                logger.warning("Falha ao aplicar guard de balloon_bbox no typesetter: %s", exc, exc_info=True)
        return original(image, text_data, *args, **kwargs)

    setattr(module, _ORIGINAL_RENDER_ATTR, original)
    setattr(module, "render_text_block", render_text_block_patched)
    setattr(module, _PATCHED_ATTR, True)


class _RendererPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:  # pragma: no cover - delegated
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module:
            return create_module(spec)
        return None

    def exec_module(self, module: Any) -> None:
        self._wrapped.exec_module(module)
        _patch_renderer_module(module)


def _is_our_finder(finder: Any) -> bool:
    return isinstance(finder, _RendererPatchFinder)


class _RendererPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        if fullname != "typesetter.renderer":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not spec or not spec.loader:
            return spec
        if isinstance(spec.loader, _RendererPatchLoader):
            return spec
        spec.loader = _RendererPatchLoader(spec.loader)
        return spec


def install() -> None:
    existing = sys.modules.get("typesetter.renderer")
    if existing is not None:
        _patch_renderer_module(existing)
        return
    if not any(_is_our_finder(finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RendererPatchFinder())


install()

__all__ = [
    "install",
    "_prepare_balloon_safe_text_container",
]
