from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_FONT_FAMILY = "ComicNeue-Bold.ttf"
DEFAULT_FONT_WEIGHT = "bold"


def _bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


@dataclass(frozen=True)
class TypesettingRenderRequest:
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TypesettingRenderRequest":
        if not isinstance(value, dict):
            raise ValueError("typesetting request must be a mapping")
        payload = dict(value)
        if not str(payload.get("bubble_mask_path") or "").strip():
            raise ValueError("bubble_mask_path is required for koharu renderer")
        payload["font_family"] = str(payload.get("font_family") or DEFAULT_FONT_FAMILY)
        payload["font_weight"] = str(payload.get("font_weight") or DEFAULT_FONT_WEIGHT)
        return cls(payload)

    @property
    def font_family(self) -> str:
        return str(self.payload.get("font_family") or DEFAULT_FONT_FAMILY)

    @property
    def font_weight(self) -> str:
        return str(self.payload.get("font_weight") or DEFAULT_FONT_WEIGHT)

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class TypesettingRenderResult:
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TypesettingRenderResult":
        if not isinstance(value, dict):
            raise ValueError("typesetting result must be a mapping")
        render_bbox = _bbox4(value.get("render_bbox"))
        if render_bbox is None:
            raise ValueError("render_bbox is required")
        payload = dict(value)
        payload["render_bbox"] = render_bbox
        return cls(payload)

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.payload)


def _number(value: Any, default: float | int | None = None) -> float | int | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _u8(value: Any) -> int | None:
    if value is None:
        return None
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= raw <= 255:
        return raw
    return None


def _style_value(style: dict[str, Any], text_data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in style and style[key] is not None:
            return style[key]
    for key in keys:
        if key in text_data and text_data[key] is not None:
            return text_data[key]
    return None


def _build_style(text_data: dict[str, Any]) -> dict[str, Any]:
    raw_style = text_data.get("style")
    raw_estilo = text_data.get("estilo")
    style: dict[str, Any] = {}
    if isinstance(raw_estilo, dict):
        style.update(raw_estilo)
    if isinstance(raw_style, dict):
        style.update(raw_style)

    font_size = _style_value(style, text_data, "fontSize", "font_size")
    stroke_width = _style_value(style, text_data, "strokeWidth", "stroke_width")
    return {
        "font_family": _style_value(style, text_data, "fontFamily", "font_family", "font"),
        "font_file": _style_value(style, text_data, "fontFile", "font_file"),
        "font_size": int(round(_number(font_size, 0) or 0)) or None,
        "color": _style_value(style, text_data, "color") or "#000000",
        "stroke_color": _style_value(style, text_data, "strokeColor", "stroke_color"),
        "stroke_width": int(round(_number(stroke_width, 0) or 0)),
        "bold": bool(_style_value(style, text_data, "bold")),
        "italic": bool(_style_value(style, text_data, "italic")),
        "align": _style_value(style, text_data, "textAlign", "align") or "center",
    }


def _layout_lines(value: Any) -> list[dict[str, float | str]]:
    if not isinstance(value, (list, tuple)):
        return []
    lines: list[dict[str, float | str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        x = _number(item.get("x"))
        y = _number(item.get("y"))
        if x is None or y is None:
            continue
        lines.append({"text": text, "x": float(x), "y": float(y)})
    return lines


def build_rust_render_request(img_size: tuple[int, int], text_data: dict[str, Any]) -> dict[str, Any]:
    width, height = img_size
    box = _bbox4(text_data.get("safe_text_box"))
    if box is None:
        raise ValueError("text_data has no valid safe_text_box for rust renderer")

    block: dict[str, Any] = {
        "id": str(text_data.get("id") or text_data.get("text_id") or ""),
        "text": str(text_data.get("translated") or ""),
        "box": box,
        "rotation_deg": float(_number(text_data.get("rotation_deg"), 0.0) or 0.0),
        "style": _build_style(text_data),
    }

    bubble_id = _u8(text_data.get("bubble_mask_value") or text_data.get("bubble_mask_id"))
    if bubble_id is None:
        bubble_id = text_data.get("bubble_id")
    if bubble_id is not None:
        block["bubble_id"] = bubble_id

    layout_lines = _layout_lines(text_data.get("_renderer_layout_lines") or text_data.get("layout_lines"))
    if layout_lines:
        block["layout_lines"] = layout_lines

    request: dict[str, Any] = {
        "image_width": int(width),
        "image_height": int(height),
        "blocks": [block],
    }

    bubble_mask_path = text_data.get("bubble_mask_path")
    if not isinstance(bubble_mask_path, str) or not bubble_mask_path.strip():
        raise ValueError("bubble_mask_path is required for koharu renderer")
    request["bubble_mask_path"] = bubble_mask_path.strip()

    return request
