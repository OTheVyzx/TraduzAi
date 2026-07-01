from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "Box | None":
        x1 = max(0, min(width, self.x1))
        y1 = max(0, min(height, self.y1))
        x2 = max(0, min(width, self.x2))
        y2 = max(0, min(height, self.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return Box(x1=x1, y1=y1, x2=x2, y2=y2)

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def parse_box(value: Any, *, bbox_format: str = "xyxy") -> Box:
    if isinstance(value, dict):
        if "bbox" in value:
            return parse_box(value["bbox"], bbox_format=str(value.get("bbox_format") or bbox_format))
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return Box(
                x1=round_int(value["x1"]),
                y1=round_int(value["y1"]),
                x2=round_int(value["x2"]),
                y2=round_int(value["y2"]),
            )
        if {"x", "y", "width", "height"}.issubset(value):
            x = round_int(value["x"])
            y = round_int(value["y"])
            return Box(x1=x, y1=y, x2=x + round_int(value["width"]), y2=y + round_int(value["height"]))

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox invalido: {value!r}")

    a, b, c, d = [round_int(part) for part in value]
    if bbox_format == "xywh":
        return Box(x1=a, y1=b, x2=a + c, y2=b + d)
    if bbox_format != "xyxy":
        raise ValueError(f"bbox_format nao suportado: {bbox_format}")
    return Box(x1=a, y1=b, x2=c, y2=d)


def collect_boxes(request: dict[str, Any]) -> list[Box]:
    bbox_format = str(request.get("bbox_format") or "xyxy")
    boxes: list[Box] = []
    for key in ("detections", "bboxes", "boxes"):
        for item in request.get(key) or []:
            boxes.append(parse_box(item, bbox_format=bbox_format))
    if request.get("bbox") is not None:
        boxes.append(parse_box(request["bbox"], bbox_format=bbox_format))
    return boxes


def round_int(value: Any) -> int:
    return int(round(float(value)))
