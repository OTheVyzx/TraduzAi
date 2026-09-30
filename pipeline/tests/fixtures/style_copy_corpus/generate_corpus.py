"""Generate deterministic owner-paired source crops and masks for tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CASES_ROOT = ROOT / "cases"


def _text_mask(
    text: str,
    *,
    shape: tuple[int, int],
    scale: float,
    thickness: int,
    condensed: float = 1.0,
) -> np.ndarray:
    height, width = shape
    raw = np.zeros((height, width), dtype=np.uint8)
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (max(2, (width - size[0]) // 2), max(size[1] + 2, (height + size[1] - baseline) // 2))
    cv2.putText(
        raw,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        255,
        thickness,
        cv2.LINE_8,
    )
    if condensed >= 0.999:
        return raw
    x, y, w, h = cv2.boundingRect(raw)
    compressed = cv2.resize(raw[y : y + h, x : x + w], None, fx=condensed, fy=1.0, interpolation=cv2.INTER_NEAREST)
    output = np.zeros_like(raw)
    target_x = (width - compressed.shape[1]) // 2
    output[y : y + compressed.shape[0], target_x : target_x + compressed.shape[1]] = compressed
    return output


def _context(shape: tuple[int, int], margin: int = 5) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[margin : shape[0] - margin, margin : shape[1] - margin] = 255
    return mask


def _write_case(
    case_id: str,
    *,
    background_rgb: tuple[int, int, int],
    fill_rgb: tuple[int, int, int],
    text: str,
    scale: float,
    thickness: int,
    condensed: float = 1.0,
    stroke_rgb: tuple[int, int, int] | None = None,
    stroke_radius: int = 0,
) -> None:
    shape = (128, 360)
    glyph = _text_mask(
        text,
        shape=shape,
        scale=scale,
        thickness=thickness,
        condensed=condensed,
    )
    image = np.full((*shape, 3), background_rgb, dtype=np.uint8)
    if stroke_rgb is not None and stroke_radius > 0:
        kernel_size = stroke_radius * 2 + 1
        expanded = cv2.dilate(glyph, np.ones((kernel_size, kernel_size), np.uint8))
        image[(expanded > 0) & (glyph == 0)] = stroke_rgb
    image[glyph > 0] = fill_rgb
    context = _context(shape)
    for suffix, value in (
        ("source", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
        ("glyph", glyph),
        ("context", context),
    ):
        target = CASES_ROOT / f"{case_id}.{suffix}.png"
        if not cv2.imwrite(str(target), value):
            raise RuntimeError(f"failed to write {target}")


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    declared = {item["case_id"] for item in manifest["cases"]}
    expected = {
        "calibration_plain_balloon",
        "calibration_condensed_card",
        "holdout_colored_card",
        "holdout_burst_effect",
    }
    if declared != expected:
        raise RuntimeError("manifest cases and generator cases diverge")
    CASES_ROOT.mkdir(parents=True, exist_ok=True)
    _write_case(
        "calibration_plain_balloon",
        background_rgb=(250, 250, 250), fill_rgb=(24, 24, 24),
        text="PLAIN BODY", scale=1.15, thickness=3,
    )
    _write_case(
        "calibration_condensed_card",
        background_rgb=(60, 150, 210), fill_rgb=(248, 248, 248),
        text="CONDENSED CARD", scale=1.2, thickness=3, condensed=0.58,
    )
    _write_case(
        "holdout_colored_card",
        background_rgb=(212, 154, 48), fill_rgb=(255, 255, 255),
        text="SYSTEM NOTICE", scale=1.1, thickness=3,
    )
    _write_case(
        "holdout_burst_effect",
        background_rgb=(22, 18, 44), fill_rgb=(242, 217, 78),
        stroke_rgb=(36, 58, 154), stroke_radius=4,
        text="BURST", scale=1.7, thickness=4,
    )


if __name__ == "__main__":
    main()
