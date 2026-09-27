"""Small deterministic raster control for the Integration V1 bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ownership.hash_contract import canonical_json_sha256
from typesetter.renderer import render_text_block


_RECIPE: dict[str, Any] = {
    "schema": "traduzai.integration.control-rerender.v1",
    "canvas": {"width": 260, "height": 120, "background_rgb": [255, 255, 255]},
    "text": {
        "id": "integration-control",
        "text": "TEST",
        "translated": "TESTE",
        "tipo": "fala",
        "layout_profile": "style_probe",
        "bbox": [30, 30, 230, 92],
        "source_bbox": [30, 30, 230, 92],
        "text_pixel_bbox": [30, 30, 230, 92],
        "target_bbox": [20, 20, 240, 105],
        "position_bbox": [20, 20, 240, 105],
        "capacity_bbox": [20, 20, 240, 105],
        "safe_text_box": [20, 20, 240, 105],
        "_debug_safe_text_box": [20, 20, 240, 105],
        "balloon_bbox": [10, 10, 250, 112],
        "background_rgb": [255, 255, 255],
        "style_origin": "user_override",
        "estilo": {
            "fonte": "ComicNeue-Bold.ttf",
            "tamanho": 48,
            "alinhamento": "center",
            "force_upper": True,
            "cor": "#111111",
        },
    },
}


def _render() -> Image.Image:
    recipe = json.loads(json.dumps(_RECIPE))
    canvas = recipe["canvas"]
    image = Image.new(
        "RGB",
        (canvas["width"], canvas["height"]),
        tuple(canvas["background_rgb"]),
    )
    recipe["text"]["style"] = dict(recipe["text"]["estilo"])
    render_text_block(image, recipe["text"])
    if np.asarray(image).min() == 255:
        raise ValueError("control rerender did not rasterize glyph pixels")
    return image


def _pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def run_control_rerender(destination: str | Path) -> dict[str, Any]:
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    first = _render()
    second = _render()
    first_sha = _pixel_sha256(first)
    second_sha = _pixel_sha256(second)
    if first_sha != second_sha:
        raise ValueError("same control recipe produced different pixels")
    image_path = output / "control-rerender.png"
    first.save(image_path, format="PNG", optimize=False)
    receipt = {
        "schema": "traduzai.integration.control-rerender-receipt.v1",
        "recipe_sha256": canonical_json_sha256(_RECIPE),
        "first_output_sha256": first_sha,
        "second_output_sha256": second_sha,
        "published_output_sha256": _pixel_sha256(Image.open(image_path).convert("RGB")),
        "same_recipe_same_pixels": first_sha == second_sha,
        "relative_output_path": image_path.name,
    }
    (output / "control-rerender-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_control_rerender(args.destination), ensure_ascii=False))


if __name__ == "__main__":
    main()
