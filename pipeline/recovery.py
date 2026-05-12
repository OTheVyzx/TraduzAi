from __future__ import annotations

from pathlib import Path
from typing import Any


def _open_image(value: Any, mode: str):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert(mode)
    return Image.open(value).convert(mode)


def bake_recovery_layer(rendered, original, mask):
    """Return rendered with original pixels restored where mask is active."""
    from PIL import Image

    rendered_img = _open_image(rendered, "RGBA")
    original_img = _open_image(original, "RGBA")
    mask_img = _open_image(mask, "L")

    if original_img.size != rendered_img.size:
        original_img = original_img.resize(rendered_img.size, Image.Resampling.LANCZOS)
    if mask_img.size != rendered_img.size:
        mask_img = mask_img.resize(rendered_img.size, Image.Resampling.NEAREST)

    if mask_img.getbbox() is None:
        return rendered_img

    composed = rendered_img.copy()
    composed.paste(original_img, (0, 0), mask_img)
    return composed


def _save_recovered_image(result, rendered_path: Path) -> None:
    suffix = rendered_path.suffix.lower()
    output = result.convert("RGB") if suffix in {".jpg", ".jpeg"} else result
    temp_path = rendered_path.with_name(f".{rendered_path.stem}.tmp{rendered_path.suffix}")
    try:
        if suffix in {".jpg", ".jpeg"}:
            output.save(temp_path, quality=95)
        else:
            output.save(temp_path)
        temp_path.replace(rendered_path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def apply_recovery_layer(rendered_path: str | Path, original_path: str | Path, mask_path: str | Path) -> bool:
    rendered_path = Path(rendered_path)
    original_path = Path(original_path)
    mask_path = Path(mask_path)
    if not rendered_path.exists() or not original_path.exists() or not mask_path.exists():
        return False

    result = bake_recovery_layer(rendered_path, original_path, mask_path)
    _save_recovered_image(result, rendered_path)
    return True
