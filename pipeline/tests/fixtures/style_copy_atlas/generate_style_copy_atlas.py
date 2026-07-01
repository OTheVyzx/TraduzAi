from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[4]
FONT_DIR = ROOT / "fonts"
OUT_DIR = Path(__file__).resolve().parent
ATLAS_PATH = OUT_DIR / "style_copy_atlas.png"
MANIFEST_PATH = OUT_DIR / "style_copy_manifest.json"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (
        FONT_DIR / name,
        FONT_DIR / "google" / name,
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(name)


def _center_text(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int],
    stroke_fill: tuple[int, int, int] | None = None,
    stroke_width: int = 0,
    shadow: tuple[tuple[int, int, int], tuple[int, int]] | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    x = bbox[0] + (bbox[2] - bbox[0] - tw) // 2 - text_bbox[0]
    y = bbox[1] + (bbox[3] - bbox[1] - th) // 2 - text_bbox[1]
    if shadow:
        color, offset = shadow
        draw.text((x + offset[0], y + offset[1]), text, font=font, fill=color)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def _text_mask(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    stroke_width: int = 0,
) -> tuple[Image.Image, tuple[int, int]]:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    x = bbox[0] + (bbox[2] - bbox[0] - tw) // 2 - text_bbox[0]
    y = bbox[1] + (bbox[3] - bbox[1] - th) // 2 - text_bbox[1]
    draw.text((x, y), text, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255)
    return mask, (x, y)


def _draw_gradient_text(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
    stroke_fill: tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    if stroke_fill and stroke_width:
        _center_text(
            image,
            bbox,
            text,
            font,
            fill=stroke_fill,
            stroke_fill=stroke_fill,
            stroke_width=stroke_width,
        )
    mask, _ = _text_mask(image.size, bbox, text, font)
    mask_bbox = mask.getbbox()
    grad_y1, grad_y2 = (mask_bbox[1], mask_bbox[3]) if mask_bbox else (bbox[1], bbox[3])
    gradient = Image.new("RGB", image.size, top)
    pixels = gradient.load()
    y1, y2 = grad_y1, grad_y2
    for y in range(y1, y2):
        t = (y - y1) / max(1, y2 - y1 - 1)
        color = tuple(int(round(top[i] * (1.0 - t) + bottom[i] * t)) for i in range(3))
        for x in range(bbox[0], bbox[2]):
            pixels[x, y] = color
    image.paste(gradient, (0, 0), mask)


def _draw_glow_text(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    radius: int,
) -> None:
    mask, _ = _text_mask(image.size, bbox, text, font)
    halo = mask.filter(ImageFilter.GaussianBlur(radius=radius))
    glow_layer = Image.new("RGB", image.size, glow)
    image.paste(glow_layer, (0, 0), halo)
    image.paste(Image.new("RGB", image.size, fill), (0, 0), mask)


def generate() -> dict:
    width = 860
    row_h = 150
    cases: list[dict] = []
    image = Image.new("RGB", (width, row_h * 14), (238, 238, 238))

    comic = _font("ComicNeue-Bold.ttf", 54)
    comic_small = _font("ComicNeue-Bold.ttf", 46)
    komika = _font("KOMIKAX_.ttf", 52)
    luck = _font("LuckiestGuy-Regular.ttf", 56)

    def case(case_id: str, row: int, bg: tuple[int, int, int], expected: dict) -> tuple[int, int, int, int]:
        pad = 22
        bbox = (pad, row * row_h + 18, width - pad, (row + 1) * row_h - 18)
        ImageDraw.Draw(image).rectangle(bbox, fill=bg)
        cases.append({"id": case_id, "bbox": list(bbox), "expected": expected})
        return bbox

    bbox = case("plain_balloon_black", 0, (255, 255, 255), {
        "font_name": "ComicNeue-Bold.ttf", "text_color": "#000000", "stroke": False, "gradient": False, "glow": False, "shadow": False
    })
    _center_text(image, bbox, "OLA TUDO BEM", comic, fill=(0, 0, 0))

    bbox = case("plain_balloon_white_on_dark", 1, (25, 30, 38), {
        "font_name": "ComicNeue-Bold.ttf", "text_color": "#FFFFFF", "stroke": False, "gradient": False, "glow": False, "shadow": False
    })
    _center_text(image, bbox, "OLA TUDO BEM", comic, fill=(255, 255, 255))

    bbox = case("white_fill_black_outline", 2, (245, 210, 200), {
        "font_family": "impact", "text_color": "#FFFFFF", "stroke_color": "#000000", "stroke_width_px_min": 2, "gradient": False, "glow": False
    })
    _center_text(image, bbox, "OLA TUDO BEM", komika, fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=5)

    bbox = case("black_fill_white_outline", 3, (145, 0, 245), {
        "font_family": "impact", "text_color": "#000000", "stroke_color": "#FFFFFF", "stroke_width_px_min": 2, "gradient": False, "glow": False
    })
    _center_text(image, bbox, "OLA TUDO BEM", komika, fill=(0, 0, 0), stroke_fill=(255, 255, 255), stroke_width=6)

    bbox = case("dark_blue_vertical_gradient", 4, (255, 255, 255), {
        "font_name": "ComicNeue-Bold.ttf", "gradient": True, "glow": False, "stroke": False
    })
    _draw_gradient_text(image, bbox, "ALCHEMY ABILITY", comic, top=(8, 8, 118), bottom=(5, 48, 48))

    bbox = case("impact_gradient_outline", 5, (110, 0, 220), {
        "font_family": "impact", "gradient": True, "stroke_color": "#FFFFFF", "stroke_width_px_min": 2, "glow": False
    })
    _draw_gradient_text(
        image,
        bbox,
        "OLA TUDO BEM",
        komika,
        top=(42, 44, 130),
        bottom=(46, 190, 195),
        stroke_fill=(255, 255, 255),
        stroke_width=5,
    )

    bbox = case("cyan_card_white_glow", 6, (85, 195, 242), {
        "font_family": "impact", "text_color": "#FFFFFF", "glow": True, "stroke": False, "gradient": False
    })
    _draw_glow_text(image, bbox, "CONSTELLATION", komika, fill=(255, 255, 255), glow=(230, 255, 255), radius=5)

    bbox = case("black_text_pink_glow", 7, (150, 0, 245), {
        "font_family": "impact", "text_color": "#000000", "glow": True, "stroke": False, "gradient": False
    })
    _draw_glow_text(image, bbox, "OLA TUDO BEM", komika, fill=(0, 0, 0), glow=(255, 210, 238), radius=6)

    bbox = case("gray_shadow_offset", 8, (255, 255, 255), {
        "font_name": "ComicNeue-Bold.ttf", "text_color": "#000000", "shadow": True, "glow": False, "stroke": False
    })
    _center_text(image, bbox, "OLA TUDO BEM", comic, fill=(0, 0, 0), shadow=((155, 155, 155), (5, 5)))

    bbox = case("thick_impact_font", 9, (255, 255, 255), {
        "font_family": "impact", "text_color": "#000000", "stroke": False, "gradient": False, "glow": False
    })
    _center_text(image, bbox, "BOOM", luck, fill=(0, 0, 0))

    bbox = case("thin_no_effect_ui_text", 10, (255, 255, 255), {
        "font_name": "ComicNeue-Bold.ttf", "text_color": "#111111", "stroke": False, "gradient": False, "glow": False, "shadow": False
    })
    _center_text(image, bbox, "MISSED CALL", comic_small, fill=(17, 17, 17))

    bbox = case("non_text_lightning_texture", 11, (32, 28, 22), {
        "no_text": True, "stroke": False, "gradient": False, "glow": False, "shadow": False
    })
    draw = ImageDraw.Draw(image)
    for x in range(bbox[0] + 25, bbox[2] - 25, 70):
        points = [
            (x, bbox[1] + 8),
            (x + 34, bbox[1] + 44),
            (x + 12, bbox[1] + 54),
            (x + 52, bbox[3] - 10),
        ]
        draw.line(points, fill=(245, 240, 165), width=8)
        draw.line(points, fill=(120, 110, 45), width=2)

    bbox = case("non_text_blue_card_texture", 12, (70, 190, 230), {
        "no_text": True, "stroke": False, "gradient": False, "glow": False, "shadow": False
    })
    draw = ImageDraw.Draw(image)
    for y in range(bbox[1], bbox[3]):
        t = (y - bbox[1]) / max(1, bbox[3] - bbox[1] - 1)
        color = (int(80 - 20 * t), int(205 + 20 * t), int(238 + 12 * t))
        draw.line([(bbox[0], y), (bbox[2], y)], fill=color)
    for x in range(bbox[0] + 12, bbox[2], 95):
        draw.rectangle((x, bbox[1] + 18, x + 46, bbox[1] + 26), fill=(235, 255, 255))
        draw.rectangle((x + 20, bbox[3] - 30, x + 78, bbox[3] - 21), fill=(215, 248, 255))

    bbox = case("non_text_vertical_art_sliver", 13, (245, 245, 245), {
        "no_text": True, "stroke": False, "gradient": False, "glow": False, "shadow": False
    })
    draw = ImageDraw.Draw(image)
    draw.rectangle((bbox[0] + 370, bbox[1] + 4, bbox[0] + 430, bbox[3] - 4), fill=(25, 22, 20))
    draw.line((bbox[0] + 382, bbox[1] + 6, bbox[0] + 422, bbox[3] - 8), fill=(248, 245, 230), width=11)
    draw.line((bbox[0] + 390, bbox[1] + 6, bbox[0] + 426, bbox[3] - 10), fill=(78, 65, 38), width=4)

    manifest = {"image": ATLAS_PATH.name, "cases": cases}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(ATLAS_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generate()
    print(str(ATLAS_PATH))
    print(str(MANIFEST_PATH))
