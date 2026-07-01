from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from server.projects.workspace import safe_path


@dataclass
class PsdRasterLayer:
    name: str
    image: Image.Image
    hidden: bool = False


def export_project_page_psd(root: Path, project: dict[str, Any], page_index: int) -> bytes:
    pages = project.get("paginas") or []
    page = pages[page_index]
    if not isinstance(page, dict):
        raise ValueError("pagina invalida")

    base_image = _load_first_existing_image(root, _page_image_candidates(page, "base"))
    if base_image is None:
        base_image = _load_first_existing_image(root, _page_image_candidates(page, "rendered"))
    if base_image is None:
        base_image = Image.new("RGBA", (1, 1), (255, 255, 255, 255))

    width, height = base_image.size
    layers = [PsdRasterLayer("Original", base_image)]

    inpaint = _load_first_existing_image(root, _page_image_candidates(page, "inpaint"), size=(width, height))
    if inpaint is not None:
        layers.append(PsdRasterLayer("Limpeza (Inpaint)", inpaint))

    image_layers = page.get("image_layers") if isinstance(page.get("image_layers"), dict) else {}
    mask = _load_layer_image(root, image_layers, "mask", size=(width, height))
    if mask is not None:
        layers.append(PsdRasterLayer("Mascara de Deteccao", mask, hidden=True))
    brush = _load_layer_image(root, image_layers, "brush", size=(width, height))
    if brush is not None:
        layers.append(PsdRasterLayer("Pincel de Edicao", brush, hidden=True))

    for index, text_layer in enumerate(_text_layers(page), 1):
        if not _text_value(text_layer).strip():
            continue
        layers.append(PsdRasterLayer(f"Texto {index}", Image.new("RGBA", (width, height), (0, 0, 0, 0))))

    return _write_psd(width, height, layers)


def _page_image_candidates(page: dict[str, Any], kind: str) -> list[str]:
    if kind == "base":
        keys = ["arquivo_original", "original_path"]
        layer_keys = ["base"]
    elif kind == "inpaint":
        keys = ["arquivo_final", "inpaint_path"]
        layer_keys = ["inpaint"]
    else:
        keys = ["arquivo_traduzido", "rendered_path", "translated_path"]
        layer_keys = ["rendered", "inpaint", "base"]

    candidates = [str(page[key]) for key in keys if isinstance(page.get(key), str) and page[key].strip()]
    image_layers = page.get("image_layers") if isinstance(page.get("image_layers"), dict) else {}
    for layer_key in layer_keys:
        layer = image_layers.get(layer_key)
        if isinstance(layer, dict) and isinstance(layer.get("path"), str) and layer["path"].strip():
            candidates.append(str(layer["path"]))
    return candidates


def _load_layer_image(root: Path, image_layers: dict[str, Any], key: str, size: tuple[int, int]) -> Image.Image | None:
    layer = image_layers.get(key)
    if not isinstance(layer, dict):
        return None
    rel = layer.get("path")
    if not isinstance(rel, str) or not rel.strip():
        return None
    return _load_first_existing_image(root, [rel], size=size)


def _load_first_existing_image(root: Path, candidates: list[str], size: tuple[int, int] | None = None) -> Image.Image | None:
    for rel in candidates:
        path = safe_path(root, rel)
        if not path.exists() or not path.is_file():
            continue
        try:
            image = Image.open(path).convert("RGBA")
            if size is not None and image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            return image
        except Exception:
            continue
    return None


def _text_layers(page: dict[str, Any]) -> list[dict[str, Any]]:
    value = page.get("text_layers")
    if not isinstance(value, list):
        value = page.get("textos")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text_value(layer: dict[str, Any]) -> str:
    for key in ["translated", "traduzido", "texto", "original"]:
        value = layer.get(key)
        if isinstance(value, str):
            return value
    return ""


def _write_psd(width: int, height: int, layers: list[PsdRasterLayer]) -> bytes:
    out = BytesIO()
    _write_header(out, width, height)
    _write_u32(out, 0)
    _write_u32(out, 0)

    layer_mask = _layer_and_mask_info(width, height, layers)
    _write_u32(out, len(layer_mask))
    out.write(layer_mask)

    composite = _merged_composite(width, height, layers)
    _write_image_data(out, composite)
    return out.getvalue()


def _write_header(out: BytesIO, width: int, height: int) -> None:
    out.write(b"8BPS")
    _write_u16(out, 1)
    out.write(b"\x00" * 6)
    _write_u16(out, 4)
    _write_u32(out, height)
    _write_u32(out, width)
    _write_u16(out, 8)
    _write_u16(out, 3)


def _layer_and_mask_info(width: int, height: int, layers: list[PsdRasterLayer]) -> bytes:
    layer_info = BytesIO()
    _write_i16(layer_info, -len(layers) if layers else 0)

    channel_payloads: list[list[bytes]] = []
    record_layers = list(reversed(layers))
    for layer in record_layers:
        channels = _layer_channels(layer.image, width, height)
        channel_payloads.append(channels)
        _write_i32(layer_info, 0)
        _write_i32(layer_info, 0)
        _write_i32(layer_info, height)
        _write_i32(layer_info, width)
        _write_u16(layer_info, 4)
        for channel_id, payload in zip([0, 1, 2, -1], channels):
            _write_i16(layer_info, channel_id)
            _write_u32(layer_info, 2 + len(payload))
        layer_info.write(b"8BIM")
        layer_info.write(b"norm")
        layer_info.write(b"\xff")
        layer_info.write(b"\x00")
        layer_info.write(b"\x0a" if layer.hidden else b"\x08")
        layer_info.write(b"\x00")
        extra = _layer_extra_data(layer.name)
        _write_u32(layer_info, len(extra))
        layer_info.write(extra)

    for channels in channel_payloads:
        for payload in channels:
            _write_u16(layer_info, 0)
            layer_info.write(payload)

    _pad(layer_info, 4)
    payload = layer_info.getvalue()

    full = BytesIO()
    _write_u32(full, len(payload))
    full.write(payload)
    _write_u32(full, 0)
    return full.getvalue()


def _layer_extra_data(name: str) -> bytes:
    extra = BytesIO()
    _write_u32(extra, 0)
    _write_u32(extra, 0)
    _write_pascal_string(extra, name, 4)
    return extra.getvalue()


def _write_pascal_string(out: BytesIO, text: str, pad_to: int) -> None:
    payload = text.encode("ascii", errors="replace")[:255]
    out.write(bytes([len(payload)]))
    out.write(payload)
    while out.tell() % pad_to:
        out.write(b"\x00")


def _layer_channels(image: Image.Image, width: int, height: int) -> list[bytes]:
    rgba = image.convert("RGBA")
    if rgba.size != (width, height):
        rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
    raw = rgba.tobytes()
    return [raw[offset::4] for offset in range(4)]


def _merged_composite(width: int, height: int, layers: list[PsdRasterLayer]) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    for layer in layers:
        if layer.hidden:
            continue
        image = layer.image.convert("RGBA")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas.alpha_composite(image)
    return canvas


def _write_image_data(out: BytesIO, image: Image.Image) -> None:
    _write_u16(out, 0)
    for channel in _layer_channels(image, image.width, image.height):
        out.write(channel)


def _write_u16(out: BytesIO, value: int) -> None:
    out.write(int(value).to_bytes(2, "big", signed=False))


def _write_i16(out: BytesIO, value: int) -> None:
    out.write(int(value).to_bytes(2, "big", signed=True))


def _write_u32(out: BytesIO, value: int) -> None:
    out.write(int(value).to_bytes(4, "big", signed=False))


def _write_i32(out: BytesIO, value: int) -> None:
    out.write(int(value).to_bytes(4, "big", signed=True))


def _pad(out: BytesIO, multiple: int) -> None:
    while out.tell() % multiple:
        out.write(b"\x00")
