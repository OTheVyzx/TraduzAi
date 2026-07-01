from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from server.config import Settings
from server.deps import current_user, get_settings
from server.models import User
from server.projects.api import _require_project_access
from server.projects.pipeline_runner import process_block, run_page_action as run_pipeline_page_action
from server.projects.workspace import load_project, load_state, page_at, project_root, save_project, save_state, write_png_layer


router = APIRouter(prefix="/api/projects/{project_id}/editor", tags=["editor"])


class LayerPatchPayload(BaseModel):
    patch: dict


class LayerCreatePayload(BaseModel):
    layer: dict


class VisibilityPayload(BaseModel):
    layer: str
    visible: bool
    page_index: int | None = None
    layer_kind: str | None = None
    layer_key: str | None = None
    layer_id: str | None = None


class BitmapPayload(BaseModel):
    png_data: str | None = None
    op: str = "replace"
    dirty_bbox: list[float] | None = None
    color: str | None = None
    opacity: float | None = None
    hardness: float | None = None
    width: int | None = None
    height: int | None = None
    brush_size: int | None = None
    clear: bool | None = None
    erase: bool | None = None
    strokes: list[list[list[float]]] | None = None


class PageActionPayload(BaseModel):
    action: str
    region: dict | None = None
    block_id: str | None = None
    mode: str | None = None


@router.get("/pages/{page_index}")
def load_editor_page(project_id: str, page_index: int, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    return {"project": project, "page": page, "page_index": page_index}


@router.patch("/pages/{page_index}/text-layers/{layer_id}")
def patch_text_layer(
    project_id: str,
    page_index: int,
    layer_id: str,
    payload: LayerPatchPayload,
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    layers = _text_layers(page)
    for layer in layers:
        if str(layer.get("id")) == layer_id:
            _patch_text_layer(layer, payload.patch)
            _sync_text_layers(page)
            save_project(root, project)
            return {"layer": layer}
    raise HTTPException(status_code=404, detail="camada nao encontrada")


@router.post("/pages/{page_index}/text-layers")
def create_text_layer(
    project_id: str,
    page_index: int,
    payload: LayerCreatePayload,
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    layers = _text_layers(page)
    layer = _normalize_text_layer(payload.layer, len(layers))
    layers.append(layer)
    _sync_text_layers(page)
    save_project(root, project)
    return {"layer": layer}


@router.delete("/pages/{page_index}/text-layers/{layer_id}")
def delete_text_layer(project_id: str, page_index: int, layer_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    layers = _text_layers(page)
    next_layers = [layer for layer in layers if str(layer.get("id")) != layer_id]
    if len(next_layers) == len(layers):
        raise HTTPException(status_code=404, detail="camada nao encontrada")
    page["text_layers"] = next_layers
    page["textos"] = next_layers
    save_project(root, project)
    return {"ok": True}


@router.post("/visibility")
def set_visibility(project_id: str, payload: VisibilityPayload, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    if payload.page_index is not None and payload.layer_kind:
        project = load_project(root)
        page = page_at(project, payload.page_index)
        if payload.layer_kind == "image":
            key = payload.layer_key or payload.layer
            image_layers = page.setdefault("image_layers", {})
            layer = image_layers.setdefault(key, {"key": key, "path": None, "visible": True, "locked": False})
            layer["visible"] = payload.visible
            save_project(root, project)
            return {"ok": True}
        if payload.layer_kind == "text":
            layer_id = payload.layer_id or payload.layer
            for layer in _text_layers(page):
                if str(layer.get("id")) == str(layer_id):
                    layer["visible"] = payload.visible
                    _sync_text_layers(page)
                    save_project(root, project)
                    return {"ok": True}
            raise HTTPException(status_code=404, detail="camada nao encontrada")
    state = load_state(root)
    visibility = state.setdefault("layer_visibility", {})
    visibility[payload.layer] = payload.visible
    save_state(root, state)
    return {"ok": True}


@router.post("/pages/{page_index}/mask")
def update_mask(project_id: str, page_index: int, payload: BitmapPayload, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    return _write_bitmap(project_id, page_index, "mask", payload, user, settings)


@router.post("/pages/{page_index}/brush")
def update_brush(project_id: str, page_index: int, payload: BitmapPayload, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    return _write_bitmap(project_id, page_index, "brush", payload, user, settings)


@router.post("/pages/{page_index}/recovery")
def update_recovery(project_id: str, page_index: int, payload: BitmapPayload, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    return _write_bitmap(project_id, page_index, "recovery", payload, user, settings)


@router.post("/pages/{page_index}/mask/png")
def write_mask_png(project_id: str, page_index: int, payload: BitmapPayload, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    return _write_bitmap(project_id, page_index, "mask", payload, user, settings)


@router.post("/pages/{page_index}/actions")
def run_page_action(project_id: str, page_index: int, payload: PageActionPayload, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    if payload.action not in {"detect", "detect_boxes", "ocr", "translate", "inpaint", "retypeset", "process-block"}:
        raise HTTPException(status_code=422, detail="acao invalida")
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    if payload.action == "process-block":
        if not payload.block_id or not payload.mode:
            raise HTTPException(status_code=422, detail="block_id e mode sao obrigatorios")
        changed_assets = process_block(root, page_index, payload.block_id, payload.mode)
    else:
        changed_assets = run_pipeline_page_action(root, page_index, payload.action, payload.region)
    project = load_project(root)
    page = page_at(project, page_index)
    history = page.setdefault("action_history", [])
    history.append({"action": payload.action, "region": payload.region})
    save_project(root, project)
    return {"ok": True, "changed_assets": changed_assets, "page": page}


def _write_bitmap(project_id: str, page_index: int, layer: str, payload: BitmapPayload, user: User, settings: Settings):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    image_layers = page.setdefault("image_layers", {})
    if layer == "recovery" and not payload.png_data:
        current_inpaint = _current_inpaint_path(page)
        if not current_inpaint:
            raise HTTPException(status_code=422, detail="png_data obrigatorio para recuperacao")
        return {"asset_path": current_inpaint, "url": f"/api/projects/{project_id}/assets/{current_inpaint}"}

    rel = write_png_layer(root, layer, page_index, payload.png_data)
    visible_key = "inpaint" if layer == "recovery" else layer
    image_layer = image_layers.setdefault(visible_key, {"key": visible_key, "path": rel, "visible": True, "locked": False})
    image_layer.update({"key": visible_key, "path": rel, "visible": True, "locked": bool(image_layer.get("locked", False))})
    if layer == "recovery":
        recovery_layer = image_layers.setdefault("recovery", {"key": "recovery", "path": rel, "visible": False, "locked": False})
        recovery_layer.update({"key": "recovery", "path": rel, "visible": False})
    save_project(root, project)
    state = load_state(root)
    state["dirty"] = True
    preview = state.setdefault("preview", {})
    preview[str(page_index)] = {"status": "stale"}
    save_state(root, state)
    return {"asset_path": rel, "url": f"/api/projects/{project_id}/assets/{rel}"}


def _current_inpaint_path(page: dict) -> str | None:
    image_layers = page.get("image_layers") if isinstance(page.get("image_layers"), dict) else {}
    inpaint_layer = image_layers.get("inpaint") if isinstance(image_layers.get("inpaint"), dict) else {}
    for value in [
        inpaint_layer.get("path"),
        page.get("inpaint_path"),
        page.get("rendered_path"),
        page.get("translated_path"),
        page.get("arquivo_traduzido"),
    ]:
        if value:
            return str(value).replace("\\", "/")
    return None


def _text_layers(page: dict) -> list[dict]:
    layers = page.setdefault("text_layers", page.get("textos") or [])
    if not isinstance(layers, list):
        layers = []
        page["text_layers"] = layers
    return layers


def _sync_text_layers(page: dict) -> None:
    layers = _text_layers(page)
    layers.sort(key=lambda item: int(item.get("order") or 0))
    page["text_layers"] = layers
    page["textos"] = layers


def _bbox(value) -> list[float]:
    if isinstance(value, list) and len(value) >= 4:
        return [float(value[0] or 0), float(value[1] or 0), float(value[2] or 32), float(value[3] or 32)]
    return [0.0, 0.0, 32.0, 32.0]


def _default_style() -> dict:
    return {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 28,
        "cor": "#000000",
        "cor_gradiente": [],
        "contorno": "",
        "contorno_px": 0,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": False,
        "sombra_cor": "",
        "sombra_offset": [0, 0],
        "bold": True,
        "italico": False,
        "rotacao": 0,
        "alinhamento": "center",
        "force_upper": False,
    }


def _normalize_text_layer(raw: dict, order: int) -> dict:
    layer = dict(raw or {})
    bbox = _bbox(layer.get("layout_bbox") or layer.get("bbox") or layer.get("source_bbox") or layer.get("balloon_bbox"))
    style = _default_style()
    if isinstance(layer.get("style"), dict):
        style.update(layer["style"])
    if isinstance(layer.get("estilo"), dict):
        style.update(layer["estilo"])
    layer.update(
        {
            "id": str(layer.get("id") or f"text-{order + 1}"),
            "kind": "text",
            "style_origin": layer.get("style_origin") or "editor",
            "source_bbox": _bbox(layer.get("source_bbox") or bbox),
            "layout_bbox": bbox,
            "render_bbox": layer.get("render_bbox"),
            "bbox": bbox,
            "balloon_bbox": _bbox(layer.get("balloon_bbox") or bbox),
            "tipo": layer.get("tipo") or "fala",
            "original": layer.get("original") or "",
            "traduzido": layer.get("traduzido") or layer.get("translated") or "",
            "translated": layer.get("translated") or layer.get("traduzido") or "",
            "confianca_ocr": layer.get("confianca_ocr", layer.get("ocr_confidence", 1)),
            "ocr_confidence": layer.get("ocr_confidence", layer.get("confianca_ocr", 1)),
            "estilo": style,
            "style": style,
            "visible": layer.get("visible", True),
            "locked": layer.get("locked", False),
            "order": int(layer.get("order", order)),
            "render_preview_path": layer.get("render_preview_path"),
            "detector": layer.get("detector"),
            "line_polygons": layer.get("line_polygons"),
            "source_direction": layer.get("source_direction"),
            "rendered_direction": layer.get("rendered_direction"),
            "source_language": layer.get("source_language"),
            "rotation_deg": layer.get("rotation_deg", 0),
            "detected_font_size_px": layer.get("detected_font_size_px"),
            "balloon_subregions": layer.get("balloon_subregions") or [],
            "layout_group_size": layer.get("layout_group_size", 1),
        }
    )
    return layer


def _patch_text_layer(layer: dict, patch: dict) -> None:
    layer.update(patch)
    if "translated" in patch and "traduzido" not in patch:
        layer["traduzido"] = patch["translated"]
    if "traduzido" in patch and "translated" not in patch:
        layer["translated"] = patch["traduzido"]
    if "style" in patch and "estilo" not in patch:
        layer["estilo"] = patch["style"]
    if "estilo" in patch and "style" not in patch:
        layer["style"] = patch["estilo"]
    if "bbox" in patch:
        layer["layout_bbox"] = patch["bbox"]
        layer["balloon_bbox"] = patch["bbox"]
