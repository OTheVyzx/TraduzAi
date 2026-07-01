from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FILE_DIGEST_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class EditorVisionCacheKey:
    kind: str
    digest: str
    cache_dir: Path

    @property
    def path(self) -> Path:
        return self.cache_dir / f"{self.kind}-{self.digest}.json"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_FILE_DIGEST_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_digest(path),
    }


def _normal_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _cache_dir(project_path: Path) -> Path:
    return project_path.resolve().parent / "layers" / "vision-cache"


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _layer_geometry(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for layer in layers:
        bbox = layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox")
        normalized.append({"id": str(layer.get("id") or ""), "bbox": bbox})
    return normalized


def build_detect_ocr_cache_key(
    *,
    project_path: Path,
    page_index: int,
    image_path: Path,
    idioma_origem: str,
    engine_preset_id: str,
    schema_version: int,
) -> EditorVisionCacheKey:
    payload = {
        "kind": "detect_ocr",
        "schema_version": schema_version,
        "project": str(project_path.resolve()),
        "page_index": int(page_index),
        "image": _stat_signature(image_path),
        "idioma_origem": _normal_text(idioma_origem),
        "engine_preset_id": _normal_text(engine_preset_id),
    }
    return EditorVisionCacheKey("detect_ocr", _digest(payload), _cache_dir(project_path))


def build_ocr_layers_cache_key(
    *,
    project_path: Path,
    page_index: int,
    image_path: Path,
    layers: list[dict[str, Any]],
    idioma_origem: str,
    engine_preset_id: str,
    schema_version: int,
) -> EditorVisionCacheKey:
    payload = {
        "kind": "ocr_layers",
        "schema_version": schema_version,
        "project": str(project_path.resolve()),
        "page_index": int(page_index),
        "image": _stat_signature(image_path),
        "layers": _layer_geometry(layers),
        "idioma_origem": _normal_text(idioma_origem),
        "engine_preset_id": _normal_text(engine_preset_id),
    }
    return EditorVisionCacheKey("ocr_layers", _digest(payload), _cache_dir(project_path))


def build_detect_ocr_payload(
    *,
    page_index: int,
    text_layers: list[dict[str, Any]],
    inpaint_blocks: list[dict[str, Any]],
    ui_layout_components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": "ready",
        "kind": "detect_ocr",
        "schema_version": 1,
        "page_index": int(page_index),
        "text_layers": text_layers,
        "inpaint_blocks": inpaint_blocks,
        "ui_layout_components": list(ui_layout_components or []),
    }


def build_ocr_layers_payload(
    *,
    page_index: int,
    layer_updates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "ready",
        "kind": "ocr_layers",
        "schema_version": 1,
        "page_index": int(page_index),
        "layer_updates": layer_updates,
    }


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_detect_ocr_payload(payload: dict[str, Any] | None, *, page_index: int) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ready"
        and payload.get("kind") == "detect_ocr"
        and payload.get("schema_version") == 1
        and _safe_int(payload.get("page_index")) == int(page_index)
        and isinstance(payload.get("text_layers"), list)
        and isinstance(payload.get("inpaint_blocks"), list)
    )


def is_ocr_layers_payload(payload: dict[str, Any] | None, *, page_index: int) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ready"
        and payload.get("kind") == "ocr_layers"
        and payload.get("schema_version") == 1
        and _safe_int(payload.get("page_index")) == int(page_index)
        and isinstance(payload.get("layer_updates"), list)
    )


def read_cache_entry(key: EditorVisionCacheKey) -> dict[str, Any] | None:
    try:
        data = json.loads(key.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("status") == "ready" else None


def write_cache_entry(key: EditorVisionCacheKey, payload: dict[str, Any]) -> None:
    key.cache_dir.mkdir(parents=True, exist_ok=True)
    temp = key.cache_dir / f".{key.kind}-{key.digest}-{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, key.path)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
