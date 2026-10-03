"""Stable identity for reusable visual analysis inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


_NON_VISUAL_KEYS = frozenset({
    "work_dir", "source_path", "capitulo", "idioma_destino",
    "glossario", "contexto", "translation_context", "ollama_host",
    "ollama_model", "style_copy_mode", "font", "font_path", "font_size",
    "text_position", "position", "debug", "strict", "export_mode",
})

_VISUAL_CODE = (
    "vision_stack/runtime.py", "vision_stack/detector.py",
    "vision_stack/ocr.py", "strip/detect_balloons.py",
    "strip/experimental_mayo.py", "ownership/coverage.py",
    "ownership/ocr_contract.py", "ownership/consensus_v2.py",
    "strip/process_bands.py", "strip/run.py",
)
_VISUAL_PACKAGES = ("paddleocr", "paddlepaddle-gpu", "torch", "opencv-python")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provider_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    pipeline_root = Path(__file__).resolve().parents[1]
    code = {
        name: _file_sha256(pipeline_root / name)
        for name in _VISUAL_CODE
        if (pipeline_root / name).is_file()
    }
    packages = {}
    for package in _VISUAL_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    model_roots = [Path(str(config.get("models_dir") or ""))]
    paddle_root = Path.home() / ".paddleocr" / "whl"
    if paddle_root.is_dir():
        model_roots.append(paddle_root)
    assets = {}
    for root in model_roots:
        if not root.is_dir():
            continue
        candidates = [root / "comic-text-detector.pt"]
        if root == paddle_root:
            candidates.extend(path for path in root.rglob("*") if path.is_file())
        else:
            huggingface = root / "huggingface"
            if huggingface.is_dir():
                for prefix in ("models--mayocream--comic-text-detector", "models--ogkalu--comic-text-and-bubble-detector", "models--mayocream--anime-text-yolo"):
                    candidate_root = huggingface / prefix / "snapshots"
                    if candidate_root.is_dir():
                        candidates.extend(path for path in candidate_root.rglob("*") if path.is_file())
        for path in candidates:
            if path.is_file():
                assets[str(path.resolve())] = _file_sha256(path)
    mayo_dir = Path(os.environ.get("TRADUZAI_EXPERIMENTAL_MAYO_MODEL_DIR", ""))
    if os.environ.get("TRADUZAI_EXPERIMENTAL_MAYO", "0").strip().lower() in {"1", "true", "yes", "on"}:
        for name in ("model.safetensors", "yolov8m-seg-local.yaml"):
            path = mayo_dir / name
            assets[str(path.resolve())] = _file_sha256(path) if path.is_file() else None
    visual_env = {name: os.environ.get(name, "") for name in (
        "TRADUZAI_OCR_CROP_FIRST", "TRADUZAI_EXPERIMENTAL_MAYO",
        "TRADUZAI_EXPERIMENTAL_MAYO_MODEL_DIR", "TRADUZAI_EXPERIMENTAL_MAYO_DEVICE",
        "TRADUZAI_STRIP_DETECT_FULL_PAGE", "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN",
        "TRADUZAI_STRIP_DARK_BALLOON_BAND_SCAN", "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN",
        "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE",
    )}
    return {"cache_version": "vision-pre-ocr-v1", "code": code, "packages": packages,
            "assets": assets, "visual_env": visual_env}


def visual_config_sha256(config: dict[str, Any]) -> str:
    """Hash visual settings without output, translation, or typography settings."""
    visual = {key: value for key, value in config.items() if key not in _NON_VISUAL_KEYS}
    encoded = json.dumps(
        {"visual_config": visual, "provider": _provider_fingerprint(config)},
        ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
