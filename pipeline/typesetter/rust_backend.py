from __future__ import annotations

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


class RustRendererError(RuntimeError):
    pass


def rust_renderer_enabled() -> bool:
    return os.environ.get("TRADUZAI_RENDERER_BACKEND", "").strip().lower() == "koharu_rust"


def rust_renderer_strict() -> bool:
    return os.environ.get("TRADUZAI_RENDERER_STRICT", "").strip().lower() in {"1", "true", "yes"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _renderer_bridge_executable() -> Path:
    configured = os.environ.get("TRADUZAI_RENDERER_BRIDGE")
    if configured:
        return Path(configured)

    root = _repo_root()
    candidates = [
        root / "src-tauri" / "renderer-bridge" / "target" / "release" / "renderer-bridge.exe",
        root / "src-tauri" / "renderer-bridge" / "target" / "debug" / "renderer-bridge.exe",
        root / "src-tauri" / "renderer-bridge" / "target" / "release" / "renderer-bridge",
        root / "src-tauri" / "renderer-bridge" / "target" / "debug" / "renderer-bridge",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def render_request_to_image(request: dict[str, Any], timeout: int = 30) -> Image.Image:
    exe = _renderer_bridge_executable()
    if not exe.exists():
        raise RustRendererError(f"renderer bridge executable not found: {exe}")

    with tempfile.TemporaryDirectory(prefix="traduzai-renderer-") as tmp:
        tmp_path = Path(tmp)
        request_path = tmp_path / "request.json"
        output_path = tmp_path / "output.png"

        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(exe), "--request", str(request_path), "--output", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RustRendererError(stderr or f"renderer bridge failed with code {exc.returncode}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RustRendererError("renderer bridge timed out") from exc
        if not output_path.exists():
            raise RustRendererError("renderer bridge did not produce output image")
        metadata: dict[str, Any] = {}
        stdout = (getattr(completed, "stdout", "") or "").strip()
        if stdout:
            try:
                parsed = json.loads(stdout.splitlines()[-1])
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {}
        with Image.open(output_path) as rendered:
            image = rendered.convert("RGBA")
            if metadata:
                image.info["renderer_bridge"] = metadata
            return image
