from __future__ import annotations

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from typesetter.backend_contract import KOHARU_RUST_CAPABILITIES, TypesettingRenderRequest, TypesettingRenderResult


KOHARU_RENDERER_CAPABILITIES = KOHARU_RUST_CAPABILITIES


class RustRendererError(RuntimeError):
    pass


class KoharuBackendUnavailable(RustRendererError):
    pass


@dataclass(frozen=True)
class KoharuRenderResult:
    render_bbox: list[int]
    font_size_px: int
    fit_status: str
    backend: str
    payload: dict[str, Any]


def resolve_koharu_bridge_path() -> Path:
    configured = os.environ.get("TRADUZAI_KOHARU_RENDERER_BIN", "").strip()
    if not configured:
        raise KoharuBackendUnavailable("TRADUZAI_KOHARU_RENDERER_BIN is not configured")
    path = Path(configured)
    if not path.exists():
        raise KoharuBackendUnavailable(f"Koharu renderer bridge not found: {path}")
    return path


def render_with_koharu_backend(
    request: TypesettingRenderRequest,
    *,
    bridge_path: Path | None = None,
    command_prefix: list[str] | None = None,
    timeout: int = 30,
) -> KoharuRenderResult:
    bridge = Path(bridge_path) if bridge_path is not None else resolve_koharu_bridge_path()
    command = [*(command_prefix or []), str(bridge)]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request.to_mapping(), ensure_ascii=False),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise KoharuBackendUnavailable(f"Koharu renderer bridge failed: {exc}") from exc
    try:
        payload = json.loads(completed.stdout.splitlines()[-1])
        normalized = TypesettingRenderResult.from_mapping(payload).to_mapping()
    except (IndexError, json.JSONDecodeError, ValueError) as exc:
        raise KoharuBackendUnavailable("Koharu renderer returned an invalid response") from exc
    return KoharuRenderResult(
        render_bbox=list(normalized["render_bbox"]),
        font_size_px=int(normalized.get("font_size_px") or 0),
        fit_status=str(normalized.get("fit_status") or "review_required"),
        backend=str(normalized.get("backend") or "koharu"),
        payload=normalized,
    )


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
