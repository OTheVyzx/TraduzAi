import json
import os
import stat
import sys
from pathlib import Path

import pytest

from typesetter.backend_contract import TypesettingRenderRequest
from typesetter.rust_backend import (
    KoharuBackendUnavailable,
    render_with_koharu_backend,
    resolve_koharu_bridge_path,
)


def _request():
    return TypesettingRenderRequest.from_mapping(
        {
            "text": "HELLO",
            "translated": "OLA",
            "bbox": [10, 12, 110, 72],
            "safe_text_box": [18, 20, 100, 64],
            "render_bbox": [18, 20, 100, 64],
            "rotation_deg": 0,
            "line_polygons": [[[18, 20], [100, 20], [100, 64], [18, 64]]],
            "bubble_mask_path": "masks/page_001.png",
            "bubble_mask_value": 7,
            "bubble_id": "bubble-7",
            "font_size_px": 24,
            "stroke_width": 2,
            "fill_rgb": [0, 0, 0],
            "stroke_rgb": [255, 255, 255],
        }
    )


def test_missing_koharu_bridge_fails_closed(monkeypatch):
    monkeypatch.delenv("TRADUZAI_KOHARU_RENDERER_BIN", raising=False)

    with pytest.raises(KoharuBackendUnavailable, match="TRADUZAI_KOHARU_RENDERER_BIN"):
        render_with_koharu_backend(_request())


def test_resolve_koharu_bridge_path_uses_configured_env(monkeypatch, tmp_path):
    bridge = tmp_path / "fake_bridge.py"
    bridge.write_text("pass", encoding="utf-8")
    monkeypatch.setenv("TRADUZAI_KOHARU_RENDERER_BIN", str(bridge))

    assert resolve_koharu_bridge_path() == bridge


def test_render_with_koharu_backend_calls_configured_bridge(tmp_path):
    bridge = tmp_path / "fake_koharu_bridge.py"
    captured = tmp_path / "captured.json"
    bridge.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "payload = json.loads(sys.stdin.read())",
                f"pathlib.Path({str(captured)!r}).write_text(json.dumps(payload), encoding='utf-8')",
                "print(json.dumps({'render_bbox': payload['render_bbox'], 'font_size_px': 20, 'fit_status': 'ok', 'backend': 'koharu'}))",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(bridge, os.stat(bridge).st_mode | stat.S_IEXEC)

    result = render_with_koharu_backend(
        _request(),
        bridge_path=bridge,
        command_prefix=[sys.executable],
    )

    assert result.backend == "koharu"
    assert result.fit_status == "ok"
    assert result.font_size_px == 20
    assert json.loads(captured.read_text(encoding="utf-8"))["font_family"] == "ComicNeue-Bold.ttf"
