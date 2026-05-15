from __future__ import annotations

from pathlib import Path

from vision_stack import detector as detector_module
from vision_stack.detector import TextDetector


def test_get_model_path_skips_empty_configured_checkpoint(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    bundled_root = tmp_path / "repo"
    configured.mkdir()
    bundled = bundled_root / "pipeline" / "models"
    bundled.mkdir(parents=True)

    empty_model = configured / "comic-text-detector.pt"
    empty_model.write_bytes(b"")
    bundled_model = bundled / "comic-text-detector.pt"
    bundled_model.write_bytes(b"x" * 2048)

    monkeypatch.setattr(detector_module, "MODELS_DIR", configured)
    monkeypatch.setattr(detector_module, "PROJECT_ROOT", bundled_root)

    detector = TextDetector.__new__(TextDetector)

    assert detector._get_model_path("comic-text-detector") == bundled_model
