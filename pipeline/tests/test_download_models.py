from __future__ import annotations

import sys
from types import SimpleNamespace

from pipeline import download_models


def test_prepare_aot_inpainting_downloads_required_files(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(download_models, "MODELS_DIR", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    download_models.prepare_aot_inpainting()

    assert calls == [
        {
            "repo_id": "mayocream/aot-inpainting",
            "local_dir": tmp_path / "aot-inpainting",
            "allow_patterns": ["config.json", "model.safetensors"],
        }
    ]
    assert (tmp_path / "aot-inpainting" / ".ready").read_text(encoding="utf-8") == "ok"
