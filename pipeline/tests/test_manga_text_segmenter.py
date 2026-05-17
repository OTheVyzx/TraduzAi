from pathlib import Path

import pytest

from vision_stack import manga_text_segmenter as mts


def test_resolve_manga_text_segmentation_model_from_configured_hf_dir(monkeypatch, tmp_path):
    model_path = tmp_path / "huggingface" / mts.MANGA_TEXT_SEGMENTATION_HF_DIR / mts.MANGA_TEXT_SEGMENTATION_FILE
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")

    monkeypatch.setattr(mts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mts, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(mts, "MIN_VALID_MODEL_BYTES", 1)
    monkeypatch.delenv("TRADUZAI_MANGA_TEXT_SEGMENTATION_MODEL", raising=False)

    assert mts.resolve_manga_text_segmentation_model(download=False) == model_path


def test_resolve_manga_text_segmentation_model_reports_missing_without_download(monkeypatch, tmp_path):
    monkeypatch.setattr(mts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mts, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.delenv("TRADUZAI_MANGA_TEXT_SEGMENTATION_MODEL", raising=False)

    with pytest.raises(mts.MangaTextSegmentationUnavailable):
        mts.resolve_manga_text_segmentation_model(download=False)
