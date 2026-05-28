from __future__ import annotations

from pipeline.vision_stack.engine_presets import (
    COMIC_TEXT_DETECTOR_HF_REPO,
    COMIC_TEXT_DETECTOR_SEGMENTER,
    engine_steps_for_preset,
    list_engine_presets,
    resolve_engine_preset,
)


def test_resolve_manga_engine_preset_from_config():
    preset = resolve_engine_preset({"engine_preset_id": "manga"})

    assert preset.to_dict() == {
        "id": "manga",
        "content_family": "manga",
        "detector": "comic-text-bubble-detector",
        "font_detector": "yuzumarker-font-detection",
        "segmenter": "comic-text-detector-seg",
        "bubble_segmenter": "speech-bubble-segmentation",
        "ocr": "paddle-ocr-vl-1.5",
        "inpainter": "aot-inpainting",
        "mask_strategy": "segmentation_assisted",
        "preserve_sfx": True,
    }


def test_resolve_manhwa_manhua_engine_preset_from_preset_settings():
    preset = resolve_engine_preset(
        {
            "preset": {
                "id": "custom_1",
                "settings": {"engine_preset_id": "manhwa_manhua"},
            }
        }
    )

    assert preset.id == "manhwa_manhua"
    assert preset.detector == "comic-text-bubble-detector"
    assert preset.segmenter == "comic-text-detector-seg"
    assert preset.bubble_segmenter == "speech-bubble-segmentation"
    assert preset.ocr == "paddle-ocr-vl-1.5"
    assert preset.inpainter == "aot-inpainting"
    assert preset.mask_strategy == "roi_segmentation_assisted"


def test_resolve_manga_ocr_guided_engine_preset_from_config():
    preset = resolve_engine_preset({"engine_preset_id": "manga_ocr_guided"})

    assert preset.id == "manga_ocr_guided"
    assert preset.mask_strategy == "ocr_guided_segmentation"
    assert preset.segmenter == "comic-text-detector-seg"
    assert preset.inpainter == "aot-inpainting"
    assert preset.preserve_sfx is False


def test_resolve_manhwa_manhua_ocr_guided_engine_preset_from_config():
    preset = resolve_engine_preset({"engine_preset_id": "manhwa_manhua_ocr_guided"})

    assert preset.id == "manhwa_manhua_ocr_guided"
    assert preset.detector == "comic-text-bubble-detector"
    assert preset.mask_strategy == "ocr_guided_roi_segmentation"
    assert preset.preserve_sfx is False


def test_cjk_mask_mode_env_can_select_ocr_guided_presets(monkeypatch):
    monkeypatch.setenv("TRADUZAI_CJK_MASK_MODE", "ocr_guided")

    assert resolve_engine_preset({"engine_preset_id": "manga"}).id == "manga_ocr_guided"
    assert resolve_engine_preset({"engine_preset_id": "manhwa_manhua"}).id == "manhwa_manhua_ocr_guided"


def test_resolve_engine_preset_from_source_language():
    assert resolve_engine_preset({}, idioma_origem="ja").id == "manga"
    assert resolve_engine_preset({}, idioma_origem="ko").id == "manhwa_manhua"
    assert resolve_engine_preset({}, idioma_origem="zh-CN").id == "manhwa_manhua"
    assert resolve_engine_preset({}, idioma_origem="en").id == "default"


def test_engine_steps_exclude_default_and_disabled_stages():
    default_preset = resolve_engine_preset({"engine_preset_id": "default"})
    manga_preset = resolve_engine_preset({"engine_preset_id": "manga"})

    assert default_preset.segmenter == "comic-text-detector-seg"
    assert default_preset.bubble_segmenter == "speech-bubble-segmentation"
    assert default_preset.inpainter == "aot-inpainting"
    assert engine_steps_for_preset(default_preset) == [
        "comic-text-detector-seg",
        "speech-bubble-segmentation",
        "aot-inpainting",
    ]
    assert engine_steps_for_preset(manga_preset) == [
        "comic-text-bubble-detector",
        "yuzumarker-font-detection",
        "comic-text-detector-seg",
        "speech-bubble-segmentation",
        "paddle-ocr-vl-1.5",
        "aot-inpainting",
    ]


def test_comic_text_detector_segmenter_records_real_model_repo():
    assert COMIC_TEXT_DETECTOR_SEGMENTER == "comic-text-detector-seg"
    assert COMIC_TEXT_DETECTOR_HF_REPO == "mayocream/comic-text-detector"


def test_all_presets_keep_segmenter_and_bubble_segmenter_enabled():
    for preset in list_engine_presets():
        assert preset.segmenter == "comic-text-detector-seg"
        assert preset.bubble_segmenter == "speech-bubble-segmentation"
        assert preset.segmenter not in {"", "default", "disabled"}
        assert preset.bubble_segmenter not in {"", "default", "disabled"}
        steps = engine_steps_for_preset(preset)
        assert "comic-text-detector-seg" in steps
        assert "speech-bubble-segmentation" in steps
        assert "aot-inpainting" in steps
