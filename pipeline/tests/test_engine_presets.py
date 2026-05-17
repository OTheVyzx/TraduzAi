from __future__ import annotations

from pipeline.vision_stack.engine_presets import engine_steps_for_preset, resolve_engine_preset


def test_resolve_manga_engine_preset_from_config():
    preset = resolve_engine_preset({"engine_preset_id": "manga"})

    assert preset.to_dict() == {
        "id": "manga",
        "content_family": "manga",
        "detector": "anime-text-yolo-n",
        "font_detector": "yuzumarker-font-detection",
        "segmenter": "manga-text-segmentation-2025",
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
    assert preset.segmenter == "manga-text-segmentation-2025"
    assert preset.bubble_segmenter == "speech-bubble-segmentation"
    assert preset.ocr == "paddle-ocr-vl-1.5"
    assert preset.inpainter == "aot-inpainting"
    assert preset.mask_strategy == "roi_segmentation_assisted"


def test_resolve_manga_ocr_guided_engine_preset_from_config():
    preset = resolve_engine_preset({"engine_preset_id": "manga_ocr_guided"})

    assert preset.id == "manga_ocr_guided"
    assert preset.mask_strategy == "ocr_guided_segmentation"
    assert preset.segmenter == "manga-text-segmentation-2025"
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

    assert engine_steps_for_preset(default_preset) == []
    assert engine_steps_for_preset(manga_preset) == [
        "anime-text-yolo-n",
        "yuzumarker-font-detection",
        "manga-text-segmentation-2025",
        "speech-bubble-segmentation",
        "paddle-ocr-vl-1.5",
        "aot-inpainting",
    ]
