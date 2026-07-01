from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


COMIC_TEXT_DETECTOR_SEGMENTER = "comic-text-detector-seg"
SPEECH_BUBBLE_SEGMENTER = "speech-bubble-segmentation"
DEFAULT_INPAINTER = "aot-inpainting"
COMIC_TEXT_DETECTOR_HF_REPO = "mayocream/comic-text-detector"
COMIC_TEXT_BUBBLE_DETECTOR = "comic-text-bubble-detector"


@dataclass(frozen=True)
class EnginePreset:
    id: str
    content_family: str
    detector: str
    font_detector: str
    segmenter: str
    bubble_segmenter: str
    ocr: str
    inpainter: str
    mask_strategy: str
    preserve_sfx: bool = True

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


_MANGA_PRESET = EnginePreset(
    id="manga",
    content_family="manga",
    detector=COMIC_TEXT_BUBBLE_DETECTOR,
    font_detector="yuzumarker-font-detection",
    segmenter=COMIC_TEXT_DETECTOR_SEGMENTER,
    bubble_segmenter=SPEECH_BUBBLE_SEGMENTER,
    ocr="paddle-ocr-vl-1.5",
    inpainter=DEFAULT_INPAINTER,
    mask_strategy="segmentation_assisted",
)

_MANGA_OCR_GUIDED_PRESET = EnginePreset(
    id="manga_ocr_guided",
    content_family="manga",
    detector=COMIC_TEXT_BUBBLE_DETECTOR,
    font_detector="yuzumarker-font-detection",
    segmenter=COMIC_TEXT_DETECTOR_SEGMENTER,
    bubble_segmenter=SPEECH_BUBBLE_SEGMENTER,
    ocr="paddle-ocr-vl-1.5",
    inpainter=DEFAULT_INPAINTER,
    mask_strategy="ocr_guided_segmentation",
    preserve_sfx=False,
)

_MANHWA_MANHUA_PRESET = EnginePreset(
    id="manhwa_manhua",
    content_family="manhwa_manhua",
    detector=COMIC_TEXT_BUBBLE_DETECTOR,
    font_detector="default",
    segmenter=COMIC_TEXT_DETECTOR_SEGMENTER,
    bubble_segmenter=SPEECH_BUBBLE_SEGMENTER,
    ocr="paddle-ocr-vl-1.5",
    inpainter=DEFAULT_INPAINTER,
    mask_strategy="roi_segmentation_assisted",
)

_MANHWA_MANHUA_OCR_GUIDED_PRESET = EnginePreset(
    id="manhwa_manhua_ocr_guided",
    content_family="manhwa_manhua",
    detector=COMIC_TEXT_BUBBLE_DETECTOR,
    font_detector="default",
    segmenter=COMIC_TEXT_DETECTOR_SEGMENTER,
    bubble_segmenter=SPEECH_BUBBLE_SEGMENTER,
    ocr="paddle-ocr-vl-1.5",
    inpainter=DEFAULT_INPAINTER,
    mask_strategy="ocr_guided_roi_segmentation",
    preserve_sfx=False,
)

_DEFAULT_PRESET = EnginePreset(
    id="default",
    content_family="default",
    detector="default",
    font_detector="default",
    segmenter=COMIC_TEXT_DETECTOR_SEGMENTER,
    bubble_segmenter=SPEECH_BUBBLE_SEGMENTER,
    ocr="default",
    inpainter=DEFAULT_INPAINTER,
    mask_strategy="roi_segmentation_assisted",
)

_PRESETS = {
    _MANGA_PRESET.id: _MANGA_PRESET,
    _MANGA_OCR_GUIDED_PRESET.id: _MANGA_OCR_GUIDED_PRESET,
    _MANHWA_MANHUA_PRESET.id: _MANHWA_MANHUA_PRESET,
    _MANHWA_MANHUA_OCR_GUIDED_PRESET.id: _MANHWA_MANHUA_OCR_GUIDED_PRESET,
    _DEFAULT_PRESET.id: _DEFAULT_PRESET,
}


def _normalize_engine_preset_id(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"manga", "manga_bw", "japanese_manga"}:
        return "manga"
    if normalized in {"manga_ocr_guided", "manga_precise", "japanese_manga_ocr_guided"}:
        return "manga_ocr_guided"
    if normalized in {"manhwa", "manhua", "manhwa_manhua", "manhwa_webtoon_color", "manhua_color"}:
        return "manhwa_manhua"
    if normalized in {
        "manhwa_ocr_guided",
        "manhua_ocr_guided",
        "manhwa_manhua_ocr_guided",
        "manhwa_precise",
        "manhua_precise",
    }:
        return "manhwa_manhua_ocr_guided"
    if normalized in {"default", "padrao", "standard"}:
        return "default"
    return ""


def _preset_id_from_config(config: dict | None) -> str:
    if not isinstance(config, dict):
        return ""

    direct = _normalize_engine_preset_id(config.get("engine_preset_id"))
    if direct:
        return direct

    preset = config.get("preset")
    if isinstance(preset, dict):
        settings = preset.get("settings")
        if isinstance(settings, dict):
            from_settings = _normalize_engine_preset_id(settings.get("engine_preset_id"))
            if from_settings:
                return from_settings
        from_preset_id = _normalize_engine_preset_id(preset.get("id"))
        if from_preset_id:
            return from_preset_id

    return ""


def _preset_id_from_language(idioma_origem: str = "") -> str:
    lang = str(idioma_origem or "").strip().lower()
    if lang in {"ja", "jp", "jpn", "japanese"}:
        return "manga"
    if lang in {"ko", "kr", "kor", "korean", "zh", "zh_cn", "zh-cn", "zh_tw", "zh-tw", "cn", "tw"}:
        return "manhwa_manhua"
    return "default"


def resolve_engine_preset(config: dict | None = None, *, idioma_origem: str = "") -> EnginePreset:
    forced = _normalize_engine_preset_id(os.getenv("TRADUZAI_ENGINE_PRESET"))
    if forced:
        return _PRESETS[forced]

    preset_id = _preset_id_from_config(config)
    if not preset_id and isinstance(config, dict):
        preset_id = _preset_id_from_language(config.get("idioma_origem", idioma_origem))
    if not preset_id:
        preset_id = _preset_id_from_language(idioma_origem)
    mask_mode = str(os.getenv("TRADUZAI_CJK_MASK_MODE") or "").strip().lower()
    if mask_mode in {"ocr_guided", "ocr-guided", "precise"}:
        if preset_id == "manga":
            preset_id = "manga_ocr_guided"
        elif preset_id == "manhwa_manhua":
            preset_id = "manhwa_manhua_ocr_guided"
    elif mask_mode in {"baseline", "default"}:
        if preset_id == "manga_ocr_guided":
            preset_id = "manga"
        elif preset_id == "manhwa_manhua_ocr_guided":
            preset_id = "manhwa_manhua"
    return _PRESETS.get(preset_id, _DEFAULT_PRESET)


def list_engine_presets() -> list[EnginePreset]:
    return list(_PRESETS.values())


def engine_steps_for_preset(preset: EnginePreset) -> list[str]:
    steps = [
        preset.detector,
        preset.font_detector,
        preset.segmenter,
        preset.bubble_segmenter,
        preset.ocr,
        preset.inpainter,
    ]
    return [step for step in steps if step and step != "disabled" and step != "default"]
