import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.ocr_probe import build_sfx_ocr_crop_variants, probe_sfx_candidate_ocr
from sfx.script_probe import probe_sfx_candidate_script


def _candidate(bbox=None):
    return {
        "id": "sfx_visual_001",
        "bbox": bbox or [10, 12, 46, 52],
        "content_class": "sfx",
        "tipo": "sfx",
        "detector": "sfx_visual",
        "route_action": "review_required",
        "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
        "sfx": {
            "visual_detector": "sfx_visual",
            "visual_confidence": 0.74,
            "inpaint_allowed": False,
            "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
        },
    }


def _page():
    image = np.full((80, 100, 3), 240, dtype=np.uint8)
    image[12:52, 10:46] = [40, 55, 120]
    return image


def test_sfx_ocr_probe_recognizes_hangul_then_script_probe_promotes_route():
    calls = []

    def recognizer(crop, lang):
        calls.append((crop.shape, lang))
        return [{"text": "쾅", "confidence": 0.82}] if lang == "ko" else []

    probed = probe_sfx_candidate_ocr(_candidate(), _page(), recognizer=recognizer)
    routed = probe_sfx_candidate_script(probed, probed.get("recognized_text", ""))

    assert calls
    assert probed["recognized_text"] == "쾅"
    assert probed["sfx_ocr"]["status"] == "recognized"
    assert probed["sfx_ocr"]["lang"] == "ko"
    assert routed["route_action"] == "translate_sfx_inpaint_render"
    assert routed["script"] == "hangul"
    assert routed["sfx"]["source_text"] == "쾅"
    assert "sfx_script_unknown" not in routed["qa_flags"]


def test_sfx_ocr_probe_keeps_empty_ocr_review_only():
    probed = probe_sfx_candidate_ocr(_candidate(), _page(), recognizer=lambda crop, lang: [])
    routed = probe_sfx_candidate_script(probed, probed.get("recognized_text", ""))

    assert probed["sfx_ocr"]["status"] == "no_confident_cjk"
    assert routed["route_action"] == "review_required"
    assert routed["script"] == "unknown"
    assert routed["sfx"]["inpaint_allowed"] is False
    assert "sfx_script_unknown" in routed["qa_flags"]


def test_sfx_ocr_probe_keeps_kana_review_only_after_script_probe():
    probed = probe_sfx_candidate_ocr(
        _candidate(),
        _page(),
        recognizer=lambda crop, lang: [{"text": "ズド", "confidence": 0.91}],
    )
    routed = probe_sfx_candidate_script(probed, probed.get("recognized_text", ""))

    assert probed["recognized_text"] == "ズド"
    assert routed["route_action"] == "review_required"
    assert routed["script"] == "cjk_unknown"
    assert routed["sfx"]["inpaint_allowed"] is False


def test_sfx_ocr_probe_crops_with_padding_and_clamps_to_image():
    shapes = []

    def recognizer(crop, lang):
        shapes.append(crop.shape)
        return [{"text": "쾅", "confidence": 0.7}]

    candidate = _candidate([0, 2, 20, 25])
    probed = probe_sfx_candidate_ocr(candidate, _page(), recognizer=recognizer)

    assert shapes
    assert shapes[0][0] > 0
    assert shapes[0][1] > 0
    assert probed["bbox"] == [0, 2, 20, 25]


def test_sfx_ocr_probe_skips_invalid_bbox_without_calling_ocr():
    calls = []

    def recognizer(crop, lang):
        calls.append((crop, lang))
        return [{"text": "쾅", "confidence": 0.7}]

    probed = probe_sfx_candidate_ocr(_candidate([30, 20, 10, 40]), _page(), recognizer=recognizer)

    assert calls == []
    assert probed["sfx_ocr"]["status"] == "invalid_bbox"


def test_sfx_ocr_probe_does_not_mutate_input_candidate():
    candidate = _candidate()
    before = deepcopy(candidate)

    probe_sfx_candidate_ocr(candidate, _page(), recognizer=lambda crop, lang: [{"text": "쾅", "confidence": 0.7}])

    assert candidate == before


def test_sfx_ocr_probe_survives_ocr_exception_as_review_only():
    def recognizer(crop, lang):
        raise RuntimeError("backend unavailable")

    probed = probe_sfx_candidate_ocr(_candidate(), _page(), recognizer=recognizer)
    routed = probe_sfx_candidate_script(probed, probed.get("recognized_text", ""))

    assert probed["sfx_ocr"]["status"] == "no_confident_cjk"
    assert routed["route_action"] == "review_required"
    assert routed["sfx"]["inpaint_allowed"] is False


def test_sfx_ocr_probe_attempts_multiple_crop_variants_and_records_variant():
    calls = []

    def recognizer(crop, lang):
        calls.append((crop.shape, lang))
        if len(calls) == 3 and lang == "ko":
            return [{"text": "\ucff5", "confidence": 0.86}]
        return []

    probed = probe_sfx_candidate_ocr(_candidate(), _page(), recognizer=recognizer, languages=("ko",))

    assert len(calls) >= 3
    assert probed["sfx_ocr"]["status"] == "recognized"
    assert probed["recognized_text"] == "\ucff5"
    variants = {attempt.get("variant") for attempt in probed["sfx_ocr"]["attempts"]}
    assert variants


def test_sfx_ocr_crop_variants_adds_deskew_for_rotated_candidate():
    candidate = _candidate()
    candidate["sfx"]["style"] = {"rotation_deg": 35}

    variants = build_sfx_ocr_crop_variants(candidate, _page())

    names = [name for name, _crop in variants]
    assert "tight_rgb" in names
    assert "deskew_rgb" in names
