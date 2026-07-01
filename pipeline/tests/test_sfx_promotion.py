import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.candidate import enrich_sfx_candidate
from sfx.promotion import promote_visual_sfx_candidate, suppress_normal_ocr_overlapping_sfx
from vision_stack.runtime import vision_blocks_to_mask


def _page() -> np.ndarray:
    image = np.full((140, 220, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (46, 42), (60, 88), (15, 15, 15), -1)
    cv2.rectangle(image, (46, 78), (92, 92), (15, 15, 15), -1)
    cv2.rectangle(image, (128, 38), (144, 96), (15, 15, 15), -1)
    cv2.rectangle(image, (128, 38), (172, 52), (15, 15, 15), -1)
    cv2.rectangle(image, (158, 68), (174, 96), (15, 15, 15), -1)
    return image


def _visual_candidate() -> dict:
    return {
        "id": "sfx_visual_001",
        "bbox": [24, 24, 190, 110],
        "text_pixel_bbox": [24, 24, 190, 110],
        "content_class": "sfx",
        "tipo": "sfx",
        "detector": "sfx_visual",
        "confidence": 0.82,
        "route_action": "review_required",
        "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
        "sfx_ocr": {"status": "no_confident_cjk"},
        "sfx": {
            "visual_detector": "sfx_visual",
            "visual_source": "local_contrast",
            "visual_confidence": 0.82,
            "inpaint_allowed": False,
            "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
        },
    }


def test_visual_sfx_promotion_does_not_depend_on_confident_ocr():
    promoted = promote_visual_sfx_candidate(_visual_candidate(), _page())

    assert promoted["route_action"] == "translate_sfx_inpaint_render"
    assert promoted["translate_policy"] == "review"
    assert promoted["render_policy"] == "sfx_style"
    assert promoted["script"] == "visual_unknown"
    assert promoted["sfx"]["visual_promotion"] is True
    assert promoted["sfx"]["inpaint_candidate_allowed"] is True
    assert promoted["sfx"]["inpaint_allowed"] is False
    assert "sfx_text_unknown" in promoted["qa_flags"]


def test_enrich_preserves_visual_sfx_promotion_without_source_text():
    promoted = promote_visual_sfx_candidate(_visual_candidate(), _page())

    enriched = enrich_sfx_candidate(promoted)

    assert enriched["route_action"] == "translate_sfx_inpaint_render"
    assert enriched["translate_policy"] == "review"
    assert enriched["render_policy"] == "sfx_style"
    assert enriched["sfx"]["visual_promotion"] is True
    assert enriched["sfx"]["inpaint_allowed"] is False


def test_sfx_inpaint_mask_skips_promoted_visual_candidate_until_allowed():
    promoted = promote_visual_sfx_candidate(_visual_candidate(), _page())

    mask = vision_blocks_to_mask(_page().shape, [promoted], image_rgb=_page(), expand_mask=True, ocr_texts=[promoted])

    assert int(np.count_nonzero(mask)) == 0


def test_suppresses_short_normal_ocr_overlapping_visual_sfx_in_english_work():
    texts = [{"id": "ocr_1", "bbox": [40, 36, 100, 96], "text": "XEV", "route_action": "translate_inpaint_render"}]

    filtered = suppress_normal_ocr_overlapping_sfx(texts, [_visual_candidate()], source_language="en")

    assert filtered[0]["route"] == "suppress"
    assert filtered[0]["route_reason"] == "visual_sfx_overlap_suppressed"
    assert filtered[0]["skip_processing"] is True
    assert filtered[0]["sfx_candidate_id"] == "sfx_visual_001"


def test_does_not_suppress_clear_english_dialogue_overlapping_sfx_candidate():
    texts = [{"id": "ocr_1", "bbox": [40, 36, 100, 96], "text": "I am here", "route_action": "translate_inpaint_render"}]

    filtered = suppress_normal_ocr_overlapping_sfx(texts, [_visual_candidate()], source_language="en")

    assert filtered[0].get("route") != "suppress"
    assert filtered[0]["route_action"] == "translate_inpaint_render"


def test_reclassifies_known_latin_sfx_word_as_sfx_in_english_work():
    texts = [{"id": "ocr_1", "bbox": [40, 36, 100, 96], "text": "BOOM", "route_action": "translate_inpaint_render"}]

    filtered = suppress_normal_ocr_overlapping_sfx(texts, [_visual_candidate()], source_language="en")

    assert filtered[0].get("route") != "suppress"
    assert filtered[0]["content_class"] == "sfx"
    assert filtered[0]["script"] == "latin_sfx"
    assert filtered[0]["route_action"] == "translate_sfx_inpaint_render"
    assert filtered[0]["sfx"]["source_text"] == "BOOM"
    assert filtered[0]["sfx"]["inpaint_allowed"] is False


def test_sfx_overlap_suppression_is_source_language_gated():
    texts = [{"id": "ocr_1", "bbox": [40, 36, 100, 96], "text": "XEV", "route_action": "translate_inpaint_render"}]

    filtered = suppress_normal_ocr_overlapping_sfx(texts, [_visual_candidate()], source_language="ko")

    assert filtered[0].get("route") != "suppress"
