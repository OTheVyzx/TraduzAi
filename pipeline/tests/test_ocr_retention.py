import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr import ocr_normalizer
from ocr.ocr_normalizer import normalize_ocr_record
from ocr.postprocess import is_ocr_truncated_or_joined
from qa.translation_qa import severity_for_flag


def test_low_confidence_short_dialogue_inside_balloon_is_retained():
    record = normalize_ocr_record(
        {
            "text": "What happened?",
            "confidence": 0.42,
            "bbox": [120, 180, 280, 236],
            "balloon_bbox": [90, 144, 330, 270],
            "tipo": "fala",
            "content_class": "dialogue",
            "skip_processing": True,
            "skip_reason": "suspicious_low_confidence",
        }
    )

    assert record["text"] == "What happened?"
    assert record["skip_processing"] is False
    assert record["skip_reason"] is None
    assert record["route_action"] == "translate_inpaint_render"
    assert record["route_reason"] == "ocr_retention_low_confidence_dialogue"
    assert "low_ocr_confidence" not in record.get("qa_flags", [])


def test_low_confidence_skip_reason_alone_does_not_drop_normal_dialogue():
    record = normalize_ocr_record(
        {
            "text": "I will explain everything when we get there.",
            "confidence": 0.50,
            "bbox": [120, 180, 420, 252],
            "tipo": "fala",
            "content_class": "dialogue",
            "skip_processing": True,
            "skip_reason": "low_confidence_noise",
        }
    )

    assert record["text"] == "I will explain everything when we get there."
    assert record["skip_processing"] is False
    assert record["skip_reason"] is None
    assert record["route_action"] == "translate_inpaint_render"
    assert record["route_reason"] == "dialogue_balloon_with_english_text"


def test_low_confidence_visual_noise_flag_is_neutral_metadata():
    record = normalize_ocr_record(
        {
            "text": "LET'S GO!!",
            "confidence": 0.12,
            "bbox": [10, 10, 130, 42],
            "qa_flags": ["low_confidence_visual_noise"],
            "skip_processing": True,
            "preserve_original": True,
            "content_class": "noise",
            "route": "noise",
            "route_action": "skip",
            "route_reason": "low_confidence_visual_noise",
            "skip_reason": "low_confidence_visual_noise",
        }
    )

    assert record["text"] == "LET'S GO!!"
    assert record.get("skip_processing") is not True
    assert record.get("preserve_original") is not True
    assert "low_confidence_visual_noise" not in record.get("qa_flags", [])
    assert record.get("content_class") in (None, "", "text", "dialogue")
    assert record["route_action"] == "translate_inpaint_render"


def test_ocr_normalizer_has_no_legacy_low_confidence_or_skip_route_helpers():
    assert not hasattr(ocr_normalizer, "_neutralize_low_confidence_visual_noise_filter")
    assert not hasattr(ocr_normalizer, "_should_preserve_legacy_skip_route")


def test_low_confidence_dialogue_is_retained_even_when_better_duplicate_exists():
    record = normalize_ocr_record(
        {
            "text": "What happened?",
            "confidence": 0.42,
            "bbox": [120, 180, 280, 236],
            "balloon_bbox": [90, 144, 330, 270],
            "tipo": "fala",
            "content_class": "dialogue",
            "skip_processing": True,
            "skip_reason": "duplicate_lower_confidence",
            "has_better_duplicate": True,
        }
    )

    assert record["skip_processing"] is False
    assert record["skip_reason"] is None
    assert record["route_action"] == "translate_inpaint_render"


def test_low_confidence_retention_neutralizes_explicit_preserve_action():
    record = normalize_ocr_record(
        {
            "text": "What happened?",
            "confidence": 0.42,
            "bbox": [120, 180, 280, 236],
            "balloon_bbox": [90, 144, 330, 270],
            "tipo": "fala",
            "content_class": "dialogue",
            "route_action": "preserve",
            "route_reason": "manual_preserve",
            "skip_processing": True,
            "skip_reason": "manual_preserve",
        }
    )

    assert record["route_action"] == "translate_inpaint_render"
    assert record["skip_processing"] is False
    assert "low_ocr_confidence" not in record.get("qa_flags", [])


def test_joined_or_truncated_ocr_is_repaired_without_review_route():
    record = normalize_ocr_record(
        {
            "text": "WEDO",
            "confidence": 0.74,
            "bbox": [120, 180, 230, 226],
            "balloon_bbox": [90, 144, 330, 270],
            "tipo": "fala",
            "content_class": "dialogue",
        }
    )

    assert record["text"] == "WE DO"
    assert "ocr_joined_repaired" in record["qa_flags"]
    assert "ocr_truncated_or_joined" not in record["qa_flags"]
    assert record.get("needs_review") is not True
    assert record["route_action"] == "translate_inpaint_render"
    assert record["skip_processing"] is False


def test_existing_joined_flag_is_removed_after_repair():
    record = normalize_ocr_record(
        {
            "text": "What!Then,why did we come to the cafe,what are you hiding?",
            "confidence": 0.82,
            "bbox": [32, 120, 312, 198],
            "balloon_bbox": [20, 88, 344, 226],
            "tipo": "fala",
            "content_class": "dialogue",
            "qa_flags": ["ocr_truncated_or_joined"],
            "route_action": "review_required",
            "route_reason": "ocr_truncated_or_joined",
            "needs_review": True,
        }
    )

    assert record["text"] == "What! Then, why did we come to the cafe, what are you hiding?"
    assert "ocr_joined_repaired" in record["qa_flags"]
    assert "ocr_truncated_or_joined" not in record["qa_flags"]
    assert record["route_action"] == "translate_inpaint_render"
    assert record.get("needs_review") is not True


def test_medical_run_on_ocr_noise_is_repaired_without_review_route():
    record = normalize_ocr_record(
        {
            "text": "We are currently bringing in a CPR TS Notecpr is patient onboard",
            "confidence": 0.78,
            "bbox": [170, 2499, 666, 2586],
            "balloon_bbox": [0, 2177, 800, 2668],
            "tipo": "narracao",
            "content_class": "narration",
        }
    )

    assert "NOTE CPR" in record["text"]
    assert "ocr_joined_repaired" in record["qa_flags"]
    assert "ocr_truncated_or_joined" not in record["qa_flags"]
    assert record.get("needs_review") is not True
    assert record["route_action"] == "translate_inpaint_render"


def test_joined_punctuation_and_known_token_are_repaired_before_review_route():
    record = normalize_ocr_record(
        {
            "text": "Why?!What's ittous?",
            "confidence": 0.78,
            "bbox": [120, 180, 310, 236],
            "balloon_bbox": [90, 144, 350, 270],
            "tipo": "fala",
            "content_class": "dialogue",
        }
    )

    assert record["text"] == "Why?! What's IT TO US?"
    assert "ocr_joined_repaired" in record["qa_flags"]
    assert "ocr_truncated_or_joined" not in record["qa_flags"]
    assert record.get("needs_review") is not True
    assert record["route_action"] == "translate_inpaint_render"


def test_joined_or_truncated_detector_covers_known_ocr_shapes():
    assert is_ocr_truncated_or_joined("Why?!What's") is True
    assert is_ocr_truncated_or_joined("WEDO") is True
    assert is_ocr_truncated_or_joined("ittous") is True
    assert is_ocr_truncated_or_joined("lyingil") is True
    assert is_ocr_truncated_or_joined("CPR TS Notecpr") is True
    assert is_ocr_truncated_or_joined("What happened?") is False


def test_long_run_on_dialogue_is_retained_for_translation_before_review():
    record = normalize_ocr_record(
        {
            "text": "BEFORE LEAVING WEREN'T YOU FOCUSING MORE ON THE OTHER MARTIALARTSMASTER MIXED IN RATHER THAN HUNWON ART?",
            "confidence": 0.90,
            "bbox": [70, 170, 441, 418],
            "balloon_bbox": [0, 78, 719, 510],
            "tipo": "fala",
            "content_class": "dialogue",
        }
    )

    assert "ocr_truncated_or_joined" in record["qa_flags"]
    assert record["route_action"] == "translate_inpaint_render"
    assert record["route_reason"] == "ocr_truncated_or_joined_retained_for_translation"
    assert record["skip_processing"] is False


def test_ocr_truncated_or_joined_is_high_severity():
    assert severity_for_flag("ocr_truncated_or_joined") == "high"
