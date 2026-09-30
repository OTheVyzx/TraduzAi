"""Owner-paired corpus contracts for mask-backed style extraction V2."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_extractor import extract_text_style_evidence_v2  # noqa: E402


CORPUS_ROOT = Path(__file__).parent / "fixtures" / "style_copy_corpus"


def _manifest() -> dict:
    return json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _case(case_id: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    case = next(item for item in _manifest()["cases"] if item["case_id"] == case_id)
    image = cv2.imread(str(CORPUS_ROOT / case["source_image"]), cv2.IMREAD_COLOR)
    glyph = cv2.imread(str(CORPUS_ROOT / case["glyph_mask"]), cv2.IMREAD_GRAYSCALE)
    context = cv2.imread(str(CORPUS_ROOT / case["context_mask"]), cv2.IMREAD_GRAYSCALE)
    assert image is not None and glyph is not None and context is not None
    return case, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), glyph, context


def _extract(case_id: str):
    case, image, glyph, context = _case(case_id)
    evidence = extract_text_style_evidence_v2(
        image,
        glyph,
        context,
        owner_id=case["owner_id"],
        semantic_role=case["semantic_role"],
    )
    return case, evidence


def test_corpus_separates_calibration_and_holdout_by_work() -> None:
    cases = _manifest()["cases"]
    calibration = {item["work"] for item in cases if item["split"] == "calibration"}
    holdout = {item["work"] for item in cases if item["split"] == "holdout"}

    assert calibration and holdout
    assert calibration.isdisjoint(holdout)


def test_every_case_pairs_source_mask_and_owner_id() -> None:
    for case in _manifest()["cases"]:
        assert case["owner_id"] and case["source_text"] and case["semantic_role"]
        image = cv2.imread(str(CORPUS_ROOT / case["source_image"]), cv2.IMREAD_COLOR)
        glyph = cv2.imread(str(CORPUS_ROOT / case["glyph_mask"]), cv2.IMREAD_GRAYSCALE)
        context = cv2.imread(str(CORPUS_ROOT / case["context_mask"]), cv2.IMREAD_GRAYSCALE)
        assert image is not None and glyph is not None and context is not None
        assert image.shape[:2] == glyph.shape == context.shape
        assert int(np.count_nonzero(glyph)) > 0
        assert int(np.count_nonzero(context)) > int(np.count_nonzero(glyph))


def test_colored_background_does_not_contaminate_fill() -> None:
    case, evidence = _extract("holdout_colored_card")
    fill = evidence.attributes["fill"]

    assert fill.value == case["expect"]["fill"]
    assert fill.value != case["expect"]["background_must_not_be_fill"]
    assert fill.confidence >= 0.7


def test_plain_balloon_abstains_from_false_stroke_and_glow() -> None:
    _case_data, evidence = _extract("calibration_plain_balloon")

    assert evidence.attributes["stroke"].value == "unknown"
    assert evidence.attributes["stroke"].abstention_reason
    assert evidence.attributes["glow"].value == "unknown"
    assert evidence.attributes["glow"].abstention_reason


def test_card_crop_preserves_condensed_white_evidence() -> None:
    case, evidence = _extract("calibration_condensed_card")

    assert evidence.attributes["fill"].value == case["expect"]["fill"]
    assert evidence.attributes["font_width"].value == "condensed"
    assert evidence.attributes["font_width"].confidence >= 0.6


def test_burst_crop_preserves_colored_effect_evidence() -> None:
    case, evidence = _extract("holdout_burst_effect")

    assert evidence.attributes["fill"].value == case["expect"]["fill"]
    assert evidence.attributes["stroke"].value["color"] == case["expect"]["stroke"]
    assert evidence.attributes["stroke"].confidence >= 0.65


def test_metrics_are_normalized_by_source_x_height() -> None:
    _case_data, evidence = _extract("holdout_burst_effect")
    metrics = evidence.attribute_provenance["typographic_metrics"]

    assert metrics["normalization_unit"] == "source_x_height"
    assert metrics["source_x_height_px"] > 0
    assert 0.0 < metrics["stroke_width_xh"] < 1.0
    assert 0.0 < metrics["glyph_occupancy"] <= 1.0
