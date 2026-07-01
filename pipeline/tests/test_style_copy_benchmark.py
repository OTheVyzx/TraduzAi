from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2

from typesetter.style_extractor import IMPACT_FONT_CHOICES, extract_text_style_evidence


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "style_copy_atlas"
GENERATOR_PATH = FIXTURE_DIR / "generate_style_copy_atlas.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("style_copy_atlas_generator", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atlas_cases():
    generator = _load_generator()
    generator.generate()
    manifest = json.loads((FIXTURE_DIR / "style_copy_manifest.json").read_text(encoding="utf-8"))
    atlas_bgr = cv2.imread(str(FIXTURE_DIR / manifest["image"]), cv2.IMREAD_COLOR)
    assert atlas_bgr is not None
    atlas_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    for case in manifest["cases"]:
        x1, y1, x2, y2 = case["bbox"]
        yield case["id"], atlas_rgb[y1:y2, x1:x2, :3], case["expected"]


def test_style_copy_atlas_extraction_matches_expected_contract():
    failures: list[str] = []

    for case_id, crop, expected in _atlas_cases():
        evidence = extract_text_style_evidence(crop)

        if expected.get("no_text") is True:
            if evidence.text_color_confidence >= 0.5:
                failures.append(f"{case_id}: unexpected text confidence {evidence.text_color_confidence:.2f}")
            if evidence.font_name != "ComicNeue-Bold.ttf":
                failures.append(f"{case_id}: unexpected font {evidence.font_name!r}")

        expected_font = expected.get("font_name")
        if expected_font and evidence.font_name != expected_font:
            failures.append(f"{case_id}: font {evidence.font_name!r} != {expected_font!r}")

        if expected.get("font_family") == "impact" and evidence.font_name not in IMPACT_FONT_CHOICES:
            failures.append(f"{case_id}: font {evidence.font_name!r} is not impact")

        expected_color = expected.get("text_color")
        if expected_color and evidence.text_color.upper() != expected_color.upper():
            failures.append(f"{case_id}: text_color {evidence.text_color} != {expected_color}")

        expected_stroke_color = expected.get("stroke_color")
        if expected_stroke_color and evidence.stroke_color.upper() != expected_stroke_color.upper():
            failures.append(f"{case_id}: stroke_color {evidence.stroke_color} != {expected_stroke_color}")

        stroke_min = expected.get("stroke_width_px_min")
        if stroke_min is not None and evidence.stroke_width_px < int(stroke_min):
            failures.append(f"{case_id}: stroke_width {evidence.stroke_width_px} < {stroke_min}")

        if expected.get("stroke") is False and evidence.stroke_color:
            failures.append(f"{case_id}: unexpected stroke {evidence.stroke_color}:{evidence.stroke_width_px}")

        if expected.get("gradient") is True and not evidence.gradient:
            failures.append(f"{case_id}: expected gradient, got none")
        if expected.get("gradient") is False and evidence.gradient:
            failures.append(f"{case_id}: unexpected gradient {evidence.gradient_colors}")

        if expected.get("glow") is True and not evidence.glow:
            failures.append(f"{case_id}: expected glow, got none")
        if expected.get("glow") is False and evidence.glow:
            failures.append(f"{case_id}: unexpected glow {evidence.glow_color}:{evidence.glow_px}")

        if expected.get("shadow") is True and not evidence.shadow:
            failures.append(f"{case_id}: expected shadow, got none")
        if expected.get("shadow") is False and evidence.shadow:
            failures.append(f"{case_id}: unexpected shadow {evidence.shadow_color}:{evidence.shadow_offset}")

        if evidence.curved:
            failures.append(f"{case_id}: curve detection should be disabled")

    assert not failures, "\n".join(failures)
