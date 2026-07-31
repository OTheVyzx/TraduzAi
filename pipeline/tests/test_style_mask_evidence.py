from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_mask_evidence import measure_masked_color_evidence
from typesetter.style_masks import build_typographic_mask_layers


def _outlined_text() -> np.ndarray:
    image = np.full((120, 300, 3), (238, 238, 238), dtype=np.uint8)
    cv2.putText(image, "TEST", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (18, 18, 18), 8, cv2.LINE_AA)
    cv2.putText(image, "TEST", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (248, 248, 248), 3, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _plain_text() -> np.ndarray:
    image = np.full((120, 300, 3), (245, 245, 245), dtype=np.uint8)
    cv2.putText(image, "PLAIN", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def test_masked_evidence_observes_high_contrast_fill_and_outline():
    image = _outlined_text()

    evidence = measure_masked_color_evidence(image, build_typographic_mask_layers(image))

    assert evidence.fill_confidence >= 0.2
    assert evidence.stroke_detected is True
    assert evidence.stroke_confidence >= 0.2
    assert evidence.fill_color != evidence.stroke_color


def test_masked_evidence_abstains_from_plain_text_outline():
    image = _plain_text()

    evidence = measure_masked_color_evidence(image, build_typographic_mask_layers(image))

    assert evidence.fill_confidence >= 0.2
    assert evidence.stroke_detected is False
    assert evidence.stroke_abstention_reason in {
        "stroke_ring_matches_fill",
        "stroke_ring_matches_background",
        "stroke_ring_not_dominant",
    }
