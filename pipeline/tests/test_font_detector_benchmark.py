from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.font_detector import FontDetector


def _loaded_detector() -> FontDetector:
    detector = FontDetector(Path("dummy.safetensors"), Path("fonts"))
    detector._loaded = True
    detector._fingerprints = {
        "Font-A.ttf": np.array([1.0, 0.0], dtype=np.float32),
        "Font-B.ttf": np.array([0.92, 0.08], dtype=np.float32),
        "Font-C.ttf": np.array([0.0, 1.0], dtype=np.float32),
    }
    return detector


def test_ranked_font_evidence_returns_top_k_calibrated_margin_and_exact_match():
    detector = _loaded_detector()
    region = np.full((32, 64, 3), 255, dtype=np.uint8)

    with patch.object(detector, "_extract_features", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        evidence = detector.detect_with_evidence(region)

    assert evidence["value"] == "Font-A.ttf"
    assert evidence["status"] == "exact"
    assert evidence["top_k"][0] == {"font_name": "Font-A.ttf", "similarity": 1.0}
    assert evidence["margin"] == 0.08
    assert evidence["confidence"] > 0.7
    assert evidence["abstention_reason"] == ""


def test_ranked_font_evidence_abstains_when_similarity_is_below_threshold():
    detector = _loaded_detector()
    region = np.full((32, 64, 3), 255, dtype=np.uint8)

    with patch.object(detector, "_extract_features", return_value=np.array([0.6, 0.4], dtype=np.float32)):
        evidence = detector.detect_with_evidence(region)

    assert evidence["value"] == "unknown"
    assert evidence["status"] == "unknown"
    assert evidence["confidence"] == 0.0
    assert evidence["abstention_reason"] == "similarity_below_threshold"
    assert len(evidence["top_k"]) == 3


def test_ranked_font_evidence_marks_low_margin_as_family_not_exact():
    detector = _loaded_detector()
    detector._fingerprints["Font-B.ttf"] = np.array([0.98, 0.02], dtype=np.float32)
    region = np.full((32, 64, 3), 255, dtype=np.uint8)

    with patch.object(detector, "_extract_features", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        evidence = detector.detect_with_evidence(region)

    assert evidence["value"] == "Font-A.ttf"
    assert evidence["status"] == "family"
    assert evidence["margin"] == 0.02
    assert evidence["confidence"] < 0.3
    assert evidence["abstention_reason"] == "low_top_k_margin"


def test_fingerprint_builder_averages_multiple_sample_texts_and_scales():
    detector = FontDetector(Path("dummy.safetensors"), Path("fonts"))
    detector._candidate_fonts = []

    with (
        patch.object(detector, "_discover_candidate_fonts", return_value=[]),
        patch.object(detector, "_render_font_sample", return_value=np.full((8, 8, 3), 255, dtype=np.uint8)) as render,
        patch.object(
            detector,
            "_extract_features",
            side_effect=[
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
            ],
        ),
    ):
        detector._build_fingerprints()

    assert render.call_count == 3
    assert len(detector._fingerprint_samples["ComicNeue-Bold.ttf"]) == 3
    assert np.isclose(np.linalg.norm(detector._fingerprints["ComicNeue-Bold.ttf"]), 1.0)
