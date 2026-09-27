from __future__ import annotations

import numpy as np


def test_raster_safety_detects_internal_protected_art_even_inside_authorized_body() -> None:
    from typesetter.raster_safety import assess_raster_safety

    authorized = np.ones((20, 30), dtype=np.uint8)
    protected = np.zeros((20, 30), dtype=np.uint8)
    protected[10, :] = 1
    alpha = np.zeros((10, 20), dtype=np.uint8)
    alpha[4:7, 2:18] = 255

    evidence = assess_raster_safety(
        alpha=alpha,
        bbox=[5, 6, 25, 16],
        authorized_body_mask=authorized,
        protected_art_mask=protected,
    )

    assert evidence["status"] == "review_required"
    assert evidence["protected_art_overlap_px"] > 0
    assert evidence["outside_authorized_body_px"] == 0


def test_raster_safety_passes_only_when_all_ink_is_authorized_and_clear_of_art() -> None:
    from typesetter.raster_safety import assess_raster_safety

    authorized = np.zeros((20, 30), dtype=np.uint8)
    authorized[4:18, 4:27] = 1
    protected = np.zeros((20, 30), dtype=np.uint8)
    alpha = np.zeros((8, 12), dtype=np.uint8)
    alpha[2:6, 2:10] = 255

    evidence = assess_raster_safety(
        alpha=alpha,
        bbox=[8, 7, 20, 15],
        authorized_body_mask=authorized,
        protected_art_mask=protected,
    )

    assert evidence["status"] == "pass"
    assert evidence["protected_art_overlap_px"] == 0
    assert evidence["outside_authorized_body_px"] == 0
    assert len(evidence["evidence_sha256"]) == 64


def test_raster_safety_reports_ink_outside_authorized_body() -> None:
    from typesetter.raster_safety import assess_raster_safety

    authorized = np.zeros((10, 10), dtype=np.uint8)
    authorized[2:8, 2:8] = 1
    protected = np.zeros((10, 10), dtype=np.uint8)
    alpha = np.full((6, 6), 255, dtype=np.uint8)

    evidence = assess_raster_safety(
        alpha=alpha,
        bbox=[0, 0, 6, 6],
        authorized_body_mask=authorized,
        protected_art_mask=protected,
    )

    assert evidence["status"] == "review_required"
    assert evidence["outside_authorized_body_px"] > 0
