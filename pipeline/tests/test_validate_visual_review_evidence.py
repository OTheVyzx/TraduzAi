from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.validate_visual_review_evidence import VisualReviewEvidenceError, _validate_items


def test_review_items_require_exact_set_and_manual_verdict(tmp_path):
    source, final = tmp_path / "source.png", tmp_path / "final.png"
    Image.new("RGB", (3, 2), "white").save(source)
    Image.new("RGB", (3, 2), "black").save(final)
    from ownership.hash_contract import canonical_page_sha256, sha256_file

    item = {
        "page_id": "page_001", "visual_verdict": "GO", "reviewer": "codex",
        "reviewed_at": "2026-08-05T12:00:00+00:00", "source_path": str(source),
        "source_file_sha256": sha256_file(source), "final_path": str(final),
        "final_file_sha256": sha256_file(final),
        "final_pixel_sha256": canonical_page_sha256(Image.open(final).convert("RGB")),
    }
    _validate_items([item], ["page_001"], label="off")
    with pytest.raises(VisualReviewEvidenceError):
        _validate_items([{**item, "visual_verdict": None}], ["page_001"], label="off")
