from __future__ import annotations


def _observation(identity: str, text: str, confidence: float, bbox: list[int]):
    return {
        "observation_id": identity,
        "text": text,
        "confidence": confidence,
        "bbox_page": bbox,
    }


def test_sparse_large_glyph_is_preserved_as_uncertain_not_negative() -> None:
    from vision_runtime.observation_evidence import assess_observation_evidence

    observations = [
        _observation("line-a", "WHAT IS THIS", 0.97, [100, 100, 350, 135]),
        _observation("line-b", "ANOTHER LINE", 0.96, [100, 145, 340, 180]),
        _observation("stroke", "W", 0.54, [80, 200, 420, 360]),
    ]

    assessed = assess_observation_evidence(observations)

    assert len(assessed) == 3
    stroke = next(row for row in assessed if row["observation_id"] == "stroke")
    assert stroke["selection_state"] == "uncertain"
    assert stroke["uncertainty_reasons"] == ["sparse_large_glyph_candidate"]
    assert stroke["text"] == "W"
    assert "confirmed" not in stroke["selection_state"]


def test_normal_text_lines_remain_selection_eligible() -> None:
    from vision_runtime.observation_evidence import assess_observation_evidence

    assessed = assess_observation_evidence([
        _observation("line", "MASTER...", 0.93, [100, 100, 260, 138]),
    ])

    assert assessed[0]["selection_state"] == "eligible"
    assert assessed[0]["uncertainty_reasons"] == []
