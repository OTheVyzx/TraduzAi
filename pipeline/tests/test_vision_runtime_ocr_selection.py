from __future__ import annotations

from types import SimpleNamespace


def _invocation(texts: list[str]):
    attempts = []
    observations = []
    for index, text in enumerate(texts):
        attempt_id = f"attempt-{index}"
        attempts.append(SimpleNamespace(attempt_id=attempt_id, variant_id=f"variant-{index}"))
        if text:
            observations.append(SimpleNamespace(
                attempt_id=attempt_id,
                observation_id=f"observation-{index}",
                text=text,
                confidence=0.8 + index * 0.01,
                bbox_page=(10, 20 + index, 100, 40 + index),
            ))
    return SimpleNamespace(attempts=tuple(attempts), observations=tuple(observations))


def test_consensus_selects_one_variant_instead_of_concatenating_repeated_ocr() -> None:
    from vision_runtime.ocr_selection import select_ocr_invocation

    result = select_ocr_invocation(_invocation(["HELLO WORLD", "HELLO WORLD", "HELLO WORLD"]))

    assert result.status == "selected"
    assert result.selected_source == "HELLO WORLD"
    assert len(result.selected_observation_ids) == 1


def test_conflicting_readings_fail_closed_without_confidence_tiebreak() -> None:
    from vision_runtime.ocr_selection import select_ocr_invocation

    result = select_ocr_invocation(_invocation(["I AM HERE", "1 AM HERE", "S AM HERE"] ))

    assert result.status == "review_required"
    assert result.selected_source is None
    assert result.uncertainty_reasons == ("conflicting_readings",)


def test_single_nonempty_variant_is_selected_with_uncertainty() -> None:
    from vision_runtime.ocr_selection import select_ocr_invocation

    result = select_ocr_invocation(_invocation(["ONLY ONE", "", ""]))

    assert result.status == "selected_with_uncertainty"
    assert result.selected_source == "ONLY ONE"
