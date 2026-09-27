from __future__ import annotations

import pytest


def _observation(identity: str, text: str, confidence: float, bbox: list[int]):
    return {
        "observation_id": identity,
        "text": text,
        "confidence": confidence,
        "bbox_page": bbox,
    }


def test_tall_page_plan_is_bounded_overlapping_and_covers_every_pixel() -> None:
    from vision_runtime.ocr_refinement import plan_vertical_refinement

    plan = plan_vertical_refinement(
        page_width=720,
        page_height=10017,
        tile_height=1800,
        overlap=180,
        max_tiles=8,
    )

    assert len(plan.regions) == 7
    assert plan.regions[0] == (0, 0, 720, 1800)
    assert plan.regions[-1][3] == 10017
    assert all(left[3] > right[1] for left, right in zip(plan.regions, plan.regions[1:]))
    assert plan.coverage_complete is True
    assert plan.selection_basis == "source_dimensions_only"


def test_refinement_budget_fails_closed_instead_of_skipping_page_tail() -> None:
    from vision_runtime.ocr_refinement import plan_vertical_refinement

    with pytest.raises(ValueError, match="budget"):
        plan_vertical_refinement(
            page_width=720,
            page_height=10017,
            tile_height=1800,
            overlap=180,
            max_tiles=3,
        )


def test_overlap_duplicate_is_reconciled_by_geometry_and_text() -> None:
    from vision_runtime.ocr_refinement import reconcile_observations

    result = reconcile_observations([
        _observation("tile-a", "MASTER...", 0.82, [100, 1700, 260, 1760]),
        _observation("tile-b", "MASTER...", 0.96, [102, 1701, 261, 1761]),
    ])

    assert len(result.observations) == 1
    assert result.observations[0]["observation_id"] == "tile-b"
    assert result.observations[0]["reconciled_from"] == ["tile-a", "tile-b"]
    assert len(result.observations[0]["alternatives"]) == 2
    assert result.duplicate_count == 1

    replayed = reconcile_observations(result.observations)
    assert replayed.observations[0]["reconciled_from"] == ["tile-a", "tile-b"]
    assert len(replayed.observations[0]["alternatives"]) == 2


def test_overlap_variant_readings_share_one_physical_line_and_keep_alternatives() -> None:
    from vision_runtime.ocr_refinement import reconcile_observations

    result = reconcile_observations([
        _observation("full", "OF LIGHTNING THAT UNLEASHES", 0.94, [107, 6947, 598, 6990]),
        _observation("tile", "OF LIGHTNING THAT UNLEASHI", 0.90, [112, 6949, 574, 7022]),
    ])

    assert len(result.observations) == 1
    assert result.duplicate_count == 1
    assert {row["text"] for row in result.observations[0]["alternatives"]} == {
        "OF LIGHTNING THAT UNLEASHES",
        "OF LIGHTNING THAT UNLEASHI",
    }


def test_repeated_text_in_distinct_locations_is_not_deduplicated() -> None:
    from vision_runtime.ocr_refinement import reconcile_observations

    result = reconcile_observations([
        _observation("top", "...", 0.9, [100, 100, 160, 130]),
        _observation("bottom", "...", 0.9, [100, 5000, 160, 5030]),
    ])

    assert len(result.observations) == 2
    assert result.duplicate_count == 0


def test_empty_provider_result_stays_unknown_not_confirmed_absent() -> None:
    from vision_runtime.ocr_refinement import reconcile_observations

    result = reconcile_observations([])

    assert result.status == "review_required"
    assert result.uncertainty_reasons == ("no_observations_not_negative_evidence",)
