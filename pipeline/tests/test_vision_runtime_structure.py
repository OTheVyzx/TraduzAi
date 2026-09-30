from __future__ import annotations


def _observation(identity: str, text: str, bbox: list[int]) -> dict[str, object]:
    return {"observation_id": identity, "text": text, "bbox_page": bbox}


def test_one_logical_unit_preserves_two_physical_lobes() -> None:
    from vision_runtime.structure import build_structural_analysis

    result = build_structural_analysis(
        observations=[
            _observation("later-id", "FIRST THOUGHT", [20, 20, 120, 60]),
            _observation("earlier-id", "SECOND THOUGHT", [180, 90, 300, 130]),
        ],
        containers=[{
            "container_id": "container-1",
            "bbox_page": [0, 0, 330, 160],
            "lobe_bboxes": [[0, 0, 150, 80], [150, 70, 330, 160]],
            "kind": "connected_balloon",
        }],
    )

    assert len(result.logical_units) == 1
    assert len(result.physical_subblocks) == 2
    assert result.logical_units[0]["observation_ids"] == ["later-id", "earlier-id"]
    assert [row["order"] for row in result.physical_subblocks] == [0, 1]
    assert {row["logical_unit_id"] for row in result.physical_subblocks} == {
        result.logical_units[0]["logical_unit_id"]
    }


def test_close_texts_in_distinct_containers_are_not_merged() -> None:
    from vision_runtime.structure import build_structural_analysis

    result = build_structural_analysis(
        observations=[
            _observation("a", "LEFT", [20, 20, 70, 50]),
            _observation("b", "RIGHT", [80, 20, 130, 50]),
        ],
        containers=[
            {"container_id": "left", "bbox_page": [0, 0, 75, 70], "kind": "balloon"},
            {"container_id": "right", "bbox_page": [76, 0, 150, 70], "kind": "balloon"},
        ],
    )

    assert len(result.logical_units) == 2
    assert {row["container_ref"] for row in result.logical_units} == {"left", "right"}


def test_uncontained_text_remains_unknown_without_invented_balloon() -> None:
    from vision_runtime.structure import build_structural_analysis

    result = build_structural_analysis(
        observations=[_observation("free", "SFX", [300, 400, 350, 450])],
        containers=[],
    )

    assert result.logical_units[0]["container_ref"] is None
    assert result.logical_units[0]["classification"] == "unknown_text"
    assert result.logical_units[0]["uncertainty_reasons"] == ["container_not_observed"]


def test_geometry_not_identifier_controls_reading_order() -> None:
    from vision_runtime.structure import build_structural_analysis

    result = build_structural_analysis(
        observations=[
            _observation("000-first-by-id", "BOTTOM", [10, 200, 100, 230]),
            _observation("999-last-by-id", "TOP", [10, 20, 100, 50]),
        ],
        containers=[],
    )

    assert result.reading_order == (
        result.logical_units[0]["logical_unit_id"],
        result.logical_units[1]["logical_unit_id"],
    )
    assert result.logical_units[0]["observation_ids"] == ["999-last-by-id"]


def test_observation_inside_nested_containers_uses_smallest_supported_container() -> None:
    from vision_runtime.structure import build_structural_analysis

    result = build_structural_analysis(
        observations=[_observation("text", "HELLO", [40, 40, 80, 70])],
        containers=[
            {"container_id": "panel", "bbox_page": [0, 0, 500, 500], "kind": "panel"},
            {"container_id": "balloon", "bbox_page": [20, 20, 100, 90], "kind": "balloon"},
        ],
    )

    assert result.logical_units[0]["container_ref"] == "balloon"


def test_overlapping_lobes_assign_each_observation_to_exactly_one_subblock() -> None:
    from vision_runtime.structure import build_structural_analysis

    result = build_structural_analysis(
        observations=[_observation("shared", "MIDDLE", [90, 70, 130, 100])],
        containers=[{
            "container_id": "connected",
            "bbox_page": [0, 0, 220, 180],
            "lobe_bboxes": [[0, 0, 150, 120], [70, 50, 220, 180]],
            "kind": "connected_balloon",
        }],
    )

    assignments = sum(
        row["observation_ids"].count("shared") for row in result.physical_subblocks
    )
    assert assignments == 1


def test_uncertain_glyph_is_retained_but_not_concatenated_into_selected_text() -> None:
    from vision_runtime.structure import build_structural_analysis

    observations = [
        {
            **_observation("stroke", "W", [20, 20, 180, 150]),
            "selection_state": "uncertain",
            "uncertainty_reasons": ["sparse_large_glyph_candidate"],
        },
        _observation("line", "REAL DIALOGUE", [30, 170, 190, 205]),
    ]
    result = build_structural_analysis(
        observations=observations,
        containers=[{
            "container_id": "balloon",
            "bbox_page": [0, 0, 220, 230],
            "kind": "balloon",
        }],
    )

    unit = result.logical_units[0]
    assert unit["observation_ids"] == ["stroke", "line"]
    assert unit["selected_observation_ids"] == ["line"]
    assert unit["text"] == "REAL DIALOGUE"
    assert unit["uncertainty_reasons"] == ["sparse_large_glyph_candidate"]
