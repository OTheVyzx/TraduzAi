from __future__ import annotations

import cv2
import numpy as np


def _outlined_bubble(image, center, radius):
    cv2.circle(image, center, radius, (0, 0, 0), 5)
    cv2.circle(image, center, radius - 5, (255, 255, 255), -1)


def test_connected_white_lobes_share_one_authenticated_container() -> None:
    from vision_runtime.white_containers import discover_white_containers

    image = np.full((220, 440, 3), 210, dtype=np.uint8)
    _outlined_bubble(image, (100, 110), 85)
    _outlined_bubble(image, (340, 110), 85)
    image[95:125, 180:260] = 255
    observations = [
        {"observation_id": "left", "bbox_page": [60, 90, 140, 125]},
        {"observation_id": "right", "bbox_page": [300, 90, 380, 125]},
    ]

    containers = discover_white_containers(image, observations)

    assert len(containers) == 1
    assert containers[0]["observation_ids"] == ["left", "right"]
    assert containers[0]["kind"] == "connected_balloon"
    assert len(containers[0]["lobe_bboxes"]) == 2


def test_close_separate_bubbles_remain_distinct_containers() -> None:
    from vision_runtime.white_containers import discover_white_containers

    image = np.full((180, 320, 3), 210, dtype=np.uint8)
    _outlined_bubble(image, (85, 90), 65)
    _outlined_bubble(image, (235, 90), 65)
    observations = [
        {"observation_id": "left", "bbox_page": [55, 75, 115, 105]},
        {"observation_id": "right", "bbox_page": [205, 75, 265, 105]},
    ]

    containers = discover_white_containers(image, observations)

    assert len(containers) == 2
    assert {tuple(row["observation_ids"]) for row in containers} == {
        ("left",), ("right",)
    }


def test_page_background_is_not_invented_as_a_balloon() -> None:
    from vision_runtime.white_containers import discover_white_containers

    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    observations = [{"observation_id": "narration", "bbox_page": [50, 50, 180, 90]}]

    assert discover_white_containers(image, observations) == ()


def test_geometric_lobes_without_text_support_do_not_claim_connected_structure() -> None:
    from vision_runtime.white_containers import discover_white_containers

    image = np.full((220, 440, 3), 210, dtype=np.uint8)
    _outlined_bubble(image, (100, 110), 85)
    _outlined_bubble(image, (340, 110), 85)
    image[95:125, 180:260] = 255
    observations = [{"observation_id": "left", "bbox_page": [60, 90, 140, 125]}]

    containers = discover_white_containers(image, observations)

    assert containers[0]["kind"] == "balloon"
    assert containers[0]["lobe_bboxes"] == []


def test_uncertain_shape_candidate_does_not_corroborate_second_lobe() -> None:
    from vision_runtime.white_containers import discover_white_containers

    image = np.full((220, 440, 3), 210, dtype=np.uint8)
    _outlined_bubble(image, (100, 110), 85)
    _outlined_bubble(image, (340, 110), 85)
    image[95:125, 180:260] = 255
    observations = [
        {"observation_id": "left", "bbox_page": [60, 90, 140, 125]},
        {
            "observation_id": "shape",
            "bbox_page": [300, 60, 390, 155],
            "selection_state": "uncertain",
        },
    ]

    containers = discover_white_containers(image, observations)

    assert set(containers[0]["observation_ids"]) == {"left", "shape"}
    assert containers[0]["kind"] == "balloon"
    assert containers[0]["lobe_bboxes"] == []


def test_aligned_line_stack_does_not_corroborate_vertical_lobes() -> None:
    from vision_runtime.white_containers import corroborate_lobes

    lobes = [[0, 0, 200, 120], [0, 100, 200, 240]]
    observations = [
        {"observation_id": "top", "bbox_page": [40, 30, 160, 60]},
        {"observation_id": "bottom", "bbox_page": [45, 170, 155, 200]},
    ]

    assert corroborate_lobes(lobes, observations, [0, 0, 200, 240]) == []


def test_diagonal_text_clusters_corroborate_connected_lobes() -> None:
    from vision_runtime.white_containers import corroborate_lobes

    lobes = [[0, 0, 180, 140], [100, 100, 300, 280]]
    observations = [
        {"observation_id": "first", "bbox_page": [20, 30, 100, 60]},
        {"observation_id": "second", "bbox_page": [190, 190, 280, 220]},
    ]

    assert corroborate_lobes(lobes, observations, [0, 0, 300, 280]) == lobes


def test_spatial_text_clusters_recover_wide_neck_connected_subblocks() -> None:
    from vision_runtime.white_containers import infer_text_cluster_lobes

    observations = [
        {"observation_id": "a1", "bbox_page": [60, 40, 170, 70]},
        {"observation_id": "a2", "bbox_page": [50, 72, 180, 102]},
        {"observation_id": "b1", "bbox_page": [230, 150, 340, 180]},
        {"observation_id": "b2", "bbox_page": [210, 182, 360, 212]},
        {"observation_id": "b3", "bbox_page": [205, 214, 365, 244]},
    ]

    lobes = infer_text_cluster_lobes([0, 0, 400, 280], observations)

    assert len(lobes) == 2
    assert lobes[0][2] < lobes[1][2]


def test_centered_multiline_text_does_not_create_cluster_lobes() -> None:
    from vision_runtime.white_containers import infer_text_cluster_lobes

    observations = [
        {"observation_id": "a", "bbox_page": [80, 30, 220, 60]},
        {"observation_id": "b", "bbox_page": [70, 65, 230, 95]},
        {"observation_id": "c", "bbox_page": [85, 100, 215, 130]},
        {"observation_id": "d", "bbox_page": [75, 135, 225, 165]},
    ]

    assert infer_text_cluster_lobes([0, 0, 300, 200], observations) == []
