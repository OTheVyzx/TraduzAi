from __future__ import annotations

import cv2
import numpy as np


def test_full_page_visual_container_recovers_independent_balloon_geometry():
    from ownership.container_evidence import recover_full_page_visual_container

    image = np.full((180, 240, 3), 35, dtype=np.uint8)
    cv2.ellipse(image, (120, 90), (84, 58), 0, 0, 360, (245, 245, 245), -1)
    cv2.ellipse(image, (120, 90), (84, 58), 0, 0, 360, (5, 5, 5), 3)

    evidence = recover_full_page_visual_container(
        image,
        owner_id="owner_a",
        semantic_body_bbox_page=(78, 65, 163, 116),
        source_replacement_bbox_page=(88, 73, 154, 108),
    )

    assert evidence is not None
    assert evidence["source"] == "full_page_visual_container"
    assert evidence["page_shape"] == (180, 240, 3)
    assert evidence["bbox_page"] != (88, 73, 154, 108)
    x1, y1, x2, y2 = evidence["bbox_page"]
    assert x1 <= 78 and y1 <= 65 and x2 >= 163 and y2 >= 116


def test_full_page_visual_container_rejects_proportional_fallback_without_boundary():
    from ownership.container_evidence import recover_full_page_visual_container

    image = np.full((180, 240, 3), 245, dtype=np.uint8)

    evidence = recover_full_page_visual_container(
        image,
        owner_id="owner_a",
        semantic_body_bbox_page=(78, 65, 163, 116),
        source_replacement_bbox_page=(88, 73, 154, 108),
    )

    assert evidence is None


def test_component_container_falls_back_to_executable_support_local_geometry():
    from ownership.container_evidence import recover_component_visual_container

    image = np.full((180, 240, 3), 245, dtype=np.uint8)
    evidence = recover_component_visual_container(
        image,
        component_id="component-a",
        glyph_bbox_page=(78, 65, 163, 116),
        glyph_polygon_page=((78, 65), (163, 65), (163, 116), (78, 116)),
    )

    assert evidence["source"] == "conservative_support_local_container"
    assert evidence["conservative"] is True
    x1, y1, x2, y2 = evidence["bbox_page"]
    assert x1 <= 78 and y1 <= 65 and x2 >= 163 and y2 >= 116


def test_multiline_body_recovers_one_canonical_visual_container():
    from ownership.container_evidence import recover_component_visual_container

    image = np.full((280, 480, 3), 28, dtype=np.uint8)
    image[150:, :] = 248
    cv2.line(image, (0, 150), (479, 150), (175, 175, 175), 2)
    cv2.ellipse(image, (240, 132), (216, 126), 0, 0, 360, (245, 245, 245), -1)
    cv2.ellipse(image, (240, 132), (216, 126), 0, 0, 360, (8, 8, 8), 3)
    line_boxes = (
        (155, 82, 325, 112),
        (108, 119, 372, 149),
        (132, 158, 348, 188),
    )
    for x1, y1, x2, y2 in line_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (35, 35, 35), 4)

    recovered = tuple(
        recover_component_visual_container(
            image,
            component_id=f"component-{index}",
            glyph_bbox_page=bbox,
            glyph_polygon_page=(
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ),
        )
        for index, bbox in enumerate(line_boxes)
    )

    assert {item["source"] for item in recovered} == {"full_page_visual_container"}
    assert len({item["evidence_id"] for item in recovered}) == 1
    assert len({item["bbox_page"] for item in recovered}) == 1
    container_bbox = recovered[0]["bbox_page"]
    assert all(
        container_bbox[0] <= bbox[0]
        and container_bbox[1] <= bbox[1]
        and container_bbox[2] >= bbox[2]
        and container_bbox[3] >= bbox[3]
        for bbox in line_boxes
    )


def test_adjacent_balloon_lines_recover_distinct_visual_containers():
    from ownership.container_evidence import recover_component_visual_container

    image = np.full((220, 520, 3), 30, dtype=np.uint8)
    for center in ((130, 110), (390, 110)):
        cv2.ellipse(image, center, (112, 82), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(image, center, (112, 82), 0, 0, 360, (5, 5, 5), 3)
    boxes = ((70, 92, 190, 122), (330, 92, 450, 122))

    recovered = tuple(
        recover_component_visual_container(
            image,
            component_id=f"component-{index}",
            glyph_bbox_page=bbox,
            glyph_polygon_page=(
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ),
        )
        for index, bbox in enumerate(boxes)
    )

    assert all(item["source"] == "full_page_visual_container" for item in recovered)
    assert recovered[0]["evidence_id"] != recovered[1]["evidence_id"]
    assert recovered[0]["bbox_page"] != recovered[1]["bbox_page"]


def test_local_containers_beat_shared_page_spanning_threshold_region():
    from ownership.container_evidence import recover_component_visual_container

    image = np.full((720, 320, 3), 255, dtype=np.uint8)
    image[12:708, 12:308] = 230
    local_boxes = ((42, 90, 278, 250), (42, 450, 278, 610))
    text_boxes = ((92, 142, 228, 198), (92, 502, 228, 558))
    for x1, y1, x2, y2 in local_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (45, 45, 45), -1)

    recovered = tuple(
        recover_component_visual_container(
            image,
            component_id=f"component-{index}",
            glyph_bbox_page=bbox,
            glyph_polygon_page=(
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ),
        )
        for index, bbox in enumerate(text_boxes)
    )

    assert recovered[0]["bbox_page"] != recovered[1]["bbox_page"]
    assert recovered[0]["evidence_id"] != recovered[1]["evidence_id"]
    for recovered_item, local_bbox in zip(recovered, local_boxes):
        assert sum(
            abs(int(actual) - int(expected))
            for actual, expected in zip(recovered_item["bbox_page"], local_bbox)
        ) <= 4


def test_nested_local_container_evidence_keeps_one_multiline_body():
    from ownership.container_evidence import canonical_component_container_ids

    evidence = {
        "title": {
            "evidence_id": "container-title",
            "source": "full_page_visual_container",
            "bbox_page": (120, 100, 480, 230),
        },
        "body": {
            "evidence_id": "container-card",
            "source": "full_page_visual_container",
            "bbox_page": (80, 80, 520, 420),
        },
        "footer": {
            "evidence_id": "container-footer",
            "source": "full_page_visual_container",
            "bbox_page": (140, 300, 460, 400),
        },
        "other": {
            "evidence_id": "container-other",
            "source": "full_page_visual_container",
            "bbox_page": (100, 520, 500, 680),
        },
    }

    ids = canonical_component_container_ids(evidence)

    assert ids["title"] == ids["body"] == ids["footer"]
    assert ids["other"] != ids["body"]


def test_page_spanning_background_groups_only_neighbouring_text_bodies():
    from ownership.container_evidence import canonical_component_container_ids

    shared = {
        "evidence_id": "container-page-background",
        "source": "full_page_visual_container",
        "bbox_page": (0, 0, 790, 900),
        "page_shape": (1000, 800, 3),
    }
    evidence = {
        "line-1": {**shared, "semantic_bbox_page": (100, 100, 300, 120)},
        "line-2": {**shared, "semantic_bbox_page": (90, 130, 310, 150)},
        "distant": {**shared, "semantic_bbox_page": (100, 500, 300, 520)},
        "other-column": {
            **shared,
            "semantic_bbox_page": (500, 100, 620, 120),
        },
    }

    ids = canonical_component_container_ids(evidence)

    assert ids["line-1"] == ids["line-2"]
    assert ids["distant"] != ids["line-1"]
    assert ids["other-column"] != ids["line-1"]
    assert len(set(ids.values())) == 3
