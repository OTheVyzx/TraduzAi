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
