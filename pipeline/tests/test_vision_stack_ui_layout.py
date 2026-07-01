import unittest
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_stack.ui_layout import attach_uied_layout_evidence, detect_uied_like_components


class VisionStackUiLayoutTests(unittest.TestCase):
    def test_detect_uied_like_components_finds_form_bars_and_fields(self):
        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        image[42:72, 24:296] = [184, 196, 224]
        image[108:128, 146:286] = [224, 221, 219]
        image[136:156, 24:296] = [92, 88, 86]

        components = detect_uied_like_components(image)

        bboxes = [component.bbox for component in components]
        self.assertTrue(any(bbox[0] <= 24 and bbox[2] >= 296 and bbox[1] <= 42 <= bbox[3] for bbox in bboxes))
        self.assertTrue(any(bbox[0] <= 146 and bbox[2] >= 286 and bbox[1] <= 108 <= bbox[3] for bbox in bboxes))
        self.assertTrue(any(component.component_type == "ui_panel" for component in components))

    def test_detect_uied_like_components_splits_horizontal_bands_connected_by_noise(self):
        image = np.full((220, 360, 3), 255, dtype=np.uint8)
        image[44:64, 40:320] = [184, 196, 224]
        image[82:112, 40:320] = [184, 196, 224]
        image[44:112, 176:180] = [184, 196, 224]

        components = detect_uied_like_components(image)

        bboxes = [component.bbox for component in components]
        self.assertTrue(any(bbox[1] <= 44 and bbox[3] >= 64 and (bbox[3] - bbox[1]) < 34 for bbox in bboxes))
        self.assertTrue(any(bbox[1] <= 82 and bbox[3] >= 112 and (bbox[3] - bbox[1]) < 44 for bbox in bboxes))

    def test_detect_uied_like_components_finds_perspective_search_panel(self):
        image = np.full((240, 420, 3), 255, dtype=np.uint8)
        panel = np.array([[82, 92], [332, 70], [350, 154], [66, 176]], dtype=np.int32)
        cv2.fillPoly(image, [panel], [86, 82, 80])
        cv2.putText(
            image,
            "Search",
            (112, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        components = detect_uied_like_components(image)

        bboxes = [component.bbox for component in components]
        self.assertTrue(
            any(bbox[0] <= 82 and bbox[2] >= 332 and bbox[1] <= 92 and bbox[3] >= 154 for bbox in bboxes),
            bboxes,
        )

    def test_attach_uied_layout_evidence_marks_text_inside_ui_component(self):
        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        image[42:72, 24:296] = [184, 196, 224]
        texts = [
            {
                "id": "ocr_001",
                "text": "Open status window",
                "bbox": [70, 54, 236, 64],
                "text_pixel_bbox": [70, 54, 236, 64],
                "line_polygons": [[[70, 54], [236, 54], [236, 64], [70, 64]]],
                "layout_profile": "white_balloon",
                "block_profile": "white_balloon",
            }
        ]

        updated, components = attach_uied_layout_evidence(image, texts)

        self.assertGreaterEqual(len(components), 1)
        evidence = updated[0].get("ui_layout_evidence")
        self.assertIsInstance(evidence, dict)
        self.assertEqual(evidence["source"], "uied_cv")
        self.assertEqual(evidence["role"], "text_inside_component")
        self.assertEqual(updated[0]["background_rgb"], [184, 196, 224])
        self.assertEqual(updated[0]["layout_profile"], "ui_form")

    def test_attach_uied_layout_evidence_marks_label_near_multiple_components(self):
        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        image[96:116, 150:292] = [224, 221, 219]
        image[126:146, 150:224] = [224, 221, 219]
        texts = [
            {
                "id": "ocr_label",
                "text": "Name Resident registration number",
                "bbox": [34, 92, 130, 146],
                "text_pixel_bbox": [34, 92, 130, 146],
                "line_polygons": [
                    [[34, 92], [130, 92], [130, 106], [34, 106]],
                    [[34, 126], [130, 126], [130, 146], [34, 146]],
                ],
                "layout_profile": "white_balloon",
                "block_profile": "white_balloon",
            }
        ]

        updated, _components = attach_uied_layout_evidence(image, texts)

        evidence = updated[0].get("ui_layout_evidence")
        self.assertIsInstance(evidence, dict)
        self.assertEqual(evidence["role"], "label_near_components")
        self.assertEqual(updated[0]["layout_profile"], "ui_form")


if __name__ == "__main__":
    unittest.main()
