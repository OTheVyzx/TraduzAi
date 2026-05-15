from types import SimpleNamespace

import numpy as np

from vision_stack.text_refiner import PPOCRv5TextRefiner


def test_ppocrv5_text_refiner_normalizes_structured_records():
    class FakeOcr:
        def recognize_blocks_from_page(self, image_rgb, blocks, **kwargs):
            assert kwargs["allow_sparse_mapping"] is True
            return [
                {
                    "text": "Hello",
                    "source_bbox": [11, 12, 21, 24],
                    "line_polygons": [[[11, 12], [21, 12], [21, 24], [11, 24]]],
                    "text_pixel_bbox": [11, 12, 21, 24],
                    "confidence": 0.91,
                }
            ]

    refiner = PPOCRv5TextRefiner(quality="ultra", ocr_engine=FakeOcr())
    image = np.zeros((40, 50, 3), dtype=np.uint8)

    refined = refiner.refine(image, [SimpleNamespace(xyxy=(10, 10, 25, 30), confidence=0.8)], quality="ultra")

    assert refined[0]["text"] == "Hello"
    assert refined[0]["source_bbox"] == [11, 12, 21, 24]
    assert refined[0]["text_pixel_bbox"] == [11, 12, 21, 24]
    assert refined[0]["confidence"] == 0.91
    assert refined[0]["text_refiner"] == "ppocrv5_text_refiner"


def test_ppocrv5_text_refiner_falls_back_to_bbox_when_engine_fails():
    class FailingOcr:
        def recognize_blocks_from_page(self, image_rgb, blocks, **kwargs):
            raise RuntimeError("missing model")

    refiner = PPOCRv5TextRefiner(ocr_engine=FailingOcr())
    image = np.zeros((40, 50, 3), dtype=np.uint8)

    refined = refiner.refine(image, [SimpleNamespace(xyxy=(1, 2, 9, 10), confidence=0.6)])

    assert refined == [
        {
            "text": "",
            "bbox": [1, 2, 9, 10],
            "source_bbox": [1, 2, 9, 10],
            "line_polygons": [],
            "text_pixel_bbox": [1, 2, 9, 10],
            "confidence": 0.6,
            "text_refiner": "ppocrv5_text_refiner",
        }
    ]
