import unittest

import cv2
import numpy as np

from vision_stack.gpu_image_ops import (
    apply_white_fill,
    connected_components_with_stats,
    expand_mask,
    probe_gpu_image_ops,
    resize_crops_batch,
)


class GpuImageOpsTests(unittest.TestCase):
    def test_apply_white_fill_updates_only_masked_pixels(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :] = [10, 20, 30]
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 3:7] = 255

        filled = apply_white_fill(image, mask, backend="cpu")

        self.assertTrue(np.all(filled[2:5, 3:7] == 255))
        self.assertTrue(np.all(filled[:2, :] == [10, 20, 30]))
        self.assertTrue(np.all(filled[:, :3][mask[:, :3] == 0] == [10, 20, 30]))

    def test_expand_mask_matches_opencv_cpu_reference(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 255
        expected = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=2,
        )

        expanded = expand_mask(mask, kernel_size=3, iterations=2, backend="cpu")

        np.testing.assert_array_equal(expanded, expected)

    def test_connected_components_returns_complex_component_stats(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:4, 2:5] = 255
        mask[7:11, 8:10] = 255

        components = connected_components_with_stats(mask, min_area=1, backend="cpu")

        self.assertEqual(len(components), 2)
        self.assertEqual(components[0]["bbox"], [2, 1, 5, 4])
        self.assertEqual(components[0]["area"], 9)
        self.assertEqual(components[1]["bbox"], [8, 7, 10, 11])
        self.assertEqual(components[1]["area"], 8)

    def test_resize_crops_batch_keeps_order_and_shape(self):
        crops = [
            np.full((4, 8, 3), 10, dtype=np.uint8),
            np.full((6, 3, 3), 200, dtype=np.uint8),
        ]

        resized = resize_crops_batch(crops, size=(5, 7), backend="cpu")

        self.assertEqual(len(resized), 2)
        self.assertEqual(resized[0].shape, (7, 5, 3))
        self.assertEqual(resized[1].shape, (7, 5, 3))
        self.assertLess(int(resized[0].mean()), int(resized[1].mean()))

    def test_resize_crops_batch_accepts_grayscale_ocr_preprocess_crops(self):
        crops = [
            np.full((4, 8), 10, dtype=np.uint8),
            np.full((6, 3), 200, dtype=np.uint8),
        ]

        resized = resize_crops_batch(crops, size=(5, 7), backend="cpu")

        self.assertEqual(len(resized), 2)
        self.assertEqual(resized[0].shape, (7, 5))
        self.assertEqual(resized[1].shape, (7, 5))
        self.assertLess(int(resized[0].mean()), int(resized[1].mean()))

    def test_probe_reports_backend_availability_without_requiring_cuda(self):
        probe = probe_gpu_image_ops()

        self.assertIn("torch_cuda", probe)
        self.assertIn("cv2_cuda", probe)
        self.assertIn("selected_backend", probe)


if __name__ == "__main__":
    unittest.main()
