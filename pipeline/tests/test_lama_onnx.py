import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from inpainter.lama_onnx import (
    build_lama_region_jobs,
    merge_inpainted_crop,
    pad_to_modulo,
    prepare_lama_dynamic_inputs,
    prepare_lama_inputs,
    resolve_windows_onnxruntime_support_dirs,
)


class LamaOnnxTests(unittest.TestCase):
    def test_prepare_lama_inputs_returns_expected_shapes(self):
        image = np.full((120, 180, 3), 200, dtype=np.uint8)
        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[30:80, 50:130] = 255

        image_input, mask_input = prepare_lama_inputs(image, mask)

        self.assertEqual(image_input.shape, (1, 3, 512, 512))
        self.assertEqual(mask_input.shape, (1, 1, 512, 512))
        self.assertAlmostEqual(float(mask_input.max()), 1.0, places=5)

    def test_prepare_lama_dynamic_inputs_preserves_content_and_pads_to_modulo(self):
        image = np.full((121, 183, 3), 180, dtype=np.uint8)
        mask = np.zeros((121, 183), dtype=np.uint8)
        mask[10:60, 22:111] = 255

        image_input, mask_input, original_size = prepare_lama_dynamic_inputs(image, mask)

        self.assertEqual(original_size, (121, 183))
        self.assertEqual(image_input.shape[2] % 8, 0)
        self.assertEqual(image_input.shape[3] % 8, 0)
        self.assertEqual(mask_input.shape[2:], image_input.shape[2:])

    def test_pad_to_modulo_keeps_original_content_in_top_left(self):
        image = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)

        padded = pad_to_modulo(image, modulo=8)

        self.assertEqual(padded.shape, (3, 8, 8))
        np.testing.assert_array_equal(padded[:, :5, :7], image)

    def test_build_lama_region_jobs_creates_masked_regions(self):
        image = np.full((180, 220, 3), 230, dtype=np.uint8)
        texts = [
            {"bbox": [70, 50, 130, 70], "tipo": "fala", "confidence": 0.9},
            {"bbox": [68, 78, 138, 100], "tipo": "fala", "confidence": 0.9},
        ]

        jobs = build_lama_region_jobs(image, texts)

        self.assertGreaterEqual(len(jobs), 1)
        self.assertTrue(np.any(jobs[0]["mask"] > 0))

    def test_merge_inpainted_crop_only_replaces_masked_pixels(self):
        base = np.full((80, 100, 3), 30, dtype=np.uint8)
        crop = np.full((20, 30, 3), 180, dtype=np.uint8)
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 8:22] = 255

        merged = merge_inpainted_crop(base, crop, mask, [10, 20, 40, 40])

        self.assertEqual(int(merged[0, 0, 0]), 30)
        self.assertEqual(int(merged[29, 25, 0]), 180)

    def test_resolve_windows_onnxruntime_support_dirs_collects_cuda_bins_from_package_root(self):
        with TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            expected_dirs = [
                package_root / "nvidia" / "cuda_runtime" / "bin",
                package_root / "nvidia" / "cublas" / "bin",
                package_root / "nvidia" / "cufft" / "bin",
                package_root / "nvidia" / "cudnn" / "bin",
                package_root / "tensorrt_libs",
            ]
            for directory in expected_dirs:
                directory.mkdir(parents=True, exist_ok=True)

            resolved = resolve_windows_onnxruntime_support_dirs(package_root=package_root)

            self.assertEqual(
                [path.resolve() for path in resolved],
                [path.resolve() for path in expected_dirs],
            )


if __name__ == "__main__":
    unittest.main()
