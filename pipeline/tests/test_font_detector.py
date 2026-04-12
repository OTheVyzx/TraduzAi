import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from typesetter.font_detector import FontDetector, _render_font_sample_textpath


class FontDetectorTests(unittest.TestCase):
    def test_render_font_sample_textpath_renders_project_font_to_rgb_image(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font_path = fonts_dir / "CCDaveGibbonsLower W00 Regular.ttf"

        sample = _render_font_sample_textpath(
            font_path=font_path,
            upper_line="ABCDEFGH123!?",
            lower_line="abcdefgh123!?",
            canvas_size=224,
        )

        self.assertEqual(sample.shape, (224, 224, 3))
        self.assertEqual(sample.dtype, np.uint8)
        self.assertLess(int(np.min(sample)), 245)
        self.assertGreater(int(np.mean(sample)), 120)

    def test_detect_can_force_non_default_candidate_for_textured_balloon(self):
        detector = FontDetector(Path("dummy.safetensors"), Path("fonts"))
        detector._loaded = True
        detector._fingerprints = {
            "DK Full Blast.otf": np.array([0.60, 0.40], dtype=np.float32),
            "SINGLE FIGHTER.otf": np.array([0.71, 0.05], dtype=np.float32),
            "Libel Suit Suit Rg.otf": np.array([0.30, 0.70], dtype=np.float32),
        }
        region = np.full((32, 64, 3), 255, dtype=np.uint8)

        with patch.object(detector, "_extract_features", return_value=np.array([1.0, 0.0], dtype=np.float32)):
            self.assertEqual(detector.detect(region, allow_default=False), "SINGLE FIGHTER.otf")
            self.assertEqual(detector.detect(region, allow_default=True), "CCDaveGibbonsLower W00 Regular.ttf")


if __name__ == "__main__":
    unittest.main()
