import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from typesetter.font_detector import FontDetector, _render_font_sample_textpath
from typesetter.google_fonts import GoogleFontSpec, google_family_slug


class FontDetectorTests(unittest.TestCase):
    def test_render_font_sample_textpath_renders_project_font_to_rgb_image(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font_path = fonts_dir / "ComicNeue-Bold.ttf"

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
            self.assertEqual(detector.detect(region, allow_default=True), "ComicNeue-Bold.ttf")

    def test_discovers_detector_fonts_from_font_map(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        detector = FontDetector(Path("dummy.safetensors"), fonts_dir)

        candidates = detector._discover_candidate_fonts()

        self.assertIn("KOMIKAX_.ttf", candidates)
        self.assertIn("CCDaveGibbonsLower W00 Regular.ttf", candidates)
        self.assertIn("LeagueGothic-Regular-VariableFont_wdth.ttf", candidates)
        self.assertNotIn("Newrotic.ttf", candidates)
        self.assertNotIn("Bangers-Regular.ttf", candidates)
        self.assertNotIn("LuckiestGuy-Regular.ttf", candidates)
        self.assertNotIn("PermanentMarker-Regular.ttf", candidates)
        self.assertNotIn("DK Full Blast.otf", candidates)

    def test_google_font_specs_use_repository_slug(self):
        self.assertEqual(google_family_slug("Comic Neue"), "comicneue")
        self.assertEqual(google_family_slug("Bangers"), "bangers")

    def test_google_fonts_can_be_added_when_enabled(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fonts_dir = Path(tmp)
            (fonts_dir / "ComicNeue-Bold.ttf").write_bytes(b"dummy")
            (fonts_dir / "font-map.json").write_text(
                json.dumps(
                    {
                        "available": [
                            {
                                "arquivo": "Bangers-Regular.ttf",
                                "detector": True,
                                "google_family": "Bangers",
                                "google_weight": 400,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            google_path = fonts_dir / "google" / "Bangers-Regular.ttf"

            detector = FontDetector(
                Path("dummy.safetensors"),
                fonts_dir,
                enable_google_fonts=True,
            )

            with patch(
                "typesetter.google_fonts.download_google_font_family",
                return_value=google_path,
            ) as download:
                candidates = detector._discover_candidate_fonts()

        download.assert_called_once_with(GoogleFontSpec("Bangers", 400, False), fonts_dir)
        self.assertIn("Bangers-Regular.ttf", candidates)


if __name__ == "__main__":
    unittest.main()
