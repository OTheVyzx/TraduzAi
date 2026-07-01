import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_stack.paddle_compat import create_paddle_ocr


class PaddleCompatTests(unittest.TestCase):
    def test_modern_constructor_with_legacy_ocr_method_falls_back_to_ocr(self):
        class ConstructorAcceptsModernArgsButLegacyRuntime:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls = []

            def ocr(self, img, *, det=True, rec=True, cls=False):
                self.calls.append((img, det, rec, cls))
                return [["legacy-result"]]

        compat = create_paddle_ocr(
            ConstructorAcceptsModernArgsButLegacyRuntime,
            lang="en",
            use_gpu=False,
            use_angle_cls=False,
            enable_mkldnn=True,
        )

        self.assertEqual(compat.ocr("image", det=True, rec=True, cls=False), [["legacy-result"]])


if __name__ == "__main__":
    unittest.main()
