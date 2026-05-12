from types import SimpleNamespace

import numpy as np

from vision_stack.runtime import _apply_adaptive_cjk_reocr


class _FakeOcr:
    _backend = "fake"

    def __init__(self):
        self.calls = 0

    def recognize_batch(self, crops):
        self.calls += 1
        return [{"text": "\ud558\ud558", "confidence": 0.95} for _ in crops]


def test_adaptive_cjk_reocr_attempts_expanded_bbox_for_partial_multiline(monkeypatch):
    monkeypatch.setenv("TRADUZAI_CJK_BBOX_EXPANDED_REOCR", "1")
    image = np.full((140, 180, 3), 255, dtype=np.uint8)
    block = SimpleNamespace(
        xyxy=(50.0, 40.0, 110.0, 70.0),
        confidence=0.8,
        mask=None,
        line_polygons=None,
        source_direction=None,
    )
    page_result = {
        "image": "page.jpg",
        "width": 180,
        "height": 140,
        "texts": [
            {
                "text": "\ud558",
                "bbox": [50, 40, 110, 70],
                "confidence": 0.4,
                "tipo": "fala",
                "line_polygons": [
                    [[50, 40], [110, 40], [110, 52], [50, 52]],
                    [[50, 56], [110, 56], [110, 70], [50, 70]],
                ],
                "skip_processing": False,
            }
        ],
        "_vision_blocks": [{"bbox": [50, 40, 110, 70], "confidence": 0.8}],
    }
    ocr = _FakeOcr()

    updated = _apply_adaptive_cjk_reocr(
        image_rgb=image,
        image_label="page.jpg",
        page_result=page_result,
        blocks=[block],
        ocr=ocr,
        profile="quality",
        backend_name="fake",
        idioma_origem="ko",
    )

    assert ocr.calls == 1
    assert any(item["stage"] == "bbox_expanded_reocr" for item in updated["route_history"])
    assert updated["page_quality"]["mode"] == "cjk_adaptive"
