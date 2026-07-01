import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_cjk_quality_run import analyze_sfx_benchmark


def _write_red_sfx(path: Path) -> None:
    image = np.full((240, 320, 3), 236, dtype=np.uint8)
    for x in range(0, 320, 7):
        cv2.line(image, (x, 20), (max(0, x - 120), 210), (38, 42, 52), 1)
    cv2.rectangle(image, (62, 48), (82, 172), (116, 32, 38), -1)
    cv2.rectangle(image, (62, 142), (164, 166), (116, 32, 38), -1)
    cv2.rectangle(image, (202, 46), (226, 178), (116, 32, 38), -1)
    cv2.rectangle(image, (202, 46), (282, 70), (116, 32, 38), -1)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_sfx_benchmark_reports_detection_metrics_from_manifest(tmp_path):
    image_path = tmp_path / "sample.png"
    _write_red_sfx(image_path)
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "expected_sfx": [
                    {"label": "red_left", "bbox": [45, 40, 175, 180]},
                    {"label": "red_right", "bbox": [190, 40, 292, 185]},
                ]
            }
        ),
        encoding="utf-8",
    )

    with patch("sfx.ocr_probe.probe_sfx_candidate_ocr", side_effect=lambda candidate, image: {
        **candidate,
        "sfx_ocr": {"status": "no_confident_cjk", "attempts": [{"lang": "ko", "text": "", "confidence": 0.0}]},
    }):
        result = analyze_sfx_benchmark(tmp_path)

    assert result["status"] == "PASS"
    assert result["expected_count"] == 2
    assert result["detected_count"] >= 1
    assert result["matched_count"] >= 1
    assert result["mean_iou"] is not None
    overlay_path = Path(result["items"][0]["overlay_path"])
    assert overlay_path.exists()
    assert overlay_path.parent.name == "sfx_benchmark_visual"
    assert Path(result["items"][0]["ocr_report_path"]).exists()
    assert any(Path(item["crop_path"]).exists() for item in result["items"][0]["detections"] if item.get("crop_path"))


def test_sfx_benchmark_keeps_unlabeled_images_supported(tmp_path):
    _write_red_sfx(tmp_path / "sample.png")

    with patch("sfx.ocr_probe.probe_sfx_candidate_ocr", side_effect=lambda candidate, image: candidate):
        result = analyze_sfx_benchmark(tmp_path)

    assert result["status"] == "PASS"
    assert result["item_count"] == 1
    assert result["route_counts"]["unlabeled"] == 1
    assert Path(result["items"][0]["overlay_path"]).exists()
    assert Path(result["items"][0]["ocr_report_path"]).exists()
