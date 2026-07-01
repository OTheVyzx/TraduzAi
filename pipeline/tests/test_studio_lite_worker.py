from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio_lite.worker import build_mask, detect_page, handle_request, inpaint_region, model_status


def test_build_mask_writes_white_rectangles_from_bboxes(tmp_path: Path) -> None:
    output_path = tmp_path / "mask.png"

    result = build_mask(
        {
            "width": 12,
            "height": 10,
            "output_path": str(output_path),
            "bboxes": [[2, 3, 6, 8], [10, 0, 20, 2]],
        }
    )

    mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (10, 12)
    assert int(mask[3, 2]) == 255
    assert int(mask[7, 5]) == 255
    assert int(mask[8, 5]) == 0
    assert int(mask[1, 11]) == 255
    assert result["boxes"] == [[2, 3, 6, 8], [10, 0, 12, 2]]
    assert result["mask_pixels"] == 24


def test_build_mask_accepts_detection_bbox_dicts_and_xywh(tmp_path: Path) -> None:
    output_path = tmp_path / "mask.png"

    build_mask(
        {
            "width": 8,
            "height": 8,
            "output_path": str(output_path),
            "bbox_format": "xywh",
            "detections": [{"bbox": [1, 2, 3, 4], "score": 0.9}],
        }
    )

    mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert int(np.count_nonzero(mask)) == 12
    assert int(mask[2, 1]) == 255
    assert int(mask[5, 3]) == 255
    assert int(mask[6, 3]) == 0


def test_inpaint_region_only_changes_supplied_roi(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "out.png"

    image = np.full((20, 24, 3), 200, dtype=np.uint8)
    image[:, 16:20] = [20, 20, 20]
    image[8:12, 6:10] = [0, 0, 0]
    cv2.imwrite(str(image_path), image)

    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[8:12, 6:10] = 255
    mask[:, 16:20] = 255
    cv2.imwrite(str(mask_path), mask)

    result = inpaint_region(
        {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "output_path": str(output_path),
            "bbox": [4, 6, 12, 14],
        }
    )

    out = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert out.shape == image.shape
    assert result["bbox"] == [4, 6, 12, 14]
    assert not np.array_equal(out[8:12, 6:10], image[8:12, 6:10])
    assert np.array_equal(out[:, 16:20], image[:, 16:20])


def test_inpaint_region_with_invalid_supplied_bbox_does_not_fall_back_to_full_page(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "out.png"

    image = np.full((10, 12, 3), 200, dtype=np.uint8)
    image[2:5, 2:5] = [0, 0, 0]
    cv2.imwrite(str(image_path), image)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    cv2.imwrite(str(mask_path), mask)

    result = inpaint_region(
        {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "output_path": str(output_path),
            "bbox": [30, 30, 40, 40],
        }
    )

    out = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert result["bbox"] is None
    assert np.array_equal(out, image)


def test_inpaint_region_accepts_rgba_png(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "out.png"

    image = np.full((14, 16, 4), 255, dtype=np.uint8)
    image[:, :, 3] = 180
    image[5:9, 6:10, :3] = [0, 0, 0]
    cv2.imwrite(str(image_path), image)
    mask = np.zeros((14, 16), dtype=np.uint8)
    mask[5:9, 6:10] = 255
    cv2.imwrite(str(mask_path), mask)

    inpaint_region(
        {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "output_path": str(output_path),
            "bbox": [5, 4, 11, 10],
        }
    )

    out = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert out.shape == image.shape
    assert np.all(out[:, :, 3] == 180)


def test_detect_page_missing_model_returns_empty_detections(monkeypatch) -> None:
    monkeypatch.delenv("TRADUZAI_STUDIO_LITE_DETECTOR_ONNX", raising=False)

    result = detect_page({"image_path": "missing-image-is-not-read.png", "model_path": "missing-model.onnx"})

    assert result["detections"] == []
    assert result["model_status"]["state"] == "missing"
    assert result["model_status"]["source"] == "manual_path"
    assert result["model_status"]["downloads"] is False


def test_model_status_reports_manual_ready_path(tmp_path: Path) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"not a real model")

    result = model_status({"model_path": str(model_path)})

    assert result["state"] == "ready"
    assert result["source"] == "manual_path"
    assert result["path"] == str(model_path)
    assert result["downloads"] is False


def test_handle_request_wraps_command_response(tmp_path: Path) -> None:
    response = handle_request(
        {
            "command": "build_mask",
            "width": 4,
            "height": 4,
            "output_path": str(tmp_path / "mask.png"),
            "bboxes": [[1, 1, 3, 3]],
        }
    )

    assert response["ok"] is True
    assert response["command"] == "build_mask"
    assert response["result"]["mask_pixels"] == 4


def test_cli_reads_json_from_stdin_and_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "mask.png"
    request = {
        "command": "build_mask",
        "width": 5,
        "height": 5,
        "output_path": str(output_path),
        "bboxes": [[0, 0, 2, 2]],
    }

    completed = subprocess.run(
        [sys.executable, "-m", "studio_lite.worker"],
        input=json.dumps(request),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=True,
    )

    response = json.loads(completed.stdout)
    assert response["ok"] is True
    assert response["result"]["mask_pixels"] == 4
    assert output_path.exists()
