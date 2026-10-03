"""Small same-process detector + OCR comparison on CH57 pages 022-024."""

import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_stack import runtime

ROOT = Path(r"N:\TraduzAI")
SOURCE = ROOT / "artifacts/e2e/optimized_e2e_20260907/full_chapter_observation_57_complete_02/source_input"
runtime._configure_model_roots(str(ROOT / "pipeline" / "models"))
detector = runtime._get_detector("max")
engine = runtime._get_ocr_engine("max", lang="en")
rows = []
for page_number in (22, 23, 24, 27):
    image = cv2.cvtColor(cv2.imread(str(SOURCE / f"{page_number:03d}.webp")), cv2.COLOR_BGR2RGB)
    started = time.perf_counter()
    blocks = detector.detect(image, conf_threshold=runtime._profile_to_detection_threshold("max"))
    detect_seconds = time.perf_counter() - started
    for mode in ("baseline", "crop_first", "baseline_repeat", "crop_first_repeat"):
        os.environ["TRADUZAI_OCR_CROP_FIRST"] = "1" if mode.startswith("crop_first") else "0"
        request = runtime._runtime_ocr_request(image, page_id=f"page_{page_number:03d}", provider_family="paddleocr", invocation_kind=f"sample-{mode}")
        started = time.perf_counter()
        result = engine.recognize_page_with_evidence(image, blocks, request=request)
        ocr_seconds = time.perf_counter() - started
        row = {
            "page": page_number, "mode": mode, "image_shape": list(image.shape),
            "detector_boxes": [list(map(float, block.xyxy)) for block in blocks],
            "detect_seconds": round(detect_seconds, 3), "ocr_seconds": round(ocr_seconds, 3),
            "detect_plus_ocr_seconds": round(detect_seconds + ocr_seconds, 3),
            "provider_calls": len(result.attempts),
            "blocks": [block.text for block in result.blocks],
            "observations": [{"text": item.text, "bbox": item.bbox_page} for item in result.observations],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

Path(__file__).with_name("ocr_crop_sample_results.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
)
