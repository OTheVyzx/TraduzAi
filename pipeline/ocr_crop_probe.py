"""Bounded OCR geometry probe for CH57 page 023; no pipeline files are changed."""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_stack import runtime

ROOT = Path(r"N:\TraduzAI")
SOURCE = ROOT / "artifacts/e2e/optimized_e2e_20260907/full_chapter_observation_57_complete_02/source_input/023.webp"
BOXES = ROOT / "artifacts/diagnostics/ch57_page023_card_20261001/detector_blocks.json"
image = cv2.cvtColor(cv2.imread(str(SOURCE)), cv2.COLOR_BGR2RGB)
boxes = [row["xyxy"] for row in json.loads(BOXES.read_text(encoding="utf-8"))]
runtime._configure_model_roots(str(ROOT / "pipeline" / "models"))
engine = runtime._get_ocr_engine("max", lang="en")


def spans(target):
    protected = [(int(box[1]) - 8, int(box[3]) + 8) for box in boxes]
    bounds = [0]
    while 3000 - bounds[-1] > target:
        ideal = bounds[-1] + target
        candidates = range(max(bounds[-1] + target // 2, ideal - target // 3), min(3000, ideal + target // 3) + 1)
        safe = [point for point in candidates if all(not (low < point < high) for low, high in protected)]
        bounds.append(min(safe, key=lambda point: abs(point - ideal)) if safe else ideal)
    bounds.append(3000)
    return [(max(0, start - (48 if index else 0)), min(3000, end + (48 if index < len(bounds) - 2 else 0)))
            for index, (start, end) in enumerate(zip(bounds, bounds[1:]))]


def read(name, crops):
    started = time.perf_counter()
    all_lines = []
    for ordinal, bbox in enumerate(crops):
        request = runtime._runtime_ocr_request(image, page_id="page_023", provider_family="paddleocr", invocation_kind=f"probe-{name}-{ordinal}")
        result = engine.recognize_region_with_evidence(image, bbox_page=bbox, request=request)
        all_lines.extend([record.text for record in result.observations])
    return {"name": name, "seconds": round(time.perf_counter() - started, 3), "calls": len(crops), "boxes": crops, "lines": all_lines}


results = []
for name, crops in ([] if "--candidate-only" in sys.argv else [
    ("full_warmup", [(0, 0, 690, 3000)]),
    ("slices_768", [(0, top, 690, bottom) for top, bottom in spans(768)]),
    ("full_control", [(0, 0, 690, 3000)]),
    ("slices_1500", [(0, top, 690, bottom) for top, bottom in spans(1500)]),
    ("slices_1800", [(0, top, 690, bottom) for top, bottom in spans(1800)]),
    ("card_only", [(10, 1353, 358, 1675)]),
    ("dialogue_control", [(113, 2384, 413, 2529)]),
]):
    row = read(name, crops)
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

os.environ["TRADUZAI_OCR_CROP_FIRST"] = "1"
request = runtime._runtime_ocr_request(image, page_id="page_023", provider_family="paddleocr", invocation_kind="probe-crop-first")
start = time.perf_counter()
candidate = engine.recognize_page_with_evidence(image, [SimpleNamespace(xyxy=box) for box in boxes], request=request)
row = {"name": "crop_first_masked_remainder", "seconds": round(time.perf_counter() - start, 3),
       "calls": len(candidate.attempts), "boxes": boxes, "blocks": [block.text for block in candidate.blocks],
       "lines": [record.text for record in candidate.observations]}
results.append(row)
print(json.dumps(row, ensure_ascii=False), flush=True)

output = Path(__file__).with_name("ocr_crop_probe_results.json")
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
