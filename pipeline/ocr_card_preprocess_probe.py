"""Read-only, bounded PaddleOCR geometry experiment on the saved CH57 card."""

import json
import sys
import time
from pathlib import Path

import cv2

from vision_stack import runtime

ROOT = Path(r"N:\TraduzAI")
SOURCE = ROOT / "artifacts/e2e/optimized_e2e_20260907/full_chapter_observation_57_complete_02/source_input/023.webp"
page = cv2.cvtColor(cv2.imread(str(SOURCE)), cv2.COLOR_BGR2RGB)
runtime._configure_model_roots(str(ROOT / "models"))
engine = runtime._get_ocr_engine("default", lang="en")


def crop(box):
    x1, y1, x2, y2 = box
    return page[y1:y2, x1:x2]


base = crop((10, 1353, 358, 1675))
margin = crop((2, 1345, 366, 1683))
scaled = cv2.resize(base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
height, width = base.shape[:2]
matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -26.5, 1.0)
cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
new_width = int(height * sine + width * cosine)
new_height = int(height * cosine + width * sine)
matrix[0, 2] += new_width / 2 - width / 2
matrix[1, 2] += new_height / 2 - height / 2
deskewed = cv2.warpAffine(base, matrix, (new_width, new_height), borderValue=(255, 255, 255))

variants = (("base", base), ("margin_8", margin), ("scale_2x", scaled), ("deskew_minus_26_5", deskewed))
if "--interpolation" in sys.argv:
    variants = tuple(
        (name, cv2.warpAffine(base, matrix, (new_width, new_height), flags=method, borderValue=(255, 255, 255)))
        for name, method in (
            ("deskew_nearest", cv2.INTER_NEAREST),
            ("deskew_linear", cv2.INTER_LINEAR),
            ("deskew_cubic", cv2.INTER_CUBIC),
            ("deskew_lanczos", cv2.INTER_LANCZOS4),
        )
    )
elif "--angles" in sys.argv:
    variants = []
    for angle in (-24.5, -25.0, -25.5, -26.0, -26.5, -27.0, -27.5, -28.0):
        angle_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        cosine, sine = abs(angle_matrix[0, 0]), abs(angle_matrix[0, 1])
        out_width = int(height * sine + width * cosine)
        out_height = int(height * cosine + width * sine)
        angle_matrix[0, 2] += out_width / 2 - width / 2
        angle_matrix[1, 2] += out_height / 2 - height / 2
        variants.append((f"angle_{angle}", cv2.warpAffine(base, angle_matrix, (out_width, out_height), borderValue=(255, 255, 255))))
elif "--contrast" in sys.argv:
    gray = cv2.cvtColor(deskewed, cv2.COLOR_RGB2GRAY)
    variants = [("deskew_gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))]
    for threshold in (140, 180, 210):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        variants.append((f"threshold_{threshold}", cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)))

rows = []
for name, image in variants:
    start = time.perf_counter()
    result = engine._model.ocr(image, det=True, rec=True, cls=False)
    lines = []
    for group in result or []:
        for entry in group or []:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                lines.append({"polygon": entry[0], "text": entry[1][0], "confidence": float(entry[1][1])})
    rows.append({"variant": name, "shape": list(image.shape), "seconds": round(time.perf_counter() - start, 3), "lines": lines})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

output = (
    "ocr_card_interpolation_probe_results.json" if "--interpolation" in sys.argv
    else "ocr_card_angle_probe_results.json" if "--angles" in sys.argv
    else "ocr_card_contrast_probe_results.json" if "--contrast" in sys.argv
    else "ocr_card_preprocess_probe_results.json"
)
Path(__file__).with_name(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
