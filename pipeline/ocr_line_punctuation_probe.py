"""Same-model recognition-only probe for a line whose detector missed endpoint ink."""

import json
from pathlib import Path

import cv2

from vision_stack import runtime

ROOT = Path(r"N:\TraduzAI")
runtime._configure_model_roots(str(ROOT / "models"))
engine = runtime._get_ocr_engine("default", lang="en")
image = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "one_page_proof/deskew_input_card.png")), cv2.COLOR_BGR2RGB)
rows = []
for name, box in (("line_margin_6", (94, 106, 362, 139)), ("line_margin_12", (88, 100, 368, 145))):
    x1, y1, x2, y2 = box
    result = engine._model.ocr(image[y1:y2, x1:x2], det=False, rec=True, cls=False)
    rows.append({"name": name, "box": box, "raw": result})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
Path(__file__).with_name("ocr_line_punctuation_probe_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
