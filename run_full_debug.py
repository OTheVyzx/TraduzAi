import sys
import json
import shutil
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

ROOT = Path("D:/TraduzAi")
PIPELINE_DIR = ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# Os módulos abaixo estão em pipeline/ e são resolvidos via sys.path.insert acima.
# O linter não consegue localizá-los estaticamente — os type: ignore são esperados.
from ocr.detector import run_ocr  # type: ignore[import-untyped]
from inpainter.lama import run_inpainting  # type: ignore[import-untyped]
from layout.balloon_layout import enrich_page_layout  # type: ignore[import-untyped]
from translator.translate import translate_pages  # type: ignore[import-untyped]
from typesetter.renderer import run_typesetting  # type: ignore[import-untyped]

import cv2
import numpy as np

def draw_ocr_boxes(img_path, ocr_result, out_path):
    img = cv2.imread(str(img_path))
    if img is None: return
    for block in ocr_result.get("texts", []):
        bbox = block.get("bbox", [0,0,0,0])
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)

def main():
    img_path = Path("D:/traduzai_data/projects/aa66aa60-255f-406e-806b-169abecd5d49/originals/003.webp")
    out_dir = ROOT / "testdebug_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("D:/traduzai_data/models")
    if not models_dir.exists():
        models_dir = ROOT / "models"
    
    print(f"[1] Running OCR on {img_path.name}...")
    ocr_result = run_ocr(
        str(img_path),
        models_dir=str(models_dir),
        profile="max"
    )
    
    # CRITICAL FIX: If ocr_result doesn't have _vision_blocks (legacy fallback), 
    # we create them from the text bboxes so the new inpainter knows what to clean.
    if not ocr_result.get("_vision_blocks"):
        print("[INFO] No vision blocks found (Legacy OCR fallback). Generating blocks from text bboxes...")
        blocks = []
        for t in ocr_result.get("texts", []):
            blocks.append({
                "bbox": t["bbox"],
                "confidence": t.get("confidence", 0.9)
            })
        ocr_result["_vision_blocks"] = blocks
        # Also ensure semantic info is present for layout
        ocr_result = enrich_page_layout(ocr_result)
    else:
        ocr_result = enrich_page_layout(ocr_result)
    
    with open(out_dir / "1_detect_output.json", "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, indent=2, ensure_ascii=False)
        
    draw_ocr_boxes(img_path, ocr_result, out_dir / "1_detect_boxes.jpg")
    
    print("[2] Running Inpainting...")
    inpaint_dir = out_dir / "2_inpaint"
    inpaint_dir.mkdir(exist_ok=True)
    
    inpainted_paths = run_inpainting(
        image_files=[img_path],
        ocr_results=[ocr_result],
        output_dir=str(inpaint_dir),
        models_dir=str(models_dir)
    )
    
    print("[3] Running Translate...")
    # Mock translation if needed or call real one
    trans_results = translate_pages(
        ocr_results=[ocr_result],
        obra="Test Manga",
        context={},
        glossario={},
        idioma_destino="pt-BR",
        idioma_origem="en"
    )
    
    with open(out_dir / "3_translate_output.json", "w", encoding="utf-8") as f:
        json.dump(trans_results, f, indent=2, ensure_ascii=False)
        
    print("[4] Running Typesetter...")
    merged_for_typeset = []
    trans_page = trans_results[0]
    merged_texts = []
    ocr_texts = ocr_result.get("texts", [])
    trans_texts = trans_page.get("texts", [])
    for idx, ocr_t in enumerate(ocr_texts):
        translated = trans_texts[idx].get("translated", ocr_t.get("text", "")) if idx < len(trans_texts) else ocr_t.get("text", "")
        merged_texts.append({**ocr_t, "translated": translated})
    merged_for_typeset.append({"texts": merged_texts})
    
    typeset_dir = out_dir / "4_typeset"
    typeset_dir.mkdir(exist_ok=True)
    
    run_typesetting(
        inpainted_paths=inpainted_paths,
        translated_results=merged_for_typeset,
        output_dir=str(typeset_dir)
    )
    
    # Save final result to easy to find location
    final_img = typeset_dir / img_path.name
    if final_img.exists():
        shutil.copy2(final_img, ROOT / "tst2.jpg")
        print(f"\n[+] SUCCESS! Final image saved as: D:\\TraduzAi\\tst2.jpg")
    
    print(f"\n[+] Script completed. Intermediate steps in: {out_dir}")

if __name__ == "__main__":
    main()
