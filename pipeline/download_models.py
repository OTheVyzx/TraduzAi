"""
Script de preparo dos modelos locais do TraduzAi.
Execute via terminal: python download_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS_DIR = Path(
    os.environ.get(
        "TRADUZAI_MODELS_DIR",
        os.environ.get(
            "MANGATL_MODELS_DIR",
            Path.home() / "AppData" / "Roaming" / "com.traduzai.app" / "models",
        ),
    )
)


def main():
    print("=" * 55)
    print("  TraduzAi - Preparo do Stack Visual")
    print("=" * 55)
    print(f"\nPasta de destino: {MODELS_DIR}\n")

    prepare_detector()
    prepare_ocr()
    prepare_inpainting()
    write_compat_markers()

    print("\n" + "=" * 55)
    print("  Stack visual pronto! Pode fechar este terminal.")
    print("=" * 55)


def prepare_detector():
    detector_dir = MODELS_DIR / "detector"
    detector_dir.mkdir(parents=True, exist_ok=True)
    ready_file = detector_dir / ".ready"

    if ready_file.exists():
        print("[Detector] comic-text-detector ja esta pronto. Pulando.")
        return

    print("[Detector] Baixando e validando comic-text-detector...")
    try:
        from vision_stack import detector as detector_module

        detector_module.MODELS_DIR = MODELS_DIR
        detector = detector_module.TextDetector(model="comic-text-detector", device="cpu", half=False)
        detector.unload()
        ready_file.write_text("ok", encoding="utf-8")
        print("[Detector] OK!")
    except Exception as exc:
        print(f"[Detector] ERRO: {exc}")
        sys.exit(1)


def prepare_ocr():
    ocr_dir = MODELS_DIR / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    ready_file = ocr_dir / ".ready"

    if ready_file.exists():
        print("[OCR] Stack visual ja esta pronto. Pulando.")
        return

    print("[OCR] Aquecendo OCR principal/fallback...")
    try:
        from vision_stack.ocr import OCREngine

        engine = OCREngine(model="manga-ocr", device="cpu", half=False)
        engine.unload()
        ready_file.write_text("ok", encoding="utf-8")
        print("[OCR] OK! Backend final:", getattr(engine, "_backend", engine.model_name))
    except Exception as exc:
        print(f"[OCR] ERRO: {exc}")
        sys.exit(1)


def prepare_inpainting():
    inpaint_dir = MODELS_DIR / "inpaint"
    inpaint_dir.mkdir(parents=True, exist_ok=True)
    ready_file = inpaint_dir / ".ready"

    if ready_file.exists():
        print("[Inpainting] Ja configurado. Pulando.")
        return

    print("[Inpainting] Baixando e validando LaMA...")
    try:
        from vision_stack import inpainter as inpainter_module

        inpainter_module.MODELS_DIR = MODELS_DIR
        inpainter = inpainter_module.Inpainter(model="lama-manga", device="cpu", half=False)
        if hasattr(inpainter, "unload"):
            inpainter.unload()
        ready_file.write_text("ok", encoding="utf-8")
        print("[Inpainting] OK!")
    except Exception as exc:
        print(f"[Inpainting] ERRO: {exc}")
        sys.exit(1)


def write_compat_markers():
    """
    Mantem os markers antigos para a UI atual do Tauri, enquanto o backend
    visual novo assume detector -> OCR -> inpaint como fluxo principal.
    """
    for relative in (
        ("easyocr", ".ready"),
        ("paddleocr", ".ready"),
        ("lama_manga_onnx", ".ready"),
    ):
        target = MODELS_DIR / relative[0]
        target.mkdir(parents=True, exist_ok=True)
        (target / relative[1]).write_text("compat", encoding="utf-8")


if __name__ == "__main__":
    main()
