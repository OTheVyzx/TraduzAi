"""Script de QA visual: gera strip + overlay de bboxes para inspeção.

Uso:
    python -m strip._debug_overlay /caminho/para/extracao /caminho/saida.png

NÃO faz parte do pipeline real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from strip.concat import build_strip
from strip.detect_balloons import detect_strip_balloons


def main(extraction_dir: str, output_path: str, models_dir: str | None = None) -> None:
    page_paths = sorted(Path(extraction_dir).glob("*.jpg")) + sorted(Path(extraction_dir).glob("*.png"))
    if not page_paths:
        raise FileNotFoundError(f"Sem páginas em {extraction_dir}")

    print(f"[debug] Construindo strip de {len(page_paths)} páginas...")
    strip = build_strip(page_paths)
    print(f"[debug] Strip: {strip.width}x{strip.height}")

    from vision_stack.detector import TextDetector
    detector = TextDetector(models_dir=Path(models_dir) if models_dir else None)

    print(f"[debug] Detectando balões...")
    balloons = detect_strip_balloons(strip, detector)
    print(f"[debug] {len(balloons)} balões detectados")

    overlay = strip.image.copy()
    for b in balloons:
        cv2.rectangle(
            overlay,
            (b.strip_bbox.x1, b.strip_bbox.y1),
            (b.strip_bbox.x2, b.strip_bbox.y2),
            color=(0, 255, 0),
            thickness=4,
        )

    if overlay.shape[0] > 8192:
        scale = 8192 / overlay.shape[0]
        new_w = int(overlay.shape[1] * scale)
        overlay = cv2.resize(overlay, (new_w, 8192))

    cv2.imwrite(output_path, overlay)
    print(f"[debug] Overlay salvo em {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python -m strip._debug_overlay <extraction_dir> <output.png> [models_dir]")
        sys.exit(1)
    models = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], models)
