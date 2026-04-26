"""Helpers de diagnóstico para o pipeline strip-based.

Não usados em produção; chamados via `STRIP_DEBUG=1` env var ou direto em testes.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from strip.types import Band, VerticalStrip


def dump_strip_debug(
    strip: VerticalStrip,
    bands: list[Band],
    out_dir: Path,
) -> None:
    """Despeja em `out_dir`:
      - strip_overlay.png (strip downscaled + bboxes coloridas)
      - bands_overview.txt (uma linha por banda)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Converter RGB → BGR para cv2
    overlay = cv2.cvtColor(strip.image.copy(), cv2.COLOR_RGB2BGR)

    for band in bands:
        # Faixa magenta por banda
        cv2.rectangle(
            overlay,
            (0, band.y_top),
            (strip.width - 1, band.y_bottom - 1),
            color=(255, 0, 255),
            thickness=3,
        )
        for balloon in band.balloons:
            # Balão verde
            cv2.rectangle(
                overlay,
                (balloon.strip_bbox.x1, balloon.strip_bbox.y1),
                (balloon.strip_bbox.x2, balloon.strip_bbox.y2),
                color=(0, 255, 0),
                thickness=2,
            )

    # Limitar a 8192 px de altura para não gerar imagens absurdas
    if overlay.shape[0] > 8192:
        scale = 8192 / overlay.shape[0]
        new_w = max(1, int(overlay.shape[1] * scale))
        overlay = cv2.resize(overlay, (new_w, 8192))

    cv2.imwrite(str(out_dir / "strip_overlay.png"), overlay)

    with open(out_dir / "bands_overview.txt", "w", encoding="utf-8") as f:
        f.write(f"strip {strip.width}x{strip.height}, {len(bands)} bands\n")
        for i, band in enumerate(bands):
            f.write(
                f"band[{i:03d}] y={band.y_top}..{band.y_bottom} "
                f"({band.y_bottom - band.y_top}px) balloons={len(band.balloons)}\n"
            )


def is_debug_enabled() -> bool:
    """Retorna True se STRIP_DEBUG=1/true/yes está no ambiente."""
    return os.environ.get("STRIP_DEBUG", "").lower() in {"1", "true", "yes"}
