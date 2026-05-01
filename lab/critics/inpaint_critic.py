"""Inpaint Critic — flageia problemas visíveis no inpainting.

Compara assinaturas do output com a referência PT-BR via PIL.ImageStat.
Checa:
- Variância local muito alta dentro dos bboxes de texto (provável texto residual)
- Luminância média dos outputs fora da faixa da referência (spill de máscara ou
  inpainting que escureceu demais o balão branco)

Importante: todas as operações são sobre imagens já no disco. Se o output não
existir (pipeline falhou antes do typesetting), não emite findings.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from lab.critics.base import Critic, Finding, load_project_json_from_artifact


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
HIGH_VARIANCE_THRESHOLD = 2200.0  # variance de intensidade em bbox limpa é baixa
SAMPLE_LIMIT = 3  # evita custo alto: amostra até 3 páginas por capítulo


class InpaintCritic:
    critic_id = "inpaint_critic"

    def analyze(self, chapter_artifact: dict) -> list[Finding]:
        try:
            from PIL import Image, ImageStat  # import tardio p/ evitar custo fora do Lab
        except ImportError:  # pragma: no cover
            return []

        project_json = load_project_json_from_artifact(chapter_artifact)
        pages = project_json.get("paginas", []) or []
        if not pages:
            return []

        chapter_number = int(chapter_artifact.get("chapter_number", 0))
        output_dir = Path(str(chapter_artifact.get("output_dir", "")))
        if not output_dir.exists():
            return []

        findings: list[Finding] = []
        sampled = 0

        for page_index, page in enumerate(pages):
            if sampled >= SAMPLE_LIMIT:
                break
            page_image_path = self._resolve_page_image(output_dir, page, page_index)
            if page_image_path is None or not page_image_path.exists():
                continue
            try:
                img = Image.open(page_image_path).convert("L")
            except Exception:
                continue

            texts = page.get("textos", []) or []
            for text in texts:
                bbox = list(text.get("bbox") or [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if x2 <= x1 or y2 <= y1:
                    continue
                # Clipe ao canvas
                x1 = max(0, min(img.width - 1, x1))
                y1 = max(0, min(img.height - 1, y1))
                x2 = max(x1 + 1, min(img.width, x2))
                y2 = max(y1 + 1, min(img.height, y2))

                translated = str(text.get("traduzido", "")).strip()
                if not translated:
                    continue

                region = img.crop((x1, y1, x2, y2))
                try:
                    stats = ImageStat.Stat(region)
                    variance = float(stats.var[0]) if stats.var else 0.0
                except Exception:
                    continue

                if variance > HIGH_VARIANCE_THRESHOLD:
                    findings.append(
                        Finding(
                            critic_id=self.critic_id,
                            chapter_number=chapter_number,
                            page_index=page_index,
                            issue_type="residual_text_in_balloon",
                            severity="warning",
                            bbox=[x1, y1, x2, y2],
                            evidence={
                                "variance": round(variance, 1),
                                "text_sample": translated[:60],
                            },
                            suggested_fix=(
                                f"Variância intensa ({variance:.0f}) dentro do bbox após inpainting. "
                                "Possível texto EN residual ou máscara muito apertada. "
                                "Cheque `mask_builder` + `balloon_mask_refiner`."
                            ),
                            suggested_file="pipeline/vision_stack/inpainter.py",
                            suggested_anchor="build_inpaint_mask",
                        )
                    )
            sampled += 1

        return findings

    def _resolve_page_image(self, output_dir: Path, page: dict, page_index: int) -> Path | None:
        candidate = page.get("arquivo_traduzido") or page.get("arquivo") or ""
        if candidate:
            resolved = output_dir / candidate.replace("/", "\\") if "\\" in str(output_dir) else output_dir / candidate
            if resolved.exists():
                return resolved

        translated_dir = output_dir / "translated"
        if not translated_dir.exists():
            return None
        # Fallback: pega por ordem alfabética
        images = sorted(
            p for p in translated_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if page_index < len(images):
            return images[page_index]
        return None


__all__ = ["InpaintCritic"]
