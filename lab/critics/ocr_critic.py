"""OCR Critic — flageia regressoes no PaddleOCR + filtro de watermark.

Checa:
- `confianca_ocr` < 0.6 (OCR instavel)
- Watermarks que escaparam do filtro (`scan|toon|lagoon|asura|discord.gg`)
- Bbox com IOU > 0.3 entre si na mesma pagina (duplicacao de deteccao)
- Texto com sequencias `111111` (artefato conhecido do PaddleOCR)
"""
from __future__ import annotations

import re
from typing import Iterable

from lab.critics.base import Critic, Finding, load_project_json_from_artifact


WATERMARK_RE = re.compile(r"(?i)\b(scan|scans|toon|toons|lagoon|asura|mangaflix|discord\.gg)\b")
REPEATED_DIGITS_RE = re.compile(r"1{5,}|0{5,}|-{5,}")
LOW_CONFIDENCE_THRESHOLD = 0.6
IOU_OVERLAP_THRESHOLD = 0.3


def _iou(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class OcrCritic:
    critic_id = "ocr_critic"

    def analyze(self, chapter_artifact: dict) -> list[Finding]:
        project_json = load_project_json_from_artifact(chapter_artifact)
        chapter_number = int(chapter_artifact.get("chapter_number", 0))
        findings: list[Finding] = []

        for page_index, page in enumerate(project_json.get("paginas", [])):
            texts = page.get("textos", []) or []
            findings.extend(self._check_confidence(chapter_number, page_index, texts))
            findings.extend(self._check_watermarks(chapter_number, page_index, texts))
            findings.extend(self._check_repeated_digits(chapter_number, page_index, texts))
            findings.extend(self._check_duplicated_boxes(chapter_number, page_index, texts))

        return findings

    def _check_confidence(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            confidence = float(text.get("confianca_ocr", 1.0) or 0.0)
            original = str(text.get("original", "")).strip()
            if not original:
                continue
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="low_confidence",
                        severity="warning" if confidence > 0.4 else "error",
                        bbox=list(text.get("bbox") or []),
                        evidence={
                            "confidence": round(confidence, 3),
                            "text_sample": original[:80],
                        },
                        suggested_fix=(
                            "Texto com confianca baixa pode indicar OCR em pagina de transicao, "
                            "letreiro estilizado ou fonte fora do corpus. Reforce o gate de "
                            "`_is_meaningful_benchmark_text` e avalie perfil PaddleOCR."
                        ),
                        suggested_file="pipeline/vision_stack/ocr.py",
                        suggested_anchor="PaddleOCR profile",
                    )
                )
        return findings

    def _check_watermarks(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            original = str(text.get("original", "")).strip()
            if not original:
                continue
            match = WATERMARK_RE.search(original)
            if match:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="watermark_leaked",
                        severity="error",
                        bbox=list(text.get("bbox") or []),
                        evidence={
                            "matched_token": match.group(0),
                            "text_sample": original[:120],
                        },
                        suggested_fix=(
                            f"Watermark '{match.group(0)}' passou pelo filtro. "
                            "Endureca o regex em `_is_meaningful_benchmark_text` ou "
                            "adicione o token ao registry de watermarks."
                        ),
                        suggested_file="lab/benchmarking.py",
                        suggested_anchor="_is_meaningful_benchmark_text",
                    )
                )
        return findings

    def _check_repeated_digits(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            original = str(text.get("original", "")).strip()
            if REPEATED_DIGITS_RE.search(original):
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="ocr_artifact_repeated_digits",
                        severity="warning",
                        bbox=list(text.get("bbox") or []),
                        evidence={"text_sample": original[:80]},
                        suggested_fix=(
                            "Sequencias '111111' ou '00000' indicam lixo PaddleOCR "
                            "em bordas de balao. Refine mask_builder ou rode pos-filtragem."
                        ),
                        suggested_file="pipeline/ocr/postprocess.py",
                        suggested_anchor="OCR_ARTIFACT_FILTERS",
                    )
                )
        return findings

    def _check_duplicated_boxes(
        self, chapter_number: int, page_index: int, texts: list[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        entries = [t for t in texts if t.get("bbox")]
        for i, a in enumerate(entries):
            for b in entries[i + 1 :]:
                score = _iou(list(a.get("bbox") or []), list(b.get("bbox") or []))
                if score >= IOU_OVERLAP_THRESHOLD:
                    findings.append(
                        Finding(
                            critic_id=self.critic_id,
                            chapter_number=chapter_number,
                            page_index=page_index,
                            issue_type="duplicated_ocr_box",
                            severity="warning",
                            bbox=list(a.get("bbox") or []),
                            evidence={
                                "iou": round(score, 3),
                                "other_bbox": list(b.get("bbox") or []),
                            },
                            suggested_fix=(
                                "Duas deteccoes com IOU alto podem indicar fusao incompleta "
                                "de linhas pelo PaddleOCR. Ajuste merge em `enrich_page_layout`."
                            ),
                            suggested_file="pipeline/layout/balloon_layout.py",
                            suggested_anchor="enrich_page_layout",
                        )
                    )
                    break  # evita explodir N^2 no mesmo page
        return findings


__all__ = ["OcrCritic"]
