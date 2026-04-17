"""Typeset Critic — flageia problemas de rendering/layout FT2Font.

Checa:
- Occupancy fora de [0.18, 0.72] (texto muito apertado ou muito esparso no balão)
- Fonte < 14px em balão com altura > 50px (ilegível)
- Texto estourando bbox (estimativa: chars*size*0.55 > bbox width)
- Lobos conectados com desequilíbrio de palavras severo (>2.5x entre lobos)
"""
from __future__ import annotations

from typing import Iterable

from lab.critics.base import Critic, Finding, load_project_json_from_artifact


MIN_OCCUPANCY = 0.18
MAX_OCCUPANCY = 0.72
MIN_FONT_SIZE = 14.0
MIN_BALLOON_HEIGHT_FOR_FONT_CHECK = 50.0
WORD_IMBALANCE_RATIO = 2.5
CHAR_WIDTH_FACTOR = 0.55  # mesma estimativa do renderer


class TypesetCritic:
    critic_id = "typeset_critic"

    def analyze(self, chapter_artifact: dict) -> list[Finding]:
        project_json = load_project_json_from_artifact(chapter_artifact)
        chapter_number = int(chapter_artifact.get("chapter_number", 0))
        findings: list[Finding] = []

        for page_index, page in enumerate(project_json.get("paginas", [])):
            texts = page.get("textos", []) or []
            findings.extend(self._check_occupancy(chapter_number, page_index, texts))
            findings.extend(self._check_font_size(chapter_number, page_index, texts))
            findings.extend(self._check_bbox_overflow(chapter_number, page_index, texts))
            findings.extend(self._check_lobe_imbalance(chapter_number, page_index, texts))

        return findings

    def _bbox_area(self, bbox: list[float]) -> float:
        if not bbox or len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    def _check_occupancy(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            translated = str(text.get("traduzido", "")).strip()
            if not translated:
                continue
            bbox = list(text.get("bbox") or [])
            area = self._bbox_area(bbox)
            if area <= 0:
                continue
            style = text.get("estilo", {}) or {}
            font_size = float(style.get("tamanho", 16) or 16)
            # Estimativa grossa: len*size*0.55 largura + size*1.2 altura por linha
            approx_text_area = max(1.0, len(translated)) * font_size * font_size * 0.55
            occupancy = min(1.0, approx_text_area / max(1.0, area))
            if occupancy < MIN_OCCUPANCY:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="occupancy_too_low",
                        severity="info",
                        bbox=[int(v) for v in bbox],
                        evidence={
                            "occupancy": round(occupancy, 3),
                            "font_size": font_size,
                            "text_sample": translated[:60],
                        },
                        suggested_fix=(
                            f"Texto ocupa só {occupancy:.0%} do balão. "
                            "Typesetter pode aumentar tamanho da fonte ou refatorar "
                            "`_resolve_connected_target_sizes` para floor maior."
                        ),
                        suggested_file="pipeline/typesetter/renderer.py",
                        suggested_anchor="_resolve_connected_target_sizes",
                    )
                )
            elif occupancy > MAX_OCCUPANCY:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="occupancy_too_high",
                        severity="warning",
                        bbox=[int(v) for v in bbox],
                        evidence={
                            "occupancy": round(occupancy, 3),
                            "font_size": font_size,
                            "text_sample": translated[:60],
                        },
                        suggested_fix=(
                            f"Texto ocupa {occupancy:.0%} do balão (acima de 72%). "
                            "Layout provavelmente estoura as bordas. Aumente margin ou "
                            "reduza font no binary search do renderer."
                        ),
                        suggested_file="pipeline/typesetter/renderer.py",
                        suggested_anchor="plan_text_layout",
                    )
                )
        return findings

    def _check_font_size(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            translated = str(text.get("traduzido", "")).strip()
            if not translated:
                continue
            bbox = list(text.get("bbox") or [])
            if len(bbox) != 4:
                continue
            balloon_height = bbox[3] - bbox[1]
            style = text.get("estilo", {}) or {}
            font_size = float(style.get("tamanho", 16) or 16)
            if balloon_height >= MIN_BALLOON_HEIGHT_FOR_FONT_CHECK and font_size < MIN_FONT_SIZE:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="font_too_small_for_balloon",
                        severity="warning",
                        bbox=[int(v) for v in bbox],
                        evidence={
                            "font_size": font_size,
                            "balloon_height": float(balloon_height),
                        },
                        suggested_fix=(
                            f"Fonte {font_size:.0f}px num balão de {balloon_height:.0f}px. "
                            "Ilegível. Renderer possivelmente escolheu tamanho menor "
                            "por coherence_penalty ou quality gate. Revise "
                            "`_font_search_floor` e `MIN_AVG_WORDS_PER_LINE`."
                        ),
                        suggested_file="pipeline/typesetter/renderer.py",
                        suggested_anchor="_resolve_text_layout",
                    )
                )
        return findings

    def _check_bbox_overflow(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        """Detecta overflow real após simular o word-wrap do renderer.

        O renderer quebra o texto em linhas de até `max_width ≈ bbox_width` px.
        Comparar `len(texto_completo) * size * 0.55` diretamente contra bbox_width
        produz falsos positivos toda vez que o texto teria sido wrapped em 2+ linhas.
        A solução é simular o word-wrap antes de estimar a largura da linha mais longa.
        Um caso genuíno de overflow só ocorre quando uma palavra individual não cabe,
        ou quando o texto não contém espaços e ocupa mais que bbox_width.
        """
        findings: list[Finding] = []
        for text in texts:
            translated = str(text.get("traduzido", "")).strip()
            if not translated:
                continue
            bbox = list(text.get("bbox") or [])
            if len(bbox) != 4:
                continue
            bbox_width = bbox[2] - bbox[0]
            style = text.get("estilo", {}) or {}
            font_size = float(style.get("tamanho", 16) or 16)

            # Simula word-wrap: calcula chars por linha com base em bbox_width.
            # O renderer usa max_width ≈ bbox_width * width_ratio (0.82–0.95).
            # Usamos bbox_width diretamente como proxy conservador.
            char_width = max(0.5, font_size * CHAR_WIDTH_FACTOR)
            chars_per_line = max(1, int(bbox_width / char_width))

            words = translated.split()
            if not words:
                continue

            cur_len = 0
            max_line_chars = 0
            for word in words:
                wlen = len(word)
                if cur_len == 0:
                    cur_len = wlen
                elif cur_len + 1 + wlen <= chars_per_line:
                    cur_len += 1 + wlen
                else:
                    max_line_chars = max(max_line_chars, cur_len)
                    cur_len = wlen
            max_line_chars = max(max_line_chars, cur_len)

            est_width = max_line_chars * char_width
            if est_width > bbox_width * 1.15:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="text_overflow_bbox",
                        severity="error",
                        bbox=[int(v) for v in bbox],
                        evidence={
                            "estimated_width_px": round(est_width, 1),
                            "bbox_width_px": float(bbox_width),
                            "longest_line_chars": max_line_chars,
                            "font_size": font_size,
                        },
                        suggested_fix=(
                            f"Texto estimado em {est_width:.0f}px num bbox de {bbox_width:.0f}px "
                            "(após simulação de wrap). Provável overflow de palavra longa. "
                            "Cheque `_best_semantic_split` e line-break fallback."
                        ),
                        suggested_file="pipeline/typesetter/renderer.py",
                        suggested_anchor="_best_semantic_split",
                    )
                )
        return findings

    def _check_lobe_imbalance(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        """Balões conectados: se há grupos com word-count muito desigual, flagear."""
        findings: list[Finding] = []
        by_group: dict[str, list[int]] = {}
        for text in texts:
            group_id = str(text.get("grupo_balao", "") or "").strip()
            if not group_id:
                continue
            translated = str(text.get("traduzido", "")).strip()
            if not translated:
                continue
            by_group.setdefault(group_id, []).append(len(translated.split()))

        for group_id, word_counts in by_group.items():
            if len(word_counts) < 2:
                continue
            lo, hi = min(word_counts), max(word_counts)
            if lo == 0:
                continue
            ratio = hi / lo
            if ratio >= WORD_IMBALANCE_RATIO:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="connected_lobe_imbalance",
                        severity="warning",
                        bbox=None,
                        evidence={
                            "group_id": group_id,
                            "word_counts": word_counts,
                            "ratio": round(ratio, 2),
                        },
                        suggested_fix=(
                            f"Lobos conectados do grupo '{group_id}' com ratio {ratio:.1f}x de palavras. "
                            "Split semântico pode ter ficado torto. Revise "
                            "`_best_semantic_split` e `coherence_penalty`."
                        ),
                        suggested_file="pipeline/typesetter/renderer.py",
                        suggested_anchor="_best_semantic_split",
                    )
                )
        return findings


__all__ = ["TypesetCritic"]
