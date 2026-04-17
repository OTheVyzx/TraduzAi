"""Translation Critic — flageia falhas no translator EN -> PT-BR.

Checa:
- `original == traduzido` (tradução não rolou — fallback Ollama/Google falhou)
- Artefatos UTF-8 mal decodificados ('VocÃª', 'Ã©', 'Ã ')
- Ratio len(traduzido)/len(original) fora de [0.5, 2.0] (texto estranhamente inflado/colapsado)
- Terminologia inconsistente (mesma palavra EN -> traduções diferentes entre páginas)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from lab.critics.base import Critic, Finding, load_project_json_from_artifact


UTF8_ARTIFACTS_RE = re.compile(r"(?:Ã[ \t]|Ã©|Ã£|Ãª|Ã¢|VocÃª|NÃ£o|NÃ­|Ã§)")
MIN_RATIO = 0.5
MAX_RATIO = 2.0


class TranslationCritic:
    critic_id = "translation_critic"

    def analyze(self, chapter_artifact: dict) -> list[Finding]:
        project_json = load_project_json_from_artifact(chapter_artifact)
        chapter_number = int(chapter_artifact.get("chapter_number", 0))
        findings: list[Finding] = []

        # Mapeia EN -> conjunto de traducoes PT-BR (p/ consistencia terminologica)
        term_map: dict[str, set[str]] = defaultdict(set)

        for page_index, page in enumerate(project_json.get("paginas", [])):
            texts = page.get("textos", []) or []
            findings.extend(self._check_untranslated(chapter_number, page_index, texts))
            findings.extend(self._check_encoding(chapter_number, page_index, texts))
            findings.extend(self._check_ratio(chapter_number, page_index, texts))

            for text in texts:
                original = self._normalize(text.get("original", ""))
                translated = self._normalize(text.get("traduzido", ""))
                if len(original) >= 3 and translated:
                    term_map[original].add(translated)

        findings.extend(self._check_terminology(chapter_number, term_map))
        return findings

    def _normalize(self, value: object) -> str:
        return " ".join(str(value or "").lower().strip().split())

    def _check_untranslated(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            original = str(text.get("original", "")).strip()
            translated = str(text.get("traduzido", "")).strip()
            if not original or not translated:
                continue
            # Ignora onomatopeias e textos muito curtos
            if len(original) < 3:
                continue
            if original.lower() == translated.lower():
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="untranslated",
                        severity="error",
                        bbox=list(text.get("bbox") or []),
                        evidence={
                            "text_sample": original[:120],
                        },
                        suggested_fix=(
                            "Tradução idêntica ao original — fallback Google falhou ou Ollama "
                            "devolveu texto em EN. Verifique `translate_with_google` e o prompt "
                            "do modelo `traduzai-translator`."
                        ),
                        suggested_file="pipeline/translator/translate.py",
                        suggested_anchor="translate_with_google",
                    )
                )
        return findings

    def _check_encoding(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            translated = str(text.get("traduzido", "")).strip()
            if not translated:
                continue
            if UTF8_ARTIFACTS_RE.search(translated):
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="encoding_artifact",
                        severity="error",
                        bbox=list(text.get("bbox") or []),
                        evidence={"text_sample": translated[:120]},
                        suggested_fix=(
                            "Texto com mojibake UTF-8 (ex: 'VocÃª'). "
                            "Verifique encoding de leitura/escrita do project.json "
                            "e o fluxo de resposta do Ollama (deve ser utf-8)."
                        ),
                        suggested_file="pipeline/translator/translate.py",
                        suggested_anchor="decode_translation_response",
                    )
                )
        return findings

    def _check_ratio(
        self, chapter_number: int, page_index: int, texts: Iterable[dict]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for text in texts:
            original = str(text.get("original", "")).strip()
            translated = str(text.get("traduzido", "")).strip()
            if len(original) < 8 or not translated:
                continue
            ratio = len(translated) / max(1, len(original))
            if ratio < MIN_RATIO or ratio > MAX_RATIO:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=page_index,
                        issue_type="length_ratio_outlier",
                        severity="warning",
                        bbox=list(text.get("bbox") or []),
                        evidence={
                            "ratio": round(ratio, 2),
                            "original_sample": original[:80],
                            "translated_sample": translated[:80],
                        },
                        suggested_fix=(
                            f"Tradução com ratio {ratio:.2f} (esperado [{MIN_RATIO}, {MAX_RATIO}]). "
                            "Pode indicar tradução truncada, parafraseada demais ou eco de prompt."
                        ),
                        suggested_file="pipeline/translator/translate.py",
                        suggested_anchor="traduzai-translator prompt",
                    )
                )
        return findings

    def _check_terminology(
        self, chapter_number: int, term_map: dict[str, set[str]]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for original, variants in term_map.items():
            # Só flageia termos que apareceram mais de 1x com 2+ traduções distintas
            if len(variants) >= 2 and len(original) >= 4:
                findings.append(
                    Finding(
                        critic_id=self.critic_id,
                        chapter_number=chapter_number,
                        page_index=-1,
                        issue_type="term_inconsistency",
                        severity="warning",
                        bbox=None,
                        evidence={
                            "term": original,
                            "variants": sorted(variants),
                        },
                        suggested_fix=(
                            f"Termo '{original}' tem {len(variants)} traduções distintas no capítulo. "
                            "Use um glossário de memória (AniList/contexto persistente) "
                            "para manter consistência."
                        ),
                        suggested_file="pipeline/translator/translate.py",
                        suggested_anchor="context_glossary",
                    )
                )
        return findings


__all__ = ["TranslationCritic"]
