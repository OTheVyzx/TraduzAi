"""
Traducao de texto EN -> PT-BR usando Google Translate via deep-translator.
"""
import re
import time
import logging
from typing import List, Optional

from deep_translator import GoogleTranslator

from . import config

logger = logging.getLogger(__name__)


class MangaTranslator:
    """Tradutor de texto de manga/manhwa com cache e preservacao de estilo."""

    def __init__(self):
        self._translator = GoogleTranslator(
            source=config.TRANSLATION_SOURCE,
            target=config.TRANSLATION_TARGET,
        )
        self._cache = {}

    def translate(self, text: str) -> str:
        """Traduz um texto de EN para PT-BR.

        Preserva uppercase, reticencias, e padroes de gaguejo.
        """
        if not text or not text.strip():
            return text

        # Cache
        cache_key = text.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        was_upper = text == text.upper()

        # Preprocessamento
        processed = self._preprocess(text)

        # Traduzir com retry
        translated = self._translate_with_retry(processed)

        if translated is None:
            logger.warning(f"Falha na traducao de: {text[:50]}...")
            return text

        # Pos-processamento
        result = self._postprocess(translated, was_upper)

        self._cache[cache_key] = result
        return result

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Traduz uma lista de textos em batch para reduzir chamadas de API.

        Agrupa textos usando separador e traduz de uma vez.
        """
        if not texts:
            return []

        results = [""] * len(texts)
        to_translate = []
        indices = []

        # Verificar cache primeiro
        for i, text in enumerate(texts):
            cached = self._cache.get(text.strip())
            if cached:
                results[i] = cached
            else:
                to_translate.append(text)
                indices.append(i)

        if not to_translate:
            return results

        # Agrupar em batches
        batches = self._create_batches(to_translate)

        for batch_texts in batches:
            was_uppers = [t == t.upper() for t in batch_texts]
            processed = [self._preprocess(t) for t in batch_texts]

            # Juntar com separador
            joined = config.TRANSLATION_BATCH_SEPARATOR.join(processed)
            translated = self._translate_with_retry(joined)

            if translated is None:
                # Fallback: traduzir individualmente
                for j, t in enumerate(batch_texts):
                    idx = indices.pop(0)
                    results[idx] = self.translate(t)
                continue

            # Separar resultados
            parts = translated.split("|||")

            # Limpar espacos extras ao redor do separador
            parts = [p.strip() for p in parts]

            # Se a separacao nao bateu, tentar traduzir individualmente
            if len(parts) != len(batch_texts):
                logger.debug(
                    f"Batch split mismatch: expected {len(batch_texts)}, got {len(parts)}. "
                    "Traduzindo individualmente."
                )
                for j, t in enumerate(batch_texts):
                    idx = indices.pop(0)
                    results[idx] = self.translate(t)
                continue

            for j, (part, was_upper, original) in enumerate(
                zip(parts, was_uppers, batch_texts)
            ):
                result = self._postprocess(part, was_upper)
                self._cache[original.strip()] = result
                idx = indices.pop(0)
                results[idx] = result

        return results

    def _preprocess(self, text: str) -> str:
        """Preprocessamento do texto antes da traducao."""
        t = text.strip()

        # Normalizar para sentenca (primeira maiuscula, resto minuscula)
        # OCR de manga retorna mistura erratica de maiusculas/minusculas
        # que confunde o tradutor. Normalizando fica mais natural.
        # Ex: "MY OFFENSE CAN'T KEEP uP WITH HOw FAST" -> "My offense can't keep up with how fast"
        if t and len(t) > 2:
            t = t[0].upper() + t[1:].lower()

        # Preservar reticencias
        t = t.replace("...", "\u2026")

        # Normalizar espacos
        t = re.sub(r"\s+", " ", t)

        return t

    def _postprocess(self, text: str, was_upper: bool) -> str:
        """Pos-processamento do texto traduzido."""
        t = text.strip()

        # Restaurar reticencias
        t = t.replace("\u2026", "...")

        # Adaptar para PT-BR natural de manga
        t = self._adapt_manga_ptbr(t)

        # Uppercase se o original era todo uppercase
        if was_upper:
            t = t.upper()

        # Corrigir espacos antes de pontuacao
        t = re.sub(r"\s+([!?.,;:])", r"\1", t)

        # Corrigir espacos duplos
        t = re.sub(r"\s{2,}", " ", t)

        return t

    def _adapt_manga_ptbr(self, text: str) -> str:
        """Adapta traducoes do Google para soar natural em PT-BR de manga.

        Google Translate produz traducoes literais que nao soam como dialogo
        de manga em portugues. Esta funcao aplica substituicoes comuns.
        """
        t = text

        # Substituicoes de termos robóticos por naturais
        adaptations = [
            # Combate
            (r"\bminha ofensa\b", "meus golpes", re.IGNORECASE),
            (r"\bmeu ataque\b", "meu golpe", re.IGNORECASE),
            (r"\bmeus ataques\b", "meus golpes", re.IGNORECASE),
            (r"\bseu ataque\b", "seu golpe", re.IGNORECASE),
            (r"\bseus ataques\b", "seus golpes", re.IGNORECASE),
            (r"\bos desvia\b", "os defende", re.IGNORECASE),
            (r"\bos desviar\b", "os defender", re.IGNORECASE),
            (r"\bos desviam\b", "os defendem", re.IGNORECASE),
            (r"\bdesfleta\b", "defende", re.IGNORECASE),
            (r"\bdesvios\b", "defesas", re.IGNORECASE),
            (r"\bos defender\b", "os defende", re.IGNORECASE),
            (r"\bgreve\b", "golpe", re.IGNORECASE),  # strike -> greve (errado) -> golpe
            (r"\bgreves\b", "golpes", re.IGNORECASE),
            (r"\bum ataque\b", "um golpe", re.IGNORECASE),
            (r"\bo ataque\b", "o golpe", re.IGNORECASE),
            # "keep up" traduzido mal
            (r"manter [Ll][Pp] com", "acompanhar", re.IGNORECASE),
            (r"manter lp", "acompanhar", re.IGNORECASE),
            # "deflects" traduzido como "desfleta"
            (r"\bos desfleta\b", "os defende", re.IGNORECASE),
            # Poder
            (r"\bnúcleos de mana\b", "núcleos de mana", re.IGNORECASE),
            (r"\bcores de mana\b", "núcleos de mana", re.IGNORECASE),
            # Expressoes
            (r"\bé inútil\b", "é inútil", re.IGNORECASE),
            (r"\bnão tem volta\b", "não há volta", re.IGNORECASE),
            # "see through" -> "ver através" (literal) -> melhor: "enxergar"
            (r"\bver através de todos\b", "enxergar todos", re.IGNORECASE),
            (r"\bver através\b", "enxergar", re.IGNORECASE),
            # Forca / poder
            (r"\bforça total\b", "força total", re.IGNORECASE),
            (r"\bpoder total\b", "poder total", re.IGNORECASE),
            (r"\bverdadeiramente monstruos[oa]\b", "realmente monstruosa", re.IGNORECASE),
            # Pronomes formais -> informais (manga usa informal)
            (r"\bvocê disse que\b", "você disse que", re.IGNORECASE),
            # Correcoes de Google Translate
            (r"\bLP\b", "", 0),  # remove "LP" solto (artifact)
        ]

        for pattern, replacement, flags in adaptations:
            t = re.sub(pattern, replacement, t, flags=flags)

        # Limpar espacos extras resultantes
        t = re.sub(r"\s{2,}", " ", t).strip()

        return t

    def _translate_with_retry(self, text: str) -> Optional[str]:
        """Traduz com retry e backoff exponencial."""
        delay = config.TRANSLATION_RETRY_DELAY

        for attempt in range(config.TRANSLATION_MAX_RETRIES):
            try:
                result = self._translator.translate(text)
                return result
            except Exception as e:
                logger.warning(
                    f"Erro na traducao (tentativa {attempt + 1}/{config.TRANSLATION_MAX_RETRIES}): {e}"
                )
                if attempt < config.TRANSLATION_MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2

        return None

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Agrupa textos em batches respeitando limite de caracteres."""
        batches = []
        current_batch = []
        current_length = 0

        for text in texts:
            text_len = len(text) + len(config.TRANSLATION_BATCH_SEPARATOR)
            if current_length + text_len > config.TRANSLATION_BATCH_MAX_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_length = 0

            current_batch.append(text)
            current_length += text_len

        if current_batch:
            batches.append(current_batch)

        return batches
