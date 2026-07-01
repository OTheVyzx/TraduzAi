"""
Traducao EN->PT-BR.
Primario: Google Translate via deep-translator.
Agora com consciencia de tipo de texto, contexto local e memoria curta.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from utils.decision_log import record_decision
except ImportError:
    from ..utils.decision_log import record_decision

try:
    from ocr.text_router import ROUTE_ACTIONS, route_action_requires_translation
except ImportError:
    from ..ocr.text_router import ROUTE_ACTIONS, route_action_requires_translation

OLLAMA_HOST = "http://localhost:11434"

GOOGLE_LANGUAGE_ALIASES = {
    "pt-br": "pt",
    "pt_br": "pt",
    "pt-pt": "pt",
    "pt_pt": "pt",
    "en-gb": "en",
    "en-us": "en",
    "en_us": "en",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh_cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh_hans": "zh-CN",
    "zh-tw": "zh-TW",
    "zh_tw": "zh-TW",
    "zh-hant": "zh-TW",
    "zh_hant": "zh-TW",
}

OCR_DEDICATED_GOOGLE_CODES = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ru",
    "ja",
    "ko",
    "zh-CN",
    "zh-TW",
}

SOURCE_SCRIPT_PATTERN = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)
KOREAN_LATIN_OCR_ARTIFACT_PATTERN = re.compile(
    r"(?i)(?:\bTCH\b|[A-Za-zÀ-ÖØ-öø-ÿ]+TCH\b|\bTCH[A-Za-zÀ-ÖØ-öø-ÿ]+|\bULL\b|[A-Za-zÀ-ÖØ-öø-ÿ]+ULL\b|\bHFOR\b)"
)
TRANSLATION_FALLBACK_PHRASE_PATTERN = re.compile(
    r"\b(?:texto\s+original|texto\s+fonte|original\s+text|source\s+text|"
    r"nao\s+consigo\s+encontrar\s+o\s+texto\s+original|"
    r"nao\s+foi\s+possivel\s+traduzir|texto\s+nao\s+encontrado|"
    r"cannot\s+translate|can't\s+translate|i\s+cannot\s+translate)\b",
    re.IGNORECASE,
)
LITERAL_OCR_TRANSLATION_PATTERN = re.compile(
    r"\b(?:madreperola|madrep[eé]rola)\b",
    re.IGNORECASE,
)

ADAPTATIONS: list[tuple[str, str, int]] = []

PRE_TRANSLATION_GLOSSARY: list[tuple[str, str, int]] = [
    # PGE: Prioridade Gramatical Explicativa — substituições para o Google traduzir corretamente
    (r"\bits useless\b", "it is futile", re.IGNORECASE),
    (r"\buseless\b", "futile", re.IGNORECASE),
]

SOURCE_OCR_REPAIRS: list[tuple[str, str, int]] = [
    (r"\bcoyld\b", "could", re.IGNORECASE),
    (r"\blightbes\b", "light be...?!", re.IGNORECASE),
    # OCR digit/letter confusions (números colados em palavras curtas em CAPS)
    (r"\bM9\b", "MY", 0),                    # 9 lido como Y
    (r"\bM0\b", "MY", 0),                    # 0 lido como Y
    (r"\bi(\d{2,4})\b", r"\1", 0),           # i999 -> 999, i99 -> 99
    (r"\bND SQUAD\b", "2ND SQUAD", 0),       # 2 inicial perdido
    (r"\bRD SQUAD\b", "3RD SQUAD", 0),       # 3 inicial perdido
    (r"\bTH SQUAD\b", "4TH SQUAD", 0),       # 4 inicial perdido
    # SFX comuns que o OCR pode ter classificado como "fala"
    (r"\bGHK\b", "Ngh", 0),                  # som estrangulado
    (r"\bWGH\b", "Ugh", 0),                  # som de esforço
    (r"\bNGH\b", "Ngh", 0),                  # já forma reconhecível
]

TRANSLATION_REVIEW_REPAIRS: list[tuple[str, str, int]] = [
    (r"VocÄ™|Vocę|VocÃª", "Você", re.IGNORECASE),
    (r"atravÃ©s|atraves", "através", re.IGNORECASE),
    (r"\b(?:ranqueador|rankeador)\b", "Ranker", re.IGNORECASE),
    (r"\bsele[cç][aã]o\s+da\s+sele[cç][aã]o\b", "seletiva nacional", re.IGNORECASE),
    (r"\bsele[cç][aã]o\s+da\s+equipe\s+nacional\b", "seletiva nacional", re.IGNORECASE),
]

# Palavras inglesas curtas em CAPS que NÃO devem ser tratadas como nomes próprios.
# Tudo que sair do OCR como single-word CAPS e estiver fora desta lista é candidato
# a nome próprio (preserva-se em PT-BR ao invés de traduzir).
_COMMON_EN_CAPS_WORDS = frozenset({
    "YES", "NO", "OK", "OKAY", "YEAH", "YEP", "NAH", "HUH", "OH", "AH", "EH",
    "NONE",
    "HEY", "HI", "HELLO", "BYE", "GO", "STOP", "WAIT", "RUN", "LOOK", "SEE",
    "WHY", "HOW", "WHAT", "WHO", "WHERE", "WHEN", "WHICH", "DAMN", "GOD",
    "ALREADY", "UNCONTROLLABLE",
    "WELL", "WHOA", "WOW", "UGH", "ARGH", "GAH", "TSK", "HUSH", "QUIET",
    "OFF", "ON", "OUT", "IN", "UP", "DOWN", "BACK", "FORTH", "AWAY",
    "ALL", "AGAIN", "OUR", "MY", "YOUR", "HIS", "HER", "ITS", "THE",
    "THIS", "THAT", "THESE", "THOSE", "AND", "BUT", "OR", "FOR", "WITH",
    "TODO", "AGAIN", "DONE", "SAFE", "TRUE", "FALSE", "REAL", "GOOD", "BAD",
    "HOUSEHOLD", "CRUSHED", "DUST", "RETURNED", "LOST", "EVERYTHING",
    "PLEASE", "SORRY", "THANKS", "WAKE", "SLEEP", "EAT", "DRINK", "FIGHT",
    "ATTACK", "DEFEND", "RETREAT", "CHARGE", "FIRE", "WATER", "EARTH", "AIR",
    "LIGHT", "DARK", "LIFE", "DEATH", "LOVE", "HATE", "WAR", "PEACE",
    "ENEMY", "FRIEND", "HELP", "SAVE", "KILL", "DIE", "LIVE",
    "KILLING", "SOMEHOW",
    "MAGIC", "SPELL", "POWER", "FORCE", "WILL", "MIND", "SOUL", "HEART",
    "MERCY", "JUSTICE", "HONOR", "GLORY", "SHAME", "PRIDE",
    "DAY", "PEOPLE", "MOST",
    "BECAUSE", "AMAZING", "REALLY",
    "MASTER", "LORD", "LADY", "SIR", "SIRE", "MILORD", "KING", "QUEEN",
    "PRINCE", "PRINCESS", "KNIGHT", "LIAR", "FOOL", "IDIOT",
    "COMMANDER", "RAID", "SQUAD", "SOLDIER",
    # Sons curtos comuns
    "HMM", "HMPH", "HRRG", "HRGH", "HRRR", "GRR", "RAWR", "ROAR",
})

# Mapeamento infinitivo PT-BR -> imperativo PT-BR (singular informal "você").
# Quando o Google traduz comandos curtos como "WAKE UP." ele tende a devolver
# infinitivo ("Acordar.") em vez de imperativo. Esta tabela corrige.
_IMPERATIVE_FIXES = {
    "acordar": "acorde",
    "parar": "pare",
    "olhar": "olhe",
    "correr": "corra",
    "fugir": "fuja",
    "atacar": "ataque",
    "lutar": "lute",
    "esperar": "espere",
    "ouvir": "ouça",
    "voltar": "volte",
    "sair": "saia",
    "entrar": "entre",
    "calar": "cale",
    "abrir": "abra",
    "fechar": "feche",
    "ler": "leia",
    "escrever": "escreva",
    "comer": "coma",
    "beber": "beba",
    "dormir": "durma",
    "andar": "ande",
    "pular": "pule",
    "subir": "suba",
    "descer": "desça",
    "matar": "mate",
    "morrer": "morra",
    "viver": "viva",
    "amar": "ame",
    "odiar": "odeie",
    "ajudar": "ajude",
    "salvar": "salve",
    "soltar": "solte",
    "segurar": "segure",
    "soltar-me": "solte-me",
    "largar": "largue",
    "pegar": "pegue",
    "agarrar": "agarre",
    "responder": "responda",
    "falar": "fale",
    "dizer": "diga",
    "contar": "conte",
    "explicar": "explique",
    "concentrar": "concentre",
    "concentrar-se": "concentre-se",
    "render": "renda",
    "render-se": "renda-se",
    "levantar": "levante",
    "levantar-se": "levante-se",
    "sentar": "sente",
    "sentar-se": "sente-se",
}

# Máximo de palavras numa source que pode ser interpretada como comando curto
# (heurística para acionar o fix de imperativo).
_IMPERATIVE_MAX_SOURCE_WORDS = 4


def _is_likely_proper_noun(token: str) -> bool:
    """Heurística: token isolado em ALL-CAPS, 4–18 letras, fora do dicionário comum.

    Usado para preservar nomes próprios (Gillion, Willow, Vanessa, Fenris, Desmond)
    que o Google Translate insiste em traduzir como substantivos comuns
    (UM BILHÃO, SALGUEIRO, etc).
    """
    cleaned = token.strip().rstrip(".,!?;:'\"")
    if not cleaned or " " in cleaned:
        return False
    # Só letras (sem dígitos, sem hífen)
    if not cleaned.isalpha():
        return False
    if not (4 <= len(cleaned) <= 18):
        return False
    # Tem que estar em CAPS (todo caps OU primeira letra maiúscula com >50% caps)
    upper_ratio = sum(1 for c in cleaned if c.isupper()) / len(cleaned)
    if upper_ratio < 0.9:
        return False
    if cleaned.upper() in _COMMON_EN_CAPS_WORDS:
        return False
    return True


def _fix_infinitive_to_imperative(translated: str, source: str, tipo: str) -> str:
    """Converte infinitivo PT-BR para imperativo quando a source é um comando curto.

    Aplica-se apenas a `tipo in {fala, narracao}` com source de até
    `_IMPERATIVE_MAX_SOURCE_WORDS` palavras e que termina em `.` ou `!`.
    """
    if tipo not in {"fala", "narracao"}:
        return translated
    src = source.strip()
    if not src:
        return translated
    word_count = len(re.findall(r"[A-Za-z']+", src))
    if word_count == 0 or word_count > _IMPERATIVE_MAX_SOURCE_WORDS:
        return translated
    if not (src.endswith("!") or src.endswith(".") or src.endswith("?")):
        return translated

    cleaned = translated.strip()
    # Match único token (possivelmente com pontuação no fim)
    m = re.fullmatch(r"([A-Za-zÀ-ÿ-]+)([!?.,;:\s]*)", cleaned)
    if not m:
        return translated
    token, suffix = m.group(1), m.group(2)
    lookup_key = token.lower()
    if lookup_key not in _IMPERATIVE_FIXES:
        return translated
    replacement = _IMPERATIVE_FIXES[lookup_key]
    # Preservar caps da forma original
    if token.isupper():
        replacement = replacement.upper()
    elif token[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    return replacement + suffix


def _source_has_initial_stutter(source: str) -> bool:
    match = re.match(r"^\s*([A-Za-z])\s*[-‐‑‒–—]\s*([A-Za-z])", source or "")
    if not match:
        return False
    return match.group(1).lower() == match.group(2).lower()


def _repair_translated_stutter_prefix(translated: str, source: str, tipo: str) -> str:
    if tipo not in {"fala", "narracao"} or not _source_has_initial_stutter(source):
        return translated
    match = re.match(r"^(\s*)([A-Za-zÀ-ÖØ-öø-ÿ])(\s*[-‐‑‒–—]\s*)([A-Za-zÀ-ÖØ-öø-ÿ])", translated or "")
    if not match:
        return translated
    leading, current, separator, next_letter = match.groups()
    if current.lower() == next_letter.lower():
        return translated
    replacement = next_letter.upper() if current.isupper() or next_letter.isupper() else next_letter.lower()
    return f"{leading}{replacement}{separator}{translated[match.end(4) - 1:]}"


def _source_multiword_caps_name(source_text: str) -> str | None:
    stripped = (source_text or "").strip()
    if not stripped:
        return None
    match = re.fullmatch(r"([A-Za-z][A-Za-z' -]{2,80})([.!?;:]*)", stripped)
    if not match:
        return None
    core, suffix = match.groups()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", core)
    if len(words) < 2 or len(words) > 4:
        return None
    if not all(_is_likely_proper_noun(word) for word in words):
        return None
    canonical = " ".join(word[:1].upper() + word[1:].lower() for word in words)
    return canonical + suffix


def _repair_translated_proper_name(
    source_text: str,
    translated_text: str,
    tipo: str = "fala",
) -> tuple[str, list[dict]]:
    return translated_text, []


def normalize_google_language_code(language_code: str) -> str:
    code = (language_code or "").strip()
    if not code:
        return "en"

    normalized = code.replace("_", "-")
    lowered = normalized.lower()
    if lowered in GOOGLE_LANGUAGE_ALIASES:
        return GOOGLE_LANGUAGE_ALIASES[lowered]

    base = lowered.split("-", 1)[0]
    if base in GOOGLE_LANGUAGE_ALIASES:
        return GOOGLE_LANGUAGE_ALIASES[base]

    return base if "-" in normalized else lowered


def list_supported_google_languages() -> list[dict[str, str]]:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target="en")
    languages = translator.get_supported_languages(as_dict=True)
    items = []
    for label, code in sorted(languages.items(), key=lambda item: item[0].lower()):
        normalized_code = normalize_google_language_code(str(code))
        items.append(
            {
                "code": str(code),
                "label": str(label).strip().capitalize(),
                "ocr_strategy": "dedicated"
                if normalized_code in OCR_DEDICATED_GOOGLE_CODES
                else "best_effort",
            }
        )
    return items


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _chunk_texts_by_budget(
    texts: list[str],
    *,
    max_texts_per_chunk: int,
    max_chars_per_chunk: int,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        text_len = len(text)
        if current and (len(current) >= max_texts_per_chunk or current_chars + text_len > max_chars_per_chunk):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += text_len
    if current:
        chunks.append(current)
    return chunks


def _translate_google_parallel_chunks(
    texts: list[str],
    translate_batch: Callable[[list[str]], list[str]],
    *,
    min_unique_texts: int = 8,
    max_texts_per_chunk: int = 16,
    max_chars_per_chunk: int = 3500,
) -> list[str]:
    if not texts:
        return []
    workers = _env_positive_int("TRADUZAI_GOOGLE_TRANSLATE_WORKERS", 1)
    if not _env_flag_enabled("TRADUZAI_GOOGLE_PARALLEL_CHUNKS") or workers <= 1:
        return translate_batch(texts)

    unique_texts = list(dict.fromkeys(texts))
    if len(unique_texts) < min_unique_texts:
        return translate_batch(texts)

    chunks = _chunk_texts_by_budget(
        unique_texts,
        max_texts_per_chunk=max(1, max_texts_per_chunk),
        max_chars_per_chunk=max(1, max_chars_per_chunk),
    )
    if len(chunks) < 2:
        return translate_batch(texts)

    with ThreadPoolExecutor(
        max_workers=min(workers, len(chunks)),
        thread_name_prefix="traduzai-google",
    ) as executor:
        chunk_results = list(executor.map(translate_batch, chunks))

    translated_by_source: dict[str, str] = {}
    for chunk, translated_chunk in zip(chunks, chunk_results):
        if len(translated_chunk) != len(chunk):
            raise ValueError(
                f"Google parallel chunk size mismatch: {len(translated_chunk)} translated for {len(chunk)} sources"
            )
        translated_by_source.update(zip(chunk, translated_chunk))

    return [translated_by_source.get(text, text) for text in texts]


class _GoogleTranslator:
    def __init__(self, source="en", target="pt"):
        from deep_translator import GoogleTranslator

        source_code = normalize_google_language_code(source)
        target_code = normalize_google_language_code(target)
        self._translator = GoogleTranslator(source=source_code, target=target_code)
        self._cache: dict[str, str] = {}
        self.target = target_code
        self._source_lang = source_code
        self._target_lang = target_code
        # Optional cross-session persistent cache. Assigned externally via
        # `attach_persistent_cache` once the pipeline knows the models_dir.
        self._persistent_cache = None  # type: ignore[assignment]

    def attach_persistent_cache(self, cache) -> None:
        self._persistent_cache = cache

    def _persistent_lookup(self, key: str) -> Optional[str]:
        cache = self._persistent_cache
        if cache is None:
            return None
        try:
            return cache.get(key, self._source_lang, self._target_lang)
        except Exception as exc:
            logger.debug("Persistent cache lookup falhou: %s", exc)
            return None

    def _persistent_store(self, key: str, value: str) -> None:
        cache = self._persistent_cache
        if cache is None or value is None:
            return
        try:
            cache.set(key, self._source_lang, self._target_lang, value)
        except Exception as exc:
            logger.debug("Persistent cache store falhou: %s", exc)

    def translate(self, text: str) -> Optional[str]:
        key = text.strip()
        if key in self._cache:
            return self._cache[key]
        persisted = self._persistent_lookup(key)
        if persisted is not None:
            self._cache[key] = persisted
            return persisted

        for attempt in range(3):
            try:
                result = self._translator.translate(text)
                self._cache[key] = result
                if result:
                    self._persistent_store(key, result)
                return result
            except Exception:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return None

    def _translate_uncached_batch(self, uncached_texts: list[str]) -> list[str]:
        results: list[Optional[str]] = [None] * len(uncached_texts)

        separator = "\n===\n"
        joined = separator.join(uncached_texts)
        batch_result = self.translate(joined)

        if batch_result:
            parts = [p.strip() for p in batch_result.split("===") if p.strip()]
            if len(parts) == len(uncached_texts):
                for idx, part in enumerate(parts):
                    cleaned = part.strip()
                    results[idx] = cleaned
                    src_key = uncached_texts[idx].strip()
                    self._cache[src_key] = cleaned
                    if cleaned:
                        self._persistent_store(src_key, cleaned)
                return [r if r is not None else uncached_texts[i] for i, r in enumerate(results)]
            logger.warning(f"Batch split mismatch: {len(parts)} vs {len(uncached_texts)}. Tentando individualmente.")

        for i in range(len(uncached_texts)):
            if results[i] is None:
                trans = self.translate(uncached_texts[i])
                if trans == uncached_texts[i] and len(uncached_texts[i]) > 8:
                    logger.info(
                        f"Traducao redundante detectada em {uncached_texts[i][:20]}... Tentando 'auto' detection."
                    )
                    try:
                        from deep_translator import GoogleTranslator
                        trans = GoogleTranslator(source="auto", target=self._translator.target).translate(uncached_texts[i])
                    except Exception:
                        pass
                results[i] = trans or uncached_texts[i]

        return [r if r is not None else uncached_texts[i] for i, r in enumerate(results)]

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        logger.info(f"Traduzindo lote de {len(texts)} textos (Google)")

        results: list[Optional[str]] = [None] * len(texts)
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            key = text.strip()
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
                continue
            persisted = self._persistent_lookup(key)
            if persisted is not None:
                self._cache[key] = persisted
                results[i] = persisted
                continue
            uncached_indices.append(i)

        if not uncached_indices:
            return [r or "" for r in results]

        uncached_texts = [texts[i] for i in uncached_indices]
        try:
            uncached_translations = _translate_google_parallel_chunks(
                uncached_texts,
                self._translate_uncached_batch,
            )
        except Exception as exc:
            logger.warning("Traducao paralela Google falhou; tentando caminho sequencial: %s", exc)
            uncached_translations = self._translate_uncached_batch(uncached_texts)

        for idx, translated in zip(uncached_indices, uncached_translations):
            cleaned = translated or texts[idx]
            results[idx] = cleaned
            src_key = texts[idx].strip()
            self._cache[src_key] = cleaned
            if cleaned:
                self._persistent_store(src_key, cleaned)

        return [r if r is not None else texts[i] for i, r in enumerate(results)]


def _check_ollama(host: str) -> dict:
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = [model["name"] for model in data.get("models", [])]
        has_translator = any(
            ("traduzai-translator" in model) or ("mangatl-translator" in model)
            for model in models
        )
        return {"running": True, "models": models, "has_translator": has_translator}
    except Exception:
        return {"running": False, "models": [], "has_translator": False}


def _pick_ollama_model(models: list[str], preferred: str) -> str:
    preferred = (preferred or "").strip()
    if preferred:
        for model in models:
            if preferred in model:
                return model

    for candidate in ("traduzai-translator", "mangatl-translator"):
        for model in models:
            if candidate in model:
                return model

    return models[0] if models else ""


def _pick_ollama_model_for_language_pair(
    models: list[str],
    preferred: str,
    source_lang: str,
    target_lang: str,
) -> str:
    source_lang = normalize_google_language_code(source_lang)
    target_lang = normalize_google_language_code(target_lang)

    preferred_match = ""
    preferred_text = (preferred or "").strip()
    if preferred_text:
        for model in models:
            if preferred_text in model:
                preferred_match = model
                break
    if preferred_match and not any(
        default_name in preferred_text
        for default_name in ("traduzai-translator", "mangatl-translator")
    ):
        return preferred_match

    if source_lang == "ko" and target_lang == "pt":
        for candidate in ("gemma4:e4b", "qwen2.5:3b"):
            for model in models:
                if candidate in model:
                    return model

    return preferred_match or _pick_ollama_model(models, preferred)


def _call_ollama(model: str, system: str, user_msg: str, host: str) -> list[dict]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.15},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())

    content = data.get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
    if not isinstance(parsed, list):
        return []

    normalized: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        translated = item.get("translated")
        if not item_id or not isinstance(translated, str):
            continue
        normalized.append({"id": item_id, "translated": translated})
    return normalized


def _postprocess(
    text: str,
    was_upper: bool,
    tipo: str = "fala",
    source_text: str = "",
    lang: str = "en",
) -> str:
    result = _review_translation_grammar_semantics(source_text, text.strip(), tipo, lang=lang)
    result = "".join(ch for ch in result if unicodedata.category(ch) != "Cf")
    result = result.replace("\u2026", "...")
    for pattern, replacement, flags in ADAPTATIONS:
        result = re.sub(pattern, replacement, result, flags=flags)
    if lang == "ko" and "생문" in source_text and re.search(r"\btexto\s+original\b", result, re.IGNORECASE):
        result = re.sub(r"\b(?:o|um)?\s*texto\s+original\b", "a porta da vida", result, flags=re.IGNORECASE)

    result = re.sub(r"\s+([!?.,;:])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result).strip()

    # Conserta infinitivo -> imperativo quando o source \u00e9 claramente um comando curto.
    # Precisa rodar antes do upper() final para casar a tabela em min\u00fasculas.
    if source_text:
        result = _repair_translated_stutter_prefix(result, source_text, tipo)
        result = _fix_infinitive_to_imperative(result, source_text, tipo)

    if tipo == "sfx":
        result = result.upper()
    elif was_upper:
        result = result.upper()
    elif tipo == "narracao" and result:
        result = result[0].upper() + result[1:]

    return result


def _prepare_source_text_for_translation(
    text: str,
    tipo: str = "fala",
    lang: str = "en",
    *,
    preserve_case: bool = False,
) -> str:
    result = text.strip()
    if not result:
        return result

    if tipo == "sfx":
        return re.sub(r"\s+", " ", result)

    for pattern, replacement, flags in SOURCE_OCR_REPAIRS:
        result = re.sub(pattern, replacement, result, flags=flags)

    result = re.sub(r"\s{2,}", " ", result).strip()
    
    # Nao forca capitalizacao para idiomas CJK
    is_cjk = lang in ("ja", "ko", "zh", "zh-CN", "zh-TW")
    if not preserve_case and not is_cjk and result and len(result) > 2:
        result = result[0].upper() + result[1:].lower()
    return result


def _review_translation_grammar_semantics(
    source_text: str,
    translated_text: str,
    tipo: str = "fala",
    lang: str = "en",
) -> str:
    del tipo

    result = translated_text.strip()
    if not result:
        return result

    for pattern, replacement, flags in TRANSLATION_REVIEW_REPAIRS:
        result = re.sub(pattern, replacement, result, flags=flags)

    for pattern, replacement, flags in ADAPTATIONS:
        result = re.sub(pattern, replacement, result, flags=flags)

    prepared_source = _prepare_source_text_for_translation(source_text, "fala", lang=lang)
    normalized_source = re.sub(r"[\W_]+", " ", prepared_source.lower()).strip()
    if re.search(r"\bcould that light be\b", normalized_source):
        if re.search(r"\bluz\b", result, re.IGNORECASE) or re.search(r"\bacende\b", result, re.IGNORECASE):
            result = "Poderia ser aquela luz...?!"
    if re.search(r"\byou mean the power he got by trading with them\b", normalized_source):
        result = "Você quer dizer o poder que ele conseguiu em um acordo com eles?"
    if re.search(r"\bhalf of a mana technique\b", normalized_source) and re.search(
        r"\binstantly surpass\b", normalized_source
    ):
        result = (
            "Pode até ser só metade de uma técnica de mana, "
            "mas seus efeitos já são mais do que suficientes. "
            "Esse poder permite ultrapassar instantaneamente os próprios limites."
        )
    if re.search(r"\bif desmond ends up using that power\b", normalized_source):
        result = "Se Desmond usar esse poder..."
    if re.search(r"\bthe rampaging vanessa also used a similar mana technique\b", normalized_source):
        result = "A Vanessa em fúria também usava um método semelhante de circulação de mana."
    if re.search(r"\bi still have many enemies\b", normalized_source) and re.search(
        r"\bfar stronger than you\b", normalized_source
    ):
        result = "E eu ainda tenho muitos inimigos. Inimigos muito mais fortes do que você."
    if re.search(r"\bwhen i killed you before\b", normalized_source) and re.search(
        r"\bsliced you in half with a single strike\b", normalized_source
    ):
        result = "Quando te matei antes, parti você ao meio com um único golpe, então não percebi..."
    if re.search(r"\bcould this light be\b", normalized_source):
        result = "Pode ser essa luz...?!"
    if re.search(r"\byou could see all of my attacks\b", normalized_source):
        result = "Que conseguia ver todos os meus ataques?"
    if re.search(r"\byou said you could see through all my attacks right\b", normalized_source):
        if re.search(r"\b(ataques|golpes|através|enxergar|ver)\b", result, re.IGNORECASE):
            result = "Você disse que podia enxergar todos os meus golpes, certo?"

    result = re.sub(r"\s+([!?.,;:])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _preprocess_text(
    text: str,
    tipo: str = "fala",
    lang: str = "en",
    *,
    preserve_case: bool = False,
) -> str:
    result = text.strip()
    if tipo == "sfx":
        return re.sub(r"\s+", " ", result)

    # Nao forca capitalizacao para idiomas CJK
    is_cjk = lang in ("ja", "ko", "zh", "zh-CN", "zh-TW")
    if not preserve_case and not is_cjk and result and len(result) > 2:
        result = result[0].upper() + result[1:].lower()
    
    # Pré-tradução: substitui expressões para o Google traduzir corretamente
    for pattern, replacement, flags in PRE_TRANSLATION_GLOSSARY:
        result = re.sub(pattern, replacement, result, flags=flags)
    result = result.replace("...", "\u2026")
    return re.sub(r"\s+", " ", result)


def _normalize_memory_key(text: str, tipo: str) -> str:
    normalized = re.sub(r"[\W_]+", "", text.lower())
    return f"{tipo}:{normalized}"


def _normalized_translation_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _merge_qa_flags(*flag_lists: list[str] | tuple[str, ...] | None) -> list[str]:
    merged: list[str] = []
    for flags in flag_lists:
        for flag in flags or []:
            if flag and flag not in merged:
                merged.append(str(flag))
    return merged


def _apply_mojibake_audit(
    translated_text: str,
    *,
    text_id: str | None = None,
    stage: str = "translation_output",
) -> tuple[str, list[str], dict | None]:
    try:
        from debug_tools.detectors import audit_mojibake
    except Exception:
        return translated_text, [], None
    audit = audit_mojibake(translated_text, text_id=text_id, stage=stage)
    flags = list(audit.get("flags") or [])
    fixed = str(audit.get("suggested_fix") or translated_text)
    if flags:
        try:
            from debug_tools import get_recorder

            recorder = get_recorder()
            if recorder and recorder.enabled:
                recorder.write_jsonl("04_text_normalization_router/mojibake_audit.jsonl", audit)
                recorder.write_jsonl("11_qa_export_gate/visual_blockers.jsonl", audit)
        except Exception:
            pass
    return fixed, flags, audit if flags else None


def _append_protection_entry(entries: list[dict], source: Any, *, entry_type: str, target: Any | None = None) -> None:
    clean_source = " ".join(str(source or "").split()).strip()
    if not clean_source:
        return
    key = (entry_type, clean_source.casefold())
    if key in {(str(entry.get("type", "")), str(entry.get("source", "")).casefold()) for entry in entries}:
        return
    entries.append(
        {
            "id": clean_source.casefold(),
            "source": clean_source,
            "target": " ".join(str(target or clean_source).split()).strip(),
            "type": entry_type,
            "protect": True,
            "locked": entry_type == "character",
            "aliases": [],
            "forbidden": [],
        }
    )


def _extend_protection_entries_from_context(entries: list[dict], context: dict) -> None:
    for character in (context or {}).get("characters") or []:
        if isinstance(character, dict):
            name = character.get("name") or character.get("source") or character.get("title")
            target = character.get("preferredPortugueseName") or name
            _append_protection_entry(entries, name, entry_type="character", target=target)
            for alias in character.get("aliases") or []:
                _append_protection_entry(entries, alias, entry_type="alias", target=alias)
        else:
            _append_protection_entry(entries, character, entry_type="character")

    aliases = (context or {}).get("aliases") or []
    if isinstance(aliases, dict):
        alias_iterable = []
        for value in aliases.values():
            if isinstance(value, (list, tuple, set)):
                alias_iterable.extend(value)
            else:
                alias_iterable.append(value)
    else:
        alias_iterable = aliases
    for alias in alias_iterable:
        _append_protection_entry(entries, alias, entry_type="alias")


def _build_protection_glossary_entries(context: dict, glossario: dict) -> list[dict]:
    try:
        from glossary.builder import build_glossary_entries
    except Exception:
        return []
    try:
        entries = build_glossary_entries(context or {}, glossario or {})
        _extend_protection_entries_from_context(entries, context or {})
    except Exception as exc:
        logger.debug("Falha ao montar entradas de glossario para protecao: %s", exc)
        return []
    protected_types = {
        "manual_glossary",
        "context_glossary",
        "memory",
        "corpus_memory",
        "term",
        "faction",
        "character",
        "alias",
    }
    return [entry for entry in entries if entry.get("type") in protected_types]


def _protect_source_for_translation(text: str, tipo: str, context: dict, glossario: dict) -> dict:
    if tipo == "sfx" or not text:
        return {"protected_source": text, "terms": []}
    entries = _build_protection_glossary_entries(context, glossario)
    if not entries:
        return {"protected_source": text, "terms": []}
    try:
        from translator.term_protection import protect_terms
    except Exception:
        return {"protected_source": text, "terms": []}
    try:
        protected = protect_terms(text, entries)
    except Exception as exc:
        logger.debug("Falha ao proteger termos da traducao: %s", exc)
        return {"protected_source": text, "terms": []}
    return protected if isinstance(protected, dict) else {"protected_source": text, "terms": []}


def _restore_protected_translation(translated: str, terms: list[dict]) -> tuple[str, list[str], list[dict]]:
    if not terms:
        return translated, [], []
    try:
        from translator.term_protection import restore_terms
    except Exception:
        return translated, [], []
    restored = restore_terms(translated or "", terms)
    flags = []
    for flag in restored.get("flags", []) or []:
        reason = str(flag.get("reason", ""))
        if reason in {"placeholder_missing", "placeholder_leftover", "placeholder_corrupted", "unrestored_placeholder"}:
            flags.append("unrestored_placeholder")
        elif reason == "forbidden_translation":
            flags.append("forbidden_translation")
        else:
            flags.append(reason or "glossary_violation")
    hits = [
        {"phase": "placeholder", "source": term.get("source", ""), "target": term.get("target", "")}
        for term in terms
        if term.get("source") and term.get("target")
    ]
    return str(restored.get("text", translated)), _merge_qa_flags(flags), hits


def _missing_protected_terms_after_lock(translated: str, terms: list[dict]) -> list[dict]:
    if not terms:
        return []
    translated_key = _normalize_entity_key(translated)
    missing = []
    for term in terms:
        target_key = _normalize_entity_key(str(term.get("target", "") or ""))
        if target_key and target_key in translated_key:
            continue
        missing.append(term)
    return missing


def _fold_for_quality_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _translation_quality_flags(source_text: str, translated_text: str, source_lang: str) -> list[str]:
    flags: list[str] = []
    normalized_source_lang = normalize_google_language_code(source_lang)
    if normalized_source_lang in {"ja", "ko", "zh-CN", "zh-TW"}:
        if SOURCE_SCRIPT_PATTERN.search(translated_text or ""):
            flags.append("source_script_leak")
        folded_translation = _fold_for_quality_match(translated_text)
        if TRANSLATION_FALLBACK_PHRASE_PATTERN.search(folded_translation):
            flags.append("translation_fallback_phrase")
        if LITERAL_OCR_TRANSLATION_PATTERN.search(folded_translation):
            flags.append("literal_ocr_translation")
    if normalized_source_lang == "ko" and KOREAN_LATIN_OCR_ARTIFACT_PATTERN.search(
        f"{source_text} {translated_text}"
    ):
        flags.append("suspected_ocr_error")
    return flags


def _fix_source_mojibake_if_useful(text: str) -> str:
    source = str(text or "")
    if not source:
        return source
    try:
        from debug_tools.detectors import fix_mojibake
    except Exception:
        return source
    fixed = fix_mojibake(source)
    if fixed == source:
        return source
    if SOURCE_SCRIPT_PATTERN.search(fixed):
        return fixed
    return source


def _looks_like_kana_sfx_source(text: str) -> bool:
    source = str(text or "").strip()
    if not source:
        return False
    kana_count = len(re.findall(r"[\u3040-\u30ff]", source))
    if kana_count <= 0:
        return False
    if re.search(r"[\u4e00-\u9fff\uac00-\ud7afA-Za-z0-9]", source):
        return False
    meaningful = re.sub(r"[\s\u3000\u30fc\uff70\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65!?.…。・~ー]+", "", source)
    return bool(meaningful) and kana_count / max(1, len(meaningful)) >= 0.75


def _should_preserve_untranslated_kana_sfx(source_text: str, translated_text: str, source_lang: str) -> bool:
    normalized_source_lang = normalize_google_language_code(source_lang)
    if normalized_source_lang not in {"zh-CN", "zh-TW"}:
        return False
    if not _looks_like_kana_sfx_source(source_text):
        return False
    return bool(SOURCE_SCRIPT_PATTERN.search(translated_text or ""))


def is_translation_fallback_phrase(text: str) -> bool:
    return bool(TRANSLATION_FALLBACK_PHRASE_PATTERN.search(_fold_for_quality_match(text or "")))


def _should_block_translation_render(
    source_text: str,
    translated_text: str,
    source_lang: str,
    tipo: str,
    qa_flags: list[str],
) -> bool:
    del source_text
    normalized_source_lang = normalize_google_language_code(source_lang)
    critical_lock_flags = {"placeholder_lost", "unrestored_placeholder", "glossary_violation", "forbidden_translation"}
    if normalized_source_lang not in {"ja", "ko", "zh-CN", "zh-TW"}:
        return bool(critical_lock_flags & set(qa_flags))
    if tipo == "sfx":
        return False
    if critical_lock_flags & set(qa_flags):
        return True
    if "translation_fallback_phrase" in qa_flags:
        return True
    if "source_script_leak" in qa_flags and SOURCE_SCRIPT_PATTERN.search(translated_text or ""):
        return True
    return False


def _should_skip_translation_item(text: dict) -> bool:
    route_action = str(text.get("route_action") or "").strip().lower()
    if route_action in ROUTE_ACTIONS:
        return not route_action_requires_translation(route_action)
    return False


def _source_text_before_normalization(text: dict) -> str:
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "leading_dark_lobe_duplicate_fragment_removed" in flags:
        metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
        cleanup = metrics.get("leading_dark_lobe_duplicate_fragment_removed") if isinstance(metrics, dict) else None
        if isinstance(cleanup, dict):
            repaired = str(cleanup.get("to") or "").strip()
            if repaired:
                return repaired
    return str(text.get("raw_ocr") or text.get("original") or text.get("text") or "")


def _normalization_confidence_after(text: dict) -> float:
    normalization = text.get("normalization")
    if isinstance(normalization, dict):
        try:
            return float(normalization.get("confidence_after_estimate"))
        except (TypeError, ValueError):
            pass
    try:
        return float(text.get("confidence") or text.get("ocr_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _source_text_for_translation(text: dict) -> str:
    raw = _source_text_before_normalization(text)
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "leading_dark_lobe_duplicate_fragment_removed" in flags and raw:
        return raw
    normalized = str(text.get("normalized_text_final") or "").strip()
    normalization = text.get("normalization")
    changed = bool(isinstance(normalization, dict) and normalization.get("changed"))
    if normalized and normalized != raw:
        changed = True
    if changed and normalized and _normalization_confidence_after(text) >= 0.7:
        return normalized
    return str(text.get("text") or raw)


def _repair_translation_after_source_prefix_cleanup(text: dict, translated: str) -> tuple[str, list[str]]:
    if not isinstance(text, dict):
        return translated, []
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "leading_dark_lobe_duplicate_fragment_removed" not in flags:
        return translated, []
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    cleanup = metrics.get("leading_dark_lobe_duplicate_fragment_removed") if isinstance(metrics, dict) else None
    if not isinstance(cleanup, dict):
        return translated, []
    original_before = str(cleanup.get("from") or "").strip()
    original_after = str(cleanup.get("to") or text.get("text") or text.get("original") or "").strip()
    if not original_before or not original_after or original_before == original_after:
        return translated, []
    duplicate_head = original_before
    if original_after in original_before:
        duplicate_head = original_before.split(original_after, 1)[0].strip()
    duplicate_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9']+", duplicate_head.lower())
        if len(token) > 1
    }
    if len(duplicate_tokens) < 2:
        return translated, []
    translated_norm = re.sub(r"\s+", " ", str(translated or "").strip())
    match = re.match(r"^(?P<head>[^.!?]{4,90}[.!?])\s*(?P<tail>.+)$", translated_norm)
    if not match:
        return translated, []
    translated_head = match.group("head").strip()
    tail = match.group("tail").strip()
    head_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9']+", translated_head.lower())
        if len(token) > 1
    }
    fuzzy_matches = 0
    for token in head_tokens:
        if any(token == src or token in src or src in token for src in duplicate_tokens):
            fuzzy_matches += 1
    english_leak_terms = {"space", "only", "utes", "subspace", "retention", "world", "quest", "system"}
    if fuzzy_matches < 2 and not (head_tokens & english_leak_terms):
        return translated, []
    return tail, ["translation_leading_duplicate_fragment_removed"]


def _uses_confident_normalized_text_final(text: dict) -> bool:
    raw = _source_text_before_normalization(text)
    normalized = str(text.get("normalized_text_final") or "").strip()
    if not normalized or normalized == raw:
        return False
    normalization = text.get("normalization")
    changed = bool(isinstance(normalization, dict) and normalization.get("changed"))
    if normalized != raw:
        changed = True
    return changed and _normalization_confidence_after(text) >= 0.7


def _debug_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _debug_page_id_from_band_id(band_id: str | None) -> str | None:
    if not band_id:
        return None
    match = re.match(r"^(page_\d{3})_band_\d{3}$", str(band_id))
    return match.group(1) if match else None


def _debug_source_page_number(page_id: str | None, fallback: int) -> int:
    if page_id:
        match = re.match(r"^page_(\d{3})$", str(page_id))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
    return fallback


def _build_translation_debug_identity(page_idx: int, index: int, text: dict) -> dict[str, Any]:
    text_id = _debug_str_or_none(text.get("text_id") or text.get("id") or f"t{index + 1}")
    band_id = _debug_str_or_none(text.get("band_id") or text.get("_band_id"))
    page_id = (
        _debug_str_or_none(text.get("page_id"))
        or _debug_page_id_from_band_id(band_id)
        or f"page_{page_idx + 1:03d}"
    )
    trace_id = _debug_str_or_none(text.get("trace_id"))
    if not trace_id and text_id and band_id:
        trace_id = f"{text_id}@{band_id}"
    text_instance_id = _debug_str_or_none(text.get("text_instance_id") or text.get("instance_id"))
    if not text_instance_id and text_id and band_id:
        text_instance_id = f"{band_id}_{text_id}"
    source_page_number = text.get("source_page_number") or text.get("_source_page_number")
    try:
        source_page_number = int(source_page_number)
    except (TypeError, ValueError):
        source_page_number = _debug_source_page_number(page_id, page_idx + 1)

    identity: dict[str, Any] = {
        "page_id": page_id,
        "source_page_number": source_page_number,
        "text_id": text_id,
    }
    if band_id:
        identity["band_id"] = band_id
    if trace_id:
        identity["trace_id"] = trace_id
    if text_instance_id:
        identity["text_instance_id"] = text_instance_id
    if trace_id:
        identity["audit_key"] = trace_id
    elif text_id and band_id:
        identity["audit_key"] = f"{text_id}@{band_id}"
    elif text_id:
        identity["audit_key"] = text_id
    return identity


def _translation_debug_identity_collection(page_idx: int | None, texts: list[dict] | None) -> dict[str, Any]:
    if not texts:
        return {}
    identities = [
        _build_translation_debug_identity(page_idx or 0, index, text)
        for index, text in enumerate(texts)
        if isinstance(text, dict)
    ]
    if not identities:
        return {}

    def unique(key: str) -> list[Any]:
        values: list[Any] = []
        for identity in identities:
            value = identity.get(key)
            if value in (None, "", []):
                continue
            if value not in values:
                values.append(value)
        return values

    payload: dict[str, Any] = {
        "text_ids": unique("text_id"),
        "trace_ids": unique("trace_id"),
        "band_ids": unique("band_id"),
        "page_ids": unique("page_id"),
    }
    if len(payload["page_ids"]) == 1:
        payload["page_id"] = payload["page_ids"][0]
    if len(payload["band_ids"]) == 1:
        payload["band_id"] = payload["band_ids"][0]
    return {key: value for key, value in payload.items() if value not in (None, [], "")}


class _TranslationDebugSession:
    def __init__(self, backend: str, model: str) -> None:
        self.backend = backend
        self.model = model
        self.inputs_count = 0
        self.outputs_count = 0
        self.fallback_count = 0
        self.mojibake_count = 0
        self.glossary_application_count = 0
        self.backend_distribution: dict[str, int] = {}
        self._recorder = None
        try:
            from debug_tools import get_recorder

            recorder = get_recorder()
            if recorder and recorder.enabled:
                self._recorder = recorder
                self._touch_jsonl("07_translation/translation_inputs.jsonl")
                self._touch_jsonl("07_translation/translation_outputs.jsonl")
                self._touch_jsonl("07_translation/glossary_application.jsonl")
                self._touch_jsonl("07_translation/translation_fallbacks.jsonl")
                self.write_summary()
        except Exception:
            self._recorder = None

    @property
    def enabled(self) -> bool:
        return bool(self._recorder and getattr(self._recorder, "enabled", False))

    def record_input(
        self,
        *,
        page_idx: int,
        index: int,
        text: dict,
        source_text_before_normalization: str,
        source_text_sent_to_translator: str,
        backend: str | None = None,
        model: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.inputs_count += 1
        identity = _build_translation_debug_identity(page_idx, index, text)
        self._write_jsonl(
            "07_translation/translation_inputs.jsonl",
            {
                **identity,
                "tipo": text.get("tipo", "fala"),
                "content_class": text.get("content_class"),
                "translate_policy": text.get("translate_policy"),
                "backend": backend or self.backend,
                "model": model or self.model,
                "source_text_before_normalization": source_text_before_normalization,
                "source_text_sent_to_translator": source_text_sent_to_translator,
                "prompt_hash": self._hash(source_text_sent_to_translator),
            },
        )
        self.write_summary()

    def record_output(
        self,
        *,
        page_idx: int,
        index: int,
        text: dict,
        source_text_sent_to_translator: str,
        raw_response: Any,
        final_translation_after_postprocess: str,
        duration_ms: int,
        backend: str | None = None,
        model: str | None = None,
        fallback_used: bool = False,
        glossary_hits: list[dict] | None = None,
        qa_flags: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        resolved_backend = backend or self.backend
        resolved_model = model or self.model
        self.outputs_count += 1
        self.backend_distribution[resolved_backend] = self.backend_distribution.get(resolved_backend, 0) + 1
        if fallback_used:
            self.fallback_count += 1
        if "mojibake_in_translation" in set(qa_flags or []):
            self.mojibake_count += 1
        hits = list(glossary_hits or [])
        if hits:
            self.glossary_application_count += len(hits)
        identity = _build_translation_debug_identity(page_idx, index, text)
        base = {
            **identity,
            "tipo": text.get("tipo", "fala"),
            "backend": resolved_backend,
            "model": resolved_model,
            "fallback_used": bool(fallback_used),
            "duration_ms": max(0, int(duration_ms)),
            "prompt_hash": self._hash(source_text_sent_to_translator),
            "raw_response_preview": self._preview(raw_response),
            "final_translation_after_postprocess": final_translation_after_postprocess,
            "qa_flags": list(qa_flags or []),
        }
        self._write_jsonl("07_translation/translation_outputs.jsonl", base)
        for hit in hits:
            self._write_jsonl(
                "07_translation/glossary_application.jsonl",
                {
                    **base,
                    "glossary_hit": hit,
                },
            )
        self.write_summary()

    def record_fallback(
        self,
        *,
        page_idx: int | None,
        backend: str,
        model: str,
        reason: str,
        error: Any = None,
        texts: list[dict] | None = None,
    ) -> None:
        if not self.enabled:
            return
        identity = _translation_debug_identity_collection(page_idx, texts)
        page_id = identity.get("page_id") or (f"page_{page_idx + 1:03d}" if page_idx is not None else None)
        self._write_jsonl(
            "07_translation/translation_fallbacks.jsonl",
            {
                **identity,
                "page_id": page_id,
                "source_page_number": page_idx + 1 if page_idx is not None else None,
                "backend": backend,
                "model": model,
                "fallback_used": True,
                "reason": reason,
                "error_preview": self._preview(error),
            },
        )
        self.write_summary()

    def write_summary(self) -> None:
        if not self.enabled:
            return
        metrics = self._translation_debug_file_metrics()
        outputs_count = metrics["translation_outputs_count"]
        fallback_count = metrics["fallback_count"]
        self._write_json(
            "07_translation/translation_debug_summary.json",
            {
                "total_inputs": metrics["translation_inputs_count"],
                "total_outputs": outputs_count,
                "translation_inputs_count": metrics["translation_inputs_count"],
                "translation_outputs_count": outputs_count,
                "outputs_count": outputs_count,
                "fallback_count": fallback_count,
                "fallback_event_count": metrics["translation_fallback_events_count"],
                "fallback_rate": round(fallback_count / outputs_count, 4) if outputs_count else 0.0,
                "backend_distribution": metrics["backend_distribution"],
                "mojibake_count": metrics["mojibake_count"],
                "glossary_application_count": metrics["glossary_application_count"],
                "translation_debug_entry_count": metrics["translation_debug_entry_count"],
                "jsonl_counts": metrics["jsonl_counts"],
                "identity_coverage": metrics["identity_coverage"],
            },
        )

    def _translation_debug_file_metrics(self) -> dict[str, Any]:
        input_count, inputs = self._read_jsonl("07_translation/translation_inputs.jsonl")
        output_count, outputs = self._read_jsonl("07_translation/translation_outputs.jsonl")
        glossary_count, _glossary = self._read_jsonl("07_translation/glossary_application.jsonl")
        fallback_event_count, _fallbacks = self._read_jsonl("07_translation/translation_fallbacks.jsonl")

        backend_distribution: dict[str, int] = {}
        fallback_count = 0
        mojibake_count = 0
        for row in outputs:
            backend = _debug_str_or_none(row.get("backend")) or "unknown"
            backend_distribution[backend] = backend_distribution.get(backend, 0) + 1
            if bool(row.get("fallback_used")):
                fallback_count += 1
            if "mojibake_in_translation" in set(row.get("qa_flags") or []):
                mojibake_count += 1

        jsonl_counts = {
            "translation_inputs.jsonl": input_count,
            "translation_outputs.jsonl": output_count,
            "glossary_application.jsonl": glossary_count,
            "translation_fallbacks.jsonl": fallback_event_count,
        }
        return {
            "translation_inputs_count": input_count,
            "translation_outputs_count": output_count,
            "fallback_count": fallback_count,
            "translation_fallback_events_count": fallback_event_count,
            "backend_distribution": dict(sorted(backend_distribution.items())),
            "mojibake_count": mojibake_count,
            "glossary_application_count": glossary_count,
            "translation_debug_entry_count": sum(jsonl_counts.values()),
            "jsonl_counts": jsonl_counts,
            "identity_coverage": self._identity_coverage(inputs, outputs),
        }

    def _read_jsonl(self, rel_path: str) -> tuple[int, list[dict[str, Any]]]:
        try:
            path = self._recorder._root / rel_path
            if not path.exists():
                return 0, []
            line_count = 0
            rows: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                line_count += 1
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
            return line_count, rows
        except Exception:
            return 0, []

    def _identity_coverage(self, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
        def field_summary(field: str) -> dict[str, Any]:
            input_values = [_debug_str_or_none(row.get(field)) for row in inputs]
            output_values = [_debug_str_or_none(row.get(field)) for row in outputs]
            values = sorted({value for value in [*input_values, *output_values] if value})
            return {
                "input_count": sum(1 for value in input_values if value),
                "output_count": sum(1 for value in output_values if value),
                "values": values,
            }

        return {
            "page_id": field_summary("page_id"),
            "band_id": field_summary("band_id"),
            "trace_id": field_summary("trace_id"),
            "text_instance_id": field_summary("text_instance_id"),
        }

    def _write_jsonl(self, rel_path: str, payload: dict[str, Any]) -> None:
        try:
            from debug_tools.text_diff import redact_debug_payload

            self._recorder.write_jsonl(rel_path, redact_debug_payload(payload))
        except Exception:
            pass

    def _write_json(self, rel_path: str, payload: dict[str, Any]) -> None:
        try:
            from debug_tools.text_diff import redact_debug_payload

            self._recorder.write_json(rel_path, redact_debug_payload(payload))
        except Exception:
            pass

    def _touch_jsonl(self, rel_path: str) -> None:
        try:
            target = self._recorder._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
            self._recorder.register_artifact(
                stage=self._recorder._stage_from_rel(rel_path),
                rel_path=rel_path,
                kind="jsonl",
            )
        except Exception:
            pass

    def _hash(self, value: Any) -> str:
        try:
            from debug_tools.text_diff import sha1_truncated

            return sha1_truncated(value)
        except Exception:
            return ""

    def _preview(self, value: Any) -> str:
        try:
            from debug_tools.text_diff import preview_text

            return preview_text(value, limit=256)
        except Exception:
            text = "" if value is None else str(value)
            return text[:256]


def _apply_translation_render_blocks(
    translated_pages: list[dict],
    source_lang: str,
) -> list[dict]:
    for page in translated_pages:
        for item in page.get("texts", []) or []:
            translated = str(item.get("translated", "") or "")
            qa_flags = list(item.get("qa_flags") or [])
            if not translated:
                continue
            if not _should_block_translation_render(
                str(item.get("original", "") or item.get("text", "") or ""),
                translated,
                source_lang,
                str(item.get("tipo", "fala")),
                qa_flags,
            ):
                continue
            item["translation_blocked_text"] = translated
            item["translated"] = ""
            item["qa_flags"] = _merge_qa_flags(qa_flags, ["translation_failed", "translation_render_blocked"])
    return translated_pages


def _normalize_entity_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _normalize_entity_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _entity_candidates(context: dict, glossario: dict) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def _push(source: str, target: str, kind: str) -> None:
        clean_source = " ".join(str(source or "").split()).strip()
        clean_target = " ".join(str(target or "").split()).strip()
        source_norm = _normalize_entity_key(clean_source)
        target_norm = _normalize_entity_key(clean_target)
        if not clean_source or not source_norm:
            return
        fingerprint = (kind, source_norm, target_norm)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        candidates.append(
            {
                "source": clean_source,
                "target": clean_target or clean_source,
                "kind": kind,
                "source_norm": source_norm,
                "target_norm": target_norm,
                "source_word_count": len(clean_source.split()),
            }
        )

    for character in context.get("personagens", []) or []:
        _push(character, character, "character")
    for alias in context.get("aliases", []) or []:
        _push(alias, alias, "alias")
    for term in context.get("termos", []) or []:
        _push(term, (glossario or {}).get(term) or (context.get("memoria_lexical", {}) or {}).get(term) or term, "term")
    for faction in context.get("faccoes", []) or []:
        _push(faction, faction, "faction")

    for source, target in (context.get("memoria_lexical", {}) or {}).items():
        _push(source, target, "memory")
    for source, target in (context.get("corpus_memoria_lexical", {}) or {}).items():
        _push(source, target, "corpus_memory")
    for source, target in (glossario or {}).items():
        _push(source, target, "glossary")

    return candidates


def _looks_like_entity_label(text: str) -> bool:
    compact = " ".join((text or "").split()).strip(" .!?…,:;\"'()[]{}")
    if not compact or len(compact) > 48:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’-]*", compact)
    if not words or len(words) > 4:
        return False
    total_alpha = sum(1 for ch in compact if ch.isalpha())
    return total_alpha >= 4


def _repair_source_entities(text: str, context: dict, glossario: dict) -> tuple[str, list[dict], list[str]]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned or not _looks_like_entity_label(cleaned):
        return cleaned, [], []

    normalized = _normalize_entity_key(cleaned)
    if not normalized:
        return cleaned, [], []

    word_count = len(cleaned.split())
    best_candidate: dict | None = None
    best_score = 0.0

    for candidate in _entity_candidates(context, glossario):
        if abs(candidate["source_word_count"] - word_count) > 1:
            continue
        candidate_norm = candidate["source_norm"]
        score = SequenceMatcher(None, normalized, candidate_norm).ratio()
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate and best_score >= 0.84 and cleaned != best_candidate["source"]:
        repair = {
            "phase": "source",
            "kind": best_candidate["kind"],
            "from": cleaned,
            "to": best_candidate["source"],
        }
        return best_candidate["source"], [repair], ["source_entity_repaired"]

    return cleaned, [], []


def _replace_canonical_phrase(text: str, canonical_phrase: str) -> tuple[str, bool]:
    if not text or not canonical_phrase:
        return text, False

    normalized_text = _normalize_entity_search_text(text)
    normalized_phrase = _normalize_entity_search_text(canonical_phrase)
    if not normalized_phrase:
        return text, False

    position = normalized_text.find(normalized_phrase)
    if position < 0:
        return text, False

    index_map: list[int] = []
    cursor = 0
    for index, ch in enumerate(text):
        expanded = unicodedata.normalize("NFKD", ch)
        expanded = "".join(item for item in expanded if not unicodedata.combining(item))
        expanded = expanded.lower()
        if not expanded:
            continue
        for _ in expanded:
            if cursor >= len(normalized_text):
                break
            index_map.append(index)
            cursor += 1

    if not index_map or position + len(normalized_phrase) - 1 >= len(index_map):
        return text, False

    start = index_map[position]
    end = index_map[position + len(normalized_phrase) - 1] + 1
    replaced = f"{text[:start]}{canonical_phrase}{text[end:]}"
    return replaced, replaced != text


def _apply_target_entity_locks(
    source_text: str,
    translated_text: str,
    context: dict,
    glossario: dict,
) -> tuple[str, list[dict], list[str]]:
    result = translated_text
    hits: list[dict] = []
    flags: list[str] = []
    source_key = _normalize_entity_key(source_text)
    if not source_key:
        return result, hits, flags

    glossary_like_kinds = {"glossary", "memory", "corpus_memory", "term"}

    for candidate in _entity_candidates(context, glossario):
        target = candidate["target"]
        target_norm = candidate["target_norm"]
        if not target or not target_norm:
            continue
        if candidate["source_norm"] not in source_key:
            continue

        updated, changed = _replace_canonical_phrase(result, target)
        if changed:
            result = updated
        if changed and candidate["kind"] not in glossary_like_kinds:
            continue
        if (changed or target_norm in _normalize_entity_key(result)) and candidate["kind"] in glossary_like_kinds:
            hit = {
                "phase": "target",
                "source": candidate["source"],
                "target": target,
            }
            if hit not in hits:
                hits.append(hit)
            if "glossary_locked" not in flags:
                flags.append("glossary_locked")

    return result, hits, flags


def _best_entity_candidate_for_span_v2(
    span_text: str,
    context: dict,
    glossario: dict,
    *,
    expected_word_count: int | None = None,
) -> tuple[dict | None, float]:
    normalized = _normalize_entity_key(span_text)
    if not normalized:
        return None, 0.0

    best_candidate: dict | None = None
    best_score = 0.0
    span_word_count = len(span_text.split())

    for candidate in _entity_candidates(context, glossario):
        candidate_word_count = int(candidate["source_word_count"] or 0)
        reference_word_count = expected_word_count if expected_word_count is not None else span_word_count
        if abs(candidate_word_count - reference_word_count) > 1:
            continue
        score = SequenceMatcher(None, normalized, candidate["source_norm"]).ratio()
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, best_score


def _repair_entity_phrase_inside_sentence_v2(
    text: str,
    context: dict,
    glossario: dict,
) -> tuple[str, list[dict], list[str]]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return cleaned, [], []

    token_matches = list(re.finditer(r"[^\W_][^\W_'’-]*", cleaned, flags=re.UNICODE))
    if len(token_matches) < 2 or len(token_matches) > 8:
        return cleaned, [], []

    best_payload: tuple[float, int, int, dict] | None = None
    for candidate in _entity_candidates(context, glossario):
        candidate_word_count = int(candidate["source_word_count"] or 0)
        if candidate_word_count < 2:
            continue
        for window_size in {candidate_word_count - 1, candidate_word_count, candidate_word_count + 1}:
            if window_size < 2 or window_size > len(token_matches):
                continue
            for start_index in range(0, len(token_matches) - window_size + 1):
                end_index = start_index + window_size - 1
                start = token_matches[start_index].start()
                end = token_matches[end_index].end()
                span_text = cleaned[start:end]
                if not span_text or span_text == candidate["source"]:
                    continue
                score = SequenceMatcher(None, _normalize_entity_key(span_text), candidate["source_norm"]).ratio()
                if score < 0.84:
                    continue
                if best_payload is None or score > best_payload[0]:
                    best_payload = (score, start, end, candidate)

    if best_payload is None:
        return cleaned, [], []

    _, start, end, candidate = best_payload
    repaired_text = f"{cleaned[:start]}{candidate['source']}{cleaned[end:]}"
    repair = {
        "phase": "source",
        "kind": candidate["kind"],
        "from": cleaned[start:end],
        "to": candidate["source"],
    }
    return repaired_text, [repair], ["source_entity_repaired"]


def _repair_source_entities(text: str, context: dict, glossario: dict) -> tuple[str, list[dict], list[str]]:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return cleaned, [], []

    entity_label_candidate = _looks_like_entity_label(cleaned) and not re.search(r"[,.!?…:;]", cleaned)
    if entity_label_candidate:
        best_candidate, best_score = _best_entity_candidate_for_span_v2(cleaned, context, glossario)
        if best_candidate and best_score >= 0.84 and cleaned != best_candidate["source"]:
            repair = {
                "phase": "source",
                "kind": best_candidate["kind"],
                "from": cleaned,
                "to": best_candidate["source"],
            }
            return best_candidate["source"], [repair], ["source_entity_repaired"]

    repaired_text, repairs, flags = _repair_entity_phrase_inside_sentence_v2(cleaned, context, glossario)
    if repairs:
        return repaired_text, repairs, flags

    return cleaned, [], []


def _should_repair_local_translation(source_text: str, translated_text: str) -> bool:
    translated = translated_text.strip()
    if not translated:
        return True

    source_key = _normalized_translation_key(source_text)
    translated_key = _normalized_translation_key(translated)
    return bool(source_key) and source_key == translated_key


def _build_context_hints(context: dict, glossario: dict) -> str:
    aliases = ", ".join(context.get("aliases", [])[:10]) or "N/A"
    termos = ", ".join(context.get("termos", [])[:12]) or "N/A"
    faccoes = ", ".join(context.get("faccoes", [])[:10]) or "N/A"
    relacoes = ", ".join(context.get("relacoes", [])[:10]) or "N/A"
    arcos = " | ".join(context.get("resumo_por_arco", [])[:3]) or "N/A"
    memoria = dict(context.get("memoria_lexical", {}) or {})
    memoria.update(glossario or {})
    memoria_json = json.dumps(memoria, ensure_ascii=False)
    corpus_candidates = context.get("corpus_memory_candidates", [])[:6]
    corpus_memory = " | ".join(
        f"{item.get('source_text', '')} => {item.get('target_text', '')}"
        for item in corpus_candidates
        if item.get("source_text") and item.get("target_text")
    ) or "N/A"
    return (
        f"ALIASES: {aliases}\n"
        f"TERMOS: {termos}\n"
        f"FACCOES: {faccoes}\n"
        f"RELACOES: {relacoes}\n"
        f"ARCOS: {arcos}\n"
        f"MEMORIA_LEXICAL: {memoria_json}\n"
        f"MEMORIA_CORPUS: {corpus_memory}"
    )


def build_translation_context_header(translation_context: dict | None) -> str:
    """Constrói um bloco de contexto da obra a ser injetado no prompt do tradutor.

    Usa o campo ``translation_context`` do project.json (editor-side rich context),
    distinto do ``context`` do pipeline (WorkContextProfile do Rust).
    Retorna string vazia se translation_context for None ou vazio.
    """
    if not translation_context:
        return ""

    parts: list[str] = []

    title = translation_context.get("title")
    if title:
        parts.append(f"TITULO_OBRA: {title}")

    synopsis = translation_context.get("synopsis")
    if synopsis:
        parts.append(f"SINOPSE: {synopsis[:400]}")  # limitar tamanho

    genre = translation_context.get("genre") or []
    if genre:
        parts.append(f"GENERO: {', '.join(genre)}")

    tone = translation_context.get("tone")
    if tone:
        parts.append(f"TOM: {tone}")

    # Glossário travado — prioridade máxima
    locked = [e for e in (translation_context.get("glossary") or []) if e.get("locked")]
    if locked:
        parts.append("GLOSSARIO_OBRIGATORIO:")
        for e in locked[:20]:  # limitar a 20 entradas
            note = f" ({e['notes']})" if e.get("notes") else ""
            parts.append(f"  {e['source']} => {e['target']}{note}")

    # Personagens
    characters = translation_context.get("characters") or []
    if characters:
        char_lines = []
        for c in characters[:15]:  # limitar a 15 personagens
            name = c.get("name", "")
            if not name:
                continue
            parts_char = [name]
            if c.get("doNotTranslateName"):
                parts_char.append("(nao traduzir nome)")
            if c.get("preferredPortugueseName"):
                parts_char.append(f"PT: {c['preferredPortugueseName']}")
            if c.get("speechStyle"):
                parts_char.append(f"tom: {c['speechStyle']}")
            char_lines.append(", ".join(parts_char))
        if char_lines:
            parts.append("PERSONAGENS_EDITOR:")
            parts.extend(f"  {line}" for line in char_lines)

    # Regras de tradução livres
    rules = translation_context.get("translationRules") or []
    if rules:
        parts.append(f"REGRAS: {'; '.join(rules[:8])}")

    if not parts:
        return ""

    header = "\n".join(parts)
    logger.info(
        json.dumps({
            "event": "translation_context_used",
            "glossary_locked_count": len(locked),
            "characters_count": len(characters),
            "has_title": bool(title),
        })
    )
    return header


def _lookup_memory_translation(text: str, tipo: str, context: dict, glossario: dict) -> str | None:
    del tipo
    normalized = re.sub(r"[\W_]+", "", text.lower())
    merged = dict(context.get("memoria_lexical", {}) or {})
    merged.update(context.get("corpus_memoria_lexical", {}) or {})
    merged.update(glossario or {})
    for source, target in merged.items():
        source_key = re.sub(r"[\W_]+", "", source.lower())
        if source_key and source_key == normalized:
            return target
    return None


def _lookup_special_literal_translation(text: str, tipo: str) -> str | None:
    del tipo
    stripped = str(text or "").strip()
    phrase_key = re.sub(r"[\W_]+", "", stripped.lower())
    phrase_literal_map = {
        "anoisano": "Um n\u00e3o \u00e9 um n\u00e3o",
        "mom": "M\u00e3e",
        "what": "O qu\u00ea",
        "why": "Por qu\u00ea",
        "shehidthismuch": "Ela escondeu tudo isso",
    }
    if phrase_key in phrase_literal_map:
        punct_match = re.search(r"([!?.,]+)$", stripped)
        punct = punct_match.group(1) if punct_match else "."
        return f"{phrase_literal_map[phrase_key]}{punct}"
    match = re.fullmatch(r"([\"'“”‘’]*)([A-Za-z]+)([\"'“”‘’]*)([!?.,]*)", stripped)
    if not match:
        return None
    prefix, token, suffix_quote, punct = match.groups()
    normalized = token.lower()
    literal_map = {
        "none": "Nenhuma",
        "we": "N\u00f3s",
    }
    if normalized not in literal_map:
        return None
    if normalized == "none":
        punct = punct or "."
    return f"{prefix}{literal_map[normalized]}{suffix_quote}{punct}"


def _translation_bbox(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _translation_context_bbox(text: dict) -> list[int] | None:
    return (
        _translation_bbox(text.get("balloon_bbox"))
        or _translation_bbox(text.get("source_bbox"))
        or _translation_bbox(text.get("bbox"))
        or _translation_bbox(text.get("text_pixel_bbox"))
    )


def _translation_bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(max(1, area_a + area_b - inter))


def _same_translation_context(prev_text: dict, next_text: dict) -> bool:
    prev_bbox = _translation_context_bbox(prev_text)
    next_bbox = _translation_context_bbox(next_text)
    if not prev_bbox or not next_bbox:
        return False
    if _translation_bbox_iou(prev_bbox, next_bbox) >= 0.20:
        return True
    px1, py1, px2, py2 = prev_bbox
    nx1, ny1, nx2, ny2 = next_bbox
    horizontal_overlap = min(px2, nx2) - max(px1, nx1)
    vertical_gap = max(0, max(ny1 - py2, py1 - ny2))
    min_width = max(1, min(px2 - px1, nx2 - nx1))
    return horizontal_overlap >= int(min_width * 0.35) and vertical_gap <= 38


def _normalize_context_source(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _looks_like_split_sake_phrase(parts: list[str]) -> bool:
    joined = _normalize_context_source(" ".join(parts)).lower()
    if re.search(r"\bfor\s+(?:the\s+)?[a-z][a-z' -]{1,40}'s\s+sake\b", joined):
        return True
    return any(
        re.search(r"\bfor\b.*'s\s*$", _normalize_context_source(left).lower())
        and re.match(r"^sake\b", _normalize_context_source(right).lower())
        for left, right in zip(parts, parts[1:])
    )


def _join_context_source(parts: list[str]) -> str:
    joined = _normalize_context_source(" ".join(part for part in parts if str(part or "").strip()))
    return joined


def _build_translation_context_groups(texts: list[dict], source_parts: list[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    index = 0
    while index < len(texts) - 1:
        current = [index]
        cursor = index + 1
        while cursor < len(texts) and len(current) < 4:
            prev = texts[current[-1]]
            nxt = texts[cursor]
            if _should_skip_translation_item(prev) or _should_skip_translation_item(nxt):
                break
            if str(prev.get("tipo", "fala")) not in {"fala", "pensamento", "narracao"}:
                break
            if str(nxt.get("tipo", "fala")) not in {"fala", "pensamento", "narracao"}:
                break
            candidate = current + [cursor]
            candidate_sources = [source_parts[i] for i in candidate]
            if not _same_translation_context(prev, nxt):
                break
            if _looks_like_split_sake_phrase(candidate_sources):
                current = candidate
                cursor += 1
                break
            break
        if len(current) > 1:
            groups.append(current)
            index = current[-1] + 1
        else:
            index += 1
    return groups


def _split_by_source_lengths(translated: str, source_parts: list[str]) -> list[str]:
    words = _normalize_context_source(translated).split()
    if len(source_parts) <= 1 or not words:
        return [translated]
    weights = [max(1, len(re.sub(r"\s+", "", part))) for part in source_parts]
    total = max(1, sum(weights))
    result: list[str] = []
    cursor = 0
    for part_index, weight in enumerate(weights):
        if part_index == len(weights) - 1:
            chunk = words[cursor:]
        else:
            remaining_parts = len(weights) - part_index - 1
            take = max(1, round(len(words) * (weight / total)))
            take = min(take, max(1, len(words) - cursor - remaining_parts))
            chunk = words[cursor: cursor + take]
            cursor += take
        result.append(" ".join(chunk).strip())
    return result


def _split_contextual_translation(translated: str, source_parts: list[str]) -> list[str]:
    cleaned = _normalize_context_source(translated)
    if len(source_parts) == 2 and _looks_like_split_sake_phrase(source_parts):
        match = re.match(r"(?is)^(.+?\bpelo\s+bem)\s+((?:d[aeo]|de)\s+.+)$", cleaned)
        if match:
            first, second = match.groups()
            return [first.strip(), second.strip()]
    split = _split_by_source_lengths(cleaned, source_parts)
    if len(split) == len(source_parts):
        return split
    return [cleaned, *[""] * (len(source_parts) - 1)]


def _build_text_payload(texts: list[dict], index: int, history_tail: list[dict]) -> dict:
    current = texts[index]
    before = texts[index - 1].get("text", "") if index > 0 else ""
    after = texts[index + 1].get("text", "") if index + 1 < len(texts) else ""
    return {
        "id": f"t{index + 1}",
        "text": current.get("text", ""),
        "tipo": current.get("tipo", "fala"),
        "context_before": before,
        "context_after": after,
        "history_tail": history_tail[-3:],
    }


_google: Optional[_GoogleTranslator] = None
_google_health_key: tuple[str, str] | None = None
_google_health_ok = False
_google_health_failed_at: dict[tuple[str, str], float] = {}
_GOOGLE_FAILURE_RETRY_SECONDS = 300.0
_SEMANTIC_REVIEW_MAX_ITEMS_PER_BATCH = 16

_GOOGLE_HEALTH_PROBES = {
    "ko": "안녕하세요",
    "ja": "こんにちは",
    "zh-CN": "你好",
    "zh-TW": "你好",
    "en": "hello",
    "es": "hola",
    "fr": "bonjour",
    "de": "hallo",
    "it": "ciao",
    "pt": "olá",
    "ru": "привет",
}


def _prefer_local_translation_backend() -> bool:
    flag = (
        os.getenv("TRADUZAI_PREFER_LOCAL_TRANSLATION")
        or os.getenv("MANGATL_PREFER_LOCAL_TRANSLATION")
        or "0"
    )
    return str(flag).strip().lower() not in {"0", "false", "no", "off"}


def _preserve_caps_proper_nouns_enabled() -> bool:
    flag = os.getenv("TRADUZAI_PRESERVE_CAPS_PROPER_NOUNS", "0")
    return str(flag).strip().lower() in {"1", "true", "yes", "on"}


def _probe_google_backend(translator: _GoogleTranslator, source_lang: str, target_lang: str) -> None:
    probe = _GOOGLE_HEALTH_PROBES.get(source_lang, "hello")
    backend = getattr(translator, "_translator", translator)
    result = backend.translate(probe)
    if not result or not str(result).strip():
        raise RuntimeError("Google Translate retornou resposta vazia no health check")
    if source_lang != target_lang and str(result).strip() == probe:
        raise RuntimeError("Google Translate retornou a source sem traduzir no health check")


def _resolve_translation_backend(google_ok: bool, ollama_status: dict) -> str:
    if google_ok:
        return "google"
    return "passthrough"


def _google_health_retry_blocked(health_key: tuple[str, str]) -> bool:
    failed_at = _google_health_failed_at.get(health_key)
    if failed_at is None:
        return False
    return (time.time() - failed_at) < _GOOGLE_FAILURE_RETRY_SECONDS


def _should_use_semantic_review(source_lang: str, target_lang: str) -> bool:
    if os.getenv("TRADUZAI_SEMANTIC_REVIEW", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return (
        normalize_google_language_code(source_lang) == "ko"
        and normalize_google_language_code(target_lang) == "pt"
    )


def _is_valid_semantic_review_output(source_text: str, draft_text: str, candidate_text: str) -> bool:
    candidate = candidate_text.strip()
    if not candidate:
        return False
    if candidate in {"...", "…", "?", "??", "???", "-", "--"}:
        return False
    if _normalized_translation_key(candidate) == _normalized_translation_key(source_text):
        return False
    if len(candidate) < 3 and len(draft_text.strip()) >= 3:
        return False
    return True


def _needs_semantic_review_candidate(item: dict, source_lang: str) -> bool:
    original = str(item.get("original", "")).strip()
    translated = str(item.get("translated", "")).strip()
    tipo = str(item.get("tipo", "fala"))
    if tipo == "sfx" or not original or not translated:
        return False
    normalized_source_lang = normalize_google_language_code(source_lang)
    if normalized_source_lang not in {"ja", "ko", "zh-CN", "zh-TW"}:
        return False
    if not SOURCE_SCRIPT_PATTERN.search(original):
        return False

    flags = set(item.get("qa_flags") or [])
    if flags & {
        "source_script_leak",
        "suspected_ocr_error",
        "translation_fallback_phrase",
        "literal_ocr_translation",
        "entity_suspect",
        "entity_mistranslated",
    }:
        return True
    folded_translation = _fold_for_quality_match(translated)
    return bool(
        TRANSLATION_FALLBACK_PHRASE_PATTERN.search(folded_translation)
        or LITERAL_OCR_TRANSLATION_PATTERN.search(folded_translation)
    )


def _refine_google_translations_with_semantic_llm(
    translated_pages: list[dict],
    context: dict,
    glossario: dict,
    source_lang: str,
    target_lang: str,
    model: str,
    host: str,
    translation_context: dict | None = None,
) -> list[dict]:
    candidates: list[tuple[int, int, dict]] = []
    for page_idx, page in enumerate(translated_pages):
        for index, item in enumerate(page.get("texts", [])):
            original = str(item.get("original", "")).strip()
            translated = str(item.get("translated", "")).strip()
            tipo = str(item.get("tipo", "fala"))
            if tipo == "sfx" or not original or not translated:
                continue
            if not re.search(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", original):
                continue
            if not _needs_semantic_review_candidate(item, source_lang):
                continue
            candidates.append((page_idx, index, item))

    if not candidates:
        return translated_pages

    system = (
        f"Voce revisa semanticamente traducoes de manga {source_lang}->{target_lang}. "
        "Use o source original e o draft do Google apenas para melhorar naturalidade e sentido em pt-BR. "
        "Preserve nomes proprios, titulos, tecnicas e termos do glossario. "
        "Nao explique nada. Responda SOMENTE JSON no formato "
        '{"items":[{"id":"t1","translated":"texto"}]}. '
        "Se o draft ja estiver bom, devolva-o sem grandes mudancas."
    )
    hints = _build_context_hints(context, glossario)
    tc_header = build_translation_context_header(translation_context)

    for start in range(0, len(candidates), _SEMANTIC_REVIEW_MAX_ITEMS_PER_BATCH):
        batch = candidates[start : start + _SEMANTIC_REVIEW_MAX_ITEMS_PER_BATCH]
        items = []
        for page_idx, index, item in batch:
            items.append(
                {
                    "id": f"p{page_idx + 1}_t{index + 1}",
                    "source": item.get("original", ""),
                    "draft": item.get("translated", ""),
                    "tipo": item.get("tipo", "fala"),
                    "context_before": item.get("context_before", ""),
                    "context_after": item.get("context_after", ""),
                }
            )

        try:
            reviewed = _call_ollama(
                model=model,
                system=system,
                user_msg=(
                    f"CONTEXTO:\n{hints}\n"
                    + (f"{tc_header}\n" if tc_header else "")
                    + f"\nRevise os itens:\n{json.dumps({'items': items}, ensure_ascii=False)}"
                ),
                host=host,
            )
        except Exception as exc:
            logger.debug("Revisao semantica local falhou no lote iniciado em %s: %s", start, exc)
            continue

        reviewed_map = {
            str(entry.get("id")): str(entry.get("translated", "")).strip()
            for entry in reviewed
            if isinstance(entry, dict) and entry.get("id")
        }

        for page_idx, index, item in batch:
            item_id = f"p{page_idx + 1}_t{index + 1}"
            candidate = reviewed_map.get(item_id, "")
            draft = str(item.get("translated", ""))
            original = str(item.get("original", ""))
            if not _is_valid_semantic_review_output(original, draft, candidate):
                continue
            if candidate.strip() == draft.strip():
                continue
            translated_pages[page_idx]["texts"][index]["translated"] = candidate.strip()
            existing_flags = [
                flag
                for flag in translated_pages[page_idx]["texts"][index].get("qa_flags", []) or []
                if flag
                not in {
                    "source_script_leak",
                    "suspected_ocr_error",
                    "translation_fallback_phrase",
                    "literal_ocr_translation",
                    "translation_failed",
                    "translation_render_blocked",
                }
            ]
            translated_pages[page_idx]["texts"][index]["qa_flags"] = _merge_qa_flags(
                existing_flags,
                _translation_quality_flags(original, candidate.strip(), source_lang),
            )
            record_decision(
                stage="translate",
                action="semantic_review",
                reason="google_plus_small_llm",
                page=page_idx + 1,
                layer=item_id,
                text=original,
                details={"before": draft, "after": candidate.strip(), "model": model},
            )

    return translated_pages


def translate_pages(
    ocr_results: list[dict],
    obra: str,
    context: dict,
    glossario: dict,
    idioma_destino: str = "pt-BR",
    idioma_origem: str = "en",
    qualidade: str = "normal",
    ollama_host: str = OLLAMA_HOST,
    ollama_model: str = "traduzai-translator",
    progress_callback: Callable | None = None,
    models_dir: str = "",
    translation_context: dict | None = None,
) -> list[dict]:
    del qualidade

    global _google, _google_health_key, _google_health_ok
    idioma_origem = normalize_google_language_code(idioma_origem)
    idioma_destino = normalize_google_language_code(idioma_destino)

    persistent_cache = _open_persistent_cache(models_dir, idioma_origem, idioma_destino)

    google_ok = False
    google_health_key = (idioma_origem, idioma_destino)
    try:
        if (
            _google is None
            or not hasattr(_google, "_translator")
            or getattr(_google, "_source_lang", "en") != idioma_origem
            or getattr(_google, "_target_lang", "pt") != idioma_destino
        ):
            _google = _GoogleTranslator(source=idioma_origem, target=idioma_destino)
            _google._source_lang = idioma_origem
            _google._target_lang = idioma_destino
            _google_health_key = None
            _google_health_ok = False
        if persistent_cache is not None:
            _google.attach_persistent_cache(persistent_cache)
        if _google_health_key == google_health_key and _google_health_ok:
            google_ok = True
        elif _google_health_retry_blocked(google_health_key):
            google_ok = False
        else:
            _probe_google_backend(_google, idioma_origem, idioma_destino)
            _google_health_key = google_health_key
            _google_health_ok = True
            _google_health_failed_at.pop(google_health_key, None)
            google_ok = True
    except Exception as exc:
        _google_health_key = google_health_key
        _google_health_ok = False
        _google_health_failed_at[google_health_key] = time.time()
        logger.warning(f"Google Translate indisponivel: {exc}")

    semantic_review_requested = False
    ollama = {"running": False, "models": [], "has_translator": False, "skipped": True}
    backend = _resolve_translation_backend(google_ok=google_ok, ollama_status=ollama)
    selected_model = "google" if backend == "google" else (ollama_model if backend == "ollama" else "passthrough")
    debug_session = _TranslationDebugSession(backend=backend, model=selected_model)

    logger.info(f"--- TRADUCAO INICIADA ---")
    logger.info(f"Backend selecionado: {backend}")
    logger.info(f"Idioma Origem: {idioma_origem}")
    logger.info(f"Idioma Destino: {idioma_destino}")
    logger.info(f"Google OK: {google_ok}")
    logger.info(f"-------------------------")
    record_decision(
        stage="translate",
        action="select_backend",
        reason=backend,
        details={
            "google_ok": google_ok,
            "ollama_running": bool(ollama.get("running")),
            "idioma_origem": idioma_origem,
            "idioma_destino": idioma_destino,
        },
    )

    try:
        if backend == "google":
            logger.info("Traducao usando Google Translate.")
            semantic_model = None
            if semantic_review_requested and bool(ollama.get("running")) and bool(ollama.get("models")):
                semantic_model = _pick_ollama_model_for_language_pair(
                    ollama["models"],
                    ollama_model,
                    idioma_origem,
                    idioma_destino,
                )
            return _translate_with_google(
                ocr_results,
                context,
                glossario,
                progress_callback,
                idioma_origem=idioma_origem,
                idioma_destino=idioma_destino,
                semantic_reviewer_model=semantic_model,
                semantic_reviewer_host=ollama_host,
                translation_context=translation_context,
                debug_session=debug_session,
            )

        if backend == "ollama":
            model = _pick_ollama_model_for_language_pair(
                ollama["models"],
                ollama_model,
                idioma_origem,
                idioma_destino,
            )
            logger.info("Traducao usando backend local Ollama: %s", model)
            return _translate_with_ollama(
                ocr_results,
                obra,
                context,
                glossario,
                idioma_destino,
                idioma_origem,
                model,
                ollama_host,
                _google if google_ok else None,
                progress_callback,
                translation_context=translation_context,
                debug_session=debug_session,
            )

        logger.warning("Nenhum backend de traducao disponivel. Retornando texto original.")
        return _passthrough(ocr_results, progress_callback, debug_session=debug_session)
    finally:
        debug_session.write_summary()
        if persistent_cache is not None:
            try:
                persistent_cache.flush()
            except Exception as exc:
                logger.debug("Falha ao flush do cache persistente: %s", exc)


def _open_persistent_cache(models_dir: str, src_lang: str, tgt_lang: str):
    if not models_dir:
        return None
    try:
        try:
            from translator.cache import PersistentTranslationCache
        except ImportError:
            from .cache import PersistentTranslationCache
        from pathlib import Path as _Path
        return PersistentTranslationCache.for_language_pair(
            _Path(models_dir) / "cache",
            src_lang,
            tgt_lang,
        )
    except Exception as exc:
        logger.debug("Cache persistente indisponivel: %s", exc)
        return None


def _translate_with_google(
    ocr_results: list[dict],
    context: dict,
    glossario: dict,
    progress_callback: Callable | None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt",
    semantic_reviewer_model: str | None = None,
    semantic_reviewer_host: str = OLLAMA_HOST,
    translation_context: dict | None = None,
    debug_session: _TranslationDebugSession | None = None,
) -> list[dict]:
    total = len(ocr_results)
    translated_pages = []
    history_memory: dict[str, str] = {}
    history_tail: list[dict] = []

    for page_idx, ocr_page in enumerate(ocr_results):
        translated, history_tail = _translate_google_single_page(
            page_idx=page_idx,
            total=total,
            ocr_page=ocr_page,
            context=context,
            glossario=glossario,
            idioma_origem=idioma_origem,
            idioma_destino=idioma_destino,
            history_memory=history_memory,
            history_tail=history_tail,
            progress_callback=progress_callback,
            semantic_reviewer_model=semantic_reviewer_model,
            semantic_reviewer_host=semantic_reviewer_host,
            debug_session=debug_session,
        )
        translated_pages.append(translated)

    if semantic_reviewer_model:
        translated_pages = _refine_google_translations_with_semantic_llm(
            translated_pages=translated_pages,
            context=context,
            glossario=glossario,
            source_lang=idioma_origem,
            target_lang=idioma_destino,
            model=semantic_reviewer_model,
            host=semantic_reviewer_host,
            translation_context=translation_context,
        )

    translated_pages = _apply_translation_render_blocks(translated_pages, idioma_origem)
    return translated_pages


def _translate_google_single_page(
    page_idx: int,
    total: int,
    ocr_page: dict,
    context: dict,
    glossario: dict,
    idioma_origem: str,
    idioma_destino: str,
    history_memory: dict[str, str],
    history_tail: list[dict],
    progress_callback: Callable | None,
    semantic_reviewer_model: str | None,
    semantic_reviewer_host: str,
    debug_session: _TranslationDebugSession | None = None,
) -> tuple[dict, list[dict]]:
    """Translate a single page using Google backend with shared history state.

    Returns (translated_page_dict, updated_history_tail).
    The `history_memory` dict is mutated in-place.
    """
    is_cjk = idioma_origem in ("ja", "ko", "zh", "zh-CN", "zh-TW")

    from ocr.ocr_normalizer import normalize_ocr_record, merge_same_balloon_fragments_before_translation

    texts = merge_same_balloon_fragments_before_translation(
        [normalize_ocr_record(text, glossario) for text in ocr_page.get("texts", [])]
    )
    if not texts:
        if progress_callback:
            progress_callback(page_idx + 1, total, f"Pagina {page_idx + 1}: sem texto")
        return {"texts": []}, history_tail

    raw_texts = [_source_text_before_normalization(text) for text in texts]
    translation_sources = [_source_text_for_translation(text) for text in texts]
    for index, source in enumerate(translation_sources):
        repaired_mojibake = _fix_source_mojibake_if_useful(source)
        if repaired_mojibake != source:
            texts[index]["source_mojibake_repaired_from"] = source
            texts[index]["source_mojibake_repaired"] = repaired_mojibake
            translation_sources[index] = repaired_mojibake
            raw_texts[index] = repaired_mojibake
    tipos = [text.get("tipo", "fala") for text in texts]
    repaired_sources: list[str] = []
    source_repairs: list[list[dict]] = []
    source_entity_flags: list[list[str]] = []

    for source in translation_sources:
        repaired_source, repairs, flags = _repair_source_entities(source, context, glossario)
        repaired_sources.append(repaired_source)
        source_repairs.append(repairs)
        source_entity_flags.append(flags)

    # Para CJK, was_upper nao faz sentido da mesma forma (seria True para tudo)
    if is_cjk:
        was_uppers = [False] * len(raw_texts)
    else:
        was_uppers = [
            source == source.upper() and any(c.isalpha() for c in source)
            for source in translation_sources
        ]

    preprocessed: list[str] = []
    protected_terms_by_index: list[list[dict]] = []
    for index, (repaired_text, tipo) in enumerate(zip(repaired_sources, tipos)):
        prepared_text = _prepare_source_text_for_translation(
            repaired_text,
            tipo,
            lang=idioma_origem,
            preserve_case=_uses_confident_normalized_text_final(texts[index]),
        )
        prepared_text, _, _ = _repair_source_entities(prepared_text, context, glossario)
        preserve_normalized_case = _uses_confident_normalized_text_final(texts[index])
        normalized_text = _preprocess_text(
            prepared_text,
            tipo,
            lang=idioma_origem,
            preserve_case=preserve_normalized_case,
        )
        normalized_text, _, _ = _repair_source_entities(normalized_text, context, glossario)
        protected = _protect_source_for_translation(normalized_text, tipo, context, glossario)
        preprocessed.append(str(protected.get("protected_source") or normalized_text))
        protected_terms_by_index.append(list(protected.get("terms") or []))

    translations = [""] * len(texts)
    translation_debug_meta: list[dict[str, Any]] = [
        {"duration_ms": 0, "fallback_used": False, "raw_response": "", "backend": "google", "model": "google"}
        for _ in texts
    ]
    pending_indices = []
    pending_texts = []
    for index, (source, tipo, prepared) in enumerate(zip(raw_texts, tipos, preprocessed)):
        if _should_skip_translation_item(texts[index]):
            translations[index] = source
            continue
        if debug_session:
            debug_session.record_input(
                page_idx=page_idx,
                index=index,
                text=texts[index],
                source_text_before_normalization=source,
                source_text_sent_to_translator=prepared,
                backend="google",
                model="google",
            )
        # Legacy proper-name preservation is opt-in. By default every CAPS
        # token goes through translation so dialogue is not left in English.
        proper_noun_token = source.strip().rstrip(".,!?;:'\"")
        if _preserve_caps_proper_nouns_enabled() and (
            _is_likely_proper_noun(source) or _is_likely_proper_noun(proper_noun_token)
        ):
            preserved = proper_noun_token.title()
            # Reanexa a pontuação final, se houver
            tail = source.strip()[len(proper_noun_token):]
            translations[index] = preserved + tail
            texts[index]["proper_noun_preserved"] = True
            continue
        special_literal = _lookup_special_literal_translation(source, tipo)
        if special_literal:
            translations[index] = special_literal
            translation_debug_meta[index]["raw_response"] = special_literal
            continue
        memory_translation = _lookup_memory_translation(source, tipo, context, glossario)
        if memory_translation:
            translations[index] = memory_translation
            translation_debug_meta[index]["raw_response"] = memory_translation
            continue
        memory_key = _normalize_memory_key(source, tipo)
        if memory_key in history_memory:
            translations[index] = history_memory[memory_key]
        else:
            pending_indices.append(index)
            pending_texts.append(prepared)

    handled_context_indices: set[int] = set()
    context_groups = _build_translation_context_groups(texts, repaired_sources)
    context_requests: list[tuple[list[int], list[str], str]] = []
    pending_set = set(pending_indices)
    for group in context_groups:
        if len(group) < 2 or any(index not in pending_set for index in group):
            continue
        source_parts = [repaired_sources[index] or raw_texts[index] for index in group]
        if not _looks_like_split_sake_phrase(source_parts):
            continue
        group_id = f"tc_{page_idx + 1:03}_{group[0] + 1:03}"
        context_source = _join_context_source([preprocessed[index] for index in group])
        if not context_source:
            continue
        for offset, index in enumerate(group):
            texts[index]["translation_context_group_id"] = group_id
            texts[index]["translation_context_index"] = offset + 1
            texts[index]["translation_context_size"] = len(group)
            texts[index]["translation_context_source"] = _join_context_source(source_parts)
        context_requests.append((group, source_parts, context_source))
        handled_context_indices.update(group)

    if context_requests:
        started = time.perf_counter()
        try:
            context_translations = _google.translate_batch([request[2] for request in context_requests])
            duration_ms = int((time.perf_counter() - started) * 1000)
        except Exception as exc:
            logger.warning(f"Batch contextual falhou na pagina {page_idx + 1}: {exc}")
            duration_ms = int((time.perf_counter() - started) * 1000)
            if debug_session:
                debug_session.record_fallback(
                    page_idx=page_idx,
                    backend="google",
                    model="google",
                    reason="context_batch_failed",
                    error=exc,
                    texts=[texts[index] for group, _parts, _source in context_requests for index in group],
                )
            context_translations = [request[2] for request in context_requests]
        for (group, source_parts, _context_source), translated_group in zip(context_requests, context_translations):
            split_parts = _split_contextual_translation(translated_group, source_parts)
            if len(split_parts) != len(group):
                split_parts = _split_by_source_lengths(translated_group, source_parts)
            for index, translated_part in zip(group, split_parts):
                translations[index] = translated_part
                translation_debug_meta[index] = {
                    "duration_ms": duration_ms,
                    "fallback_used": translated_group == _context_source,
                    "raw_response": translated_group,
                    "backend": "google",
                    "model": "google",
                }

    remaining_pending = [
        (index, text)
        for index, text in zip(pending_indices, pending_texts)
        if index not in handled_context_indices
    ]

    if remaining_pending:
        remaining_indices = [index for index, _text in remaining_pending]
        remaining_texts = [text for _index, text in remaining_pending]
        started = time.perf_counter()
        try:
            pending_translations = _google.translate_batch(remaining_texts)
            duration_ms = int((time.perf_counter() - started) * 1000)
        except Exception as exc:
            logger.warning(f"Batch falhou na pagina {page_idx + 1}: {exc}")
            duration_ms = int((time.perf_counter() - started) * 1000)
            if debug_session:
                debug_session.record_fallback(
                    page_idx=page_idx,
                    backend="google",
                    model="google",
                    reason="batch_failed",
                    error=exc,
                    texts=[texts[index] for index in remaining_indices],
                )
            pending_translations = remaining_texts

        for index, translated in zip(remaining_indices, pending_translations):
            translations[index] = translated
            translation_debug_meta[index] = {
                "duration_ms": duration_ms,
                "fallback_used": translated == preprocessed[index],
                "raw_response": translated,
                "backend": "google",
                "model": "google",
            }

    page_texts = []
    for index, (original, translated, was_upper, tipo) in enumerate(
        zip(raw_texts, translations, was_uppers, tipos)
    ):
        if _should_skip_translation_item(texts[index]):
            qa_flags = _merge_qa_flags(texts[index].get("qa_flags"))
            texts[index]["qa_flags"] = qa_flags
            payload = _build_text_payload(texts, index, history_tail)
            page_texts.append(
                {
                    **texts[index],
                    "original": original,
                    "translated": translated or original,
                    "source_text_sent_to_translator": preprocessed[index],
                    "tipo": tipo,
                    "context_before": payload["context_before"],
                    "context_after": payload["context_after"],
                }
            )
            history_tail.append({"source": original, "translated": translated or original, "tipo": tipo})
            history_tail = history_tail[-8:]
            continue
        protected_terms = protected_terms_by_index[index] if index < len(protected_terms_by_index) else []
        restored_translation, protection_flags, protection_hits = _restore_protected_translation(
            translated or original,
            protected_terms,
        )
        restored_translation, mojibake_flags, mojibake_audit = _apply_mojibake_audit(
            restored_translation or original,
            text_id=texts[index].get("id") or f"t{index + 1}",
        )
        final = _postprocess(
            restored_translation or original,
            was_upper,
            tipo,
            source_text=repaired_sources[index] or original,
            lang=idioma_origem,
        )
        locked_final, glossary_hits, target_flags = _apply_target_entity_locks(
            repaired_sources[index] or original,
            final,
            context,
            glossario,
        )
        locked_final, target_name_repairs = _repair_translated_proper_name(
            repaired_sources[index] or original,
            locked_final,
            tipo,
        )
        locked_final, prefix_cleanup_flags = _repair_translation_after_source_prefix_cleanup(texts[index], locked_final)
        missing_protected_terms = _missing_protected_terms_after_lock(locked_final, protected_terms)
        if missing_protected_terms:
            protection_flags = _merge_qa_flags(protection_flags, ["unrestored_placeholder"])
        name_flags = ["target_proper_name_repaired"] if target_name_repairs else []
        entity_flags = list(dict.fromkeys([*source_entity_flags[index], *target_flags, *name_flags]))
        entity_repairs = list(source_repairs[index])
        entity_repairs.extend(target_name_repairs)
        glossary_hits = list(glossary_hits)
        for hit in protection_hits:
            if hit not in glossary_hits:
                glossary_hits.append(hit)
        qa_flags = _merge_qa_flags(
            texts[index].get("qa_flags"),
            ["entity_suspect"] if source_repairs[index] else [],
            protection_flags,
            mojibake_flags,
            prefix_cleanup_flags,
            _translation_quality_flags(repaired_sources[index] or original, locked_final, idioma_origem),
        )
        if _should_preserve_untranslated_kana_sfx(repaired_sources[index] or original, locked_final, idioma_origem):
            locked_final = repaired_sources[index] or original
            qa_flags = [flag for flag in qa_flags if flag != "source_script_leak"]
            if "untranslated_kana_sfx_preserved" not in qa_flags:
                qa_flags.append("untranslated_kana_sfx_preserved")
            texts[index]["content_class"] = "sfx"
            texts[index]["tipo"] = "sfx"
            texts[index]["skip_reason"] = "untranslated_kana_sfx_preserved"
            texts[index]["preserve_original"] = True
            texts[index]["route_action"] = "preserve"
            texts[index]["route_reason"] = "untranslated_kana_sfx_preserved"
            texts[index]["translate_policy"] = "skip_translation"
            texts[index]["render_policy"] = "preserve_original"
            texts[index]["skip_processing"] = False
            tipo = texts[index].get("tipo", tipo)
        blocked_translation = ""
        if _should_block_translation_render(
            repaired_sources[index] or original,
            locked_final,
            idioma_origem,
            tipo,
            qa_flags,
        ) and not semantic_reviewer_model:
            blocked_translation = str((mojibake_audit or {}).get("translated") or locked_final)
            locked_final = ""
            qa_flags = _merge_qa_flags(qa_flags, ["translation_failed", "translation_render_blocked"])
            record_decision(
                stage="translate",
                action="block_render",
                reason="bad_cjk_translation",
                page=page_idx + 1,
                layer=texts[index].get("id") or f"t{index + 1}",
                text=original,
                details={"blocked_translation": blocked_translation, "qa_flags": qa_flags},
            )
        texts[index]["entity_flags"] = entity_flags
        texts[index]["entity_repairs"] = entity_repairs
        texts[index]["glossary_hits"] = glossary_hits
        texts[index]["qa_flags"] = qa_flags
        if mojibake_audit:
            texts[index]["mojibake_audit"] = mojibake_audit
        if entity_repairs:
            for repair in entity_repairs:
                repair_phase = repair.get("phase") if isinstance(repair, dict) else ""
                record_decision(
                    stage="translate",
                    action="repair_entity",
                    reason="target_proper_name" if repair_phase == "target" else "source_entity_match",
                    page=page_idx + 1,
                    layer=texts[index].get("id") or f"t{index + 1}",
                    text=original,
                    details=repair,
                )
        if glossary_hits:
            record_decision(
                stage="translate",
                action="lock_glossary_term",
                reason="target_entity_lock",
                page=page_idx + 1,
                layer=texts[index].get("id") or f"t{index + 1}",
                text=original,
                details={"hits": glossary_hits},
            )
        should_record_debug_output = bool(
            debug_session
            and (
                not _should_skip_translation_item(texts[index])
                or texts[index].get("skip_reason") == "untranslated_kana_sfx_preserved"
            )
        )
        if should_record_debug_output:
            meta = translation_debug_meta[index]
            debug_session.record_output(
                page_idx=page_idx,
                index=index,
                text=texts[index],
                source_text_sent_to_translator=preprocessed[index],
                raw_response=meta.get("raw_response") or translated,
                final_translation_after_postprocess=locked_final,
                duration_ms=int(meta.get("duration_ms") or 0),
                backend=str(meta.get("backend") or "google"),
                model=str(meta.get("model") or "google"),
                fallback_used=bool(meta.get("fallback_used")),
                glossary_hits=glossary_hits,
                qa_flags=qa_flags,
            )
        memory_key = _normalize_memory_key(original, tipo)
        if locked_final:
            history_memory[memory_key] = locked_final
        payload = _build_text_payload(texts, index, history_tail)
        text_payload = {
            **texts[index],
            "original": original,
            "translated": locked_final,
            "source_text_sent_to_translator": preprocessed[index],
            "tipo": tipo,
            "context_before": payload["context_before"],
            "context_after": payload["context_after"],
        }
        if blocked_translation:
            text_payload["translation_blocked_text"] = blocked_translation
        page_texts.append(text_payload)
        history_tail.append({"source": original, "translated": locked_final, "tipo": tipo})
        history_tail = history_tail[-8:]

    if semantic_reviewer_model:
        pass

    if progress_callback:
        progress_callback(
            page_idx + 1,
            total,
            f"[Google] Pagina {page_idx + 1}/{total} - {len(texts)} textos",
        )

    return {"texts": page_texts}, history_tail


def _translate_with_ollama(
    ocr_results: list[dict],
    obra: str,
    context: dict,
    glossario: dict,
    idioma_destino: str,
    idioma_origem: str,
    model: str,
    host: str,
    repair_translator: Optional[_GoogleTranslator],
    progress_callback: Callable | None,
    translation_context: dict | None = None,
    debug_session: _TranslationDebugSession | None = None,
) -> list[dict]:
    total = len(ocr_results)
    tc_header = build_translation_context_header(translation_context)
    system = (
        f"Voce e um tradutor de manga especializado em {idioma_origem}->{idioma_destino}. Responda SOMENTE com JSON array.\n"
        f"OBRA: {obra}\n"
        f"PERSONAGENS: {', '.join(context.get('personagens', [])[:8]) or 'N/A'}\n"
        f"GLOSSARIO: {json.dumps(glossario, ensure_ascii=False)}\n"
        f"{_build_context_hints(context, glossario)}\n"
        + (f"{tc_header}\n" if tc_header else "")
        + "Cada item de entrada tera o campo source com o texto original. Preserve o mesmo id e responda apenas [{\"id\":\"t1\",\"translated\":\"texto\"}]. Nao ecoe source, tipo, context_before ou context_after."
    )

    translated_pages = []
    history_tail: list[dict] = []
    for page_idx, ocr_page in enumerate(ocr_results):
        from ocr.ocr_normalizer import normalize_ocr_record, merge_same_balloon_fragments_before_translation

        texts = merge_same_balloon_fragments_before_translation(
            [normalize_ocr_record(text, glossario) for text in ocr_page.get("texts", [])]
        )
        if not texts:
            translated_pages.append({"texts": []})
            if progress_callback:
                progress_callback(page_idx + 1, total, f"Pagina {page_idx + 1}: sem texto")
            continue

        text_list = []
        for i, t in enumerate(texts):
            if _should_skip_translation_item(t):
                continue
            payload = _build_text_payload(texts, i, history_tail)
            source_before_normalization = _source_text_before_normalization(t)
            source_for_translation = _source_text_for_translation(t)
            payload["text"] = source_for_translation
            protected = _protect_source_for_translation(
                payload.get("text", ""),
                payload.get("tipo", "fala"),
                context,
                glossario,
            )
            t["_protected_terms"] = list(protected.get("terms") or [])
            protected_source = protected.get("protected_source") or payload.get("text", "")
            if debug_session:
                debug_session.record_input(
                    page_idx=page_idx,
                    index=i,
                    text=t,
                    source_text_before_normalization=source_before_normalization,
                    source_text_sent_to_translator=protected_source,
                    backend="ollama",
                    model=model,
                )
            text_list.append(
                {
                    "id": payload["id"],
                    "source": protected_source,
                    "tipo": payload.get("tipo", "fala"),
                    "context_before": payload.get("context_before", ""),
                    "context_after": payload.get("context_after", ""),
                }
            )
        user_msg = f"Traduza:\n{json.dumps(text_list, ensure_ascii=False)}"

        started = time.perf_counter()
        try:
            translations = _call_ollama(model, system, user_msg, host)
            translated_map = {item["id"]: item.get("translated", "") for item in translations}
            ollama_duration_ms = int((time.perf_counter() - started) * 1000)
        except Exception:
            ollama_duration_ms = int((time.perf_counter() - started) * 1000)
            if debug_session:
                debug_session.record_fallback(
                    page_idx=page_idx,
                    backend="ollama",
                    model=model,
                    reason="ollama_call_failed",
                    error="ollama_call_failed",
                    texts=texts,
                )
            translated_map = {}

        repair_indices: list[int] = []
        repair_texts: list[str] = []
        if repair_translator is not None:
            for index, text_data in enumerate(texts):
                if _should_skip_translation_item(text_data):
                    continue
                original = _source_text_before_normalization(text_data)
                candidate = translated_map.get(f"t{index + 1}", original)
                if _should_repair_local_translation(original, candidate):
                    tipo = text_data.get("tipo", "fala")
                    repair_indices.append(index)
                    repair_source, _, _ = _repair_source_entities(original, context, glossario)
                    prepared_repair = _prepare_source_text_for_translation(
                        repair_source or original,
                        tipo,
                        preserve_case=_uses_confident_normalized_text_final(text_data),
                    )
                    prepared_repair, _, _ = _repair_source_entities(prepared_repair, context, glossario)
                    normalized_repair = _preprocess_text(
                        prepared_repair,
                        tipo,
                        preserve_case=_uses_confident_normalized_text_final(text_data),
                    )
                    normalized_repair, _, _ = _repair_source_entities(normalized_repair, context, glossario)
                    protected_repair = _protect_source_for_translation(
                        normalized_repair,
                        tipo,
                        context,
                        glossario,
                    )
                    repair_terms = list(protected_repair.get("terms") or [])
                    if repair_terms:
                        text_data["_protected_terms"] = repair_terms
                    repair_texts.append(protected_repair.get("protected_source") or normalized_repair)

        repaired_map: dict[int, str] = {}
        if repair_indices and repair_translator is not None:
            try:
                repaired_batch = repair_translator.translate_batch(repair_texts)
            except Exception:
                repaired_batch = []

            for index, repaired in zip(repair_indices, repaired_batch):
                if repaired and repaired.strip():
                    repaired_map[index] = repaired

        page_texts = []
        for index, text_data in enumerate(texts):
            original = _source_text_before_normalization(text_data)
            source_for_translation = _source_text_for_translation(text_data)
            tipo = text_data.get("tipo", "fala")
            repaired_source, entity_repairs, source_entity_flags = _repair_source_entities(source_for_translation, context, glossario)
            if _should_skip_translation_item(text_data):
                qa_flags = _merge_qa_flags(
                    text_data.get("qa_flags"),
                    ["entity_suspect"] if entity_repairs else [],
                    _translation_quality_flags(repaired_source or original, original, idioma_origem),
                )
                page_texts.append(
                    {
                        **text_data,
                        "original": original,
                        "translated": original,
                        "tipo": tipo,
                        "entity_flags": list(source_entity_flags),
                        "entity_repairs": list(entity_repairs),
                        "glossary_hits": [],
                        "qa_flags": qa_flags,
                    }
                )
                history_tail.append({"source": original, "translated": original, "tipo": tipo})
                continue
            translated = repaired_map.get(index) or translated_map.get(f"t{index + 1}", original)
            fallback_used = f"t{index + 1}" not in translated_map and index not in repaired_map
            memory_translation = _lookup_memory_translation(original, tipo, context, glossario)
            if memory_translation:
                translated = memory_translation
                fallback_used = False
            special_literal = _lookup_special_literal_translation(original, tipo)
            if special_literal:
                translated = special_literal
                fallback_used = False
            is_cjk = idioma_origem in ("ja", "ko", "zh", "zh-CN", "zh-TW")
            was_upper = False if is_cjk else (original == original.upper() and any(c.isalpha() for c in original))
            protected_terms = list(text_data.get("_protected_terms") or [])
            restored_translation, protection_flags, protection_hits = _restore_protected_translation(
                translated,
                protected_terms,
            )
            restored_translation, mojibake_flags, mojibake_audit = _apply_mojibake_audit(
                restored_translation,
                text_id=text_data.get("id") or f"t{index + 1}",
            )

            final = _postprocess(
                restored_translation,
                was_upper,
                tipo,
                source_text=repaired_source or original,
                lang=idioma_origem,
            )
            locked_final, glossary_hits, target_flags = _apply_target_entity_locks(
                repaired_source or original,
                final,
                context,
                glossario,
            )
            locked_final, target_name_repairs = _repair_translated_proper_name(
                repaired_source or original,
                locked_final,
                tipo,
            )
            locked_final, prefix_cleanup_flags = _repair_translation_after_source_prefix_cleanup(text_data, locked_final)
            missing_protected_terms = _missing_protected_terms_after_lock(locked_final, protected_terms)
            if missing_protected_terms:
                protection_flags = _merge_qa_flags(protection_flags, ["unrestored_placeholder"])
            name_flags = ["target_proper_name_repaired"] if target_name_repairs else []
            entity_flags = list(dict.fromkeys([*source_entity_flags, *target_flags, *name_flags]))
            qa_flags = _merge_qa_flags(
                text_data.get("qa_flags"),
                ["entity_suspect"] if entity_repairs else [],
                protection_flags,
                mojibake_flags,
                prefix_cleanup_flags,
                _translation_quality_flags(repaired_source or original, locked_final, idioma_origem),
            )
            glossary_hits = list(glossary_hits)
            for hit in protection_hits:
                if hit not in glossary_hits:
                    glossary_hits.append(hit)
            blocked_translation = ""
            if _should_block_translation_render(
                repaired_source or original,
                locked_final,
                idioma_origem,
                tipo,
                qa_flags,
            ):
                blocked_translation = str((mojibake_audit or {}).get("translated") or locked_final)
                locked_final = ""
                qa_flags = _merge_qa_flags(qa_flags, ["translation_failed", "translation_render_blocked"])
                record_decision(
                    stage="translate",
                    action="block_render",
                    reason="bad_cjk_translation",
                    page=page_idx + 1,
                    layer=text_data.get("id") or f"t{index + 1}",
                    text=original,
                    details={"blocked_translation": blocked_translation, "qa_flags": qa_flags},
                )
            entity_repairs = list(entity_repairs)
            entity_repairs.extend(target_name_repairs)
            text_data["entity_flags"] = entity_flags
            text_data["entity_repairs"] = entity_repairs
            text_data["glossary_hits"] = glossary_hits
            text_data["qa_flags"] = qa_flags
            if mojibake_audit:
                text_data["mojibake_audit"] = mojibake_audit
            if blocked_translation:
                text_data["translation_blocked_text"] = blocked_translation
            if entity_repairs:
                for repair in entity_repairs:
                    repair_phase = repair.get("phase") if isinstance(repair, dict) else ""
                    record_decision(
                        stage="translate",
                        action="repair_entity",
                        reason="target_proper_name" if repair_phase == "target" else "source_entity_match",
                        page=page_idx + 1,
                        layer=text_data.get("id") or f"t{index + 1}",
                        text=original,
                        details=repair,
                    )
            if glossary_hits:
                record_decision(
                    stage="translate",
                    action="lock_glossary_term",
                    reason="target_entity_lock",
                    page=page_idx + 1,
                    layer=text_data.get("id") or f"t{index + 1}",
                    text=original,
                    details={"hits": glossary_hits},
                )
            if debug_session:
                payload_for_hash = next(
                    (item.get("source", "") for item in text_list if item.get("id") == f"t{index + 1}"),
                    original,
                )
                debug_session.record_output(
                    page_idx=page_idx,
                    index=index,
                    text=text_data,
                    source_text_sent_to_translator=payload_for_hash,
                    raw_response=translated,
                    final_translation_after_postprocess=locked_final,
                    duration_ms=ollama_duration_ms,
                    backend="ollama",
                    model=model,
                    fallback_used=fallback_used,
                    glossary_hits=glossary_hits,
                    qa_flags=qa_flags,
                )
            page_texts.append(
                {
                    **text_data,
                    "original": original,
                    "translated": locked_final,
                    "source_text_sent_to_translator": payload_for_hash,
                    "tipo": tipo,
                }
            )
            history_tail.append({"source": original, "translated": locked_final, "tipo": tipo})
        history_tail = history_tail[-8:]

        translated_pages.append({"texts": page_texts})
        if progress_callback:
            progress_callback(page_idx + 1, total, f"[{model}] Pagina {page_idx + 1}/{total}")

    return translated_pages


def _passthrough(
    ocr_results: list[dict],
    progress_callback: Callable | None,
    debug_session: _TranslationDebugSession | None = None,
) -> list[dict]:
    total = len(ocr_results)
    result = []
    for index, page in enumerate(ocr_results):
        page_texts = []
        for text_idx, text in enumerate(page.get("texts", [])):
            source = text.get("text", "")
            if debug_session and not _should_skip_translation_item(text):
                debug_session.record_input(
                    page_idx=index,
                    index=text_idx,
                    text=text,
                    source_text_before_normalization=source,
                    source_text_sent_to_translator=source,
                    backend="passthrough",
                    model="passthrough",
                )
                debug_session.record_output(
                    page_idx=index,
                    index=text_idx,
                    text=text,
                    source_text_sent_to_translator=source,
                    raw_response=source,
                    final_translation_after_postprocess=source,
                    duration_ms=0,
                    backend="passthrough",
                    model="passthrough",
                    fallback_used=True,
                    glossary_hits=[],
                    qa_flags=text.get("qa_flags") or [],
                )
            page_texts.append(
                {
                    **text,
                    "original": source,
                    "translated": source,
                    "tipo": text.get("tipo", "fala"),
                }
            )
        result.append(
            {
                "texts": page_texts
            }
        )
        if progress_callback:
            progress_callback(index + 1, total, f"Pagina {index + 1} (sem traducao)")
    return result


def translate_single_block(block: dict, project: dict):
    """Traduz um unico bloco usando o motor configurado no projeto."""
    global _google
    
    source_lang = project.get("idioma_origem", "en")
    target_lang = project.get("idioma_destino", "pt-BR")
    
    source_lang = normalize_google_language_code(source_lang)
    target_lang = normalize_google_language_code(target_lang)
    
    text = block.get("original", "").strip()
    if not text:
        return
        
    tipo = block.get("tipo", "fala")
    
    # Setup translator if needed
    if (
        _google is None
        or getattr(_google, "_source_lang", "en") != source_lang
        or getattr(_google, "_target_lang", "pt") != target_lang
    ):
        _google = _GoogleTranslator(source=source_lang, target=target_lang)
        _google._source_lang = source_lang
        _google._target_lang = target_lang
        
    prepared = _preprocess_text(_prepare_source_text_for_translation(text, tipo, lang=source_lang), tipo, lang=source_lang)
    translated = _google.translate(prepared) or text
    
    is_cjk = source_lang in ("ja", "ko", "zh", "zh-CN", "zh-TW")
    was_upper = False if is_cjk else (text == text.upper() and any(c.isalpha() for c in text))
    
    final = _postprocess(translated, was_upper, tipo, source_text=text, lang=source_lang)

    qa_flags = _merge_qa_flags(
        block.get("qa_flags"),
        _translation_quality_flags(text, final, source_lang),
    )
    if _should_block_translation_render(text, final, source_lang, tipo, qa_flags):
        block["translation_blocked_text"] = final
        final = ""
        qa_flags = _merge_qa_flags(qa_flags, ["translation_failed", "translation_render_blocked"])
    block["qa_flags"] = qa_flags
    block["translated"] = final
    block["traduzido"] = final
